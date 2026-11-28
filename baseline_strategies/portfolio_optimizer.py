"""3号 baseline_strategies：三资产基准组合优化模块（第二阶段整改版）。

模块边界（本次整改后）
1. 上游 ``data_processing.data_interface.get_returns_window`` 只负责按 ``as_of_date``
   返回最近 250 个共同交易日的**收益率窗口**；
2. 250 日窗口内的**样本协方差矩阵在本模块内计算**（整改要求 4），不再由上游代算；
3. 本模块负责：资产池强校验、协方差奇异/零矩阵防护、三种基准策略求解、
   风险平价的风险贡献后验校验。

统一返回格式（三种策略完全一致）
    {
        "strategy": "equal_weight" | "min_variance" | "risk_parity",
        "weight": {代码: 权重},                    # Σw = 1，w >= 0
        "risk_contribution": {代码: rc_i},          # 绝对风险贡献 w_i·(Σw)_i，合计 = 组合方差
        "risk_contribution_ratio": {代码: 占比},    # 风险贡献占比，合计 = 1
        "risk_contribution_max_deviation": float,   # max|占比 - 1/3|
        "risk_budget_tolerance": float | None,      # 仅风险平价给容差 0.005
        "risk_budget_pass": bool | None,            # 仅风险平价给校验结果
        "portfolio_variance": float,
        "covariance_shrinkage": {...},              # 协方差收缩诊断信息
    }
``run_baseline_250d`` 在此基础上追加 ``as_of_date`` / ``window_days`` /
``sample_days`` / ``asset_pool`` 四个字段。

命令行示例（仓库根目录执行）
    python -m baseline_strategies.portfolio_optimizer --as-of 2026-11-20 --strategy risk_parity
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
ASSET_POOL = ["510300", "511010", "518880"]  # 资产池：顺序、数量均不可变
N_ASSETS = len(ASSET_POOL)
WINDOW_DAYS = 250                              # 正式窗口长度（交易日）
RC_TOLERANCE = 0.005                           # 风险贡献占比相对 1/3 的最大绝对偏差容差
PD_ABS_FLOOR = 1e-14                           # 正定判定绝对下界
PD_REL_FLOOR = 1e-10                           # 正定判定相对下界（相对平均方差）
SHRINK_GRID = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
               0.1, 0.2, 0.3, 0.5, 0.75, 1.0)   # 协方差收缩强度搜索网格


class RiskBudgetError(RuntimeError):
    """风险平价后验校验失败：求解返回 success，但风险贡献偏差超过容差，禁止返回权重。"""


# --------------------------------------------------------------------------- #
# 一、资产池强校验（整改要求 2）
# --------------------------------------------------------------------------- #
def validate_asset_pool(tickers) -> list[str]:
    """资产池必须严格等于 ["510300", "511010", "518880"]（数量、顺序完全一致）。

    标的缺失、多出、重复、顺序错、非序列（含字符串/None）一律抛 ValueError。
    """
    if isinstance(tickers, (str, bytes)) or not isinstance(tickers, (list, tuple)):
        raise ValueError(f"资产池必须是列表/元组，实际类型 {type(tickers).__name__}")
    codes = list(tickers)
    if any(not isinstance(code, str) for code in codes):
        raise ValueError(f"资产池元素必须为字符串代码，实际 {codes}")
    if len(set(codes)) != len(codes):
        raise ValueError(f"资产池不允许重复标的，实际 {codes}")
    if len(codes) != N_ASSETS:
        raise ValueError(f"资产池标的数量必须为 {N_ASSETS}（{ASSET_POOL}），实际 {len(codes)} 个：{codes}")
    if codes != ASSET_POOL:
        raise ValueError(f"资产池顺序必须严格为 {ASSET_POOL}，实际 {codes}")
    return list(ASSET_POOL)


# --------------------------------------------------------------------------- #
# 二、协方差校验与奇异防护（整改要求 2、3）
# --------------------------------------------------------------------------- #
def _pd_floor(mean_variance: float) -> float:
    """正定判定下界：绝对下界与相对下界取大，避免不同量纲下判定失真。"""
    return max(PD_ABS_FLOOR, PD_REL_FLOOR * mean_variance)


def validate_covariance(cov) -> np.ndarray:
    """协方差矩阵结构校验：严格 3×3、有限值、对称、对角线方差为正。

    这里只做"能否进入优化器"的结构判定；正定性由 ``stabilize_covariance`` 处理。
    """
    try:
        arr = np.asarray(cov, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"协方差矩阵无法转换为数值数组：{exc}") from None
    if arr.ndim != 2 or arr.shape != (N_ASSETS, N_ASSETS):
        raise ValueError(f"协方差矩阵必须严格为 {N_ASSETS}×{N_ASSETS}，实际 shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("协方差矩阵存在 NaN/Inf，拒绝优化")
    if not np.allclose(arr, arr.T, rtol=1e-10, atol=1e-14):
        raise ValueError("协方差矩阵不对称，拒绝优化")
    diagonal = np.diag(arr)
    if np.any(diagonal <= 0):
        raise ValueError(f"协方差矩阵对角线（资产方差）必须为正，实际 {diagonal.tolist()}")
    return arr


def stabilize_covariance(cov) -> tuple[np.ndarray, dict]:
    """返回（可安全优化的协方差, 收缩诊断信息）。

    处理顺序：
    1. 结构非法（零矩阵、方差非正、不对称、NaN）→ 直接抛 ValueError；
    2. 最小特征值高于下界（正定）→ 原样返回，delta = 0；
    3. 奇异/近奇异/非正定 → 向 ``mu·I``（mu 为平均方差）做协方差收缩，
       取能让矩阵正定的最小收缩强度；收缩也救不回来则抛 ValueError。

    该步骤同时消除后续 ``w'Σw`` 除零、求解器发散的隐患。
    """
    arr = validate_covariance(cov)
    mean_variance = float(np.trace(arr) / N_ASSETS)
    floor = _pd_floor(mean_variance)
    eig_before = float(np.linalg.eigvalsh(arr)[0])
    info = {
        "shrunk": False,
        "shrink_delta": 0.0,
        "min_eigenvalue_before": eig_before,
        "min_eigenvalue_after": eig_before,
        "eigenvalue_floor": floor,
        "shrink_target": "mean_variance_diagonal",
    }
    if eig_before > floor:
        return arr.copy(), info

    target = mean_variance * np.eye(N_ASSETS)
    for delta in SHRINK_GRID:
        candidate = (1.0 - delta) * arr + delta * target
        candidate = (candidate + candidate.T) / 2.0
        if float(np.linalg.eigvalsh(candidate)[0]) > floor:
            info.update({
                "shrunk": True,
                "shrink_delta": float(delta),
                "min_eigenvalue_after": float(np.linalg.eigvalsh(candidate)[0]),
            })
            return candidate, info
    raise ValueError(
        f"协方差矩阵奇异/非正定且收缩修复失败（最小特征值={eig_before:.3e}），拒绝继续优化")


def is_valid_cov(cov) -> bool:
    """结构合法且正定的快速判定（不抛异常，供外部预检使用）。"""
    try:
        arr = validate_covariance(cov)
    except ValueError:
        return False
    floor = _pd_floor(float(np.trace(arr) / N_ASSETS))
    return bool(float(np.linalg.eigvalsh(arr)[0]) > floor)


# --------------------------------------------------------------------------- #
# 三、风险贡献计算（后验，整改要求 1）
# --------------------------------------------------------------------------- #
def compute_risk_contribution(weight, cov) -> tuple[np.ndarray, np.ndarray]:
    """后验风险贡献：返回（绝对贡献 rc_i = w_i·(Σw)_i, 占比 rc_i / 组合方差）。"""
    w = np.asarray(weight, dtype=float)
    sigma = np.asarray(cov, dtype=float)
    portfolio_variance = float(w @ sigma @ w)
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0:
        raise ValueError(f"组合方差为 {portfolio_variance!r}，风险贡献无法定义（除零风险）")
    rc = w * (sigma @ w)
    return rc, rc / portfolio_variance


def check_risk_budget(rc_ratio, tolerance: float = RC_TOLERANCE, budget=None) -> float:
    """风险预算后验校验：返回最大绝对偏差；超容差即抛 RiskBudgetError（禁止返回权重）。"""
    ratio = np.asarray(rc_ratio, dtype=float)
    target = np.full(N_ASSETS, 1.0 / N_ASSETS) if budget is None else np.asarray(budget, float)
    deviation = float(np.max(np.abs(ratio - target)))
    if not np.isfinite(deviation):
        raise RiskBudgetError("风险贡献占比出现 NaN/Inf，拒绝返回权重")
    if deviation > tolerance:
        raise RiskBudgetError(
            f"风险贡献后验偏差 {deviation:.6f} 超过容差 {tolerance:.6f}："
            f"占比={np.round(ratio, 6).tolist()}，目标={np.round(target, 6).tolist()}，拒绝返回权重")
    return deviation


# --------------------------------------------------------------------------- #
# 四、求解器
# --------------------------------------------------------------------------- #
def _normalize_cov(cov: np.ndarray) -> tuple[np.ndarray, float]:
    """纯尺度归一化（除以平均方差）：权重与风险贡献占比不变，只改善求解器数值条件。"""
    scale = float(np.trace(cov) / cov.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"协方差平均方差非法：{scale!r}")
    return cov / scale, scale


def _risk_budget_barrier(cov: np.ndarray, budget: np.ndarray) -> np.ndarray | None:
    """Spinu 对数障碍凸形式求解风险预算：min 0.5·y'Σy - Σb_i·ln(y_i)，y>0。

    对数障碍形式是凸问题，解唯一；归一化协方差后 L-BFGS-B 为主，BFGS 兜底。

    注意：优化器状态标记失败（如 "precision loss" / "ABNORMAL"）不等于解不可用——
    凸问题在平缓方向上常因线搜索精度断言而返回非 success，但候选解本身已收敛。
    因此这里只要候选解有限就收集，取后验偏差最小者返回；全部不可用才返回 None，
    由上层继续用 inverse-vol / 均权初值精修，最终能否采用由风险贡献后验容差裁决。
    """
    normalized, _ = _normalize_cov(cov)
    vol = np.sqrt(np.diag(normalized))
    start = budget / vol
    start = start / start.sum()
    theta0 = np.log(start)

    def objective(theta):
        y = np.exp(theta)
        return 0.5 * float(y @ normalized @ y) - float(budget @ theta)

    def gradient(theta):
        y = np.exp(theta)
        return y * (normalized @ y) - budget

    attempts = (
        ("L-BFGS-B", {"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000},
         [(-30.0, 30.0)] * normalized.shape[0]),
        ("BFGS", {"gtol": 1e-10, "maxiter": 2000}, None),
    )
    candidates = []
    for method, options, bounds in attempts:
        result = minimize(objective, theta0, jac=gradient, method=method,
                          bounds=bounds, options=options)
        if not np.all(np.isfinite(result.x)):
            continue
        y = np.exp(result.x)
        if not np.all(np.isfinite(y)) or y.sum() <= 0:
            continue
        candidates.append(y / y.sum())
        if result.success:  # 已收敛，无需再换算法重试
            break

    if not candidates:
        return None

    best, best_deviation = None, np.inf
    for w in candidates:
        rc = w * (cov @ w)
        deviation = float(np.max(np.abs(rc / rc.sum() - budget)))
        if np.isfinite(deviation) and deviation < best_deviation:
            best, best_deviation = w, deviation
    return best


def _refine_risk_budget(cov: np.ndarray, budget: np.ndarray, weight0) -> np.ndarray:
    """softmax 参数化 + 非线性最小二乘精修，把风险贡献占比压到浮点精度附近。"""
    normalized, _ = _normalize_cov(cov)

    def softmax(u):
        e = np.exp(u - np.max(u))
        return e / e.sum()

    def residual(u):
        w = softmax(u)
        rc = w * (normalized @ w)
        return rc / rc.sum() - budget

    u0 = np.log(np.maximum(np.asarray(weight0, float), 1e-12))
    solution = least_squares(residual, u0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
    return softmax(solution.x)


def _solve_risk_budget(cov: np.ndarray, budget=None) -> np.ndarray:
    """稳定求解风险预算权重：凸障碍求解 + 多起点精修，取后验偏差最小者。

    barrier 只在候选解全部不可用（NaN/Inf）时返回 None；此时仍用 inverse-vol 与均权初值精修，
    只要精修后风险贡献偏差达标就可采用，是否达标由 ``risk_parity`` 的后验校验裁决。
    """
    target = np.full(N_ASSETS, 1.0 / N_ASSETS) if budget is None else np.asarray(budget, float)
    barrier_weight = _risk_budget_barrier(cov, target)

    vol = np.sqrt(np.diag(cov))
    inverse_vol = target / vol
    inverse_vol = inverse_vol / inverse_vol.sum()
    equal = np.full(N_ASSETS, 1.0 / N_ASSETS)

    starts = [inverse_vol, equal]
    candidates = []
    if barrier_weight is not None:
        starts.insert(0, barrier_weight)  # 优先从凸解出发精修
        candidates.append(barrier_weight)
    for start in starts:
        try:
            candidates.append(_refine_risk_budget(cov, target, start))
        except Exception:  # 精修属于加强项，单项失败不影响整体
            continue

    best, best_deviation = None, np.inf
    for w in candidates:
        w = np.clip(w, 0.0, None)
        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            continue
        w = w / w.sum()
        rc = w * (cov @ w)
        deviation = float(np.max(np.abs(rc / rc.sum() - target)))
        if np.isfinite(deviation) and deviation < best_deviation:
            best, best_deviation = w, deviation
    if best is None:
        raise RuntimeError("风险预算求解未得到可行权重：凸障碍解与 inverse-vol/均权精修均不可用")
    return best


def _solve_min_variance(cov: np.ndarray) -> np.ndarray:
    """最小方差：多起点 SLSQP（多头、权重和=1），取可行解中方差最小者。"""
    normalized, _ = _normalize_cov(cov)
    vol = np.sqrt(np.diag(normalized))
    inverse_vol = (1.0 / vol) / (1.0 / vol).sum()
    starts = [np.full(N_ASSETS, 1.0 / N_ASSETS), inverse_vol]
    for index in range(N_ASSETS):
        corner = np.full(N_ASSETS, 0.02)
        corner[index] = 1.0 - 0.02 * (N_ASSETS - 1)
        starts.append(corner / corner.sum())

    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},)
    bounds = [(0.0, 1.0)] * N_ASSETS
    best_weight, best_variance = None, np.inf
    for start in starts:
        result = minimize(
            lambda w: float(w @ normalized @ w), start, jac=lambda w: 2.0 * (normalized @ w),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-14, "maxiter": 1000})
        if not np.all(np.isfinite(result.x)):
            continue
        w = np.clip(result.x, 0.0, 1.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        variance = float(w @ normalized @ w)
        if np.isfinite(variance) and variance < best_variance:
            best_weight, best_variance = w, variance
    if best_weight is None:
        raise RuntimeError("最小方差优化未找到可行解")
    return best_weight


# --------------------------------------------------------------------------- #
# 五、统一结果封装
# --------------------------------------------------------------------------- #
def _build_result(strategy: str, weight, cov, tickers, shrinkage: dict,
                  risk_budget_tolerance=None, risk_budget_pass=None) -> dict:
    w = np.asarray(weight, dtype=float)
    total = float(w.sum())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-10:
        raise ValueError(f"权重和必须为 1，实际 {total!r}")
    if np.any(w < -1e-12):
        raise ValueError(f"权重不允许为负（多头约束），实际 {w.tolist()}")

    rc, rc_ratio = compute_risk_contribution(w, cov)
    return {
        "strategy": strategy,
        "weight": {code: float(x) for code, x in zip(tickers, w)},
        "risk_contribution": {code: float(x) for code, x in zip(tickers, rc)},
        "risk_contribution_ratio": {code: float(x) for code, x in zip(tickers, rc_ratio)},
        "risk_contribution_max_deviation": float(
            np.max(np.abs(rc_ratio - np.full(N_ASSETS, 1.0 / N_ASSETS)))),
        "risk_budget_tolerance": risk_budget_tolerance,
        "risk_budget_pass": risk_budget_pass,
        "portfolio_variance": float(w @ cov @ w),
        "covariance_shrinkage": shrinkage,
    }


# --------------------------------------------------------------------------- #
# 六、三种基准策略
# --------------------------------------------------------------------------- #
def equal_weight(cov, tickers=None) -> dict:
    """等权策略：每类资产 1/3，仅做协方差安全校验后返回。"""
    codes = validate_asset_pool(ASSET_POOL if tickers is None else tickers)
    sigma, shrinkage = stabilize_covariance(cov)
    weight = np.full(N_ASSETS, 1.0 / N_ASSETS)
    return _build_result("equal_weight", weight, sigma, codes, shrinkage)


def min_variance(cov, tickers=None) -> dict:
    """最小方差策略：min w'Σw，s.t. Σw=1、w>=0。"""
    codes = validate_asset_pool(ASSET_POOL if tickers is None else tickers)
    sigma, shrinkage = stabilize_covariance(cov)
    weight = _solve_min_variance(sigma)
    return _build_result("min_variance", weight, sigma, codes, shrinkage)


def risk_parity(cov, tickers=None, rc_tolerance: float = RC_TOLERANCE) -> dict:
    """风险平价策略：各资产风险贡献占比 = 1/3，求解后做后验校验。

    求解返回 success 但后验偏差 > ``rc_tolerance``（默认 0.005）时抛 RiskBudgetError，
    绝不返回权重结果。
    """
    codes = validate_asset_pool(ASSET_POOL if tickers is None else tickers)
    sigma, shrinkage = stabilize_covariance(cov)
    weight = _solve_risk_budget(sigma)
    _, rc_ratio = compute_risk_contribution(weight, sigma)
    check_risk_budget(rc_ratio, tolerance=rc_tolerance)
    return _build_result("risk_parity", weight, sigma, codes, shrinkage,
                         risk_budget_tolerance=rc_tolerance, risk_budget_pass=True)


STRATEGIES = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
}


# --------------------------------------------------------------------------- #
# 七、策略分发（协方差由调用方传入）
# --------------------------------------------------------------------------- #
def run_baseline(cov, tickers=None, strategy: str = "risk_parity") -> dict:
    """按策略名分发：协方差已由调用方准备好（250 日入口在下方自动计算）。"""
    codes = validate_asset_pool(ASSET_POOL if tickers is None else tickers)
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError(f"策略 {strategy!r} 不存在，仅支持 {list(STRATEGIES)}")
    return STRATEGIES[strategy](cov, codes)


def run_baseline_with_rolling_cov(cov, tickers=None, strategy: str = "risk_parity") -> dict:
    """向后兼容旧函数名：等价于 ``run_baseline``（旧名中的 rolling_cov 指上游窗口口径）。"""
    return run_baseline(cov, tickers, strategy)


# --------------------------------------------------------------------------- #
# 八、250 日窗口正式入口（整改要求 4）
# --------------------------------------------------------------------------- #
def fetch_returns_window_250d(as_of_date: str, path=None):
    """调用上游 ``get_returns_window(as_of_date, window=250)`` 取 250 日收益率窗口。

    上游第二阶段实现的形参名为 ``n``，故先按 ``window=`` 调用（对齐整改要求写法），
    若形参不匹配则回退到 ``n=``；返回结果做窗口长度、列顺序、末日一致性三项契约校验。
    """
    from data_processing.data_interface import get_returns_window  # 延迟导入：本模块核心求解不依赖 pandas

    kwargs = {} if path is None else {"path": path}
    try:
        window = get_returns_window(as_of_date, window=WINDOW_DAYS, **kwargs)
    except TypeError:
        window = get_returns_window(as_of_date, n=WINDOW_DAYS, **kwargs)

    if len(window) != WINDOW_DAYS:
        raise ValueError(f"上游返回窗口长度 {len(window)} != {WINDOW_DAYS}，数据不足")
    if list(window.columns) != ASSET_POOL:
        raise ValueError(f"上游返回资产列顺序 {list(window.columns)} 与资产池 {ASSET_POOL} 不一致")
    if str(window.index[-1]) != as_of_date:
        raise ValueError(
            f"上游窗口末日 {window.index[-1]} 与 as_of_date {as_of_date} 不一致，存在未来数据泄漏风险")
    return window


def compute_sample_covariance(returns) -> np.ndarray:
    """由 250 日收益率窗口计算样本协方差矩阵（ddof=1，日频，不做年化）。

    ``returns`` 可以是上游返回的 DataFrame，也可以是 (250, 3) 的 ndarray。
    """
    if hasattr(returns, "columns"):
        if list(returns.columns) != ASSET_POOL:
            raise ValueError(f"收益率列顺序 {list(returns.columns)} 必须等于资产池 {ASSET_POOL}")
        values = np.asarray(returns[ASSET_POOL], dtype=float)
    else:
        values = np.asarray(returns, dtype=float)
        if values.ndim != 2 or values.shape[1] != N_ASSETS:
            raise ValueError(f"收益率样本必须为 (T, {N_ASSETS}) 的二维数组，实际 shape={values.shape}")
    if values.shape[0] < 2:
        raise ValueError(f"样本观测数 {values.shape[0]} 不足，无法估计协方差")
    if not np.all(np.isfinite(values)):
        raise ValueError("收益率样本存在 NaN/Inf，拒绝估计协方差")
    return np.asarray(np.cov(values, rowvar=False, ddof=1), dtype=float).reshape(N_ASSETS, N_ASSETS)


def run_baseline_250d(as_of_date: str, strategy: str = "risk_parity",
                      tickers=None, path=None) -> dict:
    """正式入口：as_of_date → 上游 250 日收益率 → 本模块算协方差 → 策略求解。"""
    codes = validate_asset_pool(ASSET_POOL if tickers is None else tickers)
    window = fetch_returns_window_250d(as_of_date, path=path)
    raw_cov = compute_sample_covariance(window)
    sigma, shrinkage = stabilize_covariance(raw_cov)

    result = run_baseline(sigma, codes, strategy)
    # 策略层对已修复矩阵再校验一次为幂等操作（不会再触发收缩），
    # 这里回传以 250 日原始样本协方差为基准的收缩诊断，便于追溯。
    result["covariance_shrinkage"] = shrinkage
    result.update({
        "as_of_date": as_of_date,
        "window_days": WINDOW_DAYS,
        "sample_days": int(window.shape[0]),
        "asset_pool": list(codes),
    })
    return result


# --------------------------------------------------------------------------- #
# 九、命令行自测
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="3号 baseline_strategies：250 日窗口基准组合优化")
    parser.add_argument("--as-of", default="2026-11-20", help="信息截止日 YYYY-MM-DD")
    parser.add_argument("--strategy", default="risk_parity", choices=list(STRATEGIES))
    parser.add_argument("--path", default=None, help="冻结矩阵路径，默认 data/unified_returns.xlsx")
    args = parser.parse_args()

    payload = run_baseline_250d(args.as_of, args.strategy, path=args.path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
