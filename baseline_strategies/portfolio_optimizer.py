import numpy as np
from scipy.optimize import minimize
from data_processing.data_interface import get_returns_window
import pandas as pd


def is_valid_cov(cov: np.ndarray) -> bool:
    """校验协方差矩阵：二维、方阵、对称、有限值、半正定【组长第2条】"""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return False
    if not np.all(np.isfinite(cov)):
        return False
    if not np.allclose(cov, cov.T):
        return False
    # 半正定检查，允许微小浮点误差
    eigvals = np.linalg.eigvalsh(cov)
    return bool(np.all(eigvals >= -1e-12))


def equal_weight(cov: np.ndarray, tickers: list[str]) -> dict:
    """等权策略，统一返回weight与risk_contribution"""
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法：非对称/非半正定/维度错误")
    n = cov.shape[0]
    w_arr = np.ones(n) / n
    weight_dict = dict(zip(tickers, w_arr))
    rc_arr = w_arr * (cov @ w_arr)
    rc_dict = dict(zip(tickers, rc_arr))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def min_variance(cov: np.ndarray, tickers: list[str]) -> dict:
    """最小方差组合，增加优化结果校验【组长第3条】"""
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法：非对称/非半正定/维度错误")
    n = cov.shape[0]

    def obj(w):
        return w.T @ cov @ w

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"最小方差优化失败: {res.message}")
    w = res.x
    if not np.isclose(np.sum(w), 1.0, atol=1e-6) or np.any(w < -1e-6):
        raise RuntimeError("最小方差输出权重不满足约束")

    weight_dict = dict(zip(tickers, w))
    rc_arr = w * (cov @ w)
    rc_dict = dict(zip(tickers, rc_arr))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def risk_parity(cov: np.ndarray, tickers: list[str]) -> dict:
    """风险平价：目标改为【风险贡献占比】均等【组长第1条核心修复】
    返回统一结构字典：{"weight":..., "risk_contribution":...}
    """
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法：非对称/非半正定/维度错误")
    n = cov.shape[0]
    target_rc_ratio = 1.0 / n

    def obj(w):
        rc = w * (cov @ w)
        rc_ratio = rc / np.sum(rc)
        # 最小化风险贡献占比 和 1/n 的残差平方和
        return np.sum((rc_ratio - target_rc_ratio) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"风险平价优化失败: {res.message}")
    w = res.x
    if not np.isclose(np.sum(w), 1.0, atol=1e-6) or np.any(w < -1e-6):
        raise RuntimeError("风险平价输出权重不满足约束")

    rc = w * (cov @ w)
    weight_dict = dict(zip(tickers, w))
    rc_dict = dict(zip(tickers, rc))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def run_strategy(as_of_date: str, tickers: list[str], strategy_name: str):
    """
    对外统一入口：
    as_of_date: 截止日期 "YYYY-MM-DD"
    tickers: 标的列表
    strategy_name: equal_weight / min_variance / risk_parity
    """
    VALID_TICKERS = {"510300", "511010", "518880"}
    for t in tickers:
        if t not in VALID_TICKERS:
            raise ValueError(f"资产代码 {t} 不合法，仅支持: {sorted(VALID_TICKERS)}")

    # 上游拿250日滚动收益
    ret_df = get_returns_window(as_of_date, n=250)
    ret_df = ret_df[tickers]
    cov = ret_df.cov().values
    # 奇异矩阵正则：先加扰动，再送入校验和优化（修复修改点B顺序bug）
    eps = 1e-8
    cov = cov + eps * np.eye(cov.shape[0])

    if strategy_name == "equal_weight":
        return equal_weight(cov, tickers)
    elif strategy_name == "min_variance":
        return min_variance(cov, tickers)
    elif strategy_name == "risk_parity":
        return risk_parity(cov, tickers)
    else:
        raise ValueError("策略名称只能是 equal_weight / min_variance / risk_parity")
