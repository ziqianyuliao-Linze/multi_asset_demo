from data_processing.data_interface import get_market_data_input
import numpy as np
from scipy.optimize import minimize


def is_valid_cov(cov: np.ndarray) -> bool:
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return False
    if not np.all(np.isfinite(cov)):
        return False
    if not np.allclose(cov, cov.T):
        return False
    eigvals = np.linalg.eigvalsh(cov)
    return bool(np.all(eigvals >= -1e-12))


def equal_weight(cov: np.ndarray, tickers: list[str]) -> dict:
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法")
    n = cov.shape[0]
    w_arr = np.ones(n) / n
    weight_dict = dict(zip(tickers, w_arr))
    rc_arr = w_arr * (cov @ w_arr)
    rc_dict = dict(zip(tickers, rc_arr))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def min_variance(cov: np.ndarray, tickers: list[str]) -> dict:
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法")
    n = cov.shape[0]

    def obj(w):
        return w.T @ cov @ w
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n
    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    if not res.success:
        raise RuntimeError("最小方差优化失败")
    w = res.x
    weight_dict = dict(zip(tickers, w))
    rc_arr = w * (cov @ w)
    rc_dict = dict(zip(tickers, rc_arr))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def risk_parity(cov: np.ndarray, tickers: list[str]) -> dict:
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法")
    n = cov.shape[0]
    target_rc_ratio = 1.0 / n

    def obj(w):
        rc = w * (cov @ w)
        rc_ratio = rc / np.sum(rc)
        return np.sum((rc_ratio - target_rc_ratio) ** 2)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n
    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    if not res.success:
        raise RuntimeError("风险平价优化失败")
    w = res.x
    rc = w * (cov @ w)
    weight_dict = dict(zip(tickers, w))
    rc_dict = dict(zip(tickers, rc))
    return {"weight": weight_dict, "risk_contribution": rc_dict}


def run_baseline_with_rolling_cov(cov, tickers, strategy: str):
    """
    【对接上游250日窗口接口的适配入口】
    参数：
        cov: 上游get_rolling_cov返回的协方差矩阵 np.ndarray
        tickers: 上游返回的资产代码列表 list[str]
        strategy: 可选 "equal_weight", "min_variance", "risk_parity"
    返回：
        字典，key: "weight", "risk_contribution"（和三个baseline策略输出保持一致）
    """
    if strategy == "equal_weight":
        return equal_weight(cov, tickers)
    elif strategy == "min_variance":
        return min_variance(cov, tickers)
    elif strategy == "risk_parity":
        return risk_parity(cov, tickers)
    else:
        raise ValueError(
            f"策略{strategy}不存在，仅支持: equal_weight / min_variance / risk_parity")

    if strategy == "equal_weight":
        return equal_weight(cov, tickers)
    elif strategy == "min_variance":
        return min_variance(cov, tickers)
    elif strategy == "risk_parity":
        return risk_parity(cov, tickers)
    else:
        raise ValueError(
            f"策略{strategy}不存在，仅支持：equal_weight / min_variance / risk_parity")

if __name__ == "__main__":
    import numpy as np
    cov_demo = np.array([
        [0.04, 0.01, 0.002],
        [0.01, 0.09, 0.001],
        [0.002, 0.001, 0.016]
    ])
    ticker_demo = ["A","B","C"]
    res = run_baseline_with_rolling_cov(cov_demo, ticker_demo, strategy="risk_parity")
    print("=====自测结果=====")
    print(res)
