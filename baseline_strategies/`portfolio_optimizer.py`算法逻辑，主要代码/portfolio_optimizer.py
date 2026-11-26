import numpy as np
from scipy.optimize import minimize

def equal_weight(cov: np.ndarray) -> np.ndarray:
    """等权策略，输入协方差矩阵，输出权重"""
    n = cov.shape[0]
    return np.ones(n) / n

def min_variance(cov: np.ndarray) -> np.ndarray:
    """最小方差组合：输入协方差矩阵，输出权重，w>=0，sum(w)=1"""
    n = cov.shape[0]
    def obj(w):
        return w.T @ cov @ w
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n)/n
    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    return res.x

def risk_parity(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    风险平价
    输入：协方差矩阵cov
    返回：weights权重, rc风险贡献
    目标：让各个资产风险贡献尽量相等
    """
    n = cov.shape[0]
    def obj(w):
        rc = w * (cov @ w)
        # 目标：最小化风险贡献之间的方差
        return np.var(rc)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-6, 1) for _ in range(n)]
    w0 = np.ones(n)/n
    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    w = res.x
    rc = w * (cov @ w)
    return w, rc
