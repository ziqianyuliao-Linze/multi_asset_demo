import numpy as np
from scipy.optimize import minimize


def is_valid_cov(cov: np.ndarray) -> bool:
    """校验协方差矩阵：二维、方阵、对称、有限值、半正定（组长第2条）"""
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
    """等权策略，返回{标的代码:权重}字典，适配回测模块（组长第7条）"""
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法：非对称/非半正定/维度错误")
    n = cov.shape[0]
    w = np.ones(n) / n
    return dict(zip(tickers, w))


def min_variance(cov: np.ndarray, tickers: list[str]) -> dict:
    """最小方差组合：增加优化结果校验（组长第3条）"""
    if not is_valid_cov(cov):
        raise ValueError("协方差矩阵不合法：非对称/非半正定/维度错误")
    n = cov.shape[0]

    def obj(w):
        return w.T @ cov @ w

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    # 优化结果校验
    if not res.success:
        raise RuntimeError(f"最小方差优化失败：{res.message}")
    w = res.x
    if not np.isclose(np.sum(w), 1.0, atol=1e-6) or np.any(w < -1e-6):
        raise RuntimeError("最小方差输出权重不满足约束")
    return dict(zip(tickers, w))


def risk_parity(cov: np.ndarray, tickers: list[str]) -> tuple[dict, dict]:
    """
    风险平价：目标改为【风险贡献占比】均等（组长第1条核心修复）
    返回：权重字典、风险贡献字典
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
        raise RuntimeError(f"风险平价优化失败：{res.message}")
    w = res.x
    if not np.isclose(np.sum(w), 1.0, atol=1e-6) or np.any(w < -1e-6):
        raise RuntimeError("风险平价输出权重不满足约束")
    rc = w * (cov @ w)
    weight_dict = dict(zip(tickers, w))
    rc_dict = dict(zip(tickers, rc))
    return weight_dict, rc_dict
