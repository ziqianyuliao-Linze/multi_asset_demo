import numpy as np
import pytest
from baseline_strategies.portfolio_optimizer import (
    equal_weight,
    min_variance,
    risk_parity,
    is_valid_cov,
    run_strategy
)

# 资产标的代码，与项目统一资产池保持一致
TICKERS = ["510300", "511010", "518880"]

# 合法半正定协方差测试矩阵
cov_valid = np.array([
    [0.05, 0.01, 0.02],
    [0.01, 0.02, 0.005],
    [0.02, 0.005, 0.03]
])

# 非法：不对称协方差矩阵，用于异常测试
cov_bad_asym = np.array([
    [0.05, 0.01, 0.02],
    [0.01, 0.002, 0.999],
    [0.02, 0.001, 0.03]
])


def test_cov_check():
    """协方差矩阵合法性单元测试"""
    assert is_valid_cov(cov_valid) is True
    assert is_valid_cov(cov_bad_asym) is False


def test_equal_weight():
    """等权策略测试"""
    res = equal_weight(cov_valid, TICKERS)
    w_dict = res["weight"]
    w = np.array(list(w_dict.values()))
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.allclose(w, 1.0 / 3.0, atol=1e-6)


def test_min_variance():
    """最小方差策略测试，校验权重约束"""
    res = min_variance(cov_valid, TICKERS)
    w_dict = res["weight"]
    w = np.array(list(w_dict.values()))
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)
    assert np.all(w <= 1.0 + 1e-6)


def test_risk_parity():
    """风险平价测试：风险贡献占比应接近均等"""
    res = risk_parity(cov_valid, TICKERS)
    w_dict = res["weight"]
    rc_dict = res["risk_contribution"]
    w = np.array(list(w_dict.values()))
    rc = np.array(list(rc_dict.values()))
    rc_ratio = rc / np.sum(rc)

    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)
    # 风险贡献占比允许5%误差
    assert np.allclose(rc_ratio, 1.0 / 3.0, atol=0.05)


def test_invalid_cov_exception():
    """传入非法协方差应当抛出ValueError"""
    with pytest.raises(ValueError):
        equal_weight(cov_bad_asym, TICKERS)
    with pytest.raises(ValueError):
        min_variance(cov_bad_asym, TICKERS)
    with pytest.raises(ValueError):
        risk_parity(cov_bad_asym, TICKERS)


def test_invalid_ticker_exception():
    """非法资产代码抛出异常"""
    with pytest.raises(ValueError):
        run_strategy("2025-12-31", ["999999"], "equal_weight")


def test_run_strategy_entry():
    """测试完整入口链路连通"""
    res = run_strategy("2025-12-31", ["510300", "511010", "518880"], "equal_weight")
    assert isinstance(res, dict)
    assert "weight" in res
    assert "risk_contribution" in res


if __name__ == "__main__":
    print("===== 测试输入协方差矩阵 =====")
    print(cov_valid)
    print("\n1.等权策略结果: ")
    w_eq = equal_weight(cov_valid, TICKERS)
    print(w_eq)

    print("\n2.最小方差策略结果: ")
    w_mv = min_variance(cov_valid, TICKERS)
    print(w_mv)

    print("\n3.风险平价策略结果: ")
    w_rp = risk_parity(cov_valid, TICKERS)
    print(w_rp)
    rc_arr = np.array(list(w_rp["risk_contribution"].values()))
    print("风险贡献占比: ", np.round(rc_arr / rc_arr.sum(), 4))
