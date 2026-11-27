import numpy as np
import pytest
from baseline_strategies.portfolio_optimizer import (
    equal_weight,
    min_variance,
    risk_parity,
    is_valid_cov
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
    """协方差矩阵合法性校验单元测试"""
    assert is_valid_cov(cov_valid) is True
    assert is_valid_cov(cov_bad_asym) is False


def test_equal_weight():
    """等权策略测试"""
    w_dict = equal_weight(cov_valid, TICKERS)
    w = np.array(list(w_dict.values()))
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.allclose(w, 1.0 / 3.0)


def test_min_variance():
    """最小方差策略测试，校验权重约束"""
    w_dict = min_variance(cov_valid, TICKERS)
    w = np.array(list(w_dict.values()))
    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)
    assert np.all(w <= 1.0 + 1e-6)


def test_risk_parity():
    """风险平价测试：风险贡献占比应接近均等"""
    w_dict, rc_dict = risk_parity(cov_valid, TICKERS)
    w = np.array(list(w_dict.values()))
    rc = np.array(list(rc_dict.values()))
    rc_ratio = rc / np.sum(rc)

    assert np.isclose(np.sum(w), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)
    # 风险贡献占比允许5%误差
    assert np.allclose(rc_ratio, 1.0 / 3.0, atol=0.05)


def test_invalid_cov_raise_exception():
    """传入非法协方差，应当抛出ValueError"""
    with pytest.raises(ValueError):
        equal_weight(cov_bad_asym, TICKERS)
    with pytest.raises(ValueError):
        min_variance(cov_bad_asym, TICKERS)
    with pytest.raises(ValueError):
        risk_parity(cov_bad_asym, TICKERS)


if __name__ == "__main__":
    print("===== 测试输入协方差矩阵 =====")
    print(cov_valid)
    print("\n1.等权策略结果：")
    w_eq = equal_weight(cov_valid, TICKERS)
    print(w_eq)

    print("\n2.最小方差策略结果：")
    w_mv = min_variance(cov_valid, TICKERS)
    print(w_mv)

    print("\n3.风险平价策略结果：")
    w_rp, rc_rp = risk_parity(cov_valid, TICKERS)
    print("权重：", w_rp)
    print("风险贡献：", rc_rp)
    rc_arr = np.array(list(rc_rp.values()))
    print("风险贡献占比：", np.round(rc_arr / rc_arr.sum(), 4))

