import numpy as np
from baseline_strategies.portfolio_optimizer import equal_weight, min_variance, risk_parity, is_valid_cov


def test_equal_weight():
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    tickers = ["A", "B"]
    res = equal_weight(cov, tickers)
    w = np.array(list(res["weight"].values()))
    assert np.allclose(w, [0.5, 0.5])
    assert np.isclose(np.sum(w), 1.0)


def test_min_variance():
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    tickers = ["A", "B"]
    res = min_variance(cov, tickers)
    w = np.array(list(res["weight"].values()))
    assert np.isclose(np.sum(w), 1.0)
    # 多头约束校验，权重不能为负
    assert np.all(w >= -1e-6)


def test_risk_parity():
    cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    tickers = ["A", "B"]
    res = risk_parity(cov, tickers)
    w = np.array(list(res["weight"].values()))
    rc = np.array(list(res["risk_contribution"].values()))
    assert np.isclose(np.sum(w), 1.0)
    # 风险贡献占比相等，容差放宽一点更稳
    rc_ratio = rc / rc.sum()
    assert np.allclose(rc_ratio[0], rc_ratio[1], atol=1e-3)
    assert np.all(w >= -1e-6)


def test_cov_check():
    bad_cov = np.array([[1, 2], [3, 4]])
    assert not is_valid_cov(bad_cov)
    # 再加一个合法协方差的正向用例
    good_cov = np.array([[0.04, 0.01], [0.01, 0.02]])
    assert is_valid_cov(good_cov)
