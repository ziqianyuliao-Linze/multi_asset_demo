# -*- coding: utf-8 -*-
"""
风险反馈模块 · 自测
===================

运行：python test_risk.py     （失败会打印明细，退出码 1）

三类测试：
  1. 手算对照：造能心算的数据，看结果对不对；
  2. 异常处理：数据太少、矩阵不对、空表时，必须报 ValueError/KeyError，不能静默；
  3. 分组红线：D 组字段不得出现在 A/B/C 包里。
"""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import packet_builder as PB
    from . import risk_calc as R
except ImportError:
    import config as C
    import packet_builder as PB
    import risk_calc as R

TOL = dict(rel_tol=1e-9, abs_tol=1e-12)


def frame(values: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    v = np.asarray(values, dtype=float).reshape(len(values), -1)
    idx = pd.date_range(start, periods=len(v), freq="B")
    df = pd.DataFrame(v, index=idx, columns=list(C.ASSETS)[: v.shape[1]])
    for a in C.ASSETS:
        if a not in df.columns:
            df[a] = 0.0
    return df[list(C.ASSETS)]


def equity(values: List[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="B"), dtype=float)


def expect_error(fn: Callable, exc: type) -> None:
    try:
        fn()
    except exc:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"期望 {exc.__name__}，实际 {type(e).__name__}: {e}")
    raise AssertionError(f"期望 {exc.__name__}，但没有抛异常")


# ======================= 一、手算对照 =======================
def test_window_return_compound() -> None:
    r = frame(np.full((30, 3), 0.001))
    w = (1 / 3,) * 3
    assert math.isclose(R.window_return(r, w, 20), 1.001 ** 20 - 1, **TOL)


def test_volatility_constant_and_alternating() -> None:
    w = (1 / 3,) * 3
    flat = frame(np.full((40, 3), 0.001))
    assert math.isclose(R.annualized_volatility(flat, w, 20), 0.0, abs_tol=1e-15)

    n = 40
    vals = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    alt = frame(np.column_stack([vals] * 3))
    expected = 0.01 * math.sqrt(n / (n - 1)) * math.sqrt(C.ANNUALIZE_DAYS)
    assert math.isclose(R.annualized_volatility(alt, w, n), expected, rel_tol=1e-9)


def test_drawdown_known() -> None:
    eq = equity([1.0, 1.2, 0.9, 1.2])
    d = R.drawdown_stats(eq)
    assert math.isclose(d["current"], 0.0, **TOL)
    assert math.isclose(d["max_drawdown"], -0.25, **TOL)
    assert d["peak_date"] == eq.index[1] and d["trough_date"] == eq.index[2]


def test_cvar_known() -> None:
    """100 天里 5 天亏 2%：k=5，CVaR 与 VaR 都应是 2%。"""
    vals = np.array([-0.02] * 5 + [0.001] * 95)
    r = frame(np.column_stack([vals] * 3))
    w = (1 / 3,) * 3
    assert math.isclose(R.cvar(r, w), 0.02, rel_tol=1e-9)
    assert math.isclose(R.var_loss(r, w), 0.02, rel_tol=1e-9)
    assert R.cvar(r, w) > 0


def test_cvar_12_5_rule() -> None:
    """250 天、95%：k=12.5，第 13 笔算半权重，不能简单取 13 笔均值。"""
    losses = np.concatenate([np.full(12, -0.02), [-0.01], np.full(237, 0.001)])
    r = frame(np.column_stack([losses] * 3))
    w = (1 / 3,) * 3
    expected = (12 * 0.02 + 0.5 * 0.01) / 12.5
    assert math.isclose(R.cvar(r, w), expected, rel_tol=1e-9)


def test_drawdown_alert_persistence() -> None:
    assert R.drawdown_alert(equity([1.0, 0.95, 0.91, 0.905, 0.99]))["triggered"] is False
    assert R.drawdown_alert(equity([1.0, 0.95, 0.91, 0.90, 0.905]))["triggered"] is True


def test_volatility_alert_spike() -> None:
    rng = np.random.default_rng(5)
    r = frame(np.vstack([rng.normal(0, 0.001, size=(400, 3)), rng.normal(0, 0.03, size=(25, 3))]))
    res = R.volatility_alert(r, (1 / 3,) * 3)
    assert res["triggered"] is True and res["current_vol"] > res["threshold"]


def test_correlation_alert_flip() -> None:
    rng = np.random.default_rng(9)
    base = rng.normal(0, 0.01, size=370)
    df = pd.DataFrame(
        {"stock": base, "bond": np.concatenate([-base[:300], base[300:]]),
         "gold": rng.normal(0, 0.005, 370)},
        index=pd.date_range("2024-01-01", periods=370, freq="B"))
    res = R.correlation_alert(df)
    assert res["corr"] > 0.5 and res["triggered"] is True and res["constant_reference"] is False


def test_risk_contribution_known() -> None:
    cov = np.diag([0.04, 0.01, 0.01])
    rc = R.risk_contribution((0.5, 0.25, 0.25), cov)["rc_pct"]
    var = 0.25 * 0.04 + 2 * 0.0625 * 0.01
    assert math.isclose(rc["stock"], 0.25 * 0.04 / var, **TOL)
    assert math.isclose(sum(rc.values()), 1.0, **TOL)


def test_budget_deviation_and_concentration() -> None:
    d = R.budget_deviation({"stock": 0.6, "bond": 0.1, "gold": 0.3},
                           {"stock": 0.4, "bond": 0.3, "gold": 0.3})
    assert math.isclose(d["stock"], 0.2, **TOL) and math.isclose(sum(d.values()), 0.0, **TOL)

    c = R.concentration({"stock": 0.8, "bond": 0.1, "gold": 0.1})
    assert c["max_asset"] == "stock"
    assert math.isclose(c["hhi"], 0.66, **TOL) and math.isclose(c["neff"], 1 / 0.66, **TOL)
    assert c["is_alert"] is True


def test_drawdown_attribution_known() -> None:
    r = frame(np.column_stack([np.full(5, -0.05), np.zeros(5), np.zeros(5)]))
    w = (0.5, 0.25, 0.25)
    ep = R.find_drawdown_episodes(R.equity_curve(r, w), 1)[0]
    a = R.drawdown_attribution(r, w, ep["peak_idx"], ep["trough_idx"])
    n_seg = ep["trough_idx"] - ep["peak_idx"]
    assert math.isclose(a["contrib"]["stock"], -0.5 * 0.05 * n_seg, rel_tol=1e-9)
    assert abs(a["contrib"]["bond"]) < 1e-12
    assert math.isclose(a["share"]["stock"], 1.0, rel_tol=1e-9)


def test_change_attribution() -> None:
    vol0, vol1 = np.array([0.2, 0.05, 0.15]), np.array([0.4, 0.05, 0.15])
    c0, c1 = np.diag(vol0 ** 2), np.diag(vol1 ** 2)
    same = R.change_attribution((1 / 3,) * 3, (1 / 3,) * 3, c0, c0)
    assert all(abs(v) < 1e-12 for v in same["total"].values())

    res = R.change_attribution((1 / 3,) * 3, (1 / 3,) * 3, c0, c1)
    assert res["total"]["stock"] > 0
    assert abs(res["weight_effect"]["stock"]) < 1e-12
    # 四项相加必须等于总变化（中心差分 + 残差）
    for a in C.ASSETS:
        ssum = (res["weight_effect"][a] + res["vol_effect"][a]
                + res["corr_effect"][a] + res["residual"][a])
        assert math.isclose(ssum, res["total"][a], abs_tol=1e-12)


# ======================= 二、异常处理 =======================
def test_error_too_few_rows() -> None:
    r = frame(np.full((5, 3), 0.001))
    w = (1 / 3,) * 3
    expect_error(lambda: R.window_return(r, w, 20), ValueError)
    expect_error(lambda: R.annualized_volatility(r, w, 20), ValueError)


def test_error_missing_column_and_bad_weight() -> None:
    df = pd.DataFrame({"stock": [0.01, 0.02], "bond": [0.0, 0.0]})
    expect_error(lambda: R.window_return(df, (1 / 3,) * 3, 2), ValueError)
    r = frame(np.full((30, 3), 0.001))
    expect_error(lambda: R.window_return(r, (0.5, 0.5), 20), ValueError)


def test_error_zero_variance_and_shape() -> None:
    r = frame(np.zeros((30, 3)))
    cov = R.covariance_matrix(r, 20)
    expect_error(lambda: R.risk_contribution((1 / 3,) * 3, cov), ValueError)
    expect_error(lambda: R.risk_contribution((1 / 3,) * 3, np.eye(2)), ValueError)


def test_error_empty_and_bad_index() -> None:
    empty = pd.DataFrame(columns=list(C.ASSETS), dtype=float)
    expect_error(lambda: R.cvar(empty, (1 / 3,) * 3), ValueError)
    r = frame(np.random.default_rng(1).normal(0, 0.01, size=(20, 3)))
    expect_error(lambda: R.drawdown_attribution(r, (1 / 3,) * 3, 10, 5), ValueError)


def test_data_health_flags_nan_and_empty() -> None:
    r = frame(np.random.default_rng(2).normal(0, 0.01, size=(50, 3)))
    r.iloc[10, 0] = np.nan
    h = PB.data_health(r)
    assert h["missing_values"] == 1 and h["rows"] == 50
    empty = pd.DataFrame(columns=list(C.ASSETS), dtype=float,
                         index=pd.DatetimeIndex([], name="date"))
    expect_error(lambda: PB.data_health(empty), ValueError)


# ======================= 三、分组红线 =======================
def _facts() -> tuple:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df = pd.DataFrame(rng.normal(0.0002, 0.01, size=(300, 3)), index=idx, columns=list(C.ASSETS))
    eq = R.equity_curve(df, (1 / 3,) * 3)
    cov = R.covariance_matrix(df)
    rc = R.risk_contribution((1 / 3,) * 3, cov)["rc_pct"]
    ep = R.find_drawdown_episodes(eq, 1)[0]
    da = R.drawdown_attribution(df, (1 / 3,) * 3, ep["peak_idx"], ep["trough_idx"])
    facts = {
        "portfolio_return": {"d20": -0.012, "d60": -0.045},
        "portfolio_volatility": {"d20": 0.10, "d60": 0.098, "d250": 0.11},
        "current_drawdown": -0.05, "max_drawdown": -0.09, "cvar_95": 0.018,
        "trigger_flags": {"drawdown": False, "volatility": True, "correlation": False},
        "risk_contribution": rc,
        "budget_deviation": R.budget_deviation(rc, (1 / 3,) * 3),
        "drawdown_attribution": {"start": da["start"], "end": da["end"],
                                 "transaction_cost": 0.0, **da["contrib"]},
        "risk_change_attribution": {"base": idx[100], **{
            a: {"weight": 0.0, "vol": 0.01, "corr": -0.01, "residual": 0.0} for a in C.ASSETS}},
        "concentration_state": {**rc, "neff": 1.8},
    }
    return facts, {"as_of_date": idx[-1]}


def test_packet_group_fields() -> None:
    facts, meta = _facts()
    assert PB.build_feedback(facts, meta, "A") == {}
    b = PB.build_feedback(facts, meta, "B")
    assert set(b) == {"as_of_date", "portfolio_return"}
    c = PB.build_feedback(facts, meta, "C")
    assert set(c) - {"as_of_date"} == PB.GROUP_FIELDS["C"]
    d = PB.build_feedback(facts, meta, "D")
    assert set(d) - {"as_of_date"} == PB.GROUP_FIELDS["D"]


def test_packet_uses_asset_codes() -> None:
    facts, meta = _facts()
    d = PB.build_feedback(facts, meta, "D")
    codes = set(PB.CODES)
    assert set(d["risk_contribution"]) == codes
    assert set(d["budget_deviation"]) == codes
    assert set(d["risk_change_attribution"]) - {"base"} == codes
    assert set(d["concentration_state"]) - {"neff"} == codes
    assert set(d["drawdown_attribution"]) - {"start", "end", "transaction_cost"} == codes


def test_packet_no_leak() -> None:
    facts, meta = _facts()
    for g in ("A", "B", "C"):
        fb = PB.build_feedback(facts, meta, g)
        assert not (set(fb) & PB.D_ONLY), g


def test_packet_missing_fact_and_group() -> None:
    facts, meta = _facts()
    bad = dict(facts)
    bad.pop("cvar_95")
    expect_error(lambda: PB.build_feedback(bad, meta, "C"), KeyError)
    expect_error(lambda: PB.build_feedback(facts, meta, "Z"), KeyError)


def test_feedback_passes_module1_validator() -> None:
    """D 组反馈必须能通过 1 号 validate_risk_feedback（缺模块则跳过）。"""
    try:
        from llm_decision.validator import validate_risk_feedback
    except Exception:
        return
    facts, meta = _facts()
    date = str(meta["as_of_date"])[:10]
    for g in ("A", "B", "C", "D"):
        fb = PB.build_feedback(facts, meta, g)
        assert validate_risk_feedback(fb, g, date), g


# ======================= 运行器 =======================
def run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failures: List[Tuple[str, str]] = []
    print(f"共 {len(tests)} 个测试\n" + "-" * 66)
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e) or "断言失败"))
            print(f"FAIL  {name}  ← {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}  ← {type(e).__name__}: {e}")
    print("-" * 66)
    print(f"通过 {len(tests) - len(failures)} / {len(tests)}，失败 {len(failures)}")
    for name, msg in failures:
        print(f"  [{name}] {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
