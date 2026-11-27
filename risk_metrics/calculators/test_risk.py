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

import json
import math
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import packet_builder as PB
    from . import pipeline
    from . import risk_calc as R
except ImportError:
    import config as C
    import packet_builder as PB
    import pipeline
    import risk_calc as R

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from llm_decision.validator import build_model_input, validate_risk_feedback  # noqa: E402

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


# ======================= 三、真实计算链路（内部分装） =======================
def _sample_returns(n: int = 700, seed: int = 7, drift: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(drift, 0.01, size=(n, 3))
    base[:, 1] *= 0.1
    base[:, 2] *= 0.7
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(base, index=idx, columns=list(C.ASSETS))


def _market_data(as_of: str, days: int = 400) -> dict:
    codes = list(PB.CODES)
    features = {k: {c: 0.05 for c in codes} for k in
                ("asset_return_20d", "asset_return_60d", "asset_volatility_20d",
                 "asset_volatility_60d", "asset_volatility_250d")}
    corr = {a: {b: (1.0 if a == b else 0.1) for b in codes} for a in codes}
    status = {k: {"required": 250, "available": days, "sufficient": True, "reason": "ok"}
              for k in ("return_20d", "return_60d", "volatility_20d", "volatility_60d",
                        "volatility_250d", "corr_60d", "corr_250d")}
    return {
        "as_of_date": as_of,
        "market_features": features,
        "asset_correlations": {"corr_60d": corr, "corr_250d": corr},
        "data_quality": {"window_status": status, "insufficient_windows": [],
                         "no_lookahead": True, "sample_days_to_date": days},
    }


W = (1 / 3,) * 3
BUDGET = (0.5, 0.3, 0.2)


def _facts(group: str = "D", n: int = 700) -> tuple:
    df = _sample_returns(n)
    as_of = str(df.index[-1])[:10]
    base = str(df.index[400])[:10]
    facts = pipeline.build_facts(df, W, group, as_of, base_date=base,
                                 previous_effective_budget=BUDGET)
    return facts, df, as_of, base


def test_facts_come_from_risk_calc() -> None:
    facts, _, _, _ = _facts()
    assert set(facts["drawdown_attribution"]) >= {"start", "end", "contrib", "share"}
    assert {"weight_effect", "vol_effect", "corr_effect", "residual"} <= set(facts["change_attribution"])
    assert {"max_rc", "max_asset", "neff", "hhi", "is_alert"} <= set(facts["concentration"])


def test_no_lookahead_after_appending_future_rows() -> None:
    """同一 as_of_date，追加未来 200 行，输出必须完全一致。"""
    df = _sample_returns(700)
    as_of = str(df.index[-1])[:10]
    base = str(df.index[400])[:10]

    future = _sample_returns(200, seed=99)
    future.index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=200, freq="B")
    extended = pd.concat([df, future])

    a = pipeline.build_facts(df, W, "D", as_of, base_date=base, previous_effective_budget=BUDGET)
    b = pipeline.build_facts(extended, W, "D", as_of, base_date=base, previous_effective_budget=BUDGET)
    assert a == b


def test_budget_deviation_uses_risk_budget() -> None:
    """偏差 = RC% − 上一期风险预算，不是 RC% − 资金权重。"""
    facts, _, _, _ = _facts()
    rc = facts["risk_contribution"]
    got = facts["budget_deviation"]
    for i, a in enumerate(C.ASSETS):
        assert abs(got[a] - (rc[a] - BUDGET[i])) < 1e-12, a
        assert abs(got[a] - (rc[a] - W[i])) > 1e-6, a   # 与用资金权重的结果明显不同


def test_group_a_short_sample_returns_empty() -> None:
    """A 组不计算任何指标，极短样本也能直接返回空反馈。"""
    short = _sample_returns(5)
    as_of = str(short.index[-1])[:10]
    assert pipeline.build_facts(short, W, "A", as_of) == {}


def test_group_b_runs_with_60_days() -> None:
    """B 组只需要 20/60 日收益，60 天样本即可运行。"""
    df = _sample_returns(60)
    as_of = str(df.index[-1])[:10]
    facts = pipeline.build_facts(df, W, "B", as_of)
    assert set(facts) == {"portfolio_return"}
    fb = PB.build_feedback(facts, {"as_of_date": as_of}, "B")
    assert set(fb) == {"as_of_date", "portfolio_return"}


def test_no_drawdown_history_does_not_crash() -> None:
    """净值单调上升、没有回撤区间时，回撤归因应为零贡献而不是报错。"""
    idx = pd.date_range("2024-01-01", periods=700, freq="B")
    rng = np.random.default_rng(3)
    rising = np.full((700, 3), 0.002) + rng.normal(0, 0.0001, size=(700, 3))
    df = pd.DataFrame(rising, index=idx, columns=list(C.ASSETS))
    as_of = str(df.index[-1])[:10]
    facts = pipeline.build_facts(df, W, "D", as_of, base_date=str(df.index[400])[:10],
                                 previous_effective_budget=BUDGET)
    assert facts["drawdown_attribution"]["start"] is None
    assert all(abs(v) < 1e-12 for v in facts["drawdown_attribution"]["contrib"].values())


def test_packet_group_fields() -> None:
    for g in ("B", "C", "D"):
        facts, df, as_of, base = _facts(g)
        meta = {"as_of_date": as_of, "base_date": base}
        assert set(PB.build_feedback(facts, meta, g)) - {"as_of_date"} == PB.GROUP_FIELDS[g], g


def test_packet_uses_asset_codes() -> None:
    facts, _, as_of, base = _facts()
    d = PB.build_feedback(facts, {"as_of_date": as_of, "base_date": base}, "D")
    codes = set(PB.CODES)
    assert set(d["risk_contribution"]) == codes
    assert set(d["budget_deviation"]) == codes
    assert set(d["risk_change_attribution"]) - {"base"} == codes
    assert set(d["concentration_state"]) - {"neff"} == codes
    assert set(d["drawdown_attribution"]) - {"start", "end", "transaction_cost"} == codes
    assert abs(d["risk_contribution"]["510300"] - facts["risk_contribution"]["stock"]) < 1e-5


def test_packet_no_leak() -> None:
    for g in ("B", "C"):
        facts, _, as_of, base = _facts(g)
        fb = PB.build_feedback(facts, {"as_of_date": as_of, "base_date": base}, g)
        assert not (set(fb) & PB.D_ONLY), g


def test_packet_missing_fact_and_group() -> None:
    facts, _, as_of, base = _facts("C")
    meta = {"as_of_date": as_of, "base_date": base}
    bad = dict(facts)
    bad.pop("cvar_95")
    expect_error(lambda: PB.build_feedback(bad, meta, "C"), KeyError)
    expect_error(lambda: PB.build_feedback(facts, meta, "Z"), KeyError)


def test_feedback_passes_module1_validator() -> None:
    """四组反馈必须通过 1 号 validate_risk_feedback（导入失败即测试失败）。"""
    for g in ("B", "C", "D"):
        facts, _, as_of, base = _facts(g)
        meta = {"as_of_date": as_of, "base_date": base}
        assert validate_risk_feedback(PB.build_feedback(facts, meta, g), g, as_of), g


def test_end_to_end_into_module1_build_model_input() -> None:
    """真实 risk_calc → packet_builder → 1 号 build_model_input（全程真实函数）。"""
    facts, df, as_of, base = _facts()
    market = _market_data(as_of, len(df))
    payload = pipeline.build_model_input_for_check(
        df, W, "D", market, as_of, base_date=base, previous_effective_budget=BUDGET)
    assert set(payload) == {"as_of_date", "market_features", "asset_correlations",
                            "data_quality", "feedback"}
    assert set(payload["feedback"]) == PB.GROUP_FIELDS["D"]
    assert "experiment_group" not in json.dumps(payload, ensure_ascii=False)


def test_risk_integration_is_the_single_official_pipeline() -> None:
    """正式入口 risk_integration 只走 risk_feedback.build_feedback，不引 packet_builder。"""
    import inspect
    from risk_metrics import risk_integration

    src = inspect.getsource(risk_integration)
    assert "build_feedback" in src
    assert "packet_builder" not in src



def test_accepts_module2_window_output() -> None:
    """兼容 2 号 get_returns_window 的真实输出：字符串日期索引 + 证券代码列。"""
    df = _sample_returns(700)
    as_of = str(df.index[-1])[:10]
    base = str(df.index[400])[:10]

    raw = df.copy()
    raw.index = pd.Index([d.strftime("%Y-%m-%d") for d in raw.index], name="date")
    raw.columns = list(PB.CODES)

    facts = pipeline.build_facts(raw, W, "D", as_of, base_date=base,
                                 previous_effective_budget=BUDGET)
    expect, _, _, _ = _facts()
    for a in C.ASSETS:
        assert abs(facts["risk_contribution"][a] - expect["risk_contribution"][a]) < 1e-12


def test_rejects_invalid_base_date() -> None:
    """基准日必须早于截止日、是真实交易日、前后各至少 250 日。"""
    df = _sample_returns(700)
    as_of = str(df.index[-1])[:10]
    later = str(df.index[500])[:10]

    # 等于截止日
    expect_error(lambda: pipeline.build_facts(
        df, W, "D", as_of, base_date=as_of, previous_effective_budget=BUDGET), ValueError)
    # 晚于截止日
    expect_error(lambda: pipeline.build_facts(
        df, W, "D", later, base_date=as_of, previous_effective_budget=BUDGET), ValueError)
    # 不是交易日
    expect_error(lambda: pipeline.build_facts(
        df, W, "D", as_of, base_date="2024-01-06", previous_effective_budget=BUDGET), ValueError)
    # 基准日之前不足 250 日
    expect_error(lambda: pipeline.build_facts(
        df, W, "D", as_of, base_date=str(df.index[10])[:10],
        previous_effective_budget=BUDGET), ValueError)


def test_module_import_path() -> None:
    """以 risk_metrics.calculators.test_risk 包路径导入时必须成立（不再递归跑子进程）。"""
    import importlib

    mod = importlib.import_module("risk_metrics.calculators.test_risk")
    assert hasattr(mod, "run_all")
    assert mod.__name__ == "risk_metrics.calculators.test_risk"

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
