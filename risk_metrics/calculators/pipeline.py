# -*- coding: utf-8 -*-
"""计算器侧的内部分装逻辑（不是正式入口）。

正式入口是 risk_metrics/risk_integration.py；本文件只服务于 calculators 目录的
自测与核对，把 risk_calc 的原始结果整理成 packet_builder 需要的形态。

三条硬约束：
  1. 所有计算先按 as_of_date 截断，末尾必须正好是截止日，绝不看未来数据；
  2. budget_deviation = RC% − 上一期有效风险预算，绝不用资金权重代替；
  3. 按组计算：A 不算任何指标，B 只算收益，C 加整体风险，D 才算协方差与归因。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from . import config as C
    from . import packet_builder as PB
    from . import risk_calc as R
except ImportError:                     # 脚本方式运行
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as C                  # type: ignore[no-redef]
    import packet_builder as PB         # type: ignore[no-redef]
    import risk_calc as R               # type: ignore[no-redef]


def normalize(returns: pd.DataFrame) -> pd.DataFrame:
    """把 2 号 get_returns_window 的原始输出转成本模块能用的形态。

    - 字符串日期索引 -> DatetimeIndex；
    - 债券代码列 510300/511010/518880 -> stock/bond/gold；
    - 已经是对内形态的原样返回。
    """
    df = returns.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(pd.Index(df.index))
    df.index.name = "date"
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    rename = {code: name for code, name in zip(PB.CODES, PB.NAMES)}
    df = df.rename(columns=rename)
    missing = [a for a in C.ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"缺少资产列（需代码或英文名）：{missing}")
    return df[list(C.ASSETS)].astype(float)


def truncate(returns: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """按截止日截断，并确认最后一行就是截止日；否则说明数据越界或日期不存在。"""
    df = normalize(returns)
    as_of = pd.to_datetime(as_of_date)
    df = df.loc[df.index <= as_of]
    if len(df) == 0:
        raise ValueError(f"截止日 {as_of_date} 之前没有可用数据")
    if pd.Timestamp(df.index[-1]) != as_of:
        raise ValueError(
            f"截止日 {as_of_date} 不是矩阵中的交易日，最后一行是 {df.index[-1]}")
    return df


def _budget_as_array(prev_budget: Mapping[str, float] | Sequence[float] | None) -> np.ndarray:
    if prev_budget is None:
        raise ValueError("budget_deviation 需要上一期有效风险预算 previous_effective_budget")
    if isinstance(prev_budget, Mapping):
        return np.array([float(prev_budget[a]) for a in C.ASSETS])
    b = np.asarray(prev_budget, dtype=float).reshape(-1)
    if b.size != len(C.ASSETS):
        raise ValueError(f"预算长度应为 {len(C.ASSETS)}，实际 {b.size}")
    return b


def _empty_drawdown(as_of_date: str) -> Dict[str, Any]:
    """没有回撤区间时用零贡献代替，避免 IndexError。"""
    return {
        "start": None,
        "end": as_of_date,
        "depth": 0.0,
        "contrib": {a: 0.0 for a in C.ASSETS},
        "share": {a: 0.0 for a in C.ASSETS},
    }


def build_facts(returns: pd.DataFrame, weights: Sequence[float], group: str,
                as_of_date: str, base_date: str | None = None,
                previous_effective_budget: Mapping[str, float] | Sequence[float] | None = None,
                prev_weights: Sequence[float] | None = None) -> Dict[str, Any]:
    """按组计算指标；只算该组需要的部分。"""
    if group not in ("A", "B", "C", "D"):
        raise KeyError(f"未知组别 {group}")
    if group == "A":
        return {}

    w = np.asarray(weights, dtype=float)
    df = truncate(returns, as_of_date)

    facts: Dict[str, Any] = {
        "portfolio_return": {
            "d20": R.window_return(df, w, C.Window.SHORT),
            "d60": R.window_return(df, w, C.Window.MID),
        }
    }
    if group == "B":
        return facts

    equity = R.equity_curve(df, w)
    dd = R.drawdown_stats(equity)
    facts.update(
        portfolio_volatility={
            "d20": R.annualized_volatility(df, w, C.Window.SHORT),
            "d60": R.annualized_volatility(df, w, C.Window.MID),
        },
        current_drawdown=dd["current"],
        max_drawdown=dd["max_drawdown"],
        cvar_95=R.cvar(df, w),
        trigger_flags={
            "drawdown": R.drawdown_alert(equity)["triggered"],
            "volatility": R.volatility_alert(df, w)["triggered"],
            "correlation": R.correlation_alert(df)["triggered"],
        },
    )
    if group == "C":
        return facts

    cov_now = R.covariance_matrix(df, C.Window.COV)
    rc = R.risk_contribution(w, cov_now)
    budget = _budget_as_array(previous_effective_budget)

    episodes = R.find_drawdown_episodes(equity, 1)
    if episodes:
        ep = episodes[0]
        attribution = R.drawdown_attribution(df, w, ep["peak_idx"], ep["trough_idx"])
    else:
        attribution = _empty_drawdown(as_of_date)

    base = base_date or facts.get("base_date")
    if base is None:
        raise ValueError("D 组需要 base_date（变化归因基准日）")
    base_ts = pd.to_datetime(base)
    as_of_ts = pd.to_datetime(as_of_date)

    if base_ts >= as_of_ts:
        raise ValueError(f"基准日 {base} 必须早于截止日 {as_of_date}")
    if base_ts not in df.index:
        raise ValueError(f"基准日 {base} 不是收益矩阵中的交易日")
    if len(df.loc[df.index < base_ts]) < C.Window.COV:
        raise ValueError(f"基准日 {base} 之前不足 {C.Window.COV} 日观测")
    if len(df) < C.Window.COV:
        raise ValueError(f"截止日 {as_of_date} 之前不足 {C.Window.COV} 日观测")
    base_df = df.loc[df.index <= base_ts]

    w0 = w if prev_weights is None else np.asarray(prev_weights, dtype=float)
    facts.update(
        risk_contribution=rc["rc_pct"],
        budget_deviation={a: float(rc["rc_pct"][a] - budget[i])
                          for i, a in enumerate(C.ASSETS)},
        drawdown_attribution=attribution,
        change_attribution=R.change_attribution(w0, w, R.covariance_matrix(base_df, C.Window.COV), cov_now),
        concentration=R.concentration(rc["rc_pct"]),
        base_date=str(base_ts)[:10],
    )
    return facts


def build_group_feedback(returns: pd.DataFrame, weights: Sequence[float], group: str,
                         as_of_date: str, base_date: str | None = None,
                         previous_effective_budget=None, prev_weights=None) -> Dict[str, Any]:
    """算指标并组装该组反馈（内部核对用，不是正式入口）。"""
    facts = build_facts(returns, weights, group, as_of_date, base_date,
                        previous_effective_budget, prev_weights)
    meta = {"as_of_date": as_of_date, "base_date": base_date or facts.get("base_date")}
    return PB.build_feedback(facts, meta, group)


def build_model_input_for_check(returns: pd.DataFrame, weights: Sequence[float], group: str,
                                market_data: Mapping[str, Any], as_of_date: str,
                                base_date: str | None = None,
                                previous_effective_budget=None, prev_weights=None) -> Dict[str, Any]:
    """仅用于自测：把结果交给 1 号校验。正式流程仍走 risk_integration。"""
    from llm_decision.validator import build_model_input

    feedback = build_group_feedback(returns, weights, group, as_of_date, base_date,
                                    previous_effective_budget, prev_weights)
    return build_model_input(dict(market_data), group, feedback)
