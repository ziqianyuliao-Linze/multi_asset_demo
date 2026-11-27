# -*- coding: utf-8 -*-
"""风险反馈的正式管线：真实 risk_calc 结果 → packet_builder → 1 号 build_model_input。

这是计算器分支唯一的对外入口，不在别处手工拼 facts。
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


def build_facts(returns: pd.DataFrame, weights: Sequence[float],
                base_date: str | None = None,
                prev_weights: Sequence[float] | None = None) -> Dict[str, Any]:
    """用 risk_calc 的真函数算出反馈所需全部指标。

    base_date 是变化归因的基准日（D 组必需）；prev_weights 缺省等于 weights。
    """
    w = np.asarray(weights, dtype=float)
    w0 = w if prev_weights is None else np.asarray(prev_weights, dtype=float)

    equity = R.equity_curve(returns, w)
    dd = R.drawdown_stats(equity)
    episode = R.find_drawdown_episodes(equity, 1)[0]
    cov_now = R.covariance_matrix(returns, C.Window.COV)

    if base_date is None:
        base_date = returns.index[-C.Window.COV * 2]
    base_ts = pd.to_datetime(base_date)
    cov_base = R.covariance_matrix(returns.loc[:base_ts], C.Window.COV)

    rc = R.risk_contribution(w, cov_now)

    return {
        "portfolio_return": {
            "d20": R.window_return(returns, w, C.Window.SHORT),
            "d60": R.window_return(returns, w, C.Window.MID),
        },
        "portfolio_volatility": {
            "d20": R.annualized_volatility(returns, w, C.Window.SHORT),
            "d60": R.annualized_volatility(returns, w, C.Window.MID),
        },
        "current_drawdown": dd["current"],
        "max_drawdown": dd["max_drawdown"],
        "cvar_95": R.cvar(returns, w),
        "trigger_flags": {
            "drawdown": R.drawdown_alert(equity)["triggered"],
            "volatility": R.volatility_alert(returns, w)["triggered"],
            "correlation": R.correlation_alert(returns)["triggered"],
        },
        "risk_contribution": rc["rc_pct"],
        "budget_deviation": R.budget_deviation(rc["rc_pct"], w),
        "drawdown_attribution": R.drawdown_attribution(
            returns, w, episode["peak_idx"], episode["trough_idx"]),
        "change_attribution": R.change_attribution(w0, w, cov_base, cov_now),
        "concentration": R.concentration(rc["rc_pct"]),
        "base_date": str(base_ts)[:10],
    }


def build_group_feedback(returns: pd.DataFrame, weights: Sequence[float], group: str,
                         as_of_date: str | None = None,
                         base_date: str | None = None,
                         prev_weights: Sequence[float] | None = None) -> Dict[str, Any]:
    """算指标 → 组装该组反馈。A 组返回空字典。"""
    meta = {
        "as_of_date": as_of_date or str(returns.index[-1])[:10],
        "base_date": base_date,
    }
    facts = build_facts(returns, weights, base_date, prev_weights)
    return PB.build_feedback(facts, meta, group)


def build_group_model_input(returns: pd.DataFrame, weights: Sequence[float], group: str,
                            market_data: Mapping[str, Any],
                            as_of_date: str | None = None,
                            base_date: str | None = None,
                            prev_weights: Sequence[float] | None = None) -> Dict[str, Any]:
    """正式入口：返回可直接送给大模型的输入（已通过 1 号校验）。"""
    feedback = build_group_feedback(returns, weights, group, as_of_date, base_date, prev_weights)
    from llm_decision.validator import build_model_input

    return build_model_input(dict(market_data), group, feedback)
