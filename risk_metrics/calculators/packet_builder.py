# -*- coding: utf-8 -*-
"""风险反馈分拣：把 risk_calc 的真实输出组装成 1 号 build_model_input 能直接吃的结构。

数据来源就是 risk_calc 的原始返回值，不做二次加工：
  drawdown_attribution       -> {start, end, depth, contrib{stock,bond,gold}, share}
  change_attribution         -> {weight_effect{...}, vol_effect{...}, corr_effect{...}, residual{...}}
  concentration              -> {max_rc, max_asset, neff, hhi, is_alert}
  risk_contribution          -> {rc_pct{stock,bond,gold}, ...}

对外仍按 1 号口径输出证券代码键。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping

try:                                    # 作为 risk_metrics.calculators 包的一部分导入
    from .. import key_map
    from . import config as C
except ImportError:                     # 直接以脚本方式运行同目录文件时
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import key_map                      # type: ignore[no-redef]
    import config as C                  # type: ignore[no-redef]

CODES = key_map.CODES
NAMES = key_map.NAMES

GROUP_FIELDS: Dict[str, set] = {
    "A": set(),
    "B": {"portfolio_return"},
    "C": {
        "portfolio_return", "portfolio_volatility", "current_drawdown",
        "max_drawdown", "cvar_95", "trigger_flags",
    },
    "D": {
        "portfolio_return", "portfolio_volatility", "current_drawdown",
        "max_drawdown", "cvar_95", "trigger_flags",
        "risk_contribution", "budget_deviation", "drawdown_attribution",
        "risk_change_attribution", "concentration_state",
    },
}
D_ONLY = {
    "risk_contribution", "budget_deviation", "drawdown_attribution",
    "risk_change_attribution", "concentration_state",
}


def _by_code(values: Mapping[str, float]) -> Dict[str, float]:
    """{stock: x, ...} -> {510300: x, ...}"""
    return {key_map.NAME_TO_CODE.get(k, k): float(v) for k, v in values.items()}


def _round(obj: Any, digits: int = 6) -> Any:
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (int, float)):
        return round(float(obj), digits)
    return obj


def _drawdown_attribution(da: Mapping[str, Any]) -> Dict[str, Any]:
    """读 risk_calc.drawdown_attribution 的真实结构（资产数值在 contrib 里）。"""
    contrib = da.get("contrib")
    if not isinstance(contrib, Mapping):
        raise KeyError("drawdown_attribution 缺少 contrib，需传 risk_calc 的原始输出")
    return {
        "start": str(da["start"])[:10] if da.get("start") else None,
        "end": str(da["end"])[:10],
        "transaction_cost": float(da.get("transaction_cost", 0.0)),
        **{key_map.NAME_TO_CODE[n]: float(contrib[n]) for n in NAMES},
    }


def _change_attribution(ch: Mapping[str, Any], base: str) -> Dict[str, Any]:
    """读 risk_calc.change_attribution 的真实结构（按效应分列）。"""
    need = ("weight_effect", "vol_effect", "corr_effect", "residual")
    missing = [k for k in need if not isinstance(ch.get(k), Mapping)]
    if missing:
        raise KeyError(f"change_attribution 缺少 {missing}，需传 risk_calc 的原始输出")
    effects = ch.get("effects", {})
    return {
        "base": str(base)[:10],
        **{
            key_map.NAME_TO_CODE[n]: {
                "weight": float(ch["weight_effect"].get(n, effects.get("weight", {}).get(n, 0.0))),
                "vol": float(ch["vol_effect"].get(n, effects.get("vol", {}).get(n, 0.0))),
                "corr": float(ch["corr_effect"].get(n, effects.get("corr", {}).get(n, 0.0))),
                "residual": float(ch["residual"].get(n, 0.0)),
            }
            for n in NAMES
        },
    }


def _concentration_state(conc: Mapping[str, Any], rc_pct: Mapping[str, float]) -> Dict[str, Any]:
    """concentration 本身不含逐资产 RC，逐资产值取 risk_contribution 的 rc_pct。"""
    return {
        **{key_map.NAME_TO_CODE[n]: float(rc_pct[n]) for n in NAMES},
        "neff": float(conc["neff"]),
    }


def build_feedback(facts: Mapping[str, Any], meta: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """按组组装反馈，输入是 risk_calc 的原始输出字典。

    facts 需要包含：
      portfolio_return, portfolio_volatility, current_drawdown, max_drawdown,
      cvar_95, trigger_flags, risk_contribution(rc_pct), budget_deviation,
      drawdown_attribution, change_attribution, concentration
    meta 需要包含 as_of_date，D 组还需要 base_date。
    """
    if group not in GROUP_FIELDS:
        raise KeyError(f"未知组别 {group}，可选：{list(GROUP_FIELDS)}")

    if group == "A":
        return {}

    date = meta.get("as_of_date")
    if date is None:
        raise KeyError("meta 里缺少 as_of_date")

    feedback: Dict[str, Any] = {
        "as_of_date": str(date)[:10],
        "portfolio_return": {
            "d20": float(facts["portfolio_return"]["d20"]),
            "d60": float(facts["portfolio_return"]["d60"]),
        },
    }
    if group == "B":
        return _assert_fields(feedback, group)

    vol = facts["portfolio_volatility"]
    feedback.update(
        portfolio_volatility={"d20": float(vol["d20"]), "d60": float(vol["d60"])},
        current_drawdown=float(facts["current_drawdown"]),
        max_drawdown=float(facts["max_drawdown"]),
        cvar_95=abs(float(facts["cvar_95"])),
        trigger_flags={
            "drawdown": bool(facts["trigger_flags"]["drawdown"]),
            "volatility": bool(facts["trigger_flags"]["volatility"]),
            "correlation": bool(facts["trigger_flags"]["correlation"]),
        },
    )
    if group == "C":
        return _assert_fields(feedback, group)

    rc_pct = facts["risk_contribution"]
    base = meta.get("base_date") or facts.get("base_date")
    if base is None:
        raise KeyError("D 组需要 meta['base_date']（变化归因的基准日）")

    feedback.update(
        risk_contribution=_by_code(rc_pct),
        budget_deviation=_by_code(facts["budget_deviation"]),
        drawdown_attribution=_drawdown_attribution(facts["drawdown_attribution"]),
        risk_change_attribution=_change_attribution(facts["change_attribution"], base),
        concentration_state=_concentration_state(facts["concentration"], rc_pct),
    )
    return _assert_fields(feedback, group)


def _assert_fields(feedback: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """字段集合必须恰好等于该组允许集合，D 字段不得出现在 A/B/C。"""
    fields = set(feedback) - {"as_of_date"}
    if group in ("A", "B", "C"):
        leaked = fields & D_ONLY
        if leaked:
            raise AssertionError(f"{group} 组出现 D 组字段：{sorted(leaked)}")
    if fields != GROUP_FIELDS[group]:
        raise AssertionError(
            f"{group} 组字段不符：多 {sorted(fields - GROUP_FIELDS[group])}，"
            f"少 {sorted(GROUP_FIELDS[group] - fields)}"
        )
    return _round(dict(feedback))


def to_model_input(market_data: Mapping[str, Any], facts: Mapping[str, Any],
                   meta: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """正式入口：risk_calc 结果 → 反馈 → 1 号 build_model_input。"""
    from llm_decision.validator import build_model_input

    return build_model_input(dict(market_data), group, build_feedback(facts, meta, group))


def data_health(returns, as_of: str | None = None) -> Dict[str, Any]:
    """数据体检（市场输入侧，不进入风险反馈字段）。"""
    import pandas as pd

    if len(returns) == 0:
        raise ValueError("空数据：没有可体检的交易日")
    idx = pd.to_datetime(returns.index)
    as_of_ts = pd.to_datetime(as_of) if as_of else idx[-1]
    return {
        "rows": int(len(returns)),
        "start": idx[0].strftime("%Y-%m-%d"),
        "end": idx[-1].strftime("%Y-%m-%d"),
        "missing_values": int(returns.isna().sum().sum()),
        "duplicate_dates": int(idx.duplicated().sum()),
        "zero_return_days": int((returns.abs() < 1e-12).all(axis=1).sum()),
        "extreme_days": int((returns.abs() > 0.20).any(axis=1).sum()),
        "stale": bool(idx[-1] < as_of_ts),
    }
