# -*- coding: utf-8 -*-
"""风险反馈分拣：把计算器结果组装成 1 号 build_model_input 能直接吃的结构。

口径以 interfaces/data_schema.py 与 llm_decision/validator.py 为准：
  A 组反馈为空；
  B 组只有组合收益；
  C 组加波动、回撤、CVaR 与三个预警；
  D 组再加风险贡献、预算偏差、回撤归因、变化归因、集中度。
资产键一律用证券代码（510300/511010/518880），转换由 key_map 完成。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

try:
    from . import config as C
    from . import key_map
except ImportError:                      # 直接当脚本运行时
    import config as C
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import key_map

CODES = key_map.CODES
NAMES = key_map.NAMES

# 每组的字段集合，和 1 号 validator.GROUP_FEEDBACK_FIELDS 保持一致
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
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        return round(float(obj), digits)
    return obj


def build_feedback(facts: Mapping[str, Any], meta: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """按组组装反馈。返回的字典可直接传给 build_model_input(..., feedback=...)。

    facts 是 risk_calc 的输出（内部用 stock/bond/gold），这里统一转成证券代码。
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

    dd_attr = facts["drawdown_attribution"]
    change = facts["risk_change_attribution"]
    conc = facts["concentration_state"]

    feedback.update(
        risk_contribution=_by_code(facts["risk_contribution"]),
        budget_deviation=_by_code(facts["budget_deviation"]),
        drawdown_attribution={
            "start": (str(dd_attr["start"])[:10] if dd_attr.get("start") else None),
            "end": str(dd_attr["end"])[:10],
            "transaction_cost": float(dd_attr.get("transaction_cost", 0.0)),
            **{code: float(dd_attr[name]) for name, code in zip(NAMES, CODES)},
        },
        risk_change_attribution={
            "base": str(change["base"])[:10],
            **{
                code: {
                    "weight": float(change[name]["weight"]),
                    "vol": float(change[name]["vol"]),
                    "corr": float(change[name]["corr"]),
                    "residual": float(change[name].get("residual", 0.0)),
                }
                for name, code in zip(NAMES, CODES)
            },
        },
        concentration_state={
            **{code: float(conc[name]) for name, code in zip(NAMES, CODES)},
            "neff": float(conc["neff"]),
        },
    )
    return _assert_fields(feedback, group)


def _assert_fields(feedback: Mapping[str, Any], group: str) -> Dict[str, Any]:
    """红线：字段集合必须恰好等于该组允许的集合，D 字段不得出现在 A/B/C。"""
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
    """组装并交给 1 号校验：返回可直接喂给模型的输入。"""
    from llm_decision.validator import build_model_input   # 延迟导入，避免循环依赖

    feedback = build_feedback(facts, meta, group)
    return build_model_input(dict(market_data), group, feedback)


def data_health(returns, as_of: str | None = None) -> Dict[str, Any]:
    """数据体检（属于市场输入侧，不进入风险反馈字段）。"""
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
