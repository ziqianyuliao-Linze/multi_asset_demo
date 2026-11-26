# -*- coding: utf-8 -*-
"""
风险反馈模块 · 分拣打包
=======================

把计算器算出的数字分拣成四个包：

  A：日期 + 数据体检
  B：A + 近 20/60 日收益
  C：B + 波动、回撤、CVaR、三个报警
  D：C + 风险贡献、预算偏差、回撤归因、变化归因、集中度

红线：D 独有字段不得出现在 A/B/C。
实现：白名单构建（按 config.GROUPS 逐项取）+ 构建后断言（assert_no_leak）。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:
    import config as C

D_ONLY_FIELDS: Sequence[str] = (
    "risk_contribution", "budget_deviation",
    "drawdown_attribution", "risk_change_attribution", "concentration_state",
)
MARKET_BASIC_FIELDS: Sequence[str] = ("as_of_date", "data_health")
FIELD_ORDER: Sequence[str] = C.FEEDBACK_FIELDS


# ---------------------------------------------------------------- 清洗
def _clean(obj: Any) -> Any:
    """numpy / Timestamp / NaN → 可直接 json.dump 的类型。"""
    if isinstance(obj, Mapping):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else round(obj, 4)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _ordered(packet: Mapping[str, Any]) -> Dict[str, Any]:
    out = {k: packet[k] for k in FIELD_ORDER if k in packet}
    for k, v in packet.items():
        out.setdefault(k, v)
    return out


# ---------------------------------------------------------------- 数据体检
def data_health(returns: pd.DataFrame, as_of: str | None = None) -> Dict[str, Any]:
    """只体检不改数据：行数、首末日期、缺失、重复、零收益、异常涨跌、是否滞后。"""
    df = returns.copy()
    if len(df) == 0:
        raise ValueError("空数据：没有可体检的交易日")
    idx = pd.to_datetime(df.index)
    as_of_ts = pd.to_datetime(as_of) if as_of else idx[-1]
    return {
        "rows": int(len(df)),
        "start": idx[0],
        "end": idx[-1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_dates": int(idx.duplicated().sum()),
        "zero_return_days": int((df.abs() < 1e-12).all(axis=1).sum()),
        "extreme_days": int((df.abs() > 0.20).any(axis=1).sum()),
        "stale": bool(idx[-1] < as_of_ts),
    }


# ---------------------------------------------------------------- 分拣
def _allowed_fields(group: str) -> List[str]:
    if group not in C.GROUPS:
        raise KeyError(f"未知组别 {group}，可选：{list(C.GROUPS)}")
    fields: List[str] = []
    for item in C.GROUPS[group]["fields"]:            # type: ignore[index]
        fields.extend(MARKET_BASIC_FIELDS if item == "market_basic" else [item])
    return fields


def build_packets(facts: Mapping[str, Any], meta: Mapping[str, Any],
                  groups: Iterable[str] = C.GROUP_ORDER) -> Dict[str, Dict[str, Any]]:
    """把同一份 facts 分拣成 A/B/C/D。"""
    packets: Dict[str, Dict[str, Any]] = {}
    for g in groups:
        packet: Dict[str, Any] = {}
        for field in _allowed_fields(g):
            if field in MARKET_BASIC_FIELDS:
                if field not in meta:
                    raise KeyError(f"{g} 包需要 {field}，但 meta 里没有")
                packet[field] = meta[field]
            else:
                if field not in facts:
                    raise KeyError(f"{g} 包需要 {field}，但 facts 里没有（上游漏算？）")
                packet[field] = facts[field]
        packets[g] = _ordered(_clean(packet))
    assert_no_leak(packets)
    return packets


def assert_no_leak(packets: Mapping[str, Mapping[str, Any]]) -> None:
    """红线检查。"""
    for g in ("A", "B", "C"):
        leaked = [f for f in D_ONLY_FIELDS if f in packets.get(g, {})]
        if leaked:
            raise AssertionError(f"字段泄漏：{g} 包出现了 D 组专属字段 {leaked}")
    for g, p in packets.items():
        extra = [k for k in p if k not in C.FEEDBACK_FIELDS]
        if extra:
            raise AssertionError(f"{g} 包含未登记字段 {extra}")


def packet_summary(packets: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """字段对照表：每组拿到了哪些字段。"""
    return pd.DataFrame([
        {"字段": f, **{g: ("✓" if f in packets.get(g, {}) else "×") for g in C.GROUP_ORDER}}
        for f in FIELD_ORDER
    ])


# ---------------------------------------------------------------- 落盘
def save_packets(packets: Mapping[str, Mapping[str, Any]], outdir: Path | None = None) -> List[Path]:
    outdir = Path(C.FEEDBACK_DIR if outdir is None else outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for g, packet in packets.items():
        path = outdir / f"{packet.get('as_of_date', 'unknown')}_{g}.json"
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


# ---------------------------------------------------------------- 自检
def _demo() -> tuple:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    returns = pd.DataFrame(rng.normal(0.0002, 0.01, size=(300, 3)), index=idx, columns=list(C.ASSETS))
    meta = {"as_of_date": idx[-1], "data_health": data_health(returns)}
    facts = {
        "portfolio_return": {"d20": -0.0123, "d60": -0.0456},
        "portfolio_volatility": {"d20": 0.1032, "d60": 0.0987, "d250": 0.1120},
        "current_drawdown": -0.0512,
        "max_drawdown": -0.0876,
        "cvar_95": 0.0178,
        "risk_contribution": {"stock": 0.58, "bond": 0.05, "gold": 0.37},
        "budget_deviation": {"stock": 0.25, "bond": -0.28, "gold": 0.03},
        "drawdown_attribution": {"start": idx[100], "end": idx[130],
                                 "stock": -0.032, "bond": 0.004, "gold": -0.023},
        "risk_change_attribution": {
            "base": idx[50],
            "stock": {"vol": -0.03, "corr": 0.01, "weight": 0.0},
            "bond": {"vol": 0.0, "corr": 0.0, "weight": 0.0},
            "gold": {"vol": 0.02, "corr": -0.01, "weight": 0.0},
        },
        "trigger_flags": {"drawdown": False, "volatility": False, "correlation": False},
        "concentration_state": {"max_asset": "stock", "max_rc": 0.58, "neff": 2.06, "is_alert": False},
    }
    return meta, facts


if __name__ == "__main__":
    meta, facts = _demo()
    packets = build_packets(facts, meta)
    print(packet_summary(packets).to_string(index=False))
    print("\nA 包：", json.dumps(packets["A"], ensure_ascii=False))
    print("\nD 包字段：", list(packets["D"]))
    print("\n已写入：", [p.name for p in save_packets(packets)])
    print("泄漏检查：通过")
