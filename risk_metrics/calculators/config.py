# -*- coding: utf-8 -*-
"""
风险反馈模块 · 规则备忘录
=========================

只放固定数字和路径，不做计算。改规则只改这里。

约定：比例用小数（0.08 = 8%）；窗口单位是交易日；资产顺序固定 stock/bond/gold。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "risk"
FEEDBACK_DIR = OUTPUT_DIR / "feedback"

# ---------------------------------------------------------------- 资产池
ASSETS: Tuple[str, str, str] = ("stock", "bond", "gold")
ASSET_CODES: Dict[str, str] = {"stock": "510300", "bond": "511010", "gold": "518880"}
ASSET_CN: Dict[str, str] = {"stock": "沪深300ETF", "bond": "国债ETF", "gold": "黄金ETF"}
N_ASSETS: int = 3

# ---------------------------------------------------------------- 窗口
class Window:
    SHORT: int = 20
    MID: int = 60
    LONG: int = 250
    COV: int = 250
    CVAR: int = 250
    ZSCORE: int = 250
    PERCENTILE: int = 250

ANNUALIZE_DAYS: int = 250
RISK_FREE_RATE: float = 0.015

# ---------------------------------------------------------------- 尾部风险
TAIL: Dict[str, object] = {
    "var_level": 0.95,
    "cvar_level": 0.95,
    "window": Window.CVAR,
    "method": "historical",
    "sign": "positive_loss",
}

# ---------------------------------------------------------------- 回撤报警
DRAWDOWN: Dict[str, object] = {
    "alert_level": 0.08,
    "persistence_days": 3,
    "measure": "close_to_peak",
    "alert_only": True,
}

# ---------------------------------------------------------------- 波动报警
VOLATILITY: Dict[str, object] = {
    "window": Window.SHORT,
    "reference_window": Window.PERCENTILE,
    "quantile": 0.80,
    "measure": "std",
    "persistence_days": 1,
}

# ---------------------------------------------------------------- 相关性报警
CORRELATION: Dict[str, object] = {
    "pair": ("stock", "bond"),
    "window": Window.MID,
    "z_window": Window.ZSCORE,
    "z_threshold": 2.0,
}

# ---------------------------------------------------------------- 集中度（只给 D 组）
CONCENTRATION: Dict[str, object] = {
    "rc_threshold": 0.50,
    "neff_threshold": 1.8,
    "role": "state_only",
    "group": "D",
}

# ---------------------------------------------------------------- 四组可见字段
GROUPS: Dict[str, Dict[str, object]] = {
    "A": {"name": "无反馈", "fields": ["market_basic"]},
    "B": {"name": "收益反馈", "fields": ["market_basic", "portfolio_return"]},
    "C": {
        "name": "整体风险反馈",
        "fields": [
            "market_basic", "portfolio_return", "portfolio_volatility",
            "current_drawdown", "max_drawdown", "cvar_95", "trigger_flags",
        ],
    },
    "D": {
        "name": "风险归因反馈",
        "fields": [
            "market_basic", "portfolio_return", "portfolio_volatility",
            "current_drawdown", "max_drawdown", "cvar_95", "trigger_flags",
            "risk_contribution", "budget_deviation",
            "drawdown_attribution", "risk_change_attribution",
            "concentration_state",
        ],
    },
}
GROUP_ORDER: Tuple[str, str, str, str] = ("A", "B", "C", "D")

# ---------------------------------------------------------------- 接口字段
FEEDBACK_FIELDS: Tuple[str, ...] = (
    "as_of_date", "data_health", "portfolio_return", "portfolio_volatility",
    "current_drawdown", "max_drawdown", "cvar_95", "risk_contribution",
    "budget_deviation", "drawdown_attribution", "risk_change_attribution",
    "trigger_flags", "concentration_state",
)
JSON_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------- 自检
def print_rule_sheet() -> None:
    print("风险反馈模块 · 规则表")
    print("-" * 64)
    print(f"资产池     : {', '.join(ASSET_CODES[a] + ' ' + ASSET_CN[a] for a in ASSETS)}")
    print(f"监测窗口   : {Window.SHORT} / {Window.MID} / {Window.LONG} 个交易日")
    print(f"协方差窗口 : {Window.COV} 个交易日；年化 √{ANNUALIZE_DAYS}")
    print(f"无风险利率 : {RISK_FREE_RATE:.1%}")
    print(f"CVaR       : {TAIL['method']}，{TAIL['cvar_level']:.0%} 分位，窗口 {TAIL['window']} 日，正数表示损失")
    print(f"            k = (1−95%)×250 = 12.5 笔：前 12 笔全权重 + 第 13 笔半权重，合计 ÷ 12.5")
    print(f"回撤报警   : 回撤 < −{DRAWDOWN['alert_level']:.0%}，连续 {DRAWDOWN['persistence_days']} 日")
    print(f"波动报警   : {VOLATILITY['window']} 日波动 > 之前 {VOLATILITY['reference_window']} 个观测的 {VOLATILITY['quantile']:.0%} 分位")
    print(f"相关报警   : {CORRELATION['window']} 日相关，Z > {CORRELATION['z_threshold']:g}")
    print(f"集中度     : RC > {CONCENTRATION['rc_threshold']:.0%} 或 N_eff < {CONCENTRATION['neff_threshold']:g}（只给 {CONCENTRATION['group']} 组）")
    print("-" * 64)
    for g in GROUP_ORDER:
        print(f"{g}（{GROUPS[g]['name']}）：{', '.join(GROUPS[g]['fields'])}")


if __name__ == "__main__":
    print_rule_sheet()
