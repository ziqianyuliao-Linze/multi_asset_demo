# from_interface.py
"""接 2 号的 data_processing.data_interface：不自己解 xlsx，也不改它的代码。

主实验（连续权重）只需要收益率矩阵，不需要价格。
决策输入统一成 {信号日: {"weights": {...}, "budget": {...} 或 None}}：
  - weights 是 1 号模型或 3 号基准策略给出的目标资金权重；
  - budget 是同一期生效的风险预算，与资金权重分开传递，缺省时沿用上一期，
    绝不把 target weight 复制成 budget（评审口径第 3 条）。
"""

import csv

from .ledger import (CODES, FORMAL_START, WARM_START_WEIGHTS, InputError,
                     run_backtest)


def load_returns(path=None):
    """返回 (dates, returns)。dates 是 YYYY-MM-DD 列表，returns 是 {代码: [小数收益]}。"""
    from data_processing.data_interface import load_unified_returns

    df = load_unified_returns(path)
    dates = [str(d)[:10] for d in df.index]
    if len(set(dates)) != len(dates):
        raise InputError("矩阵日期有重复")
    returns = {}
    for c in CODES:
        if c not in df.columns:
            raise InputError(f"矩阵里没有 {c} 列，现有列：{list(df.columns)}")
        col = df[c].astype(float).tolist()
        if any(v is None or v != v for v in col):
            raise InputError(f"{c} 有缺失值，先让 2 号补")
        returns[c] = col
    return dates, returns


def month_end_signal_dates(dates):
    """各完整月份最后一个共同交易日（数据末尾不完整的月份不计入）。"""
    groups = {}
    for d in dates:
        groups.setdefault(d[:7], []).append(d)
    signals = []
    for month in sorted(groups):
        y, m = int(month[:4]), int(month[5:7])
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        if f"{ny}-{nm:02d}" in groups:
            signals.append(groups[month][-1])
    return signals


def next_trading_day(date_string, dates):
    later = [d for d in dates if d > date_string]
    return later[0] if later else None


def month_end_plan(weights, dates, path=None):
    """（兼容旧接口）{信号日: 权重} → ({成交日: 权重}, 缺决策的信号日)。

    path 参数保留给调用方传矩阵路径，本函数不再自己读文件。
    """
    plan = {}
    missing = []
    for s in month_end_signal_dates(dates):
        if s not in weights:
            missing.append(s)
            continue
        nxt = next_trading_day(s, dates)
        if nxt is None:          # 数据末尾那个月的信号还没到成交日
            continue
        plan[nxt] = weights[s]
    return plan, missing


def latest_signal_date(dates):
    signals = month_end_signal_dates(dates)
    return signals[-1] if signals else None


def load_decisions(path):
    """读决策表（CSV，UTF-8）。

    必需列：date, 510300, 511010, 518880
    可选列：budget_510300, budget_511010, budget_518880（缺省表示该期没有新预算）
    """
    decisions = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        need = {"date", *CODES}
        if not need.issubset(fields):
            raise InputError(f"决策表缺少列 {sorted(need - fields)}，实际列 {sorted(fields)}")
        budget_cols = {f"budget_{c}" for c in CODES}
        has_budget = budget_cols.issubset(fields)
        for row in reader:
            d = (row.get("date") or "").strip()
            if not d:
                continue
            if d in decisions:
                raise InputError(f"决策表日期重复：{d}")
            weights = {c: float(row[c]) for c in CODES}
            budget = None
            if has_budget and all((row.get(f"budget_{c}") or "").strip() for c in CODES):
                budget = {c: float(row[f"budget_{c}"]) for c in CODES}
            decisions[d] = {"weights": weights, "budget": budget}
    return decisions


def formal_signals(dates, start=FORMAL_START):
    """正式绩效起点当日成交，其前一交易日即首个模型信号日（口径 9.2）。"""
    signals = month_end_signal_dates(dates)
    first = [d for d in dates if d < start]
    if not first:
        raise InputError(f"交易日内没有早于 {start} 的日期，无法确定首个信号日")
    first_signal = first[-1]
    selected = [s for s in signals if s >= first_signal]
    if not selected or selected[0] != first_signal:
        raise InputError(f"{first_signal} 不是月末信号日，检查交易日历")
    return selected


def build_formal_plan(decisions, dates, start=FORMAL_START):
    """返回 (signals, plan, missing)：signals 覆盖全部信号期，plan 只含有决策的期。"""
    signals = formal_signals(dates, start)
    plan = {}
    missing = []
    for s in signals:
        if s not in decisions:
            missing.append(s)
            continue
        nxt = next_trading_day(s, dates)
        if nxt is None:
            continue
        item = decisions[s]
        plan[nxt] = {"weights": item["weights"], "budget": item.get("budget"),
                     "signal_date": s}
    return signals, plan, missing


def run_formal(returns, dates, decisions, initial=None, rate=None, group="A",
               start=FORMAL_START, warm_start=None):
    """跑正式主实验：暖启动 + 月末信号次日成交 + 缺决策回退 + 归一化绩效。"""
    from .ledger import COST_RATE, INITIAL_CAPITAL

    signals, plan, missing = build_formal_plan(decisions, dates, start)
    led = run_backtest(
        returns, plan, dates,
        initial=INITIAL_CAPITAL if initial is None else initial,
        rate=COST_RATE if rate is None else rate,
        mode="continuous", group=group,
        warm_start=dict(WARM_START_WEIGHTS) if warm_start is None else warm_start,
        signals=signals,
    )
    led.meta["formal_start"] = start
    led.meta["missing_signals"] = list(missing)
    return led
