"""接 2 号的 data_processing.data_interface：不自己解 xlsx，也不改它的代码。

主实验（连续权重）只需要收益率矩阵，不需要价格。
"""

from .ledger import CODES, InputError


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


def month_end_plan(weights, path=None):
    """月末信号日 → 次一交易日收盘成交（分工规范 5.4 与设计方案口径）。

    weights: {信号日: {代码: 权重}}，通常由 4 号提案或 3 号策略给出。
    """
    from data_processing.data_interface import (month_end_signal_dates,
                                                next_trading_day)

    signals = month_end_signal_dates(path)
    plan = {}
    missing = []
    for s in signals:
        if s not in weights:
            missing.append(s)
            continue
        nxt = next_trading_day(s, path)
        if nxt is None:
            continue          # 数据末尾那个月的信号还没到成交日
        plan[nxt] = weights[s]
    return plan, missing


def latest_signal_date(path=None):
    from data_processing.data_interface import month_end_signal_dates

    ss = month_end_signal_dates(path)
    return ss[-1] if ss else None