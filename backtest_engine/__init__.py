"""backtest_engine —— 5 号 lys 负责的回测引擎（第二阶段）。

主实验：2013-07-29 收盘等权暖启动、连续权重、月末信号次日收盘成交、
        双边万三按隐式方程扣费、逐日账本（字段对齐 interfaces.BacktestRecord）。
稳健性：100 份取整 + 现金余额（mode="lots"）、可选的 5%–60% 限仓（bounds）。
数据统一走 2 号 data_processing.data_interface，不另开数据源。
"""

from .ledger import (CODES, COST_RATE, FORMAL_START, HOLDING_COLUMNS,
                     INITIAL_CAPITAL, LOT, RECORD_FIELDS, RISK_FREE_ANNUAL,
                     TRADING_DAYS, WARM_START_DATE, WARM_START_WEIGHTS, InputError,
                     Ledger, commission, lots_for, max_affordable, run_backtest,
                     solve_fee, summarize, summarize_formal, trade_amounts,
                     turnover_ratio)

__all__ = [
    "CODES", "LOT", "COST_RATE", "INITIAL_CAPITAL", "TRADING_DAYS",
    "RISK_FREE_ANNUAL", "WARM_START_DATE", "WARM_START_WEIGHTS", "FORMAL_START",
    "RECORD_FIELDS", "HOLDING_COLUMNS", "InputError", "Ledger",
    "run_backtest", "summarize", "summarize_formal", "commission", "lots_for",
    "max_affordable", "solve_fee", "trade_amounts", "turnover_ratio",
]
__version__ = "0.2.0"
