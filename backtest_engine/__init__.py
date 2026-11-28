"""backtest_engine —— 5 号 lys 负责的回测引擎（第二阶段）。

主实验：连续权重、收盘成交、双边万三、逐日账本。
稳健性：100 份取整 + 现金余额（mode="lots"）。
数据统一走 2 号 data_processing.data_interface，不另开数据源。
"""

from .ledger import (CODES, COST_RATE, INITIAL_CAPITAL, LOT, InputError, Ledger,
                     commission, lots_for, max_affordable, run_backtest, summarize)

__all__ = ["CODES", "LOT", "COST_RATE", "INITIAL_CAPITAL", "InputError", "Ledger",
           "run_backtest", "summarize", "commission", "lots_for", "max_affordable"]
__version__ = "0.1.0"