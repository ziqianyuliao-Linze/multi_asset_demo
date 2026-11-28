"""回测引擎单元测试（不依赖真数据）：

    python -m unittest discover -s tests -v

2026-11-28 评审口径变更：交易费改为隐式方程求解、换手率除以 2V⁻、
账本字段与 interfaces.BacktestRecord 对齐、主实验取消 5%–60% 限仓。
本文件里对应的期望值已按新口径重算（原口径：cost=300.00、turnover=1,000,000）。
"""

import unittest

from backtest_engine import (CODES, InputError, lots_for, max_affordable,
                             run_backtest, summarize, summarize_formal)
from interfaces.data_schema import BacktestRecord

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
RETS = {
    "510300": [0.0, 0.01, -0.01, 0.02, 0.0],
    "511010": [0.0, 0.001, 0.0, 0.001, 0.0],
    "518880": [0.0, -0.01, 0.02, -0.01, 0.0],
}
PLAN = {"2026-01-05": {"510300": 0.40, "511010": 0.30, "518880": 0.30},
        "2026-01-08": {"510300": 0.50, "511010": 0.25, "518880": 0.25}}
PRICES = {"2026-01-05": {"510300": 4.00, "511010": 100.0, "518880": 8.00},
          "2026-01-06": {"510300": 4.04, "511010": 100.1, "518880": 7.92},
          "2026-01-07": {"510300": 4.00, "511010": 100.0, "518880": 8.08},
          "2026-01-08": {"510300": 4.08, "511010": 100.2, "518880": 8.00},
          "2026-01-09": {"510300": 4.08, "511010": 100.2, "518880": 8.00}}


class TestContinuous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.led = run_backtest(RETS, PLAN, DATES, mode="continuous")
        cls.rows = cls.led.rows

    def test_01_建仓日扣掉隐式交易费(self):
        # F = 0.0003 × (1,000,000 − F)  =>  F = 300 / 1.0003
        self.assertAlmostEqual(self.rows[0]["ending_nav"], 999700.089973008, places=6)
        self.assertAlmostEqual(self.rows[0]["cost_amount"], 300 / 1.0003, places=6)
        # 换手率 = Σ|Trade| / (2V⁻) = (1,000,000 − F) / 2,000,000
        self.assertAlmostEqual(self.rows[0]["turnover"], 0.5 / 1.0003, places=12)
        self.assertAlmostEqual(self.rows[0]["net_return"], -300 / 1.0003 / 1e6, places=12)

    def test_02_第二天只看收益不再收费(self):
        self.assertAlmostEqual(self.rows[1]["ending_nav"], 1000999.700089973, places=6)
        self.assertAlmostEqual(self.rows[1]["gross_return"], 0.0013, places=12)
        self.assertAlmostEqual(self.rows[1]["cost_amount"], 0.0, places=12)
        self.assertAlmostEqual(self.rows[1]["turnover"], 0.0, places=12)

    def test_03_调仓日换手等于成交额除以二倍成交前净值(self):
        r = self.rows[3]
        self.assertAlmostEqual(r["turnover"], 0.095467232990, places=10)
        self.assertAlmostEqual(r["ending_nav"], 1008109.901519544, places=5)
        # 费用与换手率的恒等关系：F = rate × 2V⁻ × turnover
        self.assertAlmostEqual(r["cost_amount"],
                               r["turnover"] * 2 * r["pre_trade_nav"] * 0.0003, places=9)

    def test_04_净值等于市值加现金(self):
        for r in self.rows:
            self.assertAlmostEqual(r["ending_nav"],
                                   r["pre_trade_nav"] - r["cost_amount"], places=9)
            self.assertAlmostEqual(r["gross_return"],
                                   r["pre_trade_nav"] / r["beginning_nav"] - 1, places=12)

    def test_05_调仓后权重等于目标(self):
        # 2026-01-08 收盘调仓，01-09 开盘（收益发生前）的权重就是目标权重
        for c, w in PLAN["2026-01-08"].items():
            self.assertAlmostEqual(self.rows[4]["actual_weights"][c], w, places=12)

    def test_06_不调仓的日子没有成本(self):
        self.assertAlmostEqual(self.rows[1]["cost_amount"] + self.rows[2]["cost_amount"], 0.0,
                               places=12)

    def test_07_汇总字段齐全(self):
        s = summarize(self.led)
        for k in ("days", "final_nav", "total_return", "annual_vol", "max_drawdown", "total_cost"):
            self.assertIn(k, s)
        self.assertEqual(s["days"], 5)

    def test_08_账本列与BacktestRecord一致(self):
        row = self.led.to_rows()[0]
        self.assertEqual(set(row), set(BacktestRecord.__annotations__))
        self.assertNotIn("trade_amount", row)
        self.assertIsInstance(row["asset_returns"], dict)
        self.assertIsInstance(row["actual_weights"], dict)
        self.assertIsInstance(row["effective_budget"], dict)


class TestLots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.led = run_backtest(RETS, {"2026-01-05": PLAN["2026-01-05"]}, DATES,
                               prices=PRICES, mode="lots")
        cls.rows = cls.led.rows

    def test_09_现金不够就少买一手(self):
        self.assertAlmostEqual(self.rows[0]["cost_amount"], 299.76, places=2)
        self.assertAlmostEqual(self.rows[0]["ending_nav"], 999700.24, places=2)

    def test_10_持仓是整百份(self):
        shares = self.rows[0]["shares"]
        self.assertEqual(shares["510300"], 100000)
        self.assertEqual(shares["511010"], 3000)
        self.assertEqual(shares["518880"], 37400)
        for r in self.rows:
            for c in CODES:
                self.assertEqual(r["shares"][c] % 100, 0)

    def test_11_取整函数(self):
        self.assertEqual(lots_for(400000, 4.00), 100000)
        self.assertEqual(lots_for(299999, 8.00), 37400)
        self.assertEqual(lots_for(7.99, 8.00), 0)
        self.assertEqual(max_affordable(8.00, 400.0, 100), 0)
        self.assertEqual(max_affordable(8.00, 1000.0, 100), 100)


class TestInput(unittest.TestCase):
    def test_12_口径不对必须报错(self):
        with self.assertRaises(InputError):          # 权重合计不为 1
            run_backtest(RETS, {"2026-01-05": {"510300": 0.7, "511010": 0.2, "518880": 0.2}},
                         DATES)
        with self.assertRaises(InputError):          # 调仓日不在交易日历
            run_backtest(RETS, {"2026-01-09": PLAN["2026-01-08"]}, DATES[:-1], mode="continuous")
        with self.assertRaises(InputError):          # lots 模式缺价格
            run_backtest(RETS, PLAN, DATES, mode="lots")
        with self.assertRaises(InputError):          # 负数权重
            run_backtest(RETS, {"2026-01-05": {"510300": -0.1, "511010": 0.55, "518880": 0.55}},
                         DATES)

    def test_13_截断样本不改变历史(self):
        full = run_backtest(RETS, PLAN, DATES, mode="continuous")
        part = run_backtest({c: RETS[c][:2] for c in CODES},
                            {"2026-01-05": PLAN["2026-01-05"]}, DATES[:2], mode="continuous")
        for i in range(2):
            self.assertAlmostEqual(full.rows[i]["ending_nav"], part.rows[i]["ending_nav"],
                                   places=9)


class TestBounds(unittest.TestCase):
    def test_14_主实验不设五到六十的限制(self):
        plan = {"2026-01-05": {"510300": 0.70, "511010": 0.20, "518880": 0.10}}
        led = run_backtest(RETS, plan, DATES, mode="continuous")
        self.assertAlmostEqual(led.rows[0]["actual_weights"]["510300"], 0.0)  # 收益前为空仓
        self.assertGreater(led.rows[0]["cost_amount"], 0)

    def test_15_稳健性实验可以显式开限仓(self):
        plan = {"2026-01-05": {"510300": 0.70, "511010": 0.20, "518880": 0.10}}
        with self.assertRaises(InputError):
            run_backtest(RETS, plan, DATES, mode="continuous", bounds=(0.05, 0.60))


class TestFormalSummary(unittest.TestCase):
    def test_16_归一化起点与首次费用(self):
        led = run_backtest(RETS, PLAN, DATES, mode="continuous",
                           warm_start={"510300": 1 / 3, "511010": 1 / 3, "518880": 1 / 3})
        s = summarize_formal(led, start=DATES[0])
        self.assertAlmostEqual(s["start_nav"], 1.0, places=12)
        self.assertAlmostEqual(s["end_nav"], led.rows[-1]["ending_nav"] / led.rows[0]["pre_trade_nav"],
                               places=12)
        self.assertAlmostEqual(s["days"], len(DATES), places=12)

    def test_17_夏普扣除年化无风险利率(self):
        led = run_backtest(RETS, PLAN, DATES, mode="continuous",
                           warm_start={"510300": 1 / 3, "511010": 1 / 3, "518880": 1 / 3})
        s = summarize_formal(led, start=DATES[0])
        self.assertAlmostEqual(s["risk_free_annual"], 0.015, places=12)
        self.assertLess(s["sharpe"], summarize(led)["sharpe"] or 1e9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
