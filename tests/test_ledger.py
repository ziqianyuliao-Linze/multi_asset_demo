"""回测引擎测试（不依赖真数据）：

    python -m unittest discover -s tests -v
"""

import unittest

from backtest_engine import (CODES, InputError, lots_for, max_affordable,
                             run_backtest, summarize)

DATES = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
RETS = {
    "510300": [0.0, 0.01, -0.01, 0.02],
    "511010": [0.0, 0.001, 0.0, 0.001],
    "518880": [0.0, -0.01, 0.02, -0.01],
}
PLAN = {"2026-01-05": {"510300": 0.40, "511010": 0.30, "518880": 0.30},
        "2026-01-08": {"510300": 0.50, "511010": 0.25, "518880": 0.25}}
PRICES = {"2026-01-05": {"510300": 4.00, "511010": 100.0, "518880": 8.00},
          "2026-01-06": {"510300": 4.04, "511010": 100.1, "518880": 7.92},
          "2026-01-07": {"510300": 4.00, "511010": 100.0, "518880": 8.08},
          "2026-01-08": {"510300": 4.08, "511010": 100.2, "518880": 8.00}}


class TestContinuous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.led = run_backtest(RETS, PLAN, DATES, mode="continuous")
        cls.rows = cls.led.rows

    def test_01_建仓日扣掉双边成本(self):
        self.assertAlmostEqual(self.rows[0]["nav_end"], 999700.00, places=2)
        self.assertAlmostEqual(self.rows[0]["cost"], 300.00, places=6)
        self.assertAlmostEqual(self.rows[0]["turnover"], 1000000.00, places=4)
        self.assertAlmostEqual(self.rows[0]["net_return"], -0.0003, places=9)

    def test_02_第二天只看收益不再收费(self):
        self.assertAlmostEqual(self.rows[1]["nav_end"], 1000999.61, places=2)
        self.assertAlmostEqual(self.rows[1]["gross_return"], 0.0013, places=9)
        self.assertAlmostEqual(self.rows[1]["cost"], 0.0, places=9)

    def test_03_调仓日换手等于权重变化(self):
        r = self.rows[3]
        self.assertAlmostEqual(r["turnover"], 192493.93, places=1)
        self.assertAlmostEqual(r["nav_end"], 1008109.81, places=2)
        self.assertAlmostEqual(r["cost"], r["turnover"] * 0.0003, places=6)

    def test_04_净值等于市值加现金(self):
        for r in self.rows:
            self.assertAlmostEqual(r["nav_end"], r["market_value"] + r["cash"], places=6)
            self.assertAlmostEqual(r["cash"], 0.0, places=9)

    def test_05_调仓后权重等于目标(self):
        for c, w in PLAN["2026-01-08"].items():
            self.assertAlmostEqual(self.rows[3]["weights"][c], w, places=9)

    def test_06_不调仓的日子没有成本(self):
        self.assertAlmostEqual(self.rows[1]["cost"] + self.rows[2]["cost"], 0.0, places=9)

    def test_07_汇总字段齐全(self):
        s = summarize(self.led)
        for k in ("days", "final_nav", "total_return", "annual_vol", "max_drawdown", "total_cost"):
            self.assertIn(k, s)
        self.assertEqual(s["days"], 4)

    def test_08_账本列按规范5_4齐全(self):
        row = self.led.to_rows()[0]
        for k in ("as_of_date", "group", "nav_begin", "nav_before_trade", "nav_end",
                  "gross_return", "net_return", "cost", "turnover"):
            self.assertIn(k, row)
        for c in CODES:
            self.assertIn(f"w_{c}", row)
            self.assertIn(f"r_{c}", row)
            self.assertIn(f"budget_{c}", row)


class TestLots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.led = run_backtest(RETS, {"2026-01-05": PLAN["2026-01-05"]}, DATES,
                               prices=PRICES, mode="lots")
        cls.rows = cls.led.rows

    def test_09_现金不够就少买一手(self):
        self.assertAlmostEqual(self.rows[0]["cost"], 299.76, places=2)
        self.assertAlmostEqual(self.rows[0]["cash"], 500.24, places=2)
        self.assertAlmostEqual(self.rows[0]["nav_end"], 999700.24, places=2)

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
        with self.assertRaises(InputError):
            run_backtest(RETS, {"2026-01-05": {"510300": 0.7, "511010": 0.2, "518880": 0.1}},
                         DATES)
        with self.assertRaises(InputError):
            run_backtest(RETS, {"2026-01-09": PLAN["2026-01-05"]}, DATES)
        with self.assertRaises(InputError):
            run_backtest(RETS, PLAN, DATES, mode="lots")

    def test_13_截断样本不改变历史(self):
        full = run_backtest(RETS, PLAN, DATES, mode="continuous")
        part = run_backtest({c: RETS[c][:2] for c in CODES},
                            {"2026-01-05": PLAN["2026-01-05"]}, DATES[:2], mode="continuous")
        for i in range(2):
            self.assertAlmostEqual(full.rows[i]["nav_end"], part.rows[i]["nav_end"], places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)