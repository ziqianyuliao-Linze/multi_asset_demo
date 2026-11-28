"""九条评审口径的回归测试（用仓库里的真实收益矩阵）：

    python -m unittest discover -s tests -v

覆盖：
  1. 2013-07-29 收盘等权暖启动 + 固定日期（2014-11-28 信号 / 2014-12-01 成交 /
     2014-12-02 新权重首次承接收益）；
  2. 交易费隐式方程、换手率 Σ|TradeAmount|/(2V⁻)；
  3. 风险预算与资金权重分开传递；
  4. 账本字段与 interfaces.BacktestRecord 一致 + 可直接交给 risk_metrics.read_holdings；
  5. 主实验无 5%–60% 限制（限仓只在稳健性实验显式开启）；
  6. 缺决策回退：沿用上期预算与持仓，首期回退暖启动等权，并记录状态与原因；
  7. 模拟权重只作测试夹具（本文件用内存字典，不读 data/weights.csv）；
  8. 绩效按 2014-12-01 成交前净值归一、计入首次交易费、夏普扣 1.5% 无风险利率；
  9. 手算测试 + 正式日期回归 + risk_integration / 基准策略输出的端到端接口测试。
"""

import csv
import math
import os
import tempfile
import unittest

from backtest_engine import (CODES, FORMAL_START, HOLDING_COLUMNS,
                             RISK_FREE_ANNUAL, WARM_START_DATE, summarize_formal)
from backtest_engine.from_interface import (build_formal_plan, load_returns,
                                            run_formal)
from data_processing.feature_builder import read_matrix
from interfaces.data_schema import BacktestRecord
from risk_metrics.risk_feedback import (build_feedback, portfolio_history,
                                        read_holdings, risk_contributions)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "data", "unified_returns.xlsx")

# 模拟决策（synthetic_test，只用于验证口径，不得作为正式主实验输入）
DECISION_1128 = {
    "weights": {"510300": 0.50, "511010": 0.30, "518880": 0.20},
    "budget": {"510300": 0.40, "511010": 0.35, "518880": 0.25},
}


class RealDataBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates, cls.rets = load_returns(MATRIX)
        cls.records = read_matrix(MATRIX)
        cls.decisions = {"2014-11-28": DECISION_1128}
        cls.led = run_formal(cls.rets, cls.dates, cls.decisions)
        cls.rows = {r["date"]: r for r in cls.led.rows}


class TestWarmStart(RealDataBase):
    def test_01_暖启动日与首日等权(self):
        self.assertEqual(WARM_START_DATE, "2013-07-29")
        self.assertEqual(self.led.rows[0]["date"], "2013-07-30")
        self.assertEqual(self.led.rows[0]["beginning_nav"], 1_000_000.0)
        for c in CODES:
            self.assertAlmostEqual(self.led.rows[0]["actual_weights"][c], 1 / 3, places=12)
        self.assertEqual(self.led.rows[0]["cost_amount"], 0.0)
        self.assertEqual(len(self.led.rows), len(self.dates))

    def test_02_暖启动期不调仓(self):
        # 2014-11-28 之前没有任何调仓：成本必须为 0
        before = [r for r in self.led.rows if r["date"] < FORMAL_START]
        self.assertTrue(before)
        self.assertEqual(sum(r["cost_amount"] for r in before), 0.0)
        self.assertTrue(all(r["fallback_status"] == "none" for r in before))


class TestFixedDates(RealDataBase):
    def test_03_首个信号日与首次成交日(self):
        signals, plan, missing = build_formal_plan(self.decisions, self.dates)
        self.assertEqual(signals[0], "2014-11-28")
        self.assertEqual(list(plan)[0], "2014-12-01")
        self.assertEqual(len(signals), 144)

    def test_04_2014_12_01成交日仍由旧权重承接收益(self):
        r = self.rows["2014-12-01"]
        prev = self.rows["2014-11-28"]
        # 暖启动持仓一路持有到 2014-12-01，权重已随收益漂移；成交前不改变持仓
        nav = 1.0 + prev["gross_return"]
        for c in CODES:
            drifted = prev["actual_weights"][c] * (1 + prev["asset_returns"][c]) / nav
            self.assertAlmostEqual(r["actual_weights"][c], drifted, places=12)
        expected = math.fsum(r["actual_weights"][c] * r["asset_returns"][c] for c in CODES)
        self.assertAlmostEqual(r["gross_return"], expected, places=12)
        self.assertGreater(r["cost_amount"], 0)

    def test_05_2014_12_02新权重首次承接收益(self):
        r = self.rows["2014-12-02"]
        for c in CODES:
            self.assertAlmostEqual(r["actual_weights"][c], DECISION_1128["weights"][c], places=12)
        expected = math.fsum(DECISION_1128["weights"][c] * r["asset_returns"][c] for c in CODES)
        self.assertAlmostEqual(r["gross_return"], expected, places=12)
        self.assertEqual(r["cost_amount"], 0.0)

    def test_06_首次交易费满足隐式方程(self):
        r = self.rows["2014-12-01"]
        fee = r["cost_amount"]
        # 换手率恒等式：F = rate × Σ|Trade| = rate × 2V⁻ × turnover
        self.assertAlmostEqual(fee, 0.0003 * 2 * r["pre_trade_nav"] * r["turnover"], places=6)
        # 成交后净值 = 成交前净值 − 费用
        self.assertAlmostEqual(r["ending_nav"], r["pre_trade_nav"] - fee, places=9)
        self.assertAlmostEqual(r["cost_rate"], fee / r["beginning_nav"], places=14)


class TestSeparateBudget(RealDataBase):
    def test_07_预算不等于资金权重(self):
        r = self.rows["2014-12-01"]
        self.assertAlmostEqual(r["effective_budget"]["510300"], 0.40, places=12)
        self.assertAlmostEqual(r["effective_budget"]["511010"], 0.35, places=12)
        self.assertAlmostEqual(r["effective_budget"]["518880"], 0.25, places=12)
        self.assertNotAlmostEqual(r["actual_weights"]["510300"],
                                  r["effective_budget"]["510300"], places=3)

    def test_08_缺省预算不复制目标权重(self):
        decisions = {"2014-11-28": {"weights": {"510300": 0.60, "511010": 0.20, "518880": 0.20},
                                     "budget": None}}
        led = run_formal(self.rets, self.dates, decisions)
        row = [r for r in led.rows if r["date"] == "2014-12-01"][0]
        # 决策只给权重、不给预算 → 预算保持暖启动等权，不能复制成 0.60/0.20/0.20
        for c in CODES:
            self.assertAlmostEqual(row["effective_budget"][c], 1 / 3, places=12)
        self.assertAlmostEqual(row["actual_weights"]["510300"], 0.4025162771, places=9)


class TestFallback(RealDataBase):
    def test_09_缺决策沿用上期持仓(self):
        # 2014-12-31 是月末信号但没有决策 → 2015-01-05 不成交
        periods = {p["trade_date"]: p for p in self.led.periods}
        self.assertIn("2015-01-05", periods)
        p = periods["2015-01-05"]
        self.assertEqual(p["status"], "hold_previous")
        self.assertIn("2014-12-31", p["reason"])
        r = self.rows["2015-01-05"]
        self.assertEqual(r["cost_amount"], 0.0)
        self.assertEqual(r["turnover"], 0.0)

    def test_10_首期失败回退暖启动等权(self):
        led = run_formal(self.rets, self.dates, {})
        first = led.periods[0]
        self.assertEqual(first["status"], "warm_start")
        self.assertIn("2014-11-28", first["reason"])
        row = [r for r in led.rows if r["date"] == first["trade_date"]][0]
        for c in CODES:
            self.assertAlmostEqual(first["weights"][c], row["actual_weights"][c], places=12)
            self.assertAlmostEqual(sum(first["weights"].values()), 1.0, places=12)
        self.assertEqual(sum(r["cost_amount"] for r in led.rows), 0.0)

    def test_11_每期都有状态和原因(self):
        self.assertEqual(len(self.led.periods), 144)
        for p in self.led.periods:
            self.assertIn(p["status"], ("applied", "hold_previous", "warm_start"))
            self.assertTrue(p["reason"])
            self.assertIn(p["signal_date"], self.dates)


class TestRecordContract(RealDataBase):
    def test_12_账本字段与BacktestRecord完全一致(self):
        expected = set(BacktestRecord.__annotations__)
        for row in self.led.to_rows():
            self.assertEqual(set(row), expected)
            self.assertEqual(set(row["asset_returns"]), set(CODES))
            self.assertEqual(set(row["actual_weights"]), set(CODES))
            self.assertEqual(set(row["effective_budget"]), set(CODES))

    def test_13_持仓列与read_holdings一致(self):
        rows = self.led.to_holdings_rows()
        self.assertEqual(tuple(rows[0]), HOLDING_COLUMNS)


class TestRiskIntegrationE2E(RealDataBase):
    def test_14_持仓CSV可直接交给read_holdings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "holdings_A.csv")
            rows = self.led.to_holdings_rows()
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(HOLDING_COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
            groups = read_holdings(path)
        self.assertEqual(set(groups), set("ABCD"))
        self.assertEqual(len(groups["A"]), len(self.dates))
        self.assertEqual(groups["A"][0]["date"], "2013-07-30")

    def test_15_风险模块复算的净收益与账本一致(self):
        as_of = "2016-12-30"
        groups = {"A": [dict(date=r["date"],
                             weights=[r["actual_weights"][c] for c in CODES],
                             budget=[r["effective_budget"][c] for c in CODES],
                             cost=r["cost_rate"]) for r in self.led.rows]}
        path = portfolio_history(self.records, groups["A"], as_of)
        window = [r for r in self.led.rows if groups["A"][0]["date"] <= r["date"] <= as_of]
        self.assertEqual(len(path["returns"]), len(window))
        for got, row in zip(path["returns"], window):
            self.assertAlmostEqual(got, row["net_return"], places=12)

    def test_16_风险贡献合计为一(self):
        as_of = "2016-12-30"
        groups = {"B": [dict(date=r["date"],
                             weights=[r["actual_weights"][c] for c in CODES],
                             budget=[r["effective_budget"][c] for c in CODES],
                             cost=r["cost_rate"]) for r in self.led.rows]}
        feedback, audit = build_feedback(self.records, groups["B"], as_of, "B", "2014-12-01")
        self.assertIn("portfolio_return", feedback)
        self.assertEqual(feedback["as_of_date"], as_of)
        self.assertIsInstance(audit, dict)

    def test_17_基准策略输出端到端(self):
        # 3 号等权基准策略在 2014-11-28 给出的输出：{资产: 1/3}
        baseline = {c: 1 / 3 for c in CODES}
        led = run_formal(self.rets, self.dates, {"2014-11-28": {"weights": baseline,
                                                                "budget": None}})
        row = [r for r in led.rows if r["date"] == "2014-12-02"][0]
        for c in CODES:
            self.assertAlmostEqual(row["actual_weights"][c], 1 / 3, places=12)
        # 等权基准与漂移后的暖启动持仓不同 → 首笔换手与费用都为正
        trade = [r for r in led.rows if r["date"] == "2014-12-01"][0]
        self.assertGreater(trade["turnover"], 0)
        self.assertAlmostEqual(trade["cost_amount"],
                               0.0003 * 2 * trade["pre_trade_nav"] * trade["turnover"], places=6)

        cov = [[0.0004, 0.0001, 0.0000],
               [0.0001, 0.0002, 0.0000],
               [0.0000, 0.0000, 0.0006]]
        contributions = risk_contributions([1 / 3, 1 / 3, 1 / 3], cov)
        self.assertAlmostEqual(math.fsum(contributions), 1.0, places=12)


class TestFormalSummary(RealDataBase):
    def test_18_归一化与首次费用(self):
        s = summarize_formal(self.led)
        self.assertEqual(s["start_date"], FORMAL_START)
        self.assertEqual(s["days"], len(self.dates) - self.dates.index(FORMAL_START))
        self.assertAlmostEqual(s["start_nav"], 1.0, places=12)
        first = self.rows[FORMAL_START]
        self.assertAlmostEqual(s["end_nav"],
                               self.led.rows[-1]["ending_nav"] / first["pre_trade_nav"], places=9)
        # 首个归一化收益只含首次交易费：−F / V⁻
        self.assertAlmostEqual(first["ending_nav"] / first["pre_trade_nav"] - 1,
                               -first["cost_amount"] / first["pre_trade_nav"], places=12)

    def test_19_夏普扣除年化无风险利率(self):
        s = summarize_formal(self.led)
        self.assertEqual(s["risk_free_annual"], 1.5 / 100)
        rows = [r for r in self.led.rows if r["date"] >= FORMAL_START]
        rets = [rows[0]["ending_nav"] / rows[0]["pre_trade_nav"] - 1]
        rets += [r["net_return"] for r in rows[1:]]
        rf = (1 + RISK_FREE_ANNUAL) ** (1 / 250) - 1
        excess = [r - rf for r in rets]
        n = len(excess)
        mean = math.fsum(excess) / n
        var = math.fsum((x - mean) ** 2 for x in excess) / (n - 1)
        expected = mean / math.sqrt(var) * math.sqrt(250)
        self.assertAlmostEqual(s["sharpe"], expected, places=12)

    def test_20_最大回撤与Calmar(self):
        s = summarize_formal(self.led)
        self.assertLess(s["max_drawdown"], 0)
        self.assertAlmostEqual(s["calmar"], s["annual_return"] / abs(s["max_drawdown"]), places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
