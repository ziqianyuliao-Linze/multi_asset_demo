"""2号数据处理模块独立测试（标准库 unittest + pandas）。

运行（仓库根目录）：python tests/test_data_processing.py
覆盖：冻结矩阵读取与质量、防未来泄漏、手工复算对照、与1号 feature_builder 数值一致性、
窗口不足状态、非法截止日、交易日历与月末信号、接口 schema 形状。
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from statistics import stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_processing"))

import feature_builder as legacy  # noqa: E402  1号既有实现，仅做数值对照
from interfaces import data_schema  # noqa: E402
from data_processing import data_interface as di  # noqa: E402
from data_processing.data_loader import (ASSET_CODES, check_matrix,  # noqa: E402
                                         file_sha256_12, load_unified_returns)
from data_processing.market_features import compute_market_features  # noqa: E402

AS_OF = "2026-11-20"


class TestLoaderAndQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_unified_returns()

    def test_frozen_shape_and_range(self):
        self.assertEqual(len(self.df), 3236)
        self.assertEqual(self.df.index[0], "2013-07-30")
        self.assertEqual(self.df.index[-1], AS_OF)
        self.assertEqual(list(self.df.columns), list(ASSET_CODES))

    def test_frozen_hash(self):
        self.assertEqual(file_sha256_12(), "c8ae2dad3039")

    def test_quality_all_pass(self):
        report = check_matrix(self.df)
        for key in ("asset_order_ok", "dates_unique", "dates_ascending",
                    "no_missing_values", "no_infinite_values",
                    "common_calendar", "no_forward_fill"):
            self.assertTrue(report[key], key)

    def test_source_label_relative(self):
        self.assertEqual(check_matrix(self.df)["source"]["file"],
                         "data/unified_returns.xlsx")

    def test_bad_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_unified_returns(ROOT / "data" / "not_exist.xlsx")


class TestNoLookahead(unittest.TestCase):
    def test_truncation_equals_full(self):
        df = load_unified_returns()
        cut = "2024-06-28"
        full = compute_market_features(df, cut)
        trunc = compute_market_features(df.loc[:cut], cut)
        self.assertEqual(full, trunc)

    def test_window_slice(self):
        win = di.get_returns_window(AS_OF, 250)
        self.assertEqual(len(win), 250)
        self.assertEqual(win.index[-1], AS_OF)
        self.assertTrue((win.index <= AS_OF).all())


class TestKnownValues(unittest.TestCase):
    """用独立手工公式复算 2026-11-20 全部特征与相关矩阵。"""

    @classmethod
    def setUpClass(cls):
        cls.df = load_unified_returns()
        cls.payload = di.get_market_data_input(AS_OF)
        cls.series = {c: cls.df[c].tolist() for c in ASSET_CODES}

    def test_returns_and_vols(self):
        mf = self.payload["market_features"]
        for c, name in di.CODE_TO_NAME.items():
            s = self.series[c]
            for w in (20, 60):
                expect = math.prod(1 + x for x in s[-w:]) - 1
                self.assertAlmostEqual(mf[f"asset_return_{w}d"][name], expect, places=12)
            for w in (20, 60, 250):
                expect = stdev(s[-w:]) * math.sqrt(250)
                self.assertAlmostEqual(mf[f"asset_volatility_{w}d"][name], expect, places=12)

    def test_correlation_matrices(self):
        ac = self.payload["asset_correlations"]
        for w in (60, 250):
            matrix = ac[f"corr_{w}d"]
            self.assertEqual(list(matrix), list(ASSET_CODES))
            for i, a in enumerate(ASSET_CODES):
                self.assertEqual(matrix[a][a], 1.0)
                for b in ASSET_CODES[i + 1:]:
                    xs, ys = self.series[a][-w:], self.series[b][-w:]
                    mx, my = sum(xs) / w, sum(ys) / w
                    xc = [x - mx for x in xs]
                    yc = [y - my for y in ys]
                    expect = sum(x * y for x, y in zip(xc, yc)) / math.sqrt(
                        sum(x * x for x in xc) * sum(y * y for y in yc))
                    self.assertAlmostEqual(matrix[a][b], expect, places=12)
                    self.assertAlmostEqual(matrix[b][a], expect, places=12)

    def test_legacy_parity(self):
        records = legacy.read_matrix(ROOT / "data" / "unified_returns.xlsx")
        old = legacy.build_features(records, AS_OF)
        mf = self.payload["market_features"]
        for c, name in di.CODE_TO_NAME.items():
            for w in (20, 60):
                self.assertAlmostEqual(mf[f"asset_return_{w}d"][name],
                                       old["assets"][c][f"return_{w}d"], places=12)
            for w in (20, 60, 250):
                self.assertAlmostEqual(mf[f"asset_volatility_{w}d"][name],
                                       old["assets"][c][f"volatility_{w}d_ann"], places=12)
        for i, a in enumerate(ASSET_CODES):
            for b in ASSET_CODES[i + 1:]:
                self.assertAlmostEqual(self.payload["asset_correlations"]["corr_60d"][a][b],
                                       old["correlations_60d"][f"{a}_{b}"], places=12)

    def test_json_serializable(self):
        json.dumps(self.payload, ensure_ascii=False, allow_nan=False)


class TestInsufficientWindow(unittest.TestCase):
    def test_mid_window(self):
        payload = di.get_market_data_input("2013-12-31")
        q = payload["data_quality"]
        self.assertGreaterEqual(q["sample_days_to_date"], 60)
        self.assertLess(q["sample_days_to_date"], 250)
        self.assertIsNotNone(payload["market_features"]["asset_return_60d"]["stock"])
        self.assertIsNone(payload["market_features"]["asset_volatility_250d"]["stock"])
        self.assertIsNone(payload["asset_correlations"]["corr_250d"])
        self.assertIn("volatility_250d", q["insufficient_windows"])
        self.assertIn("corr_250d", q["insufficient_windows"])
        self.assertEqual(q["window_status"]["corr_250d"]["reason"], "insufficient_window")

    def test_short_window(self):
        payload = di.get_market_data_input("2013-08-30")
        q = payload["data_quality"]
        self.assertLess(q["sample_days_to_date"], 60)
        self.assertIsNotNone(payload["market_features"]["asset_return_20d"]["bond"])
        self.assertIsNone(payload["market_features"]["asset_return_60d"]["bond"])
        self.assertIn("return_60d", q["insufficient_windows"])


class TestInvalidAsOf(unittest.TestCase):
    def test_non_trading_day(self):
        for bad in ("2026-11-21", "2015-01-01", "2013-07-29"):
            with self.assertRaises(ValueError):
                di.get_market_data_input(bad)

    def test_bad_format(self):
        with self.assertRaises(ValueError):
            di.get_market_data_input("2015/01/05")


class TestCalendar(unittest.TestCase):
    def test_month_end_signals(self):
        signals = di.month_end_signal_dates()
        self.assertIn("2014-11-28", signals)
        self.assertEqual(signals[-1], "2026-10-30")  # 2026-11 不完整，不计入
        self.assertEqual(di.next_trading_day("2014-11-28"), "2014-12-01")
        self.assertIsNone(di.next_trading_day(AS_OF))


class TestSchemaShape(unittest.TestCase):
    def test_four_fields(self):
        payload = di.get_market_data_input(AS_OF)
        self.assertEqual(set(payload), set(data_schema.MarketDataInput.__annotations__))
        self.assertEqual(set(payload["market_features"]),
                         set(data_schema.MarketFeatures.__annotations__))
        for key, ann in data_schema.MarketFeatures.__annotations__.items():
            self.assertEqual(set(payload["market_features"][key]),
                             set(data_schema.AssetValues.__annotations__))
        self.assertTrue(data_schema.validate_date(payload["as_of_date"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)