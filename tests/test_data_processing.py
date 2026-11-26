"""2号数据处理模块独立测试（标准库 unittest + pandas）。

运行（仓库根目录）：python tests/test_data_processing.py
覆盖：冻结矩阵读取与质量、防未来泄漏、手工复算对照、与1号 feature_builder 数值一致性、
窗口不足状态、零方差相关状态、严格日期、公开输出证券代码键（方案A）、
非法截止日、交易日历与月末信号、接口 schema 形状。
零方差用例使用程序生成的 synthetic_test 数据，仅验证逻辑，不进入正式研究。
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from statistics import stdev

import pandas as pd

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
        for c in ASSET_CODES:
            s = self.series[c]
            for w in (20, 60):
                expect = math.prod(1 + x for x in s[-w:]) - 1
                self.assertAlmostEqual(mf[f"asset_return_{w}d"][c], expect, places=12)
            for w in (20, 60, 250):
                expect = stdev(s[-w:]) * math.sqrt(250)
                self.assertAlmostEqual(mf[f"asset_volatility_{w}d"][c], expect, places=12)

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
        for c in ASSET_CODES:
            for w in (20, 60):
                self.assertAlmostEqual(mf[f"asset_return_{w}d"][c],
                                       old["assets"][c][f"return_{w}d"], places=12)
            for w in (20, 60, 250):
                self.assertAlmostEqual(mf[f"asset_volatility_{w}d"][c],
                                       old["assets"][c][f"volatility_{w}d_ann"], places=12)
        for i, a in enumerate(ASSET_CODES):
            for b in ASSET_CODES[i + 1:]:
                self.assertAlmostEqual(self.payload["asset_correlations"]["corr_60d"][a][b],
                                       old["correlations_60d"][f"{a}_{b}"], places=12)

    def test_json_serializable(self):
        json.dumps(self.payload, ensure_ascii=False, allow_nan=False)


class TestPublicCodeKeys(unittest.TestCase):
    """方案A：公开输出资产键只允许证券代码，stock/bond/gold 不得出现。"""

    @classmethod
    def setUpClass(cls):
        cls.payload = di.get_market_data_input(AS_OF)

    def test_code_keys_everywhere(self):
        for key, values in self.payload["market_features"].items():
            self.assertEqual(set(values), set(ASSET_CODES), key)
        for w in ("corr_60d", "corr_250d"):
            matrix = self.payload["asset_correlations"][w]
            self.assertEqual(set(matrix), set(ASSET_CODES))
            for row in matrix.values():
                self.assertEqual(set(row), set(ASSET_CODES))
        self.assertNotIn("asset_code_mapping", self.payload["data_quality"])

    def test_no_names_in_json(self):
        text = json.dumps(self.payload, ensure_ascii=False)
        for name in ("stock", "bond", "gold"):
            self.assertNotIn(f'"{name}"', text)


class TestStrictDate(unittest.TestCase):
    def test_rejects_non_plain_dates(self):
        bad = ["2026-11-20T00:00:00", "2026-11-20 00:00:00", "2026-11-20Z",
               "2026-11-20+08:00", "20261120", "2026-1-5", "2026-02-30",
               " 2026-11-20", "2026-11-20\n", 20261120, None,
               date(2026, 11, 20), datetime(2026, 11, 20), ["2026-11-20"]]
        for value in bad:
            with self.assertRaises(ValueError, msg=repr(value)):
                di.get_market_data_input(value)
        with self.assertRaises(ValueError):
            di.next_trading_day("2014-11-28T00:00:00")

    def test_accepts_plain_date(self):
        payload = di.get_market_data_input("2026-11-20")
        self.assertEqual(payload["as_of_date"], "2026-11-20")
        self.assertEqual(di.next_trading_day("2014-11-28"), "2014-12-01")


class TestZeroVariance(unittest.TestCase):
    """synthetic_test：零方差资产的相关窗口必须 sufficient=False 并进 insufficient_windows。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        dates = pd.date_range("2020-01-01", periods=260, freq="B").strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame({
            "日期": dates,
            "510300": [0.0] * 260,
            "511010": [0.001 * ((i % 3) - 1) for i in range(260)],
            "518880": [0.001 * ((i % 5) - 2) for i in range(260)],
        })
        cls.path = Path(cls.tmp.name) / "synthetic_test_matrix.xlsx"
        frame.to_excel(cls.path, sheet_name="日收益率矩阵", index=False)
        cls.as_of = dates[-1]
        cls.payload = di.get_market_data_input(cls.as_of, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_status_and_insufficient(self):
        q = self.payload["data_quality"]
        for w in (60, 250):
            st = q["window_status"][f"corr_{w}d"]
            self.assertFalse(st["sufficient"])
            self.assertEqual(st["reason"], "zero_variance_asset")
            self.assertIn(f"corr_{w}d", q["insufficient_windows"])
        self.assertTrue(q["window_status"]["return_60d"]["sufficient"])

    def test_null_entries_and_zero_vol(self):
        for w in (60, 250):
            matrix = self.payload["asset_correlations"][f"corr_{w}d"]
            self.assertIsNone(matrix["510300"]["511010"])
            self.assertIsNone(matrix["518880"]["510300"])
            self.assertEqual(matrix["510300"]["510300"], 1.0)
            self.assertIsNotNone(matrix["511010"]["518880"])
        mf = self.payload["market_features"]
        self.assertEqual(mf["asset_volatility_20d"]["510300"], 0.0)
        self.assertEqual(mf["asset_return_20d"]["510300"], 0.0)


class TestInsufficientWindow(unittest.TestCase):
    def test_mid_window(self):
        payload = di.get_market_data_input("2013-12-31")
        q = payload["data_quality"]
        self.assertGreaterEqual(q["sample_days_to_date"], 60)
        self.assertLess(q["sample_days_to_date"], 250)
        self.assertIsNotNone(payload["market_features"]["asset_return_60d"]["510300"])
        self.assertIsNone(payload["market_features"]["asset_volatility_250d"]["510300"])
        self.assertIsNone(payload["asset_correlations"]["corr_250d"])
        self.assertIn("volatility_250d", q["insufficient_windows"])
        self.assertIn("corr_250d", q["insufficient_windows"])
        self.assertEqual(q["window_status"]["corr_250d"]["reason"], "insufficient_window")

    def test_short_window(self):
        payload = di.get_market_data_input("2013-08-30")
        q = payload["data_quality"]
        self.assertLess(q["sample_days_to_date"], 60)
        self.assertIsNotNone(payload["market_features"]["asset_return_20d"]["511010"])
        self.assertIsNone(payload["market_features"]["asset_return_60d"]["511010"])
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
        # 方案A（2026-11-26 审查确认）：内层键为证券代码；
        # interfaces/data_schema.py 的 AssetValues 仍为 stock/bond/gold，待1号同步。
        for key in data_schema.MarketFeatures.__annotations__:
            self.assertEqual(set(payload["market_features"][key]), set(ASSET_CODES))
        self.assertTrue(data_schema.validate_date(payload["as_of_date"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)