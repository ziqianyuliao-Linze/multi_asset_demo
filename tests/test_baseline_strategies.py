"""3号 baseline_strategies 单元测试（第二阶段整改版，pytest）。

运行（仓库根目录）：
    python -m pytest tests/test_baseline_strategies.py -v

覆盖范围（对应 6 条整改要求）：
1. 风险平价：权重可行、风险贡献后验占比与 1/3 的最大绝对偏差 <= 0.005；
   偏差超容差必须抛异常、绝不返回权重；
2. 资产池强校验：缺失/多出/重复/顺序错/非序列全部拒绝；协方差严格 3×3；
3. 奇异/零矩阵防护：零矩阵直接拒绝，奇异矩阵收缩修复后可解、不出现除零；
4. 250 日窗口正式入口：真实冻结矩阵、历史不足、未来日期、格式非法、截断无泄漏；
5. 新增用例：难解（高条件数）正定矩阵、后验独立复算；
6. 统一返回格式键集合。
"""
import json
from importlib.util import find_spec

import numpy as np
import pytest

from baseline_strategies import portfolio_optimizer as po
from baseline_strategies.portfolio_optimizer import (ASSET_POOL, N_ASSETS,
                                                    RC_TOLERANCE, RiskBudgetError,
                                                    WINDOW_DAYS, check_risk_budget,
                                                    compute_risk_contribution,
                                                    compute_sample_covariance,
                                                    equal_weight, is_valid_cov,
                                                    min_variance, risk_parity,
                                                    run_baseline, run_baseline_250d,
                                                    stabilize_covariance,
                                                    validate_asset_pool,
                                                    validate_covariance)

HAS_PANDAS = find_spec("pandas") is not None and find_spec("openpyxl") is not None
needs_matrix = pytest.mark.skipif(not HAS_PANDAS, reason="需要 pandas/openpyxl 读取 Excel 冻结矩阵")

REAL_AS_OF = "2026-11-20"  # 冻结矩阵末日，与 2 号数据模块 TEST_REPORT 一致

# 全样本 147 个月末信号日中，求解器（L-BFGS-B/BFGS）会返回非 success 的 6 个日期：
# 凸问题在平缓方向上触发线搜索精度断言，但候选解有限且后验偏差已达标，必须照常产出。
PRECISION_LOSS_DATES = ["2015-10-30", "2022-06-30", "2024-08-30",
                        "2024-12-31", "2025-05-30", "2025-06-30"]

# 常规（良态、正定）协方差：日频量纲，三资产波动率约 2%、0.2%、1.7%，条件数约 123
CALM_COV = np.outer(np.array([0.020, 0.002, 0.017]), np.array([0.020, 0.002, 0.017])) * np.array([
    [1.00, 0.25, 0.30],
    [0.25, 1.00, 0.10],
    [0.30, 0.10, 1.00],
])

# 难解但正定：债券波动率极低 + 股票/黄金高相关，条件数约 2.4e3
HARD_COV = np.outer(np.array([0.020, 0.003, 0.045]), np.array([0.020, 0.003, 0.045])) * np.array([
    [1.00, 0.95, 0.15],
    [0.95, 1.00, 0.10],
    [0.15, 0.10, 1.00],
])

# 奇异（秩 1，最小特征值为 0）：三资产完全同向
SINGULAR_COV = np.array([
    [0.040, 0.020, 0.040],
    [0.020, 0.010, 0.020],
    [0.040, 0.020, 0.040],
])

SUMMARY_KEYS = {
    "strategy", "weight", "risk_contribution", "risk_contribution_ratio",
    "risk_contribution_max_deviation", "risk_budget_tolerance", "risk_budget_pass",
    "portfolio_variance", "covariance_shrinkage",
}


def weights_of(result) -> np.ndarray:
    return np.array([result["weight"][code] for code in ASSET_POOL], dtype=float)


def ratio_of(result) -> np.ndarray:
    return np.array([result["risk_contribution_ratio"][code] for code in ASSET_POOL], dtype=float)


# --------------------------------------------------------------------------- #
# 1. 资产池强校验（整改要求 2）
# --------------------------------------------------------------------------- #
class TestAssetPool:
    def test_accepts_exact_pool(self):
        assert validate_asset_pool(list(ASSET_POOL)) == list(ASSET_POOL)
        assert validate_asset_pool(tuple(ASSET_POOL)) == list(ASSET_POOL)

    @pytest.mark.parametrize("bad", [
        ["510300", "511010"],                        # 缺失
        ["510300", "511010", "518880", "159915"],    # 多标的
        ["510300", "511010", "518880", "518880"],    # 重复
        ["511010", "510300", "518880"],              # 顺序错
        ["510300", "511010", "518881"],              # 错代码
        "510300",                                    # 字符串
        None,                                        # 空值
        "510300,511010,518880",                      # 逗号字符串
    ])
    def test_rejects_bad_pool(self, bad):
        with pytest.raises(ValueError):
            validate_asset_pool(bad)

    def test_strategies_reject_bad_pool(self):
        for strategy in (equal_weight, min_variance, risk_parity):
            with pytest.raises(ValueError):
                strategy(CALM_COV, ["510300", "511010"])
            with pytest.raises(ValueError):
                strategy(CALM_COV, ["511010", "510300", "518880"])


# --------------------------------------------------------------------------- #
# 2. 协方差 3×3 与结构强校验（整改要求 2、3）
# --------------------------------------------------------------------------- #
class TestCovarianceGuard:
    @pytest.mark.parametrize("shape", [(2, 2), (4, 4), (3, 2), (1, 1)])
    def test_shape_must_be_3x3(self, shape):
        with pytest.raises(ValueError):
            validate_covariance(np.full(shape, 0.01))

    def test_zero_matrix_rejected(self):
        with pytest.raises(ValueError):
            validate_covariance(np.zeros((N_ASSETS, N_ASSETS)))
        # 零矩阵不得进入任何策略（防止后续 w'Σw 除零）
        for strategy in (equal_weight, min_variance, risk_parity):
            with pytest.raises(ValueError):
                strategy(np.zeros((N_ASSETS, N_ASSETS)))

    def test_asymmetric_rejected(self):
        bad = CALM_COV.copy()
        bad[0, 1] = 0.5
        with pytest.raises(ValueError):
            validate_covariance(bad)

    def test_nan_inf_rejected(self):
        bad = CALM_COV.copy()
        bad[1, 1] = np.nan
        with pytest.raises(ValueError):
            validate_covariance(bad)

    def test_non_positive_diagonal_rejected(self):
        bad = CALM_COV.copy()
        bad[2, 2] = -0.0003
        with pytest.raises(ValueError):
            validate_covariance(bad)

    def test_is_valid_cov_flags(self):
        assert is_valid_cov(CALM_COV) is True
        assert is_valid_cov(HARD_COV) is True
        assert is_valid_cov(SINGULAR_COV) is False      # 奇异 -> 非正定
        assert is_valid_cov(np.zeros((3, 3))) is False  # 零矩阵
        assert is_valid_cov(np.full((2, 2), 0.01)) is False

    def test_well_conditioned_matrix_not_shrunk(self):
        cov, info = stabilize_covariance(CALM_COV)
        assert info["shrunk"] is False
        assert info["shrink_delta"] == 0.0
        assert np.allclose(cov, CALM_COV)

    def test_singular_matrix_gets_shrinkage_repair(self):
        cov, info = stabilize_covariance(SINGULAR_COV)
        assert info["shrunk"] is True
        assert info["shrink_delta"] > 0
        assert info["min_eigenvalue_before"] <= info["eigenvalue_floor"]
        assert info["min_eigenvalue_after"] > info["eigenvalue_floor"]
        assert np.linalg.eigvalsh(cov)[0] > 0
        assert np.allclose(cov, cov.T)
        # 收缩后仍须处在合理量纲内（不得把方差改到离谱数量级）
        assert np.all(np.diag(cov) > 0)
        assert 0.5 * np.trace(SINGULAR_COV) < np.trace(cov) < 2.0 * np.trace(SINGULAR_COV)

    def test_singular_matrix_strategies_run_without_division_error(self):
        for strategy in ("equal_weight", "min_variance", "risk_parity"):
            result = run_baseline(SINGULAR_COV, strategy=strategy)
            assert np.isclose(np.sum(weights_of(result)), 1.0)
            assert np.all(np.isfinite(weights_of(result)))
            assert result["covariance_shrinkage"]["shrunk"] is True
            assert np.isfinite(result["portfolio_variance"])
            assert result["portfolio_variance"] > 0

    def test_strategy_name_validated(self):
        with pytest.raises(ValueError):
            run_baseline(CALM_COV, strategy="max_sharpe")
        with pytest.raises(ValueError):
            run_baseline(CALM_COV, strategy=None)


# --------------------------------------------------------------------------- #
# 3. 三种策略：可行性与最小方差性质
# --------------------------------------------------------------------------- #
class TestBaselineStrategies:
    def test_equal_weight(self):
        result = equal_weight(CALM_COV)
        np.testing.assert_allclose(weights_of(result), np.full(N_ASSETS, 1.0 / 3.0), atol=1e-12)
        assert np.isclose(np.sum(weights_of(result)), 1.0)
        assert np.isclose(np.sum(ratio_of(result)), 1.0)
        assert result["risk_budget_tolerance"] is None
        assert result["risk_budget_pass"] is None

    def test_min_variance_feasible_and_optimal(self):
        result = min_variance(CALM_COV)
        weights = weights_of(result)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        # 最小方差性质：目标函数不劣于等权组合
        equal_w = np.full(N_ASSETS, 1.0 / 3.0)
        assert result["portfolio_variance"] <= float(equal_w @ CALM_COV @ equal_w) + 1e-15
        # 风险贡献合计 = 组合方差（绝对贡献定义自洽）
        assert np.isclose(sum(result["risk_contribution"].values()),
                          result["portfolio_variance"])

    def test_min_variance_on_hard_matrix(self):
        assert np.linalg.cond(HARD_COV) > 1e3  # 确认这确实是难解矩阵
        result = min_variance(HARD_COV)
        weights = weights_of(result)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        equal_w = np.full(N_ASSETS, 1.0 / 3.0)
        assert result["portfolio_variance"] <= float(equal_w @ HARD_COV @ equal_w) + 1e-15

    def test_risk_parity_posterior_within_tolerance(self):
        result = risk_parity(CALM_COV)
        weights = weights_of(result)
        ratio = ratio_of(result)

        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights > 0)
        assert np.isclose(np.sum(ratio), 1.0)
        assert np.max(np.abs(ratio - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE
        assert result["risk_contribution_max_deviation"] <= RC_TOLERANCE
        assert result["risk_budget_tolerance"] == RC_TOLERANCE
        assert result["risk_budget_pass"] is True

        # 独立后验复算：由返回权重重新算出占比，必须与返回值一致
        _, recomputed = compute_risk_contribution(weights, CALM_COV)
        np.testing.assert_allclose(ratio, recomputed, atol=1e-12)

    def test_risk_parity_on_hard_matrix(self):
        result = risk_parity(HARD_COV)
        weights = weights_of(result)
        _, recomputed = compute_risk_contribution(weights, HARD_COV)
        assert np.max(np.abs(recomputed - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE
        assert np.all(weights > 0)
        assert np.isclose(np.sum(weights), 1.0)

    def test_risk_parity_on_singular_repaired_matrix(self):
        result = risk_parity(SINGULAR_COV)
        ratio = ratio_of(result)
        assert np.isclose(np.sum(ratio), 1.0)
        assert np.max(np.abs(ratio - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE

    def test_uniform_return_schema(self):
        for strategy, func in (("equal_weight", equal_weight),
                               ("min_variance", min_variance),
                               ("risk_parity", risk_parity)):
            result = func(CALM_COV)
            assert set(result) == SUMMARY_KEYS
            assert result["strategy"] == strategy
            assert list(result["weight"]) == list(ASSET_POOL)
            assert list(result["risk_contribution"]) == list(ASSET_POOL)
            assert list(result["risk_contribution_ratio"]) == list(ASSET_POOL)
            json.dumps(result, ensure_ascii=False, allow_nan=False)


# --------------------------------------------------------------------------- #
# 4. 风险预算后验校验必须拦截（整改要求 1）
# --------------------------------------------------------------------------- #
class TestRiskBudgetGate:
    def test_check_risk_budget_pass_and_fail(self):
        ok = np.array([0.334, 0.333, 0.333])
        assert check_risk_budget(ok) <= RC_TOLERANCE

        bad = np.array([0.360, 0.320, 0.320])  # 最大偏差 0.0267 > 0.005
        with pytest.raises(RiskBudgetError):
            check_risk_budget(bad)

        nan_case = np.array([np.nan, 0.5, 0.5])
        with pytest.raises(RiskBudgetError):
            check_risk_budget(nan_case)

    def test_risk_parity_raises_when_deviation_exceeds_tolerance(self):
        # 容差取负值：任何非零偏差都必须拦截，验证"超容差即抛异常、不返回权重"
        with pytest.raises(RiskBudgetError):
            risk_parity(CALM_COV, rc_tolerance=-1e-6)

    def test_risk_parity_does_not_swallow_guard_failure(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RiskBudgetError("模拟：求解 success 但后验偏差超容差")

        monkeypatch.setattr(po, "check_risk_budget", boom)
        with pytest.raises(RiskBudgetError):
            risk_parity(CALM_COV)

    def test_exception_type_is_runtime_error(self):
        # 便于上游按异常类型统一处理链路
        assert issubclass(RiskBudgetError, RuntimeError)


# --------------------------------------------------------------------------- #
# 5. 250 日窗口正式入口（整改要求 4）
# --------------------------------------------------------------------------- #
@needs_matrix
class TestWindow250D:
    def test_fetch_window_contract(self):
        window = po.fetch_returns_window_250d(REAL_AS_OF)
        assert window.shape == (WINDOW_DAYS, N_ASSETS)
        assert list(window.columns) == list(ASSET_POOL)
        assert str(window.index[-1]) == REAL_AS_OF
        assert (window.index <= REAL_AS_OF).all()

    def test_sample_covariance_matches_numpy(self):
        window = po.fetch_returns_window_250d(REAL_AS_OF)
        cov = compute_sample_covariance(window)
        expected = np.cov(np.asarray(window[ASSET_POOL], dtype=float), rowvar=False, ddof=1)
        np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-15)
        assert cov.shape == (N_ASSETS, N_ASSETS)

        # 同一矩阵以 ndarray 传入结果必须一致
        np.testing.assert_allclose(
            compute_sample_covariance(np.asarray(window[ASSET_POOL], dtype=float)), cov)

    def test_all_strategies_on_real_data(self):
        for strategy in ("equal_weight", "min_variance", "risk_parity"):
            result = run_baseline_250d(REAL_AS_OF, strategy)
            assert result["as_of_date"] == REAL_AS_OF
            assert result["window_days"] == WINDOW_DAYS
            assert result["sample_days"] == WINDOW_DAYS
            assert result["asset_pool"] == list(ASSET_POOL)
            assert np.isclose(np.sum(weights_of(result)), 1.0)
            assert np.all(weights_of(result) >= 0)
            assert np.isfinite(result["portfolio_variance"])
            json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_risk_parity_covariance_computed_in_module(self):
        """协方差必须由本模块从 250 日收益率算出，并用返回值独立复算风险贡献。"""
        window = po.fetch_returns_window_250d(REAL_AS_OF)
        cov, shrinkage = stabilize_covariance(compute_sample_covariance(window))

        result = run_baseline_250d(REAL_AS_OF, "risk_parity")
        weights = weights_of(result)
        _, recomputed = compute_risk_contribution(weights, cov)

        assert np.max(np.abs(recomputed - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE
        np.testing.assert_allclose(ratio_of(result), recomputed, atol=1e-10)
        assert result["covariance_shrinkage"]["shrunk"] == shrinkage["shrunk"]

    def test_risk_parity_regression_20241231(self):
        """队长指定回归用例：该日期 L-BFGS-B/BFGS 均返回 precision loss，
        但候选解有限且后验偏差达标，必须正常返回而不是抛 RuntimeError。"""
        result = run_baseline_250d("2024-12-31", "risk_parity")

        weights = weights_of(result)
        ratio = ratio_of(result)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        assert np.isclose(np.sum(ratio), 1.0)
        assert np.max(np.abs(ratio - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE
        assert result["risk_budget_pass"] is True
        assert result["sample_days"] == WINDOW_DAYS

    @pytest.mark.parametrize("as_of", PRECISION_LOSS_DATES)
    def test_risk_parity_precision_loss_dates(self, as_of):
        """全样本扫描发现 147 个月末日中有 6 个会触发求解器精度断言，
        这些日期都必须正常产出且后验偏差达标。"""
        result = run_baseline_250d(as_of, "risk_parity")
        ratio = ratio_of(result)
        assert np.max(np.abs(ratio - np.full(N_ASSETS, 1.0 / 3.0))) <= RC_TOLERANCE
        assert np.isclose(np.sum(weights_of(result)), 1.0)

    def test_insufficient_history_raises(self):
        # 2013-08-30 仅有 24 个共同交易日，凑不满 250 日窗口
        with pytest.raises(ValueError):
            run_baseline_250d("2013-08-30", "risk_parity")

    @pytest.mark.parametrize("bad_date", ["2026-11-21", "2027-12-31", "2013-07-29"])
    def test_future_or_nonexistent_date_raises(self, bad_date):
        with pytest.raises(ValueError):
            run_baseline_250d(bad_date, "risk_parity")

    @pytest.mark.parametrize("bad_date", ["2026-11-20T00:00:00", "2026-11-20 ", "20261120", None])
    def test_malformed_date_raises(self, bad_date):
        with pytest.raises(ValueError):
            run_baseline_250d(bad_date, "risk_parity")

    def test_bad_pool_raises_in_250d_entry(self):
        with pytest.raises(ValueError):
            run_baseline_250d(REAL_AS_OF, "risk_parity", tickers=["510300", "511010"])


# --------------------------------------------------------------------------- #
# 6. 截断即稳定：未来数据不得影响历史日结果（整改要求 5）
# --------------------------------------------------------------------------- #
@needs_matrix
class TestNoLookahead:
    SIGNAL_INDEX = 249  # 0-based：第 250 个交易日，恰好够 250 日窗口
    TOTAL = 300

    @staticmethod
    def _build_frames():
        pd = pytest.importorskip("pandas")
        dates = pd.bdate_range("2020-01-01", periods=TestNoLookahead.TOTAL).strftime("%Y-%m-%d")
        rng = np.random.default_rng(20261128)
        signal = rng.normal(0.0, 0.004, size=(TestNoLookahead.SIGNAL_INDEX + 1, 3))
        future = rng.normal(0.0, 0.060, size=(TestNoLookahead.TOTAL - signal.shape[0], 3))
        values = np.vstack([signal, future])
        frame = pd.DataFrame(values, columns=ASSET_POOL)
        frame.insert(0, "日期", dates)
        return frame, dates[TestNoLookahead.SIGNAL_INDEX], signal

    def _write(self, path, frame):
        frame.to_excel(path, sheet_name="日收益率矩阵", index=False)
        return path

    def test_truncation_equals_full(self, tmp_path):
        frame, as_of, _ = self._build_frames()
        full_path = self._write(tmp_path / "full_matrix.xlsx", frame)
        truncated = frame.iloc[:self.SIGNAL_INDEX + 1]
        truncated_path = self._write(tmp_path / "truncated_matrix.xlsx", truncated)

        full = run_baseline_250d(as_of, "risk_parity", path=full_path)
        cut = run_baseline_250d(as_of, "risk_parity", path=truncated_path)

        np.testing.assert_allclose(weights_of(full), weights_of(cut), atol=1e-15)
        assert full["portfolio_variance"] == pytest.approx(cut["portfolio_variance"], rel=1e-15)
        assert full["risk_contribution_max_deviation"] == pytest.approx(
            cut["risk_contribution_max_deviation"], abs=1e-15)
        assert full["sample_days"] == cut["sample_days"] == WINDOW_DAYS

    def test_window_ignores_future_rows(self, tmp_path):
        """若上游误把未来行算进窗口，权重必然不同——用于证明当前窗口未泄漏。"""
        frame, as_of, signal = self._build_frames()
        full_path = self._write(tmp_path / "full_matrix.xlsx", frame)

        window_cov = compute_sample_covariance(signal)
        actual = run_baseline_250d(as_of, "risk_parity", path=full_path)
        expected = run_baseline(window_cov, strategy="risk_parity")

        np.testing.assert_allclose(weights_of(actual), weights_of(expected), atol=1e-10)

        # 反向对照：把未来数据混入窗口，权重会明显改变
        leaked_cov = compute_sample_covariance(np.vstack([signal, frame.iloc[:, 1:].to_numpy()[-50:]]))
        assert not np.allclose(weights_of(run_baseline(leaked_cov, strategy="risk_parity")),
                              weights_of(expected), atol=1e-3)

    def test_zero_variance_column_rejected(self, tmp_path):
        """某资产收益率恒为 0（零方差）时，协方差对角线为 0，必须在入口被拒绝。"""
        pd = pytest.importorskip("pandas")
        dates = pd.bdate_range("2020-01-01", periods=260).strftime("%Y-%m-%d")
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 0.004, size=(260, 3))
        values[:, 0] = 0.0
        frame = pd.DataFrame(values, columns=ASSET_POOL)
        frame.insert(0, "日期", dates)
        path = self._write(tmp_path / "zero_variance.xlsx", frame)

        with pytest.raises(ValueError):
            run_baseline_250d(str(dates[-1]), "risk_parity", path=path)
