"""真实收益矩阵 + 明确标识的测试持仓，验证本地对接入口和关键边界。"""
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from feature_builder import ASSETS, read_matrix, write_json
from risk_feedback import (HOLDING_COLUMNS, NAMES, build_feedback, covariance,
                           drawdown_attribution, historical_cvar, ledoit_wolf,
                           portfolio_history, positive_definite, read_holdings,
                           risk_contributions, solve_weights)
from risk_integration import decision_to_weights, read_decisions, run_integration

ROOT = Path(__file__).resolve().parent


def make_fixture(records, target):
    targets = {'A': [1 / 3] * 3, 'B': [0.15, 0.70, 0.15],
               'C': [0.20, 0.30, 0.50], 'D': [0.50, 0.25, 0.25]}
    weights = copy.deepcopy(targets)
    last_month = None
    with target.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(HOLDING_COLUMNS)
        for day, returns in records[-800:]:
            month = day[:7]
            if month != last_month:
                weights = copy.deepcopy(targets)
            for group in 'ABCD':
                current = weights[group]
                writer.writerow([day, group, *current, 1 / 3, 1 / 3, 1 / 3, 0.0])
                grown = [w * (1 + r) for w, r in zip(current, returns)]
                weights[group] = [value / sum(grown) for value in grown]
            last_month = month


class RiskIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_matrix(MATRIX)
        cls.day = cls.records[-1][0]
        cls.base = '2025-11-20'
        cls.holdings_file = OUTPUT / 'synthetic_holdings.csv'
        make_fixture(cls.records, cls.holdings_file)
        cls.groups = read_holdings(cls.holdings_file)
        cls.decisions_file = OUTPUT / 'synthetic_decisions.json'
        decisions = {g: dict(as_of_date=cls.day, decision_status='ok',
                            risk_budget=dict(zip(ASSETS, [0.4, 0.4, 0.2])),
                            decision_summary='离线测试预算，未调用模型') for g in 'ABCD'}
        write_json(decisions, cls.decisions_file)
        command = [sys.executable, '-B', '-E', '-S', '-X', 'utf8', str(ROOT / 'risk_integration.py'),
                   str(MATRIX), '--holdings', str(cls.holdings_file), '--as-of', cls.day,
                   '--base-date', cls.base, '--decisions', str(cls.decisions_file),
                   '--data-kind', 'synthetic_test', '--output-dir', str(OUTPUT / 'runs')]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        if result.returncode:
            raise AssertionError(f'正常命令入口失败，{result.stderr}')
        cls.console = json.loads(result.stdout)
        cls.run_dir = Path(cls.console['output_dir'])
        cls.report = cls.load('integration_report.json')

    @classmethod
    def load(cls, name):
        return json.loads((cls.run_dir / name).read_text(encoding='utf-8'))

    def test_cli_and_fresh_saved_files(self):
        self.assertEqual(self.report['data_kind'], 'synthetic_test')
        self.assertEqual(self.report['matrix_rows'], 3236)
        self.assertFalse(self.report['api_called'])
        self.assertFalse(self.report['trades_executed'])
        self.assertEqual(self.report, {k: v for k, v in self.console.items() if k != 'output_dir'})
        for name, digest in self.report['output_sha256'].items():
            self.assertEqual(hashlib.sha256((self.run_dir / name).read_bytes()).hexdigest(), digest)
        self.assertTrue(all(self.report['solver_converged'].values()))

    def test_exact_group_isolation(self):
        basic = {'as_of_date', 'assets', 'correlations_60d', 'market_data_quality'}
        feedback = {'portfolio_return'}
        risk = {'portfolio_volatility', 'current_drawdown', 'max_drawdown', 'cvar_95', 'trigger_flags'}
        attribution = {'risk_contribution', 'budget_deviation', 'drawdown_attribution',
                       'risk_change_attribution', 'concentration_state'}
        allowed = [basic, basic | feedback, basic | feedback | risk,
                   basic | feedback | risk | attribution]
        for group, keys in zip('ABCD', allowed):
            inp = self.load(f'input_group_{group}.json')
            self.assertEqual(set(inp), keys)
            if group in 'CD':
                self.assertEqual(set(inp['trigger_flags']), {'drawdown', 'volatility', 'correlation'})
        self.assertNotEqual(self.load('input_group_B.json')['portfolio_return'],
                            self.load('input_group_C.json')['portfolio_return'])

    def test_independent_group_history(self):
        changed = copy.deepcopy(self.groups)
        changed['B'][-1]['cost'] = 0.001
        original, _ = build_feedback(self.records, self.groups['B'], self.day, 'B')
        mutated, _ = build_feedback(self.records, changed['B'], self.day, 'B')
        self.assertNotEqual(original['portfolio_return'], mutated['portfolio_return'])
        for group in 'CD':
            before, _ = build_feedback(self.records, self.groups[group], self.day, group, self.base)
            after, _ = build_feedback(self.records, changed[group], self.day, group, self.base)
            self.assertEqual(before, after)

    def test_no_future_observations(self):
        cutoff = self.records[-10][0]
        prefix = [(d, r) for d, r in self.records if d <= cutoff]
        history = [row for row in self.groups['D'] if row['date'] <= cutoff]
        expected, _ = build_feedback(prefix, history, cutoff, 'D', self.base)
        altered = [(d, r if d <= cutoff else [100.0, -0.9, 10.0]) for d, r in self.records]
        actual, _ = build_feedback(altered, self.groups['D'], cutoff, 'D', self.base)
        self.assertEqual(expected, actual)

    def test_hand_calculated_returns_and_cost_attribution(self):
        records = [('2020-01-02', [0.1, 0, 0]), ('2020-01-03', [-0.2, 0, 0])]
        rows = [dict(date=d, weights=[1, 0, 0], budget=[1 / 3] * 3, cost=0) for d, _ in records]
        rows[1]['cost'] = 0.01
        path = portfolio_history(records, rows, records[-1][0])
        self.assertAlmostEqual(path['navs'][-1], 1.1 * 0.79)
        attribution = drawdown_attribution(path, records)
        self.assertEqual(attribution['start'], '2020-01-02')
        self.assertAlmostEqual(attribution['stock'], -0.20)
        self.assertAlmostEqual(attribution['transaction_cost'], -0.01)
        self.assertAlmostEqual(sum(attribution[k] for k in (*NAMES, 'transaction_cost')),
                               path['drawdowns'][-1])

    def test_current_drawdown_and_change_reconcile(self):
        feedback = self.load('feedback_group_D.json')
        attribution = feedback['drawdown_attribution']
        self.assertAlmostEqual(sum(attribution[k] for k in (*NAMES, 'transaction_cost')),
                               feedback['current_drawdown'], places=11)
        current = feedback['risk_contribution']
        baseline, _ = build_feedback(self.records, self.groups['D'], self.base, 'C')
        self.assertTrue(math.isfinite(baseline['current_drawdown']))
        from risk_feedback import covariance_at, closing_weights
        cov, _ = covariance_at(self.records, self.base)
        row = next(row for row in self.groups['D'] if row['date'] == self.base)
        old = risk_contributions(closing_weights(row, dict(self.records)[self.base]), cov)
        changes = feedback['risk_change_attribution']
        for name, previous in zip(NAMES, old):
            self.assertAlmostEqual(sum(changes[name].values()), current[name] - previous, places=10)

    def test_cvar_fractional_tail(self):
        returns = [-float(i) / 10000 for i in range(250)]
        expected = (sum(range(238, 250)) + 237 * 0.5) / 12.5 / 10000
        self.assertAlmostEqual(historical_cvar(returns), expected)

    def test_missing_duplicate_and_future_base_rejected(self):
        missing = copy.deepcopy(self.groups['D'])
        del missing[-20]
        with self.assertRaisesRegex(ValueError, '连续'):
            build_feedback(self.records, missing, self.day, 'D', self.base)
        duplicate = copy.deepcopy(self.groups['D'])
        duplicate.insert(-20, copy.deepcopy(duplicate[-20]))
        with self.assertRaisesRegex(ValueError, '连续'):
            build_feedback(self.records, duplicate, self.day, 'D', self.base)
        with self.assertRaisesRegex(ValueError, '早于'):
            build_feedback(self.records, self.groups['D'], self.day, 'D', self.day)
        with self.assertRaisesRegex(ValueError, '270'):
            build_feedback(self.records, self.groups['C'][-250:], self.day, 'C')

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as directory:
            target = Path(directory) / 'bad.csv'
            for weights in ([0.2, 0.2, 0.2], [float('nan'), 0.5, 0.5], [-0.1, 0.5, 0.6]):
                with target.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(HOLDING_COLUMNS)
                    writer.writerow([self.day, 'A', *weights, 1 / 3, 1 / 3, 1 / 3, 0])
                with self.assertRaises(ValueError):
                    read_holdings(target)

    def test_budget_diagonal_analytic_solution(self):
        cov = [[0.04, 0, 0], [0, 0.0004, 0], [0, 0, 0.01]]
        result = solve_weights([1 / 3] * 3, cov, tol=1e-10)
        inverse_vol = [5, 50, 10]
        self.assertTrue(result['converged'])
        for actual, inverse in zip(result['weights'], inverse_vol):
            self.assertAlmostEqual(actual, inverse / sum(inverse_vol), places=9)
        boundary = solve_weights([1, 0, 0], cov)
        self.assertEqual(boundary['weights'], [1, 0, 0])

    def test_negative_rc_and_random_positive_covariances(self):
        cov = [[1, -0.8, 0], [-0.8, 1, 0], [0, 0, 0.2]]
        result = solve_weights([1 / 3] * 3, cov, [0.01, 0.98, 0.01])
        self.assertTrue(result['converged'])
        self.assertEqual(result['method'], 'coordinate_descent')
        rng = random.Random(42)
        for _ in range(20):
            matrix = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
            cov = [[sum(matrix[i][k] * matrix[j][k] for k in range(3)) + (0.1 if i == j else 0)
                    for j in range(3)] for i in range(3)]
            result = solve_weights([0.4, 0.4, 0.2], cov)
            self.assertTrue(result['converged'])
            self.assertLessEqual(result['max_budget_error'], 0.005)
            self.assertAlmostEqual(sum(result['weights']), 1)
            self.assertTrue(all(0 <= v <= 1 for v in result['weights']))

    def test_shrinkage_and_degenerate_inputs(self):
        samples = [[i / 1000] * 3 for i in range(250)]
        cov = covariance(samples)
        self.assertFalse(positive_definite(cov))
        shrunk, intensity = ledoit_wolf(samples)
        self.assertTrue(positive_definite(shrunk))
        self.assertTrue(0 < intensity <= 1)
        self.assertTrue(solve_weights([0.4, 0.4, 0.2], cov, samples=samples)['converged'])
        with self.assertRaisesRegex(ValueError, '原始收益'):
            solve_weights([1 / 3] * 3, cov)
        with self.assertRaises(ValueError):
            ledoit_wolf([[0, 0, 0]] * 250)

    def test_smoothing_and_nonconvergence_fallback(self):
        decision = dict(as_of_date=self.day, decision_status='ok', risk_budget=dict(zip(ASSETS, [0.4, 0.4, 0.2])))
        cov = [[0.04, 0, 0], [0, 0.0004, 0], [0, 0, 0.01]]
        result = decision_to_weights(decision, [1 / 3] * 3, [1 / 3] * 3, cov, self.day)
        self.assertEqual(result['status'], 'proposal')
        for asset, value in zip(ASSETS, [0.4, 0.4, 0.2]):
            self.assertAlmostEqual(result['risk_budget'][asset], 0.5 * (1 / 3 + value))
        hard_cov = [[1, -0.95, 0.1], [-0.95, 1, -0.1], [0.1, -0.1, 0.1]]
        fallback = decision_to_weights(decision, [1 / 3] * 3, [0.01, 0.98, 0.01], hard_cov, self.day, max_iter=1)
        self.assertEqual(fallback['status'], 'fallback_previous')
        self.assertEqual(list(fallback['weights'].values()), [0.01, 0.98, 0.01])
        self.assertEqual(list(fallback['risk_budget'].values()), [1 / 3] * 3)

    def test_invalid_decision_and_no_partial_run(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT) as directory:
            bad = Path(directory) / 'bad.json'
            original = json.loads(self.decisions_file.read_text(encoding='utf-8'))
            for invalid in (True, float('inf'), -0.1, 0.3):
                candidate = copy.deepcopy(original['A'])
                candidate['risk_budget']['518880'] = invalid
                with self.assertRaises(ValueError):
                    decision_to_weights(candidate, [1 / 3] * 3, [1 / 3] * 3,
                                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], self.day)
            original['D']['as_of_date'] = self.base
            write_json(original, bad)
            destination = Path(directory) / 'runs'
            with self.assertRaises(ValueError):
                run_integration(MATRIX, self.holdings_file, self.day, self.base, destination, bad)
            self.assertFalse(destination.exists())
            bad.write_text('{"A":{},"A":{}}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '重复'):
                read_decisions(bad)

    def test_repeated_runs_get_separate_output(self):
        destination, report = run_integration(MATRIX, self.holdings_file, self.day, self.base,
                                              OUTPUT / 'runs', data_kind='synthetic_test')
        self.assertNotEqual(destination, self.run_dir)
        self.assertTrue(self.run_dir.exists())
        self.assertFalse(report['decisions_supplied'])
        self.assertFalse((destination / 'weight_proposals.json').exists())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('用法 python test_risk_integration.py 收益率矩阵.xlsx')
    MATRIX = Path(sys.argv[1]).resolve()
    parent = ROOT / 'risk_test_runs'
    parent.mkdir(exist_ok=True)
    OUTPUT = Path(tempfile.mkdtemp(prefix='test-', dir=parent))
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RiskIntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = dict(mode='synthetic_test', created_at=datetime.now().astimezone().isoformat(),
                   tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   passed=result.wasSuccessful(), api_called=False, real_holdings_validated=False,
                   matrix_sha256=hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
                   source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ('risk_feedback.py', 'risk_integration.py', 'test_risk_integration.py')},
                   integration_report=str(getattr(RiskIntegrationTests, 'run_dir', OUTPUT) / 'integration_report.json'))
    write_json(summary, OUTPUT / 'test_report.json')
    print(json.dumps(dict(report_path=str(OUTPUT / 'test_report.json'), **summary), ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
