"""离线验证 A/B/C/D 的反馈字段隔离、输出预算校验与保存；不调用API。"""
import json
import math
import sys
from pathlib import Path
from feature_builder import ASSETS, build_features, read_matrix, write_json

ROOT = Path(__file__).resolve().parent
BASIC = ('as_of_date', 'assets', 'correlations_60d', 'market_data_quality')
B_KEYS = ('portfolio_return',)
C_KEYS = ('portfolio_volatility', 'current_drawdown', 'max_drawdown', 'cvar_95', 'trigger_flags')
D_KEYS = ('risk_contribution', 'budget_deviation', 'drawdown_attribution',
          'risk_change_attribution', 'concentration_state')


def build_group_input(market, feedback, group):
    if group not in 'ABCD' or len(group) != 1:
        raise ValueError('组别必须是A/B/C/D')
    result = {k: market[k] for k in BASIC}
    keys = (() if group == 'A' else
            B_KEYS if group == 'B' else
            B_KEYS + C_KEYS if group == 'C' else
            B_KEYS + C_KEYS + D_KEYS)
    for key in keys:
        if key not in feedback:
            raise ValueError(f'{group}组缺少反馈字段: {key}')
        result[key] = feedback[key]
    if 'trigger_flags' in result:
        result['trigger_flags'] = {k: result['trigger_flags'][k]
                                   for k in ('drawdown', 'volatility', 'correlation')}
    return result


def validate_decision(result, expected_date):
    if result.get('decision_status') != 'ok' or result.get('as_of_date') != expected_date:
        raise ValueError('状态或日期不合法')
    b = result.get('risk_budget')
    if not isinstance(b, dict) or set(b) != set(ASSETS):
        raise ValueError('资产字段不合法')
    values = list(b.values())
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 or x >= 1 for x in values):
        raise ValueError('风险预算非合法正数')
    if abs(sum(values) - 1) > 1e-6:
        raise ValueError('预算总和不为1')
    return True


def main():
    if len(sys.argv) < 2:
        raise SystemExit('用法: python offline_test.py "2：统一口径日收益率矩阵(1).xlsx"')
    records = read_matrix(sys.argv[1])
    assert len(records) == 3236, f'预期3236，实得{len(records)}'
    market = build_features(records)
    feedback = {
        'portfolio_return': {'d20': -0.05, 'd60': -0.1171},
        'portfolio_volatility': {'d20': 0.1297, 'd60': 0.1064},
        'current_drawdown': -0.1851, 'max_drawdown': -0.1851,
        'cvar_95': 0.0138,
        'trigger_flags': {'drawdown': True, 'volatility': True, 'correlation': False,
                          'concentration': True},
        'risk_contribution': {'stock': 0.322, 'bond': 0.002, 'gold': 0.676},
        'budget_deviation': {'stock': -0.011, 'bond': -0.331, 'gold': 0.343},
        'drawdown_attribution': {'start': '2026-01-29', 'end': market['as_of_date'],
                                 'stock': -0.0527, 'bond': 0.0070, 'gold': -0.151},
        'risk_change_attribution': {'base': '2025-11-20', 'gold': {'vol': 0.129, 'corr': -0.049, 'weight': 0}},
        'concentration_state': {'gold': 0.676, 'neff': 1.78},
    }
    report = {'matrix_rows': len(records), 'as_of_date': market['as_of_date'],
              'mode': 'offline_mock', 'groups': {}}
    for group in 'ABCD':
        inp = build_group_input(market, feedback, group)
        assert ('portfolio_return' in inp) == (group != 'A')
        assert ('portfolio_volatility' in inp) == (group in 'CD')
        assert ('risk_contribution' in inp) == (group == 'D')
        assert ('concentration_state' in inp) == (group == 'D')
        assert 'concentration' not in inp.get('trigger_flags', {})
        assert 'experiment_group' not in inp
        write_json(inp, ROOT / f'input_group_{group}.json')
        report['groups'][group] = {'field_count': len(inp), 'passed': True}
    mock = {'as_of_date': market['as_of_date'], 'decision_status': 'ok',
            'risk_budget': {'510300': 0.4, '511010': 0.4, '518880': 0.2},
            'market_state': 'risk_elevated', 'confidence': 0.8,
            'decision_summary': '离线模拟，不是真实模型输出',
            'failure_assumption': '模拟数据不具预测意义'}
    validate_decision(mock, market['as_of_date'])
    write_json(mock, ROOT / 'mock_decision_output.json')
    report['mock_decision_validation'] = 'passed'
    report['api_called'] = False
    write_json(report, ROOT / 'offline_test_report.json')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
