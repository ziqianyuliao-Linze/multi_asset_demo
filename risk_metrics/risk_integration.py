"""本地命令入口。四组按各自历史生成输入，可读取已有模型JSON完成预算平滑和求解。"""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from feature_builder import ASSETS, build_features, read_matrix, write_json
from offline_test import build_group_input, validate_decision
from risk_feedback import (allocation, build_feedback, check_date, closing_weights,
                           covariance_at, read_holdings, solve_weights)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_decisions(path):
    def invalid_number(value):
        raise ValueError(f'非法JSON数值 {value}')
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'JSON键重复 {key}')
            result[key] = value
        return result
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=invalid_number, object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or set(data) != set('ABCD'):
        raise ValueError('决策JSON顶层须恰为 A/B/C/D 四组')
    return data


def decision_to_weights(decision, previous_budget, current_weights, cov, signal_date,
                        max_iter=200):
    if not isinstance(decision, dict):
        raise ValueError('决策须为JSON对象')
    validate_decision(decision, signal_date)
    previous_budget = allocation(previous_budget, '上一期预算')
    current_weights = allocation(current_weights, '收盘权重')
    budget = [0.5 * old + 0.5 * decision['risk_budget'][asset]
              for old, asset in zip(previous_budget, ASSETS)]
    result = solve_weights(budget, cov, current_weights, max_iter=max_iter)
    accepted = result['converged']
    applied_weights = result['weights'] if accepted else current_weights
    applied_budget = budget if accepted else previous_budget
    return dict(as_of_date=signal_date, status='proposal' if accepted else 'fallback_previous',
                risk_budget=dict(zip(ASSETS, applied_budget)),
                weights=dict(zip(ASSETS, applied_weights)),
                proposed_smoothed_budget=dict(zip(ASSETS, budget)),
                solver=result, executed=False)


def run_integration(matrix, holdings, as_of_date, base_date, output_root,
                    decisions=None, data_kind='user_supplied'):
    if data_kind not in ('synthetic_test', 'user_supplied'):
        raise ValueError('data_kind不合法')
    check_date(as_of_date)
    check_date(base_date)
    source_hashes = {'matrix': file_hash(matrix), 'holdings': file_hash(holdings)}
    if decisions is not None:
        source_hashes['decisions'] = file_hash(decisions)
    records = read_matrix(matrix)
    groups = read_holdings(holdings)
    market = build_features(records, as_of_date)
    results = {}
    report = dict(mode='offline_risk_integration', data_kind=data_kind,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  as_of_date=as_of_date, base_date=base_date,
                  matrix_rows=len(records), market_rows_to_date=market['market_data_quality']['sample_days_to_date'],
                  source_paths={'matrix': str(Path(matrix).resolve()), 'holdings': str(Path(holdings).resolve())},
                  source_sha256=source_hashes, groups={}, api_called=False,
                  decisions_supplied=decisions is not None, trades_executed=False)
    for group in 'ABCD':
        feedback, audit = build_feedback(records, groups[group], as_of_date, group, base_date)
        inp = build_group_input(market, feedback, group)
        results[f'input_group_{group}.json'] = inp
        results[f'feedback_group_{group}.json'] = feedback
        report['groups'][group] = dict(field_count=len(inp), passed=True, audit=audit)
    if decisions is not None:
        parsed_decisions = read_decisions(decisions)
        report['source_paths']['decisions'] = str(Path(decisions).resolve())
        results['decisions_received.json'] = parsed_decisions
        cov, shrinkage = covariance_at(records, as_of_date)
        report['solver_covariance_shrinkage'] = shrinkage
        daily_returns = dict(records)[as_of_date]
        proposals = {}
        for group in 'ABCD':
            rows = [row for row in groups[group] if row['date'] <= as_of_date]
            if not rows or rows[-1]['date'] != as_of_date:
                raise ValueError(f'{group}组缺少截止日实际持仓和上一期预算')
            row = rows[-1]
            proposals[group] = decision_to_weights(parsed_decisions[group], row['budget'],
                                                   closing_weights(row, daily_returns), cov, as_of_date)
        results['weight_proposals.json'] = proposals
        report['solver_converged'] = {g: p['solver']['converged'] for g, p in proposals.items()}
    future = [day for day, _ in records if day > as_of_date]
    next_day = future[0] if future else None
    report['execution_calendar'] = dict(next_trading_date=next_day,
        monthly_rebalance_due=(next_day[:7] != as_of_date[:7]) if next_day else None,
        policy='仅T+1为月首交易日时可交回测端执行，当前入口只生成提案')
    # 所有计算通过后才保存。每次独立目录，最终重命名发布完整结果。
    for key, path in report['source_paths'].items():
        if file_hash(path) != source_hashes[key]:
            raise ValueError(f'运行期间输入文件发生变化 {key}')
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=output_root) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != output_root:
            raise ValueError('输出临时目录越界')
        for name, data in results.items():
            write_json(data, staging / name)
        report['output_sha256'] = {name: file_hash(staging / name) for name in results}
        write_json(report, staging / 'integration_report.json')
        destination = output_root / ('run-' + staging.name.removeprefix('.pending-'))
        os.rename(staging, destination)
    return destination, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=Path)
    parser.add_argument('--holdings', type=Path, required=True)
    parser.add_argument('--as-of', required=True)
    parser.add_argument('--base-date', required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--data-kind', choices=('user_supplied', 'synthetic_test'), default='user_supplied')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'risk_runs')
    args = parser.parse_args()
    try:
        destination, report = run_integration(args.matrix, args.holdings, args.as_of, args.base_date,
                                             args.output_dir, args.decisions, args.data_kind)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'对接失败，{error}\n')
    print(json.dumps(dict(output_dir=str(destination), **report), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
