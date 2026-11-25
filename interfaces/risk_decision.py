"""离线风险联调的分组输入与模型预算校验。"""
import math

from interfaces.data_schema import ASSET_CODES


BASIC = ('as_of_date', 'assets', 'correlations_60d', 'market_data_quality')
B_KEYS = ('portfolio_return',)
C_KEYS = ('portfolio_volatility', 'current_drawdown', 'max_drawdown', 'cvar_95', 'trigger_flags')
D_KEYS = ('risk_contribution', 'budget_deviation', 'drawdown_attribution',
          'risk_change_attribution', 'concentration_state')


def build_group_input(market, feedback, group):
    """按实验组构建模型输入，并保留该组允许的反馈字段。"""
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
    """校验离线权重求解所需的决策状态、日期和预算。"""
    if result.get('decision_status') != 'ok' or result.get('as_of_date') != expected_date:
        raise ValueError('状态或日期不合法')
    b = result.get('risk_budget')
    if not isinstance(b, dict) or set(b) != set(ASSET_CODES):
        raise ValueError('资产字段不合法')
    values = list(b.values())
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 or x >= 1 for x in values):
        raise ValueError('风险预算非合法正数')
    if abs(sum(values) - 1) > 1e-6:
        raise ValueError('预算总和不为1')
    return True

