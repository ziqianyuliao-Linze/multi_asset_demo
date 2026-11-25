"""4号风险计算与预算求解。Python 标准库实现，所有计算截至信号日。"""
import csv
import math
from datetime import date
from statistics import mean, stdev

from data_processing.feature_builder import (
    ASSETS,
    compound,
    corr,
)

NAMES = ('stock', 'bond', 'gold')
HOLDING_COLUMNS = ('as_of_date', 'group') + tuple('w_' + a for a in ASSETS) + tuple(
    'budget_' + a for a in ASSETS) + ('cost_rate',)


def finite_number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{label} 必须是有限数字')
    return value


def check_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('日期须为 YYYY-MM-DD')
    return value


def allocation(values, label):
    if len(values) != 3:
        raise ValueError(f'{label} 须按 510300/511010/518880 提供三个值')
    values = [finite_number(v, label) for v in values]
    if any(v < 0 or v > 1 for v in values) or abs(sum(values) - 1) > 1e-8:
        raise ValueError(f'{label} 须非负、至多为1且合计为1')
    return values


def read_holdings(path):
    """权重为当日收益发生前的实际权重，cost_rate 为当日期初净值扣费比例。"""
    groups = {g: [] for g in 'ABCD'}
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(HOLDING_COLUMNS):
            raise ValueError('持仓CSV列须为 ' + ','.join(HOLDING_COLUMNS))
        for line, row in enumerate(reader, 2):
            try:
                group = row['group']
                if group not in groups or None in row:
                    raise ValueError('组别或列数错误')
                day = check_date(row['as_of_date'])
                weights = allocation([float(row['w_' + a]) for a in ASSETS], '资金权重')
                budget = allocation([float(row['budget_' + a]) for a in ASSETS], '风险预算')
                cost = finite_number(float(row['cost_rate']), 'cost_rate')
                if not 0 <= cost < 1:
                    raise ValueError('cost_rate 须在 [0,1) 内')
                if groups[group] and groups[group][-1]['date'] >= day:
                    raise ValueError('同组日期重复或未升序')
                groups[group].append(dict(date=day, weights=weights, budget=budget, cost=cost))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'持仓第 {line} 行错误，{error}') from error
    return groups


def portfolio_history(records, history, as_of_date):
    check_date(as_of_date)
    market = {d: values for d, values in records if d <= as_of_date}
    rows = [row for row in history if row['date'] <= as_of_date]
    if not rows or rows[-1]['date'] != as_of_date:
        raise ValueError('持仓必须覆盖指定截止日')
    expected = [d for d in market if d >= rows[0]['date']]
    if [row['date'] for row in rows] != expected:
        raise ValueError('持仓日期须连续覆盖矩阵交易日，不能缺日、重复或含非交易日')
    returns, contributions, navs, drawdowns = [], [], [1.0], []
    peak = 1.0
    for row in rows:
        weights = allocation(row['weights'], '资金权重')
        allocation(row['budget'], '风险预算')
        cost = finite_number(row['cost'], 'cost_rate')
        if not 0 <= cost < 1:
            raise ValueError('cost_rate 须在 [0,1) 内')
        components = [w * r for w, r in zip(weights, market[row['date']])]
        net = sum(components) - cost
        if not math.isfinite(net) or net <= -1:
            raise ValueError('组合日收益须有限且大于 -1')
        returns.append(net)
        contributions.append(components + [-cost])
        nav = navs[-1] * (1 + net)
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError('净值超出可计算范围')
        navs.append(nav)
        peak = max(peak, nav)
        drawdowns.append(nav / peak - 1)
    return dict(rows=rows, returns=returns, contributions=contributions,
                navs=navs, drawdowns=drawdowns)


def closing_weights(row, daily_returns):
    values = [w * (1 + r) for w, r in zip(row['weights'], daily_returns)]
    if min(values) < 0 or sum(values) <= 0:
        raise ValueError('无法从期初持仓计算收盘权重')
    # 费用按收盘资产价值比例扣除，因此不改变资产间相对权重。
    return [v / sum(values) for v in values]


def covariance(samples):
    if len(samples) < 2 or any(len(row) != 3 for row in samples):
        raise ValueError('协方差至少需要两行三资产观测')
    columns = list(zip(*samples))
    centers = [mean(col) for col in columns]
    return [[sum((row[i] - centers[i]) * (row[j] - centers[j]) for row in samples)
             / (len(samples) - 1) for j in range(3)] for i in range(3)]


def positive_definite(cov):
    lower = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(i + 1):
            value = cov[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if value <= max(cov[i][i] * 1e-12, 0):
                    return False
                lower[i][j] = math.sqrt(value)
            else:
                lower[i][j] = value / lower[j][j]
    return True


def checked_covariance(cov):
    if len(cov) != 3 or any(len(row) != 3 for row in cov):
        raise ValueError('协方差须为 3×3')
    for i in range(3):
        for j in range(3):
            finite_number(cov[i][j], '协方差')
            if abs(cov[i][j] - cov[j][i]) > 1e-12:
                raise ValueError('协方差须对称')
        if cov[i][i] < 0:
            raise ValueError('协方差对角线须非负')
    return cov


def ledoit_wolf(samples):
    """对中心化样本作 Ledoit-Wolf 球形目标收缩，再恢复样本协方差尺度。"""
    count = len(samples)
    centers = [mean(col) for col in zip(*samples)]
    centered = [[v - m for v, m in zip(row, centers)] for row in samples]
    empirical = [[sum(row[i] * row[j] for row in centered) / count
                  for j in range(3)] for i in range(3)]
    mu = sum(empirical[i][i] for i in range(3)) / 3
    delta = sum((empirical[i][j] - (mu if i == j else 0)) ** 2
                for i in range(3) for j in range(3))
    beta = sum((row[i] * row[j] - empirical[i][j]) ** 2
               for row in centered for i in range(3) for j in range(3)) / count ** 2
    shrinkage = min(1.0, beta / delta) if delta else 0.0
    result = [[((1 - shrinkage) * empirical[i][j] +
                (shrinkage * mu if i == j else 0)) * count / (count - 1)
               for j in range(3)] for i in range(3)]
    if not positive_definite(result):
        raise ValueError('Ledoit-Wolf收缩后仍非正定，无法求解')
    return result, shrinkage


def covariance_at(records, as_of_date):
    samples = [row for day, row in records if day <= as_of_date][-250:]
    if len(samples) != 250:
        raise ValueError(f'{as_of_date} 之前不足250条市场观测')
    cov = covariance(samples)
    shrinkage = 0.0
    if not positive_definite(cov):
        cov, shrinkage = ledoit_wolf(samples)
    return cov, shrinkage


def risk_contributions(weights, cov):
    marginal = [sum(cov[i][j] * weights[j] for j in range(3)) for i in range(3)]
    variance = sum(w * m for w, m in zip(weights, marginal))
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError('组合方差须为正数')
    return [w * m / variance for w, m in zip(weights, marginal)]


def solve_weights(b, cov, w0=None, max_iter=200, tol=0.005, *, samples=None):
    """风险预算转资金权重，负RC或固定点停滞时用坐标下降，未收敛由调用方回退。"""
    budget = allocation(b, '风险预算')
    checked_covariance(cov)
    if type(max_iter) is not int or max_iter < 1:
        raise ValueError('max_iter 须为正整数')
    finite_number(tol, 'tol')
    if not 0 < tol < 1:
        raise ValueError('tol 须在 (0,1) 内')
    shrinkage = 0.0
    if not positive_definite(cov):
        if samples is None:
            raise ValueError('非正定协方差需要原始收益样本才能计算Ledoit-Wolf收缩')
        sample_cov = covariance(samples)
        if any(abs(cov[i][j] - sample_cov[i][j]) > 1e-10 for i in range(3) for j in range(3)):
            raise ValueError('收缩样本与输入协方差不匹配')
        cov, shrinkage = ledoit_wolf(samples)
    initial = allocation(w0, '初始权重') if w0 is not None else [1 / 3] * 3
    weights = [max(w, 1e-8) if b > 0 else 0.0 for w, b in zip(initial, budget)]
    weights = [w / sum(weights) for w in weights]
    # 统一尺度改善坐标下降的数值条件，不改变RC占比。
    scale = max(cov[i][i] for i in range(3))
    normalized_cov = [[v / scale for v in row] for row in cov]
    use_coordinate = False
    method = 'fixed_point'
    for iteration in range(max_iter + 1):
        rc = risk_contributions(weights, cov)
        error = max(abs(r - b) for r, b in zip(rc, budget))
        if error <= tol or iteration == max_iter:
            return dict(weights=weights, rc_pct=rc, iters=iteration,
                        converged=error <= tol, max_budget_error=error,
                        method=method, shrinkage=shrinkage)
        if any(r <= 0 and b > 0 for r, b in zip(rc, budget)) or iteration >= max_iter // 2:
            use_coordinate = True
        if use_coordinate:
            method = 'coordinate_descent'
            variance = sum(weights[i] * normalized_cov[i][j] * weights[j]
                           for i in range(3) for j in range(3))
            x = [w / math.sqrt(variance) for w in weights]
            for i in range(3):
                if budget[i] == 0:
                    x[i] = 0.0
                    continue
                cross = sum(normalized_cov[i][j] * x[j] for j in range(3) if j != i)
                diagonal = normalized_cov[i][i]
                root = math.sqrt(cross * cross + 4 * diagonal * budget[i])
                x[i] = 2 * budget[i] / (root + cross) if cross >= 0 else (root - cross) / (2 * diagonal)
            weights = [value / sum(x) for value in x]
        else:
            updated = [w * math.sqrt(b / r) if b > 0 else 0.0
                       for w, b, r in zip(weights, budget, rc)]
            weights = [w / sum(updated) for w in updated]


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def historical_cvar(returns):
    """最差5%损失的均值，边界观测按比例加权，精确保留12.5/250的尾部质量。"""
    losses = sorted((-r for r in returns), reverse=True)
    mass = len(losses) * 0.05
    whole = int(mass)
    return (sum(losses[:whole]) + (mass - whole) * losses[whole]) / mass


def drawdown_attribution(path, records):
    navs, rows = path['navs'], path['rows']
    peak_index = max(range(len(navs)), key=lambda i: (navs[i], i))
    dates = [day for day, _ in records]
    if peak_index == 0:
        start_index = dates.index(rows[0]['date']) - 1
        start = dates[start_index] if start_index >= 0 else None
    else:
        start = rows[peak_index - 1]['date']
    totals = [0.0] * 4
    for i in range(peak_index, len(rows)):
        factor = navs[i] / navs[peak_index]
        for j, contribution in enumerate(path['contributions'][i]):
            totals[j] += factor * contribution
    result = dict(start=start, end=rows[-1]['date'], **dict(zip(NAMES, totals[:3])))
    result['transaction_cost'] = totals[3]
    return result


def change_attribution(old_weights, new_weights, old_cov, new_cov, base_date):
    """RC% 在基准点的一阶方向导数，分别扰动权重、波动、相关，单列近似残差。"""
    old_vol = [math.sqrt(old_cov[i][i]) for i in range(3)]
    new_vol = [math.sqrt(new_cov[i][i]) for i in range(3)]
    old_corr = [[old_cov[i][j] / (old_vol[i] * old_vol[j]) for j in range(3)] for i in range(3)]
    new_corr = [[new_cov[i][j] / (new_vol[i] * new_vol[j]) for j in range(3)] for i in range(3)]
    effects = {}
    step = 1e-5
    for component in ('weight', 'vol', 'corr'):
        results = []
        for direction in (-step, step):
            weights = list(old_weights)
            vols = list(old_vol)
            correlations = [list(row) for row in old_corr]
            if component == 'weight':
                weights = [a + direction * (b - a) for a, b in zip(old_weights, new_weights)]
            elif component == 'vol':
                vols = [a + direction * (b - a) for a, b in zip(old_vol, new_vol)]
            else:
                correlations = [[old_corr[i][j] + direction * (new_corr[i][j] - old_corr[i][j])
                                 for j in range(3)] for i in range(3)]
            cov = [[vols[i] * vols[j] * correlations[i][j] for j in range(3)] for i in range(3)]
            results.append(risk_contributions(weights, cov))
        effects[component] = [(b - a) / (2 * step) for a, b in zip(*results)]
    old_rc = risk_contributions(old_weights, old_cov)
    new_rc = risk_contributions(new_weights, new_cov)
    result = {'base': base_date}
    for i, name in enumerate(NAMES):
        result[name] = {key: values[i] for key, values in effects.items()}
        result[name]['residual'] = new_rc[i] - old_rc[i] - sum(values[i] for values in effects.values())
    return result


def build_feedback(records, history, as_of_date, group, base_date=None):
    if group not in ('A', 'B', 'C', 'D'):
        raise ValueError('组别须为 A/B/C/D')
    market = [(d, v) for d, v in records if d <= as_of_date]
    if not market or market[-1][0] != as_of_date:
        raise ValueError('截止日须为矩阵交易日')
    if group == 'A':
        return {'as_of_date': as_of_date}, {'history_required': False}
    path = portfolio_history(market, history, as_of_date)
    returns = path['returns']
    if len(returns) < 60:
        raise ValueError('收益反馈至少需要60日组合历史')
    result = dict(as_of_date=as_of_date, portfolio_return={
        'd20': compound(returns[-20:]), 'd60': compound(returns[-60:])})
    audit = dict(history_start=path['rows'][0]['date'], history_days=len(returns),
                 nav=path['navs'][-1], cost_days=sum(r['cost'] > 0 for r in path['rows']))
    if group == 'B':
        return result, audit
    if len(returns) < 270 or len(market) < 310:
        raise ValueError('风险预警需要至少270日组合历史、310日市场历史')
    vol20 = stdev(returns[-20:]) * math.sqrt(250)
    vol_reference = [stdev(returns[end - 20:end]) * math.sqrt(250)
                     for end in range(len(returns) - 250, len(returns))]
    volatility_threshold = quantile(vol_reference, 0.8)
    def stock_bond_correlation(end):
        window = [v for _, v in market[end - 60:end]]
        return corr([v[0] for v in window], [v[1] for v in window])
    correlation = stock_bond_correlation(len(market))
    correlation_reference = [stock_bond_correlation(end)
                             for end in range(len(market) - 250, len(market))]
    reference_std = stdev(correlation_reference)
    if reference_std == 0:
        raise ValueError('历史相关系数标准差为0，无法计算Z分数')
    correlation_z = (correlation - mean(correlation_reference)) / reference_std
    result.update(portfolio_volatility={'d20': vol20, 'd60': stdev(returns[-60:]) * math.sqrt(250)},
                  current_drawdown=path['drawdowns'][-1], max_drawdown=min(path['drawdowns']),
                  cvar_95=historical_cvar(returns[-250:]),
                  trigger_flags={'drawdown': all(dd < -0.08 for dd in path['drawdowns'][-3:]),
                                 'volatility': vol20 > volatility_threshold,
                                 'correlation': correlation_z > 2})
    audit.update(volatility_threshold=volatility_threshold, correlation_z=correlation_z,
                 reference_observations=250, reference_excludes_signal_day=True)
    if group == 'C':
        return result, audit
    if base_date is None or check_date(base_date) >= as_of_date:
        raise ValueError('D组须明确提供早于截止日的变化归因基准日')
    rows_by_date = {row['date']: row for row in path['rows']}
    if base_date not in rows_by_date:
        raise ValueError('D组基准日缺少实际持仓')
    market_by_date = dict(market)
    current_weights = closing_weights(path['rows'][-1], market[-1][1])
    old_weights = closing_weights(rows_by_date[base_date], market_by_date[base_date])
    cov, shrinkage = covariance_at(market, as_of_date)
    old_cov, old_shrinkage = covariance_at(market, base_date)
    rc = risk_contributions(current_weights, cov)
    result.update(risk_contribution=dict(zip(NAMES, rc)),
                  budget_deviation=dict(zip(NAMES, (r - b for r, b in zip(rc, path['rows'][-1]['budget'])))),
                  drawdown_attribution=drawdown_attribution(path, market),
                  risk_change_attribution=change_attribution(old_weights, current_weights, old_cov, cov, base_date),
                  concentration_state={**dict(zip(NAMES, rc)), 'neff': 1 / sum(r * r for r in rc)})
    audit.update(current_weights=current_weights, covariance_shrinkage=shrinkage,
                 base_covariance_shrinkage=old_shrinkage)
    return result, audit
