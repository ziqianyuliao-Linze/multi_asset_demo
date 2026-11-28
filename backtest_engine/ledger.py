# ledger.py
"""回测账本：100 万初始资金，逐日净值、成交、费用、换手率。

口径依据《统一数据口径说明》（V1.0）第八、九节：

- 9.2 暖启动：2013-07-29 收盘三资产等权建仓；首个模型信号 2014-11-28；
      首次成交 2014-12-01；新权重自 2014-12-02 起承接收益。
- 9.4 交易费按隐式方程求解：F = 0.0003 × Σ|w_i(V⁻ − F) − A_i⁻|。
- 9.5 单边换手率：Turnover = Σ|TradeAmount_i| / (2V⁻)；交易金额不单独入账。
- 8   主实验权重只受 0 ≤ w ≤ 1、Σw = 1 约束，不设 5%–60% 限制；
      限仓只作为稳健性实验的可选参数（bounds=(0.05, 0.60)）。

两种模式：
  mode="continuous"  主实验。连续权重，按目标权重精确持仓，不取整，现金为 0。
  mode="lots"        可交易性稳健性实验。100 份整数倍、现金余额、买不起就少买一手。

账本行与 interfaces.data_schema.BacktestRecord 的字段一一对应；持仓行可直接交给
risk_metrics.risk_feedback.read_holdings（w_* 为当日收益发生前的权重，避免风险模块
重复计算当天收益）。
"""

from dataclasses import dataclass, field

import math

CODES = ("510300", "511010", "518880")
INITIAL_CAPITAL = 1_000_000.0
COST_RATE = 0.0003      # 买卖双边各 0.03%
LOT = 100

TRADING_DAYS = 250
RISK_FREE_ANNUAL = 0.015
WARM_START_DATE = "2013-07-29"
FORMAL_START = "2014-12-01"
WARM_START_WEIGHTS = {c: 1.0 / 3.0 for c in CODES}

# 账本列名固定按 interfaces.data_schema.BacktestRecord 的顺序。
RECORD_FIELDS = (
    "date", "group",
    "beginning_nav", "pre_trade_nav", "ending_nav",
    "gross_return", "net_return",
    "cost_amount", "cost_rate", "turnover",
    "asset_returns", "actual_weights", "effective_budget",
)

# 与 risk_metrics.risk_feedback.HOLDING_COLUMNS 完全一致。
HOLDING_COLUMNS = (
    ("as_of_date", "group")
    + tuple("w_" + c for c in CODES)
    + tuple("budget_" + c for c in CODES)
    + ("cost_rate",)
)


class InputError(ValueError):
    """数据或口径不对就抛错，不静默补值。"""


# --------------------------------------------------------------------------- #
# 交易费用与取整
# --------------------------------------------------------------------------- #

def commission(amount, rate=COST_RATE):
    """显式佣金（仅 lots 模式按实际成交额使用）。"""
    return amount * rate


def solve_fee(pre_trade_nav, holdings, target, rate=COST_RATE):
    """解隐式方程 F = rate × Σ|w_i(V⁻ − F) − A_i⁻|。

    g(F) = rate × Σ|w_i(V⁻ − F) − A_i⁻| − F 严格单调递减
    （g' = −rate·Σw_i − 1 = −(1 + rate) < 0），在 [0, V⁻] 上有唯一根。
    用二分法求解，绝对误差 ≤ 1e-9 元。
    """
    nav = float(pre_trade_nav)
    if not math.isfinite(nav) or nav <= 0:
        raise InputError(f"成交前净值必须为正的有限数，收到 {pre_trade_nav!r}")
    if not math.isfinite(rate) or rate < 0:
        raise InputError(f"费率必须非负，收到 {rate!r}")
    if rate == 0:
        return 0.0
    if abs(math.fsum(target.values()) - 1.0) > 1e-6:
        raise InputError("目标权重合计必须等于 1")

    def g(fee):
        traded = math.fsum(abs(target[c] * (nav - fee) - holdings[c]) for c in CODES)
        return rate * traded - fee

    low, high = 0.0, nav
    if g(low) < 0 or g(high) > 0:          # 理论上不可能，出错就报，不静默
        raise InputError(f"费用方程在 [0, {nav}] 上无根，检查权重与持仓")
    for _ in range(200):
        mid = 0.5 * (low + high)
        if g(mid) > 0:
            low = mid
        else:
            high = mid
        if high - low <= 1e-9:
            break
    return 0.5 * (low + high)


def trade_amounts(pre_trade_nav, holdings, target, fee):
    """成交金额 TradeAmount_i = w_i(V⁻ − F) − A_i⁻（只在内存里算，不单独入账）。"""
    after = float(pre_trade_nav) - fee
    return {c: target[c] * after - holdings[c] for c in CODES}


def turnover_ratio(amounts, pre_trade_nav):
    """单边换手率 Σ|TradeAmount_i| / (2V⁻)。"""
    nav = float(pre_trade_nav)
    if nav <= 0:
        raise InputError(f"成交前净值必须为正，收到 {pre_trade_nav!r}")
    return math.fsum(abs(v) for v in amounts.values()) / (2.0 * nav)


def lots_for(amount, price, lot=LOT):
    if price <= 0:
        raise InputError(f"价格必须大于 0，收到 {price}")
    return math.floor(amount / price + 1e-9) // lot * lot


def max_affordable(price, cash, want, rate=COST_RATE, lot=LOT):
    qty = want
    while qty > 0:
        amount = qty * price
        if amount + commission(amount, rate) <= cash + 1e-9:
            return qty
        qty -= lot
    return 0


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

def _check_weights(weights, bounds, where):
    if not isinstance(weights, dict) or set(weights) != set(CODES):
        raise InputError(f"{where} 的权重字段必须恰为 {list(CODES)}，收到 {weights!r}")
    for c in CODES:
        w = weights[c]
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
            raise InputError(f"{where} 的 {c} 权重必须是有限数字，收到 {w!r}")
    if abs(math.fsum(weights.values()) - 1.0) > 1e-6:
        raise InputError(f"{where} 的权重合计 {math.fsum(weights.values()):.6f}，必须等于 1")
    low, high = bounds
    for c in CODES:
        w = weights[c]
        if w < low - 1e-12 or w > high + 1e-12:
            raise InputError(f"{where} 的 {c} 权重 {w} 超出 [{low}, {high}]")


def _check_bounds(bounds):
    if bounds is None:
        return 0.0, 1.0
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise InputError(f"bounds 必须是 (下限, 上限)，收到 {bounds!r}")
    low, high = float(bounds[0]), float(bounds[1])
    if not (0.0 <= low <= high <= 1.0):
        raise InputError(f"bounds 必须满足 0 ≤ 下限 ≤ 上限 ≤ 1，收到 {bounds!r}")
    return low, high


def _normalize_item(value, where, bounds):
    """把 plan 的一项统一成 {"weights":..., "budget":..., "signal_date":...}。"""
    if not isinstance(value, dict):
        raise InputError(f"{where} 的决策必须是 dict，收到 {value!r}")
    if "weights" in value:
        weights = value["weights"]
        budget = value.get("budget")
        signal_date = value.get("signal_date")
    else:
        weights = value
        budget = None
        signal_date = None
    _check_weights(weights, bounds, where)
    if budget is not None:
        _check_weights(budget, (0.0, 1.0), f"{where} 的风险预算")
        if any(budget[c] <= 0 for c in CODES):
            raise InputError(f"{where} 的风险预算必须全部为正")
    return {"weights": dict(weights), "budget": None if budget is None else dict(budget),
            "signal_date": signal_date}


# --------------------------------------------------------------------------- #
# 账本
# --------------------------------------------------------------------------- #

@dataclass
class Ledger:
    """一行对应一个交易日；periods 按信号期记录 fallback 状态与原因。"""

    dates: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    periods: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    group: str = "A"

    # 严格按 BacktestRecord 导出（不含扩展字段）。
    def to_rows(self, group=None):
        group = self.group if group is None else group
        out = []
        for r in self.rows:
            out.append({
                "date": r["date"], "group": group,
                "beginning_nav": r["beginning_nav"],
                "pre_trade_nav": r["pre_trade_nav"],
                "ending_nav": r["ending_nav"],
                "gross_return": r["gross_return"],
                "net_return": r["net_return"],
                "cost_amount": r["cost_amount"],
                "cost_rate": r["cost_rate"],
                "turnover": r["turnover"],
                "asset_returns": dict(r["asset_returns"]),
                "actual_weights": dict(r["actual_weights"]),
                "effective_budget": dict(r["effective_budget"]),
            })
        return out

    # 直接喂给 risk_metrics.risk_feedback.read_holdings。
    def to_holdings_rows(self, group=None):
        group = self.group if group is None else group
        out = []
        for r in self.rows:
            row = {"as_of_date": r["date"], "group": group}
            for c in CODES:
                row["w_" + c] = r["actual_weights"][c]
            for c in CODES:
                row["budget_" + c] = r["effective_budget"][c]
            row["cost_rate"] = r["cost_rate"]
            out.append(row)
        return out


def _next_trading_day(signal, dates, index):
    if signal not in index:
        raise InputError(f"信号日 {signal} 不在交易日历里")
    i = index[signal]
    return dates[i + 1] if i + 1 < len(dates) else None


def run_backtest(returns, plan, dates, prices=None, initial=INITIAL_CAPITAL,
                 rate=COST_RATE, mode="continuous", lot=LOT, group="A",
                 warm_start=None, bounds=None, signals=None):
    """returns: {代码: [每日收益]}；plan: {成交日: {代码: 目标权重} 或
    {"weights":..., "budget":..., "signal_date":...}}。

    warm_start: 建仓权重（暖启动）；给定时第一天开盘即持有该组合，不收建仓费。
    bounds:     权重区间，默认 (0.0, 1.0)（主实验不设 5%–60% 限制）。
    signals:    信号日列表；给出时按信号期记录 fallback 状态（缺决策就沿用上期）。

    mode="continuous" 不需要 prices；mode="lots" 必须给 prices。
    """
    if mode not in ("continuous", "lots"):
        raise InputError(f"mode 只能是 continuous 或 lots，收到 {mode!r}")
    if not dates:
        raise InputError("dates 不能为空")
    if len(set(dates)) != len(dates) or list(dates) != sorted(dates):
        raise InputError("dates 必须升序且不重复")
    for c in CODES:
        if c not in returns or len(returns[c]) != len(dates):
            raise InputError(f"{c} 的收益长度和 dates 对不上")
    if mode == "lots" and not prices:
        raise InputError("lots 模式必须给 prices（真实收盘价）")

    low, high = _check_bounds(bounds)
    plan_norm = {d: _normalize_item(v, f"{d} 的决策", (low, high))
                 for d, v in plan.items()}
    for d in plan_norm:
        if d not in dates:
            raise InputError(f"调仓日 {d} 不在交易日历里")

    index = {d: i for i, d in enumerate(dates)}
    fallback_trades = {}
    if signals is not None:
        seen = set()
        for s in signals:
            if s in seen:
                raise InputError(f"信号日 {s} 重复")
            seen.add(s)
            t = _next_trading_day(s, dates, index)
            if t is None:            # 数据末尾那个月的信号还没到成交日
                continue
            if t not in plan_norm:
                fallback_trades[t] = s

    led = Ledger(dates=list(dates), group=group,
                 meta={"mode": mode, "initial_capital": float(initial),
                       "cost_rate": float(rate), "bounds": [low, high],
                       "warm_start": None if warm_start is None else dict(warm_start),
                       "signals": None if signals is None else list(signals)})

    cash = float(initial)
    value = {c: 0.0 for c in CODES}
    shares = {c: 0.0 for c in CODES}
    if warm_start is not None:
        _check_weights(warm_start, (low, high), "暖启动权重")
        value = {c: float(initial) * warm_start[c] for c in CODES}
        cash = float(initial) - math.fsum(value.values())
        current_weights = dict(warm_start)
        effective_budget = dict(warm_start)
    else:
        current_weights = {c: 0.0 for c in CODES}
        effective_budget = {c: 0.0 for c in CODES}

    prev_nav = float(initial)
    applied_any = False
    fallback_seen = False

    for i, d in enumerate(dates):
        nav_begin = prev_nav
        if nav_begin <= 0:
            raise InputError(f"{d} 的期初净值必须为正，收到 {nav_begin}")
        rets = {c: returns[c][i] for c in CODES}
        for c in CODES:
            if not isinstance(rets[c], (int, float)) or not math.isfinite(rets[c]):
                raise InputError(f"{d} 的 {c} 收益不是有限数字：{rets[c]!r}")
            if rets[c] <= -1:
                raise InputError(f"{d} 的 {c} 收益必须大于 -1，收到 {rets[c]}")

        # ① 昨天收盘的持仓按今天的收益漂移；今日收益发生前的权重就是 current_weights
        if mode == "continuous":
            value = {c: value[c] * (1.0 + rets[c]) for c in CODES}
        else:
            px = prices[d]
            value = {c: shares[c] * px[c] for c in CODES}
        pre_trade = math.fsum(value.values()) + cash
        if pre_trade <= 0:
            raise InputError(f"{d} 的成交前净值必须为正，收到 {pre_trade}")

        # ② 今天收盘成交
        cost = 0.0
        turnover = 0.0
        status = "none"
        reason = ""

        if d in plan_norm:
            item = plan_norm[d]
            target = item["weights"]
            if item["budget"] is not None:
                effective_budget = dict(item["budget"])
            elif not applied_any and warm_start is None:
                reason = "决策未给风险预算，且无历史预算，沿用等权"
            if mode == "continuous":
                fee = solve_fee(pre_trade, value, target, rate)
                amounts = trade_amounts(pre_trade, value, target, fee)
                turnover = turnover_ratio(amounts, pre_trade)
                cost = fee
                after = pre_trade - fee
                value = {c: target[c] * after for c in CODES}
                cash = 0.0
            else:
                px = prices[d]
                amounts = {c: 0.0 for c in CODES}
                for c in CODES:
                    want = lots_for(pre_trade * target[c], px[c], lot)
                    if want < shares[c]:
                        qty = shares[c] - want
                        amount = qty * px[c]
                        fee = commission(amount, rate)
                        cash += amount - fee
                        shares[c] -= qty
                        cost += fee
                        amounts[c] -= amount
                for c in CODES:
                    need = lots_for(pre_trade * target[c], px[c], lot) - shares[c]
                    if need > 0:
                        qty = max_affordable(px[c], cash, need, rate, lot)
                        if qty:
                            amount = qty * px[c]
                            fee = commission(amount, rate)
                            cash -= amount + fee
                            shares[c] += qty
                            cost += fee
                            amounts[c] += amount
                turnover = turnover_ratio(amounts, pre_trade)
                value = {c: shares[c] * px[c] for c in CODES}
            status = "applied"
            reason = f"应用 {item['signal_date']} 期决策" if item["signal_date"] else "应用决策"
            applied_any = True
        elif d in fallback_trades:
            signal = fallback_trades[d]
            if not applied_any and not fallback_seen:
                status = "warm_start"
                reason = f"首期缺少 {signal} 期决策，回退暖启动等权持仓"
            else:
                status = "hold_previous"
                reason = f"缺少 {signal} 期决策，沿用上一期有效预算与实际持仓"
            fallback_seen = True

        nav_end = math.fsum(value.values()) + cash
        if nav_end <= 0:
            raise InputError(f"{d} 的期末净值必须为正，收到 {nav_end}")

        led.rows.append({
            "date": d,
            "group": group,
            "beginning_nav": nav_begin,
            "pre_trade_nav": pre_trade,
            "ending_nav": nav_end,
            "gross_return": pre_trade / nav_begin - 1.0,
            "net_return": nav_end / nav_begin - 1.0,
            "cost_amount": cost,
            "cost_rate": cost / nav_begin,
            "turnover": turnover,
            "asset_returns": dict(rets),
            "actual_weights": dict(current_weights),
            "effective_budget": dict(effective_budget),
            # 扩展字段：只留在内存，不进入 BacktestRecord 导出
            "signal_date": plan_norm[d]["signal_date"] if d in plan_norm else None,
            "fallback_status": status,
            "fallback_reason": reason,
            "shares": {c: shares[c] for c in CODES},
        })
        if status != "none":
            led.periods.append({
                "signal_date": plan_norm[d]["signal_date"] if d in plan_norm
                else fallback_trades[d],
                "trade_date": d,
                "status": status,
                "reason": reason,
                "weights": dict(current_weights),
                "budget": dict(effective_budget),
            })

        prev_nav = nav_end
        current_weights = {c: (value[c] / nav_end if nav_end else 0.0) for c in CODES}

    return led


# --------------------------------------------------------------------------- #
# 绩效汇总
# --------------------------------------------------------------------------- #

def _stdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    var = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def _max_drawdown(navs):
    peak, mdd = navs[0], 0.0
    for nav in navs:
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1.0)
    return mdd


def _cvar_95(rets):
    """7.4 口径：最近 250 日净收益，12 笔全额 + 第 13 笔半额，除以 12.5。"""
    window = rets[-250:]
    losses = sorted(-r for r in window)[::-1]
    if len(losses) < 13:
        return 0.0
    return (math.fsum(losses[:12]) + 0.5 * losses[12]) / 12.5


def summarize(led):
    """未归一化的全样本汇总（兼容旧接口；正式绩效用 summarize_formal）。"""
    navs = [r["ending_nav"] for r in led.rows]
    rets = [r["net_return"] for r in led.rows[1:]]
    n = len(rets)
    total = navs[-1] / navs[0] - 1 if len(navs) > 1 else 0.0
    mean = math.fsum(rets) / n if n else 0.0
    var = math.fsum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    vol = math.sqrt(var) * math.sqrt(TRADING_DAYS) if var > 0 else 0.0
    years = n / TRADING_DAYS
    annual = (navs[-1] / navs[0]) ** (1 / years) - 1 if years > 0 and navs[0] > 0 else 0.0
    return {"days": len(navs), "final_nav": navs[-1], "total_return": total,
            "annual_return": annual, "annual_vol": vol,
            "sharpe": (mean * TRADING_DAYS / vol) if vol > 0 else 0.0,
            "max_drawdown": _max_drawdown(navs),
            "total_cost": math.fsum(r["cost_amount"] for r in led.rows),
            "total_turnover": math.fsum(r["turnover"] for r in led.rows)}


def summarize_formal(led, start=FORMAL_START, risk_free=RISK_FREE_ANNUAL):
    """9.2 / 第十节口径：以 start 日成交前净值归一，计入首次交易费。"""
    rows = [r for r in led.rows if r["date"] >= start]
    if not rows:
        raise InputError(f"账本里没有 {start} 及之后的交易日")
    base = rows[0]["pre_trade_nav"]
    if base <= 0:
        raise InputError(f"{start} 的成交前净值必须为正，收到 {base}")

    rets = [rows[0]["ending_nav"] / base - 1.0]
    navs = [1.0, 1.0 * (1.0 + rets[0])]
    for r in rows[1:]:
        rets.append(r["net_return"])
        navs.append(navs[-1] * (1.0 + r["net_return"]))

    n = len(rets)
    rf_daily = (1.0 + risk_free) ** (1.0 / TRADING_DAYS) - 1.0
    excess = [r - rf_daily for r in rets]
    vol = _stdev(rets) * math.sqrt(TRADING_DAYS)
    excess_vol = _stdev(excess)
    annual = navs[-1] ** (TRADING_DAYS / n) - 1.0
    mdd = _max_drawdown(navs)

    months = {}
    for r in rows:
        month = r["date"][:7]
        months[month] = months.get(month, 1.0) * (1.0 + r["net_return"])
    wins = sum(1 for v in months.values() if v > 1.0)

    last_w = rows[-1]["actual_weights"]
    concentration = math.fsum(w * w for w in last_w.values())
    return {
        "start_date": rows[0]["date"],
        "end_date": rows[-1]["date"],
        "days": n,
        "start_nav": 1.0,
        "end_nav": navs[-1],
        "total_return": navs[-1] - 1.0,
        "annual_return": annual,
        "annual_vol": vol,
        "sharpe": (math.fsum(excess) / n / excess_vol * math.sqrt(TRADING_DAYS)
                   if excess_vol > 0 else 0.0),
        "risk_free_annual": risk_free,
        "max_drawdown": mdd,
        "calmar": (annual / abs(mdd)) if mdd < 0 else 0.0,
        "cvar_95": _cvar_95(rets),
        "total_cost": math.fsum(r["cost_amount"] for r in rows),
        "total_turnover": math.fsum(r["turnover"] for r in rows),
        "annual_turnover": math.fsum(r["turnover"] for r in rows) * TRADING_DAYS / n,
        "monthly_win_rate": wins / len(months) if months else 0.0,
        "effective_assets": (1.0 / concentration) if concentration > 0 else 0.0,
    }
