"""回测账本：100 万初始资金，逐日净值、成交、费用、换手率。

两种模式，对应分工规范里「主实验」和「稳健性实验」两件事：
  mode="continuous"  主实验。连续权重，按目标权重精确持仓，不取整，现金为 0，
                     当天费用按成交额比例从净值里扣（口径与 4 号一致）。
  mode="lots"        可交易性稳健性实验。100 份整数倍、现金余额、买不起就少买一手。
"""

from dataclasses import dataclass, field

import math

CODES = ("510300", "511010", "518880")
INITIAL_CAPITAL = 1_000_000.0
COST_RATE = 0.0003      # 买卖双边各 0.03%
LOT = 100


class InputError(ValueError):
    """数据或口径不对就抛错，不静默补值。"""


def commission(amount, rate=COST_RATE):
    return amount * rate


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


@dataclass
class Ledger:
    """一行对应一个交易日，字段按分工规范 5.4 回测账本接口。"""
    dates: list = field(default_factory=list)
    rows: list = field(default_factory=list)

    def to_rows(self, group="A"):
        out = []
        for r in self.rows:
            row = {"as_of_date": r["date"], "group": group,
                   "nav_begin": r["nav_begin"], "nav_before_trade": r["nav_before_trade"],
                   "nav_end": r["nav_end"],
                   "gross_return": r["gross_return"], "net_return": r["net_return"],
                   "cost": r["cost"], "cost_rate": r["cost_rate"], "turnover": r["turnover"],
                   "holdings_value": r["market_value"], "cash": r["cash"]}
            for c in CODES:
                row[f"r_{c}"] = r["asset_returns"][c]
                row[f"w_{c}"] = r["weights"][c]
                row[f"budget_{c}"] = r["budget"][c]
                row[f"shares_{c}"] = r["shares"][c]
            out.append(row)
        return out


def run_backtest(returns, plan, dates, prices=None, initial=INITIAL_CAPITAL,
                 rate=COST_RATE, mode="continuous", lot=LOT, group="A"):
    """returns: {代码: [每日收益]}；plan: {成交日: {代码: 目标权重}}。

    mode="continuous" 不需要 prices；mode="lots" 必须给 prices。
    """
    if mode not in ("continuous", "lots"):
        raise InputError(f"mode 只能是 continuous 或 lots，收到 {mode!r}")
    if not dates:
        raise InputError("dates 不能为空")
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise InputError("dates 必须升序且不重复")
    for c in CODES:
        if c not in returns or len(returns[c]) != len(dates):
            raise InputError(f"{c} 的收益长度和 dates 对不上")
    if mode == "lots" and not prices:
        raise InputError("lots 模式必须给 prices（真实收盘价）")
    for d, w in plan.items():
        if d not in dates:
            raise InputError(f"调仓日 {d} 不在交易日历里")
        if abs(sum(w.values()) - 1.0) > 1e-6:
            raise InputError(f"{d} 的权重合计 {sum(w.values()):.6f}，必须等于 1")
        for c, w in w.items():
            if w < -1e-12:
                raise InputError(f"{d} 的 {c} 权重是负数 {w}")
            if 0 < w < 0.05 - 1e-9:
                raise InputError(f"{d} 的 {c} 权重 {w} 低于 5%，要么清仓写 0，要么给到 5% 以上")
            if w > 0.60 + 1e-9:
                raise InputError(f"{d} 的 {c} 权重 {w} 超过 60%")

    led = Ledger()
    cash = float(initial)
    value = {c: 0.0 for c in CODES}
    shares = {c: 0.0 for c in CODES}
    prev_nav = float(initial)
    last_w = {c: 0.0 for c in CODES}
    last_budget = {c: 0.0 for c in CODES}

    for i, d in enumerate(dates):
        nav_begin = prev_nav
        rets = {c: returns[c][i] for c in CODES}

        # ① 昨天的持仓按今天的收益漂移
        if mode == "continuous":
            value = {c: value[c] * (1 + rets[c]) for c in CODES}
            nav_before = sum(value.values()) + cash
        else:
            px = prices[d]
            value = {c: shares[c] * px[c] for c in CODES}
            nav_before = sum(value.values()) + cash

        # ② 今天收盘成交：先算换手，再扣费
        cost = turnover = 0.0
        if d in plan:
            target = plan[d]
            if mode == "continuous":
                turnover = sum(abs(target[c] - value[c] / nav_before) for c in CODES) * nav_before
                cost = commission(turnover, rate)
                nav_after = nav_before - cost
                value = {c: nav_after * target[c] for c in CODES}
                cash = 0.0
                last_budget = dict(target)
            else:
                px = prices[d]
                for c in CODES:
                    want = lots_for(nav_before * target[c], px[c], lot)
                    if want < shares[c]:
                        qty = shares[c] - want
                        amount = qty * px[c]
                        fee = commission(amount, rate)
                        cash += amount - fee
                        shares[c] -= qty
                        cost += fee
                        turnover += amount
                for c in CODES:
                    need = lots_for(nav_before * target[c], px[c], lot) - shares[c]
                    if need > 0:
                        qty = max_affordable(px[c], cash, need, rate, lot)
                        if qty:
                            amount = qty * px[c]
                            fee = commission(amount, rate)
                            cash -= amount + fee
                            shares[c] += qty
                            cost += fee
                            turnover += amount
                value = {c: shares[c] * px[c] for c in CODES}
                last_budget = dict(target)

        nav_end = sum(value.values()) + cash
        weights = {c: (value[c] / nav_end if nav_end else 0.0) for c in CODES}
        led.dates.append(d)
        led.rows.append({
            "date": d, "nav_begin": nav_begin, "nav_before_trade": nav_before,
            "nav_end": nav_end,
            "gross_return": (nav_before / prev_nav - 1) if prev_nav else 0.0,
            "net_return": (nav_end / prev_nav - 1) if prev_nav else 0.0,
            "cost": cost, "cost_rate": (cost / prev_nav) if prev_nav else 0.0,
            "turnover": turnover, "market_value": sum(value.values()), "cash": cash,
            "asset_returns": rets, "weights": weights, "budget": dict(last_budget),
            "shares": {c: (value[c] / prices[d][c] if mode == "lots" else 0.0) for c in CODES},
        })
        prev_nav = nav_end
        last_w = weights
    led.group = group
    return led


def summarize(led):
    navs = [r["nav_end"] for r in led.rows]
    rets = [r["net_return"] for r in led.rows[1:]]
    n = len(rets)
    total = navs[-1] / navs[0] - 1 if len(navs) > 1 else 0.0
    mean = sum(rets) / n if n else 0.0
    var = sum((r - mean) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    vol = math.sqrt(var) * math.sqrt(250) if var > 0 else 0.0
    years = n / 250
    annual = (navs[-1] / navs[0]) ** (1 / years) - 1 if years > 0 and navs[0] > 0 else 0.0
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    return {"days": len(navs), "final_nav": navs[-1], "total_return": total,
            "annual_return": annual, "annual_vol": vol,
            "sharpe": (mean * 250 / vol) if vol > 0 else 0.0, "max_drawdown": mdd,
            "total_cost": sum(r["cost"] for r in led.rows),
            "total_turnover": sum(r["turnover"] for r in led.rows)}