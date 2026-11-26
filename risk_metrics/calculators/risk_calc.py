# -*- coding: utf-8 -*-
"""
风险反馈模块 · 计算器
=====================

只做一件事：给输入，算数字。不分组、不打包、不判断 A/B/C/D。

 1 window_return           近 20/60 日组合收益
 2 annualized_volatility   年化波动（20/60/250）
 3 drawdown_stats          当前回撤、最大回撤
 4 cvar / var_loss         尾部损失（250 日，正数表示损失）
 5 drawdown_alert          回撤报警
 6 volatility_alert        波动报警
 7 correlation_alert       股债相关报警
 8 risk_contribution       风险贡献 RC
 9 budget_deviation        预算偏差 Δ = RC% − b
10 concentration           集中度
11 drawdown_attribution    回撤归因
12 change_attribution      变化归因
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:                                     # 作为包导入时
    from .config import (
        ANNUALIZE_DAYS, ASSETS, CONCENTRATION, CORRELATION,
        DRAWDOWN, TAIL, VOLATILITY, Window,
    )
except ImportError:                      # 直接当脚本运行时
    from config import (
        ANNUALIZE_DAYS, ASSETS, CONCENTRATION, CORRELATION,
        DRAWDOWN, TAIL, VOLATILITY, Window,
    )

EPS = 1e-12


# ---------------------------------------------------------------- 小工具
def _as_frame(returns) -> pd.DataFrame:
    df = returns if isinstance(returns, pd.DataFrame) else pd.DataFrame(
        np.asarray(returns, dtype=float), columns=list(ASSETS))
    missing = [a for a in ASSETS if a not in df.columns]
    if missing:
        raise ValueError(f"缺少资产列：{missing}")
    return df[list(ASSETS)].astype(float)


def _as_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != len(ASSETS):
        raise ValueError(f"权重长度应为 {len(ASSETS)}，实际 {w.size}")
    return w


def _portfolio_returns(returns, weights) -> pd.Series:
    df, w = _as_frame(returns), _as_weights(weights)
    return pd.Series(df.values @ w, index=df.index, name="portfolio")


def equity_curve(returns, weights) -> pd.Series:
    return (1.0 + _portfolio_returns(returns, weights)).cumprod()


def covariance_matrix(returns, window: int = Window.COV) -> pd.DataFrame:
    df = _as_frame(returns).tail(int(window))
    if len(df) < 3:
        raise ValueError("样本不足，至少需要 3 个交易日")
    return df.cov(ddof=1)


# ---------------------------------------------------------------- 1 收益
def window_return(returns, weights, window: int) -> float:
    """最近 window 日累计收益：Π(1+r) − 1。"""
    r = _portfolio_returns(returns, weights).tail(int(window))
    if len(r) < int(window):
        raise ValueError(f"样本不足：需要 {window} 日，实际 {len(r)} 日")
    return float(np.prod(1.0 + r.values) - 1.0)


def window_returns(returns, weights, windows: Sequence[int] = (20, 60)) -> Dict[int, float]:
    return {int(w): window_return(returns, weights, int(w)) for w in windows}


# ---------------------------------------------------------------- 2 波动
def annualized_volatility(returns, weights, window: int) -> float:
    """年化波动 = std(ddof=1) × √250。"""
    r = _portfolio_returns(returns, weights).tail(int(window))
    if len(r) < int(window) or len(r) < 2:
        raise ValueError(f"样本不足：需要 {window} 日，实际 {len(r)} 日")
    return float(r.std(ddof=1) * math.sqrt(ANNUALIZE_DAYS))


def volatility_by_window(returns, weights, windows: Sequence[int] = (20, 60, 250)) -> Dict[int, float]:
    return {int(w): annualized_volatility(returns, weights, int(w)) for w in windows}


def rolling_volatility(returns, weights, window: int = Window.SHORT) -> pd.Series:
    r = _portfolio_returns(returns, weights)
    return r.rolling(int(window), min_periods=int(window)).std(ddof=1) * math.sqrt(ANNUALIZE_DAYS)


# ---------------------------------------------------------------- 3 回撤
def drawdown_series(equity) -> pd.Series:
    eq = equity.astype(float) if isinstance(equity, pd.Series) else pd.Series(equity, dtype=float)
    return eq / eq.cummax() - 1.0


def drawdown_stats(equity) -> Dict[str, object]:
    """当前回撤、最大回撤及其峰谷日期（回撤为负）。"""
    eq = equity.astype(float) if isinstance(equity, pd.Series) else pd.Series(equity, dtype=float)
    dd = drawdown_series(eq)
    trough = dd.idxmin()
    return {
        "current": float(dd.iloc[-1]),
        "max_drawdown": float(dd.min()),
        "peak_date": eq.loc[:trough].idxmax(),
        "trough_date": trough,
    }


# ---------------------------------------------------------------- 4 CVaR
def _tail_losses(returns, weights, level: float, window: int) -> Tuple[np.ndarray, int, float, float]:
    """尾部损失：k = (1−level)×window，返回 (损失降序, k 整数部分, 小数部分, k)。"""
    r = _portfolio_returns(returns, weights).tail(int(window)).values
    if r.size == 0:
        raise ValueError("没有可用收益")
    losses = np.sort(-r)[::-1]
    k = (1.0 - float(level)) * r.size
    if k < 1e-9:
        raise ValueError("level 过高，尾部不足 1 笔")
    k_int = int(math.floor(k + 1e-9))
    k_frac = float(k - k_int)
    if k_frac < 1e-9:
        k_frac = 0.0
    if k_int >= losses.size:
        raise ValueError("样本不足以覆盖尾部")
    return losses, k_int, k_frac, float(k)


def cvar(returns, weights, level: float = None, window: int = None) -> float:
    """
    CVaR（历史模拟法），正数表示损失。

    方案口径：250 日、95% 时 k = 12.5，前 12 笔全权重 + 第 13 笔半权重，合计 ÷ 12.5。
    """
    level = TAIL["cvar_level"] if level is None else level
    window = TAIL["window"] if window is None else window
    losses, k_int, k_frac, k = _tail_losses(returns, weights, level, window)
    return float((losses[:k_int].sum() + k_frac * losses[k_int]) / k)


def var_loss(returns, weights, level: float = None, window: int = None) -> float:
    """VaR：第 ⌈k⌉ 笔损失，正数表示损失。"""
    level = TAIL["var_level"] if level is None else level
    window = TAIL["window"] if window is None else window
    losses, k_int, k_frac, _ = _tail_losses(returns, weights, level, window)
    idx = (k_int - 1) if k_frac == 0 else k_int
    return float(losses[max(0, min(idx, losses.size - 1))])


# ---------------------------------------------------------------- 5 回撤报警
def drawdown_alert(equity, level: float = None, persistence: int = None) -> Dict[str, object]:
    """当前回撤跌破 −8%，且最近 3 个交易日都在阈值下才触发。"""
    level = DRAWDOWN["alert_level"] if level is None else level
    persistence = DRAWDOWN["persistence_days"] if persistence is None else persistence
    dd = drawdown_series(equity)
    hit = dd < -abs(float(level))
    streak = 0
    for v in reversed(hit.values):
        if not v:
            break
        streak += 1
    return {
        "triggered": bool(streak >= int(persistence)),
        "current": float(dd.iloc[-1]),
        "streak": int(streak),
        "since": dd.index[-streak] if streak else None,
        "level": -abs(float(level)),
    }


# ---------------------------------------------------------------- 6 波动报警
def volatility_alert(returns, weights, window: int = None, ref_window: int = None,
                     quantile: float = None) -> Dict[str, object]:
    """当日 20 日波动 > 之前 250 个同类观测的 80 分位（分位线性插值）。"""
    window = VOLATILITY["window"] if window is None else window
    ref_window = VOLATILITY["reference_window"] if ref_window is None else ref_window
    quantile = VOLATILITY["quantile"] if quantile is None else quantile
    vol = rolling_volatility(returns, weights, int(window)).dropna()
    if len(vol) <= int(ref_window):
        raise ValueError(f"样本不足：需要超过 {ref_window} 个观测，实际 {len(vol)}")
    current = float(vol.iloc[-1])
    hist = vol.iloc[-(int(ref_window) + 1):-1].values
    threshold = float(np.quantile(hist, float(quantile)))
    return {
        "triggered": bool(current > threshold),
        "current_vol": current,
        "threshold": threshold,
        "percentile": float((hist <= current).mean()),
        "window": int(window),
    }


# ---------------------------------------------------------------- 7 相关报警
def rolling_correlation(returns, pair: Tuple[str, str] = None, window: int = None) -> pd.Series:
    pair = CORRELATION["pair"] if pair is None else pair
    window = CORRELATION["window"] if window is None else window
    df = _as_frame(returns)
    return df[pair[0]].rolling(int(window), min_periods=int(window)).corr(df[pair[1]])


def correlation_alert(returns, pair: Tuple[str, str] = None, window: int = None,
                      z_window: int = None, z_threshold: float = None) -> Dict[str, object]:
    """当前 60 日相关相对之前 250 个观测的 Z 分数，Z > 2 触发。"""
    pair = CORRELATION["pair"] if pair is None else pair
    window = CORRELATION["window"] if window is None else window
    z_window = CORRELATION["z_window"] if z_window is None else z_window
    z_threshold = CORRELATION["z_threshold"] if z_threshold is None else z_threshold
    corr = rolling_correlation(returns, pair, int(window)).dropna()
    if len(corr) <= int(z_window):
        raise ValueError(f"样本不足：需要超过 {z_window} 个观测，实际 {len(corr)}")
    hist = corr.iloc[-(int(z_window) + 1):-1]
    sd = float(hist.std(ddof=1))
    current = float(corr.iloc[-1])
    if sd < 1e-12:
        return {"triggered": False, "corr": current, "z": None,
                "threshold": float(z_threshold), "constant_reference": True}
    z = float((current - float(hist.mean())) / sd)
    return {"triggered": bool(z > float(z_threshold)), "corr": current, "z": z,
            "threshold": float(z_threshold), "constant_reference": False}


# ---------------------------------------------------------------- 8 风险贡献
def risk_contribution(weights, cov) -> Dict[str, object]:
    """RC_i = w_i(Σw)_i/σ_p；RC_i% = w_i(Σw)_i/σ_p²，三项和为 1。"""
    w = _as_weights(weights)
    sigma = np.asarray(cov, dtype=float)
    if sigma.shape != (len(ASSETS), len(ASSETS)):
        raise ValueError(f"协方差矩阵形状应为 {len(ASSETS)}×{len(ASSETS)}")
    var = float(w @ sigma @ w)
    if var <= EPS:
        raise ValueError("组合方差接近 0，无法分解风险贡献")
    sigma_w = sigma @ w
    rc_abs = w * sigma_w / math.sqrt(var)
    rc_pct = rc_abs / math.sqrt(var)
    return {
        "rc_pct": {a: float(rc_pct[i]) for i, a in enumerate(ASSETS)},
        "rc_abs": {a: float(rc_abs[i]) for i, a in enumerate(ASSETS)},
        "portfolio_vol_daily": math.sqrt(var),
        "portfolio_vol_annual": math.sqrt(var * ANNUALIZE_DAYS),
    }


# ---------------------------------------------------------------- 9 预算偏差
def budget_deviation(rc_pct, budget) -> Dict[str, float]:
    """Δ_i = RC_i% − b_i（比的是风险贡献与目标风险预算，不是资金权重）。"""
    rc = rc_pct if isinstance(rc_pct, dict) else {a: float(v) for a, v in zip(ASSETS, rc_pct)}
    b = budget if isinstance(budget, dict) else {a: float(v) for a, v in zip(ASSETS, budget)}
    return {a: float(rc[a] - b[a]) for a in ASSETS}


# ---------------------------------------------------------------- 10 集中度
def concentration(rc_pct) -> Dict[str, object]:
    """max RC%、N_eff = 1/Σ(RC%)²、HHI，以及是否超过阈值。"""
    rc = rc_pct if isinstance(rc_pct, dict) else {a: float(v) for a, v in zip(ASSETS, rc_pct)}
    vals = np.array([rc[a] for a in ASSETS], dtype=float)
    hhi = float(np.sum(vals ** 2))
    neff = 1.0 / hhi if hhi > EPS else float(len(ASSETS))
    top = max(rc, key=rc.get)
    return {
        "max_rc": float(rc[top]),
        "max_asset": top,
        "neff": neff,
        "hhi": hhi,
        "is_alert": bool(rc[top] > CONCENTRATION["rc_threshold"] or neff < CONCENTRATION["neff_threshold"]),
    }


# ---------------------------------------------------------------- 11 回撤归因
def drawdown_attribution(returns, weights, peak_idx: int, trough_idx: int) -> Dict[str, object]:
    """区间内各资产贡献 = Σ w_i·r_i,t；占比 = 贡献 / 区间跌幅。"""
    df, w = _as_frame(returns), _as_weights(weights)
    if not (0 <= peak_idx < trough_idx < len(df)):
        raise ValueError("peak_idx / trough_idx 越界")
    seg = df.iloc[peak_idx + 1: trough_idx + 1]
    contrib = (seg.values * w).sum(axis=0)
    depth = float(contrib.sum())
    return {
        "start": df.index[peak_idx],
        "end": df.index[trough_idx],
        "depth": depth,
        "contrib": {a: float(contrib[i]) for i, a in enumerate(ASSETS)},
        "share": {a: float(contrib[i] / depth) if abs(depth) > EPS else 0.0 for i, a in enumerate(ASSETS)},
    }


def find_drawdown_episodes(equity, top_n: int = 3) -> List[Dict[str, object]]:
    """找出最深的若干回撤区间（峰值、谷值、修复日、深度）。"""
    eq = equity.astype(float) if isinstance(equity, pd.Series) else pd.Series(equity, dtype=float)
    peak = eq.cummax()
    under = eq < peak
    episodes, i, n = [], 0, len(eq)
    while i < n:
        if under.iloc[i]:
            start = max(i - 1, 0)
            j = trough = i
            while j < n and eq.iloc[j] < eq.iloc[start]:
                if eq.iloc[j] < eq.iloc[trough]:
                    trough = j
                j += 1
            episodes.append({
                "peak_idx": start, "trough_idx": trough,
                "recover_idx": j if j < n else None,
                "peak_date": eq.index[start], "trough_date": eq.index[trough],
                "depth": float(eq.iloc[trough] / eq.iloc[start] - 1.0),
                "days_to_trough": int(trough - start),
                "days_to_recover": int(j - start),
            })
            i = j
        else:
            i += 1
    episodes.sort(key=lambda d: d["depth"])
    return episodes[:int(top_n)]


# ---------------------------------------------------------------- 12 变化归因
def change_attribution(w0, w1, cov0, cov1, step: float = 1e-5) -> Dict[str, object]:
    """
    风险贡献变化的中心差分分解（统一口径：步长 1e-5，残差单列）。

    对 RC% 沿三类输入的实际变化方向做中心差分：
      权重效应  → 权重从 w0 向 w1 走 ±step
      波动效应  → 波动率从 σ0 向 σ1 走 ±step
      相关效应  → 相关系数从 ρ0 向 ρ1 走 ±step
    实际变化扣除三项后得到 residual，四项相加等于总变化。
    """
    w0, w1 = _as_weights(w0), _as_weights(w1)
    s0, s1 = np.asarray(cov0, dtype=float), np.asarray(cov1, dtype=float)
    v0, v1 = np.sqrt(np.diag(s0)), np.sqrt(np.diag(s1))
    r0 = s0 / np.outer(v0, v0)
    r1 = s1 / np.outer(v1, v1)

    def rc_of(w, v, corr):
        cov = corr * np.outer(v, v)
        return np.array([risk_contribution(w, cov)["rc_pct"][a] for a in ASSETS])

    def central(perturb):
        plus = rc_of(*perturb(+step))
        minus = rc_of(*perturb(-step))
        return (plus - minus) / (2.0 * step)

    def p_weight(d):
        return (list(w0 + d * (w1 - w0)), v0, r0)

    def p_vol(d):
        return (w0, v0 + d * (v1 - v0), r0)

    def p_corr(d):
        return (w0, v0, r0 + d * (r1 - r0))

    eff = {"weight": central(p_weight), "vol": central(p_vol), "corr": central(p_corr)}
    a0, a1 = rc_of(w0, v0, r0), rc_of(w1, v1, r1)
    resid = a1 - a0 - sum(eff.values())

    def as_dict(vec):
        return {a: float(vec[i]) for i, a in enumerate(ASSETS)}

    return {
        "rc_start": as_dict(a0),
        "rc_end": as_dict(a1),
        "total": as_dict(a1 - a0),
        "weight_effect": as_dict(eff["weight"]),
        "vol_effect": as_dict(eff["vol"]),
        "corr_effect": as_dict(eff["corr"]),
        "residual": as_dict(resid),
        "step": float(step),
    }
