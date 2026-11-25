"""2号数据处理模块：市场特征计算（口径说明 第五节）。

- 20/60 日累计收益：几何累计 prod(1+r)-1。
- 20/60/250 日年化波动：滚动样本标准差 * sqrt(250)。
- 60/250 日滚动 Pearson 相关矩阵（3x3，资产顺序 510300/511010/518880）。
- 全部计算严格截至 as_of_date，不使用未来收益。
- 窗口观测不足时对应字段置 None，并在 window_status 记录 required/available/reason，
  不得把不足窗口伪装成完整窗口（口径说明 5.4）。
仅依赖标准库 math/statistics，数值口径与1号 feature_builder 保持一致。
"""
from __future__ import annotations

import math
from statistics import stdev

from data_processing.data_loader import ASSET_CODES, slice_as_of

RETURN_WINDOWS = (20, 60)
VOL_WINDOWS = (20, 60, 250)
CORR_WINDOWS = (60, 250)
ANNUAL_FACTOR = math.sqrt(250)


def compound(returns) -> float:
    return math.prod(1.0 + x for x in returns) - 1.0


def ann_vol(returns) -> float:
    return stdev(returns) * ANNUAL_FACTOR


def pearson(xs, ys):
    """样本 Pearson 相关；任一侧方差为0时返回 None（不静默填0）。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    xc = [x - mx for x in xs]
    yc = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in xc) * sum(y * y for y in yc))
    if denom == 0.0:
        return None
    return sum(x * y for x, y in zip(xc, yc)) / denom


def corr_matrix(series_by_code, window):
    codes = list(ASSET_CODES)
    tails = {c: series_by_code[c][-window:] for c in codes}
    matrix = {}
    for i, a in enumerate(codes):
        row = {}
        for j, b in enumerate(codes):
            row[b] = 1.0 if i == j else pearson(tails[a], tails[b])
        matrix[a] = row
    return matrix


def compute_market_features(df, as_of_date: str) -> dict:
    """返回 market_features / asset_correlations / window_status / sample_days。

    market_features 内层键为 stock/bond/gold（interfaces/data_schema.py AssetValues）。
    asset_correlations 为 corr_60d / corr_250d 两个代码键 3x3 矩阵。
    """
    selected = slice_as_of(df, as_of_date)
    n = len(selected)
    series = {c: selected[c].tolist() for c in ASSET_CODES}
    names = {"510300": "stock", "511010": "bond", "518880": "gold"}

    status = {}
    features = {}
    for w in RETURN_WINDOWS:
        ok = n >= w
        status[f"return_{w}d"] = {
            "required": w, "available": n, "sufficient": ok,
            "reason": "ok" if ok else "insufficient_window",
        }
        features[f"asset_return_{w}d"] = {
            names[c]: (compound(series[c][-w:]) if ok else None) for c in ASSET_CODES
        }
    for w in VOL_WINDOWS:
        ok = n >= w
        status[f"volatility_{w}d"] = {
            "required": w, "available": n, "sufficient": ok,
            "reason": "ok" if ok else "insufficient_window",
        }
        features[f"asset_volatility_{w}d"] = {
            names[c]: (ann_vol(series[c][-w:]) if ok else None) for c in ASSET_CODES
        }
    correlations = {}
    for w in CORR_WINDOWS:
        ok = n >= w
        status[f"corr_{w}d"] = {
            "required": w, "available": n, "sufficient": ok,
            "reason": "ok" if ok else "insufficient_window",
        }
        if ok:
            matrix = corr_matrix(series, w)
            if any(v is None for row in matrix.values() for v in row.values()):
                status[f"corr_{w}d"]["reason"] = "zero_variance_asset"
            correlations[f"corr_{w}d"] = matrix
        else:
            correlations[f"corr_{w}d"] = None
    return {
        "market_features": features,
        "asset_correlations": correlations,
        "window_status": status,
        "sample_days": n,
    }