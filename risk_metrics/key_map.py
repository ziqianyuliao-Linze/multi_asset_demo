# -*- coding: utf-8 -*-
"""资产键名转换：模块内部用 stock/bond/gold，对外接口用 510300/511010/518880。

统一口径要求所有收益矩阵、权重数组、风险贡献数组按资产代码顺序排列；
本模块内部沿用英文名，进出接口时在此转换，避免两边各写一套。
"""

from __future__ import annotations

from typing import Any, Dict

NAMES = ("stock", "bond", "gold")
CODES = ("510300", "511010", "518880")
NAME_TO_CODE: Dict[str, str] = dict(zip(NAMES, CODES))
CODE_TO_NAME: Dict[str, str] = dict(zip(CODES, NAMES))


def to_codes(value: Any) -> Any:
    """把字典/列表里的 stock/bond/gold 递归改成资产代码。"""
    if isinstance(value, dict):
        return {NAME_TO_CODE.get(k, k): to_codes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_codes(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_codes(v) for v in value)
    return value


def to_names(value: Any) -> Any:
    """把字典/列表里的资产代码递归改回 stock/bond/gold。"""
    if isinstance(value, dict):
        return {CODE_TO_NAME.get(k, k): to_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_names(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_names(v) for v in value)
    return value


def weights_to_codes(weights) -> Dict[str, float]:
    """把仓位列表/字典统一成 {代码: 权重}。"""
    if isinstance(weights, dict):
        return {NAME_TO_CODE.get(k, k): float(v) for k, v in weights.items()}
    return {code: float(v) for code, v in zip(CODES, weights)}
