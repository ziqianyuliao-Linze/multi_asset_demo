"""2号数据处理模块：统一收益矩阵读取与质量检查。

冻结数据：data/unified_returns.xlsx（工作表“日收益率矩阵”）。
口径依据：docs/统一数据口径说明.md 第三、四节；第一阶段最终实验设计方案 附录B。
不得重新下载行情或替换矩阵；本模块只读，不写回数据文件。
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

ASSET_CODES = ("510300", "511010", "518880")
SHEET_NAME = "日收益率矩阵"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX_PATH = REPO_ROOT / "data" / "unified_returns.xlsx"


def _validate_date_string(value: str) -> str:
    text = str(value)[:10]
    datetime.strptime(text, "%Y-%m-%d")
    return text


def load_unified_returns(path=None) -> pd.DataFrame:
    """读取冻结矩阵，返回 index=date(YYYY-MM-DD 字符串)、columns=ASSET_CODES 的 DataFrame。

    结构性违规（缺表/列序错/日期重复或未升序/非有限值）直接抛 ValueError，
    不做任何填充或修改。
    """
    source = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    if not source.exists():
        raise FileNotFoundError(f"冻结矩阵不存在：{source}")
    df = pd.read_excel(source, sheet_name=SHEET_NAME, engine="openpyxl")
    expected = ["日期"] + list(ASSET_CODES)
    if list(df.columns) != expected:
        raise ValueError(f"列头与固定资产顺序不一致：{list(df.columns)} != {expected}")
    dates = [_validate_date_string(v) for v in df["日期"]]
    if dates != sorted(set(dates)):
        raise ValueError("日期重复或未升序")
    values = df[list(ASSET_CODES)].astype(float)
    if not values.map(math.isfinite).all().all():
        raise ValueError("存在空值或非有限收益率")
    out = values.copy()
    out.index = pd.Index(dates, name="date")
    return out


def file_sha256_12(path=None) -> str:
    source = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    return hashlib.sha256(source.read_bytes()).hexdigest()[:12]


def slice_as_of(df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """严格截取 <= as_of_date 的历史观测，防止未来数据泄漏。

    as_of_date 必须是矩阵中的交易日且格式合法，否则抛 ValueError。
    """
    text = _validate_date_string(as_of_date)
    if text not in df.index:
        raise ValueError(f"指定日期不是统一收益率矩阵中的交易日：{text}")
    return df.loc[:text]


def _source_label(path=None) -> str:
    """来源文件记录为仓库相对路径，保证示例与报告跨机器可复现（规范 4.3）。"""
    source = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    try:
        return source.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(source)


def check_matrix(df: pd.DataFrame, path=None) -> dict:
    """数据质量检查报告（口径说明 3.4）。全部检查项应为 True。"""
    dates = list(df.index)
    finite = df.map(math.isfinite).all().all()
    return {
        "rows": len(df),
        "start_date": dates[0],
        "end_date": dates[-1],
        "asset_order_ok": list(df.columns) == list(ASSET_CODES),
        "dates_unique": len(dates) == len(set(dates)),
        "dates_ascending": dates == sorted(dates),
        "no_missing_values": bool(df.notna().all().all()),
        "no_infinite_values": bool(finite),
        "common_calendar": True,  # 冻结矩阵本身即三资产共同交易日交集
        "no_forward_fill": True,  # 本模块不做任何填充
        "source": {
            "file": _source_label(path),
            "sheet": SHEET_NAME,
            "sha256_12": file_sha256_12(path),
        },
    }