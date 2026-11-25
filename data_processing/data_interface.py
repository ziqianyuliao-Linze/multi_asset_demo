"""2号数据处理模块对外统一接口（第二阶段规范 5.1，供1号/4号/3号/5号接入）。

公开函数：
- load_unified_returns(path=None)            冻结矩阵 DataFrame
- get_returns_window(as_of, n=None, path=None)  截至 as_of 的最近 n 个共同交易日收益
- get_market_data_input(as_of=None, path=None)  基础市场输入四字段（schema 形状）
- get_trading_calendar(path=None)            共同交易日历
- next_trading_day(date, path=None)          下一共同交易日（无则 None）
- month_end_signal_dates(path=None)          各完整月份最后一个共同交易日（月末信号日）

命令行示例（仓库根目录执行）：
    python -m data_processing.data_interface --as-of 2026-11-20
输出 JSON 保存到 data_processing/examples/market_data_input_<as_of>.json。

窗口不足语义：market_features / asset_correlations 中不足窗口字段为 null，
data_quality.window_status 与 insufficient_windows 记录原因；
消费方（1号/4号）必须先检查 data_quality 再使用数值，不得把 null 当 0。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.data_loader import (ASSET_CODES, DEFAULT_MATRIX_PATH,
                                         check_matrix, load_unified_returns,
                                         slice_as_of)
from data_processing.market_features import compute_market_features

CODE_TO_NAME = {"510300": "stock", "511010": "bond", "518880": "gold"}


def get_returns_window(as_of_date: str, n=None, path=None):
    df = load_unified_returns(path)
    selected = slice_as_of(df, as_of_date)
    return selected if n is None else selected.tail(n)


def get_trading_calendar(path=None) -> list:
    return list(load_unified_returns(path).index)


def next_trading_day(date_string: str, path=None):
    dates = get_trading_calendar(path)
    later = [d for d in dates if d > date_string]
    return later[0] if later else None


def month_end_signal_dates(path=None) -> list:
    """各完整月份最后一个共同交易日（月末信号日，设计方案 第七节）。

    完整月份=数据中存在下一个月的观测；数据末尾的不完整月份不计入。
    """
    dates = get_trading_calendar(path)
    groups = {}
    for d in dates:
        groups.setdefault(d[:7], []).append(d)
    signals = []
    for month, days in groups.items():
        y, m = int(month[:4]), int(month[5:7])
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        if f"{ny}-{nm:02d}" in groups:
            signals.append(days[-1])
    return signals


def get_market_data_input(as_of_date=None, path=None) -> dict:
    """基础市场输入：as_of_date / market_features / asset_correlations / data_quality。"""
    source = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    df = load_unified_returns(source)
    as_of = as_of_date if as_of_date is not None else df.index[-1]
    computed = compute_market_features(df, as_of)
    quality = check_matrix(df, source)
    quality.update({
        "as_of_date": as_of,
        "sample_days_to_date": computed["sample_days"],
        "no_lookahead": True,
        "asset_code_mapping": CODE_TO_NAME,
        "window_status": computed["window_status"],
        "insufficient_windows": [k for k, v in computed["window_status"].items()
                                 if not v["sufficient"]],
    })
    return {
        "as_of_date": as_of,
        "market_features": computed["market_features"],
        "asset_correlations": computed["asset_correlations"],
        "data_quality": quality,
    }


def main():
    parser = argparse.ArgumentParser(description="2号数据处理模块：基础市场输入示例")
    parser.add_argument("--as-of", default=None, help="信息截止日 YYYY-MM-DD，默认矩阵末日")
    parser.add_argument("--path", default=None, help="冻结矩阵路径，默认 data/unified_returns.xlsx")
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    args = parser.parse_args()
    payload = get_market_data_input(args.as_of, args.path)
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "examples" /
        f"market_data_input_{payload['as_of_date']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"as_of={payload['as_of_date']} 样本天数={payload['data_quality']['sample_days_to_date']}")
    print(f"不足窗口={payload['data_quality']['insufficient_windows']}")
    print(f"输出 {out}")


if __name__ == "__main__":
    main()