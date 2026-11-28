"""命令行示例：python -m backtest_engine.run_demo

读 data/unified_returns.xlsx（通过 2 号接口）和 data/weights.csv，
跑主实验，把账本写到 results/backtest_ledger_A.csv。
"""

import csv
import json
import os

from .ledger import CODES, InputError, run_backtest, summarize
from .from_interface import load_returns, month_end_plan

MATRIX = os.environ.get("BT_MATRIX", "data/unified_returns.xlsx")
WEIGHTS = os.environ.get("BT_WEIGHTS", "data/weights.csv")
OUT_DIR = os.environ.get("BT_OUT", "results")


def read_weights(path):
    plan = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or r.get("signal_date") or "").strip()
            if not d:
                continue
            plan[d] = {c: float(r[c]) for c in CODES}
    return plan


def main():
    dates, returns = load_returns(MATRIX)
    print(f"收益矩阵：{len(dates)} 个交易日，{dates[0]} → {dates[-1]}")

    weights = read_weights(WEIGHTS)
    plan, missing = month_end_plan(weights, MATRIX)
    print(f"权重表 {len(weights)} 行，落到成交日 {len(plan)} 个"
          + (f"，缺 {len(missing)} 个月末信号" if missing else ""))

    led = run_backtest(returns, plan, dates, mode="continuous")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = led.to_rows(group="A")
    path = os.path.join(OUT_DIR, "backtest_ledger_A.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})

    summary = summarize(led)
    with open(os.path.join(OUT_DIR, "backtest_summary_A.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"{'日期':<12}{'净值':>14}{'净收益':>10}{'费用':>8}{'换手':>14}")
    for r in led.rows[:3] + led.rows[-2:]:
        print(f"{r['date']:<12}{r['nav_end']:>14,.2f}{r['net_return']:>10.4%}"
              f"{r['cost']:>8,.2f}{r['turnover']:>14,.0f}")
    print("\n汇总")
    for k, v in summary.items():
        print(f"  {k:<16}{v if isinstance(v, int) else round(v, 6)}")
    print(f"\n账本 {path}")


if __name__ == "__main__":
    main()