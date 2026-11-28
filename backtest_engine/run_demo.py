# run_demo.py
"""命令行示例：python -m backtest_engine.run_demo

读 data/unified_returns.xlsx（通过 2 号接口）和 examples/mock_weights.csv（模拟决策），
按《统一数据口径说明》9.2 的暖启动口径跑主实验，输出：

  results/backtest_ledger_A.jsonl 账本（每行一条 interfaces.BacktestRecord）
  results/holdings_A.csv          持仓（列名 = risk_metrics.risk_feedback.HOLDING_COLUMNS）
  results/backtest_summary_A.json 绩效（2014-12-01 成交前净值归一）

examples/mock_weights.csv 只有两期模拟权重，不能当成正式主实验输入；
正式结果要把 1 号或 3 号的全样本决策按同一列名放进去再跑。
"""

import csv
import json
import os

from .from_interface import load_decisions, load_returns, run_formal
from .ledger import FORMAL_START, HOLDING_COLUMNS, summarize_formal

MATRIX = os.environ.get("BT_MATRIX", "data/unified_returns.xlsx")
DECISIONS = os.environ.get("BT_WEIGHTS", "examples/mock_weights.csv")
OUT_DIR = os.environ.get("BT_OUT", "results")


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 10) if isinstance(v, float) else v)
                             for k, v in row.items()})


def main():
    dates, returns = load_returns(MATRIX)
    print(f"收益矩阵：{len(dates)} 个交易日，{dates[0]} → {dates[-1]}")

    decisions = load_decisions(DECISIONS)
    print(f"决策表：{DECISIONS}（模拟数据 synthetic_test，不得作为正式主实验输入）")
    print(f"       {len(decisions)} 期决策；正式绩效起点 {FORMAL_START}")

    led = run_formal(returns, dates, decisions)
    missing = led.meta["missing_signals"]
    print(f"信号期 {len(led.meta['signals'])} 个，缺决策 {len(missing)} 期"
          + (f"（首个缺失 {missing[0]}）" if missing else ""))
    status_count = {}
    for period in led.periods:
        status_count[period["status"]] = status_count.get(period["status"], 0) + 1
    print(f"回退状态：{status_count}")

    os.makedirs(OUT_DIR, exist_ok=True)
    ledger_path = os.path.join(OUT_DIR, "backtest_ledger_A.jsonl")
    holdings_path = os.path.join(OUT_DIR, "holdings_A.csv")
    summary_path = os.path.join(OUT_DIR, "backtest_summary_A.json")
    with open(ledger_path, "w", encoding="utf-8") as f:
        for record in led.to_rows(group="A"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_csv(holdings_path, led.to_holdings_rows(group="A"))

    summary = summarize_formal(led)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'日期':<12}{'期初净值':>14}{'期末净值':>14}{'净收益':>12}{'费用':>10}{'换手':>10}")
    for r in led.rows[:2] + led.rows[324:326] + led.rows[-2:]:
        print(f"{r['date']:<12}{r['beginning_nav']:>14,.2f}{r['ending_nav']:>14,.2f}"
              f"{r['net_return']:>12.4%}{r['cost_amount']:>10,.2f}{r['turnover']:>10.4f}")

    print("\n绩效（2014-12-01 成交前净值归一，已扣 1.5% 无风险利率）")
    for k, v in summary.items():
        shown = v if isinstance(v, (int, str)) else round(v, 6)
        print(f"  {k:<18}{shown}")
    print(f"\n账本 {ledger_path}\n持仓 {holdings_path}\n绩效 {summary_path}")


if __name__ == "__main__":
    main()
