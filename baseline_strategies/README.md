# baseline_strategies（3号模块）

> 多资产基准组合优化模块，服务于《面向多资产配置的大语言模型风险反馈与自适应决策研究》。
> 第二阶段整改版：资产池强校验、协方差奇异防护、风险平价后验校验、250 日窗口正式入口。

## 一、模块边界
1. 上游 `data_processing.data_interface.get_returns_window(as_of_date, n=250)` **只提供收益率窗口**；
2. **250 日窗口内的样本协方差矩阵在本模块内计算**（`compute_sample_covariance`，ddof=1，日频不年化）；
3. 本模块负责：资产池强校验 → 协方差校验与收缩防护 → 三策略求解 → 风险贡献后验校验。

## 二、固定资产池与窗口

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `ASSET_POOL` | `["510300", "511010", "518880"]` | 顺序、数量不可变 |
| `WINDOW_DAYS` | `250` | 交易日窗口 |
| `RC_TOLERANCE` | `0.005` | 风险贡献占比相对 1/3 的最大绝对偏差 |

## 三、公开函数

| 函数 | 作用 |
| --- | --- |
| `run_baseline_250d(as_of_date, strategy, tickers=None, path=None)` | 正式入口：取 250 日收益率 → 本模块算协方差 → 策略求解 |
| `fetch_returns_window_250d(as_of_date, path=None)` | 调用上游窗口接口，并做窗口长度/列序/末日一致性契约校验 |
| `compute_sample_covariance(returns)` | 由 250×3 收益率算样本协方差 |
| `validate_asset_pool(tickers)` | 资产池强校验（缺失/多出/重复/顺序错一律 ValueError） |
| `validate_covariance(cov)` | 结构校验：严格 3×3、有限值、对称、方差为正 |
| `stabilize_covariance(cov)` | 奇异/非正定 → 向 `mu·I` 收缩修复；零矩阵直接拒绝 |
| `check_risk_budget(rc_ratio, tolerance=0.005)` | 风险贡献后验校验，超容差抛 `RiskBudgetError` |
| `equal_weight / min_variance / risk_parity(cov, tickers=None)` | 三种基准策略 |
| `run_baseline(cov, tickers=None, strategy="risk_parity")` | 按策略名分发（协方差由调用方传入） |
| `run_baseline_with_rolling_cov(cov, tickers=None, strategy=...)` | 旧函数名兼容入口，等价于 `run_baseline` |

## 四、统一返回格式
```python
{
    "strategy": "equal_weight" | "min_variance" | "risk_parity",
    "weight": {代码: 权重},                    # Σw = 1，w >= 0
    "risk_contribution": {代码: rc_i},          # 绝对风险贡献 w_i·(Σw)_i，合计 = 组合方差
    "risk_contribution_ratio": {代码: 占比},    # 风险贡献占比，合计 = 1
    "risk_contribution_max_deviation": float,   # max|占比 - 1/3|
    "risk_budget_tolerance": float | None,      # 仅风险平价给 0.005
    "risk_budget_pass": bool | None,            # 仅风险平价给校验结果
    "portfolio_variance": float,
    "covariance_shrinkage": {...},              # 协方差收缩诊断
}
```
`run_baseline_250d` 额外返回 `as_of_date` / `window_days` / `sample_days` / `asset_pool`。

## 五、异常约定
- `ValueError`：资产池非法、协方差非 3×3/不对称/NaN/方差非正、零矩阵、策略名非法、日期非法、窗口不足；
- `RiskBudgetError`（`RuntimeError` 子类）：风险平价求解返回 success 但风险贡献偏差 > 0.005，**不返回权重**；
- `RuntimeError`：优化器无可行解。

## 六、单元测试
```bash
python -m pytest tests/test_baseline_strategies.py -v
```
50 个用例全部通过（资产池、协方差、奇异防护、三策略、后验校验、250 日窗口、截断无泄漏）。

## 七、命令行自测
```bash
python -m baseline_strategies.portfolio_optimizer --as-of 2026-11-20 --strategy risk_parity
```

## 八、尚待解决问题
1. 样本协方差估计噪声较大，高维情形下易病态（当前仅做收缩兜底，未做 Ledoit-Wolf 最优强度估计）。
2. 窗口固定 250 日，尚未随市场风险状态自适应调整。
3. 优化目标未纳入交易成本与换手率约束。
4. 风险平价仅均衡波动率贡献，尚未扩展到因子风险/尾部风险维度。
