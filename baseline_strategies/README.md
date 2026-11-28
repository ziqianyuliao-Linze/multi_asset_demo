# baseline_strategies（3号模块）

> 多资产基准组合优化模块，服务于《面向多资产配置的大语言模型风险反馈与自适应决策研究》。
> 第二阶段整改版：资产池强校验、协方差奇异防护、风险平价后验校验、250 日窗口正式入口。

## 一、模块边界
1. 上游 `data_processing.data_interface.get_returns_window(as_of_date, n=250)` **只提供收益率窗口**；
2. **250 日窗口内的样本协方差矩阵在本模块内计算**（`compute_sample_covariance`，ddof=1，日频不年化）；
3. 本模块负责：资产池强校验 → 协方差校验与收缩防护 → 三策略求解 → 风险贡献后验校验。

三个核心概念（中英对照）：
- **风险平价（risk parity）**：让每项资产对组合风险的**贡献占比（risk contribution ratio）**都等于 1/3，而不是让权重相等；
- **风险贡献（risk contribution）**：`rc_i = w_i·(Σw)_i`，组合方差 = 各项 rc_i 之和；
- **协方差收缩（covariance shrinkage）**：把奇异/病态矩阵向 `mu·I` 拉近，让它变成可解的正定矩阵。

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
| `compute_risk_contribution(weight, cov)` | 后验风险贡献与占比复算 |
| `equal_weight / min_variance / risk_parity(cov, tickers=None)` | 三种基准策略 |
| `run_baseline(cov, tickers=None, strategy="risk_parity")` | 按策略名分发（协方差由调用方传入） |
| `run_baseline_with_rolling_cov(cov, tickers=None, strategy=...)` | 旧函数名兼容入口，等价于 `run_baseline` |

## 四、快速开始

```bash
# 1) 单日 250 日窗口组合（命令行，输出 JSON）
python -m baseline_strategies.portfolio_optimizer --as-of 2026-11-20 --strategy risk_parity

# 2) 直接运行本文件（文件底部自测入口，默认同日同策略）
python baseline_strategies/portfolio_optimizer.py

# 3) 单元测试（50 个用例）
python -m pytest tests/test_baseline_strategies.py -v
```

Python 调用等价写法：

```python
from baseline_strategies.portfolio_optimizer import run_baseline_250d

result = run_baseline_250d("2026-11-20", strategy="risk_parity")
print(result["weight"])                        # {'510300': ..., '511010': ..., '518880': ...}
print(result["risk_contribution_ratio"])       # 三项均约 0.3333
```

## 五、统一返回格式
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
    "covariance_shrinkage": {                   # 协方差诊断
        "shrunk": bool, "shrink_delta": float,
        "min_eigenvalue_before": float, "min_eigenvalue_after": float,
        "eigenvalue_floor": float, "shrink_target": str,
    },
}
```
`run_baseline_250d` 额外返回 `as_of_date` / `window_days` / `sample_days` / `asset_pool`。

## 六、异常约定
- `ValueError`：资产池非法、协方差非 3×3/不对称/NaN/方差非正、零矩阵、策略名非法、日期非法、窗口不足；
- `RiskBudgetError`（`RuntimeError` 子类）：风险平价求解返回 success 但风险贡献偏差 > 0.005，**不返回权重**；
- `RuntimeError`：优化器无可行解。

## 七、与队长 6 条整改的对应关系

| 整改要求 | 实现位置 | 单元测试 |
| --- | --- | --- |
| 1 风险平价后验校验，偏差容差 0.005，超限抛异常 | `_risk_budget_barrier`（凸对数障碍）+ `_refine_risk_budget`（精修）+ `check_risk_budget` + `RiskBudgetError` | `TestRiskBudgetGate` |
| 2 资产池强校验、协方差严格 3×3 | `validate_asset_pool`、`validate_covariance` | `TestAssetPool`、`TestCovarianceGuard::test_shape_must_be_3x3` |
| 3 零/奇异/非正定矩阵防护，防除零 | `stabilize_covariance`（收缩修复）、`compute_risk_contribution`（方差<=0 即拒绝） | `TestCovarianceGuard` |
| 4 250 日窗口正式入口，协方差在本模块算 | `run_baseline_250d`、`fetch_returns_window_250d`、`compute_sample_covariance` | `TestWindow250D` |
| 5 新增单元测试（难解矩阵、后验误差、资产池异常、零/奇异矩阵、窗口不足、未来日期、截断） | — | 50 个用例全覆盖 |
| 6 清理冗余：删未用导入、删不可达分支、修正不存在的 `get_rolling_cov` 注释 | `portfolio_optimizer.py` 顶部与分发函数 | 全量测试回归通过 |

## 八、实测结果（as_of_date = 2026-11-20，250 日窗口）

| 策略 | 510300 | 511010 | 518880 | 风险贡献占比 | 组合方差 |
| --- | --- | --- | --- | --- | --- |
| 等权 | 33.33% | 33.33% | 33.33% | 32.20% / 0.16% / 67.65% | 7.35e-05 |
| 最小方差 | 0.22% | 99.78% | 0.00% | 0.22% / 99.78% / 0.00% | 2.15e-07 |
| 风险平价 | 3.51% | 94.58% | 1.91% | 33.33% / 33.33% / 33.33% | 6.22e-07 |

> 债券 511010 权重极高属正常现象：其日波动率约为股票的十分之一，要让三类资产风险贡献各占 1/3，必须把主要资金配置在低波动资产上。

## 九、目录结构
```
baseline_strategies/
├── __init__.py
├── portfolio_optimizer.py    # 策略核心代码
├── README.md                 # 本说明文档
└── 运行结果及问题.doc         # 测试结果与待解决问题记录
```

## 十、尚待解决问题
1. 样本协方差估计噪声较大，高维情形下易病态（当前仅做收缩兜底，未做 Ledoit-Wolf 最优收缩强度估计）。
2. 窗口固定 250 日，尚未随市场风险状态自适应调整。
3. 优化目标未纳入交易成本与换手率约束。
4. 风险平价仅均衡波动率贡献，尚未扩展到因子风险/尾部风险维度。
