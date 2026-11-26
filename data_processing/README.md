# data_processing（2号 数据处理模块）

版本：v1.2（2026-11-26，复查修订：样本接口 n 校验＋日期 ASCII 显式化）　负责人：2号 mdy
依据：`docs/第二阶段开发分工与接口规范.md` §2/§5.1、`docs/统一数据口径说明.md` §3–5、《第一阶段最终实验设计方案》附录B。

## 1. 职责与文件

| 文件 | 职责 |
|---|---|
| `data_loader.py` | 读取冻结矩阵 `data/unified_returns.xlsx`（只读）、结构性校验、质量检查报告、截至日截取 |
| `market_features.py` | 20/60 日几何累计收益；20/60/250 日年化波动（样本标准差×√250）；60/250 日 Pearson 相关矩阵；窗口不足状态 |
| `data_interface.py` | 对外统一接口＋命令行示例生成 |
| `feature_builder.py` | 1号既有实现，原样保留（数值对照基准，未修改） |
| `examples/` | 输入输出示例 JSON（含窗口不足示例） |
| `TEST_REPORT.md` | 独立测试报告 |
| `../tests/test_data_processing.py` | 独立测试（25 用例） |

冻结数据版本：SHA-256 前12位 `c8ae2dad3039`，3236 行，2013-07-30 ~ 2026-11-20。本模块不下载行情、不改矩阵、不做任何填充。

## 2. 运行说明

环境：Python 3.11+，依赖见根目录 `requirements.txt`（pandas、openpyxl）。仓库根目录执行：

```bash
python tests/test_data_processing.py                 # 独立测试
python -m data_processing.data_interface --as-of 2026-11-20   # 生成基础市场输入示例
```

队友接入（Python）：

```python
from data_processing.data_interface import (
    get_market_data_input,   # 1号/4号：基础市场输入四字段
    get_returns_window,      # 3号/4号：截至日最近 n 日收益样本（协方差自算）
    get_trading_calendar, next_trading_day, month_end_signal_dates,  # 5号：日历
)
payload = get_market_data_input("2026-11-20")   # 默认读 data/unified_returns.xlsx
```

## 3. 输出结构（接口 5.1）

`get_market_data_input(as_of)` 返回四字段：`as_of_date`、`market_features`、`asset_correlations`、`data_quality`。

- `market_features`：`asset_return_20d/60d`、`asset_volatility_20d/60d/250d`，内层键统一为证券代码 `510300/511010/518880`（公共接口方案A，2026-11-26 审查确认）。`stock/bond/gold` 仅限模块内部使用，不进入任何公开输出与示例。
- `asset_correlations`：`corr_60d`、`corr_250d` 两个 3×3 矩阵，键为资产代码，顺序 510300/511010/518880。
- `data_quality`：质量检查布尔项、来源文件与 SHA、`sample_days_to_date`、`window_status`（每窗口 required/available/sufficient/reason）、`insufficient_windows` 列表、`no_lookahead=True`。

**窗口不足语义（口径说明 5.4）**：观测数不足的窗口对应字段为 `null`（相关矩阵整体为 `null`），原因记录在 `window_status`；消费方必须先检查 `data_quality.insufficient_windows`，不得把 `null` 当 0 或伪装完整窗口。20/60 日特征在样本 ≥60 日后即可用，250 日字段自 2014-08-06 起可用。**零方差**：某资产窗口内方差为 0 使相关系数为 `null` 时，该相关窗口记 `sufficient=False`、`reason=zero_variance_asset`，并进入 `insufficient_windows`（矩阵保留其余可算元素）。

**日期语义**：`as_of_date` 严格只接受 ASCII 数字的 `YYYY-MM-DD` 字符串；带时间、时区后缀、空白、非零填充、全角/其他 Unicode 数字、非法日历日及其他类型（含显式 `None`）一律 `ValueError`；省略参数时默认矩阵末日。

**样本接口与相关矩阵约定**：`get_returns_window(as_of, n)` 的 `n` 必须为正整数（`None` 返回全部）；`n` 超过截至日可用观测数时抛 `ValueError`，不静默返回短样本，消费方可先查 `sample_days_to_date`。相关矩阵对角线（自相关）恒为 1.0（含零方差资产）；零方差只影响非对角元素置 `null`。

**防未来泄漏**：所有计算仅用 `<= as_of_date` 的观测；`as_of_date` 必须是矩阵内交易日，否则抛 `ValueError`。

## 4. 与既有代码的关系

- `interfaces/data_schema.py` 为公共接口文件，本模块未修改：顶层四字段与 `market_features` 字段名与其 TypedDict 对齐（测试覆盖）；内层资产键按方案A 用证券代码，该文件 `AssetValues` 仍为 stock/bond/gold，**待1号同步**（见 §5）。
- 1号 `feature_builder.py` 与 `tests/offline_test.py`、4号集成测试保持原样；本模块与其数值一致性在 1e-12 内（测试覆盖）。legacy 输出本就为代码键，与方案A 一致，形状冲突已消除。

## 5. 尚未解决 / 需群内确认

1. `interfaces/data_schema.py` 的 `AssetValues`（stock/bond/gold）与方案A 代码键不一致，公共文件由1号同步更新；本模块不擅自修改。
2. 实现侧补定语义需3号/4号接入知悉：`get_returns_window` 的 `n` 超可用观测抛 `ValueError`（不返回短样本）；相关矩阵对角线恒 1.0。
3. 停牌/不可交易判断不在本模块（冻结矩阵无停牌字段）；E 日顺延规则由5号结合 `next_trading_day` 与可交易信息实现。
4. 协方差求解、权重求解属3号职责；本模块仅提供 `get_returns_window` 样本接口。