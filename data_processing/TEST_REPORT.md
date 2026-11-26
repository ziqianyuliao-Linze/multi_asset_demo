# 数据处理模块独立测试报告

日期：2026-11-26　负责人：2号 mdy　模块版本：v1.1（审查修订：公共接口方案A）
环境：Python 3.13.18（负责人本机）；v1.0 曾于 Python 3.12.11 云验证机复跑；pandas / openpyxl 见根目录 requirements.txt。
命令：仓库根目录 `python tests/test_data_processing.py`
结果：**Ran 23 tests — OK（23/23 通过，0 失败 0 错误）**

## 用例覆盖

| 组 | 用例 | 核验点 |
|---|---|---|
| TestLoaderAndQuality | 5 | 3236 行、2013-07-30~2026-11-20、列序 510300/511010/518880；SHA-256 前12位 c8ae2dad3039；质量七项全 True；source.file 为仓库相对路径；坏路径抛 FileNotFoundError |
| TestNoLookahead | 2 | 2024-06-28 全量计算与截断计算结果逐键相等；250 日窗口切片末行=as_of 且无未来日期 |
| TestKnownValues | 4 | 2026-11-20 全部收益/波动/60d与250d相关矩阵用独立手工公式复算，误差 <1e-12；与1号 feature_builder 同日期数值一致 <1e-12；JSON 可 allow_nan=False 序列化 |
| TestPublicCodeKeys | 2 | 方案A：market_features 与相关矩阵全部资产键=证券代码；整份 JSON 不含 "stock"/"bond"/"gold"；data_quality 无 asset_code_mapping |
| TestStrictDate | 2 | 拒绝带时间/时区后缀/空白/非零填充/非法日历日/整数/None/date/datetime/列表等 14 类输入；next_trading_day 同样严格；纯 YYYY-MM-DD 正常接受 |
| TestZeroVariance | 2 | synthetic_test（260 日、510300 恒零收益）：corr_60d/corr_250d 的 sufficient=False、reason=zero_variance_asset、进入 insufficient_windows；涉及零方差资产的相关元素为 null、其余可算；零方差资产波动=0 |
| TestInsufficientWindow | 2 | 2013-12-31（104 观测）：20/60 可用、250 波动与 250 相关为 null 且 reason=insufficient_window；2013-08-30（<60）：60 日收益为 null |
| TestInvalidAsOf | 2 | 非交易日/节假日/矩阵首日之前/非法格式均抛 ValueError |
| TestCalendar | 1 | 月末信号含 2014-11-28；next_trading_day(2014-11-28)=2014-12-01（与设计方案首信号/首成交一致）；末尾不完整月 2026-11 不计入；末日无下一交易日返回 None |
| TestSchemaShape | 1 | 顶层四字段与 market_features 字段名同 interfaces/data_schema.py 对齐；内层键按方案A 为证券代码（AssetValues 待1号同步） |

## 示例输出摘要（examples/market_data_input_2026-11-20.json）

- 资产键：510300 / 511010 / 518880（方案A）
- 20 日累计收益：510300 −0.038296、511010 +0.001799、518880 −0.111843
- 250 日年化波动：510300 0.171382、511010 0.007349、518880 0.296192
- 60 日股债相关 +0.011781；250 日股债相关 −0.009434（corr_60d / corr_250d 保留）
- data_quality：rows=3236、sha256_12=c8ae2dad3039、insufficient_windows=[]

窗口不足示例：`examples/market_data_input_2013-12-31_窗口不足示例.json`（104 观测，250 日字段 null，reason=insufficient_window）。

## 修订记录

- v1.0（2026-11-25）：首版 16 用例；后补相对路径用例至 17。
- v1.1（2026-11-26）：按审查意见修订——公开输出资产键统一证券代码（方案A）；as_of_date 严格 YYYY-MM-DD（显式 None 亦拒绝）；零方差相关窗口 sufficient=False 并入 insufficient_windows；保留 corr_60d/corr_250d 与窗口不足 null 语义；README、两份示例、测试同步更新；用例 17→23。

## 结论

模块可独立运行；读取冻结矩阵、质量检查、市场特征、滚动窗口提取接口均通过独立测试与手工复算对照；未修改冻结数据、公共接口文件及其他成员代码。遗留确认项见 README §5（schema 文件 AssetValues 待1号同步）。