# 数据处理模块独立测试报告

日期：2026-11-25　负责人：2号 mdy　模块版本：v1.0
环境：Python 3.12.11（云验证机）与 Python 3.13.18（负责人本机）双环境复跑；pandas / openpyxl 见根目录 requirements.txt。
命令：仓库根目录 `python tests/test_data_processing.py`
结果：**Ran 16 tests — OK（16/16 通过，0 失败 0 错误）**

## 用例覆盖

| 组 | 用例 | 核验点 |
|---|---|---|
| TestLoaderAndQuality | 4 | 3236 行、2013-07-30~2026-11-20、列序 510300/511010/518880；SHA-256 前12位 c8ae2dad3039；质量七项全 True；坏路径抛 FileNotFoundError |
| TestNoLookahead | 2 | 2024-06-28 全量计算与截断计算结果逐键相等；250 日窗口切片末行=as_of 且无未来日期 |
| TestKnownValues | 4 | 2026-11-20 全部收益/波动/60d与250d相关矩阵用独立手工公式复算，误差 <1e-12；与1号 feature_builder 同日期数值一致 <1e-12；JSON 可 allow_nan=False 序列化 |
| TestInsufficientWindow | 2 | 2013-12-31（104 观测）：20/60 可用、250 波动与 250 相关为 null 且 reason=insufficient_window；2013-08-30（<60）：60 日收益为 null |
| TestInvalidAsOf | 2 | 非交易日/节假日/矩阵首日之前/非法格式均抛 ValueError |
| TestCalendar | 1 | 月末信号含 2014-11-28；next_trading_day(2014-11-28)=2014-12-01（与设计方案首信号/首成交一致）；末尾不完整月 2026-11 不计入；末日无下一交易日返回 None |
| TestSchemaShape | 1 | 四字段与 interfaces/data_schema.MarketDataInput 逐键对齐；market_features 内层键与 AssetValues 对齐 |

## 示例输出摘要（examples/market_data_input_2026-11-20.json）

- 20 日累计收益：stock −0.038296、bond +0.001799、gold −0.111843
- 250 日年化波动：stock 0.171382、bond 0.007349、gold 0.296192
- 60 日股债相关 +0.011781；250 日股债相关 −0.009434
- data_quality：rows=3236、sha256_12=c8ae2dad3039、insufficient_windows=[]

窗口不足示例：`examples/market_data_input_2013-12-31_窗口不足示例.json`（104 观测，250 日字段 null）。

## 结论

模块可独立运行；读取冻结矩阵、质量检查、市场特征、滚动窗口提取接口均通过独立测试与手工复算对照；未修改冻结数据、公共接口及其他成员代码。遗留确认项见 README §5。