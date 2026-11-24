
# Multi-Asset Allocation Research

## 1. Project Overview

面向多资产配置的大语言模型风险反馈与自适应决策研究。

本项目研究大语言模型在多资产配置中的决策能力，以及收益反馈、整体风险反馈和风险归因反馈对动态配置策略的影响。

## 2. Development Environment

- Python 3.11
- NumPy
- Pandas
- SciPy
- OpenPyXL
- Requests
- Python-dotenv

## 3. Project Structure

- data: 统一研究数据
- interfaces: 公共数据接口
- config: 实验参数
- data_processing: 数据处理模块（2号）
- baseline_strategies: 传统量化策略（3号）
- risk_metrics: 风险计算与归因（4号）
- llm_decision: 大模型决策（1号）
- backtest_engine: 回测引擎（5号）
- tests: 模块测试
- results: 实验结果

## 4. Unified Research Settings

- Assets: 510300, 511010, 518880
- Initial Capital: 1,000,000 CNY
- Buy Transaction Cost: 0.03%
- Sell Transaction Cost: 0.03%
- Risk Window: 250 trading days
- Rebalancing: Monthly
- Python Version: 3.11

All experimental settings follow the approved First-Stage Experimental Design.

## 5. Development Rules

1. Each member develops within their assigned module.
2. All modules must follow interfaces/data_schema.py.
3. Do not independently modify the frozen dataset or experiment parameters.
4. Each module must include independent tests.
5. API keys and environment secrets must not be committed.

## 6. Project Workflow

Stage 1: Research Design and Feasibility Validation

Stage 2: Basic Module Development

Stage 3: Risk Feedback Loop Integration

Stage 4: Complete Backtesting and Comparison

Stage 5: Results Analysis and Research Improvement

Stage 6: Paper Writing and Finalization
