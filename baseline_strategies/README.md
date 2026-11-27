```
# baseline_strategies 基线策略模块
实现三种基准资产配置策略：等权、最小方差、风险平价。

## 接口
输入：
1. cov：np.ndarray，正定对称协方差矩阵
2. tickers：标的代码列表 ["510300","511010","518880"]

输出：
1. equal_weight(cov,tickers) → dict[str,float] 权重字典
2. min_variance(cov,tickers) → dict[str,float] 权重字典
3. risk_parity(cov,tickers) → (weight_dict, rc_dict)，权重字典 + 风险贡献字典

### 重要约定（对接项目规范）
1. 本模块**不读取原始收益率、不做滚动窗口切片**；
2. 250日求解窗口、数据切片、防止未来数据由上游`data_processing`模块完成，计算得到协方差矩阵再传入本模块；
3. 权重约束：$0 \le w_i \le 1，\sum w_i=1$；不允许做空；
4. 协方差输入必须为对称半正定矩阵，模块内置合法性校验，非法输入抛出ValueError。

## 本地测试运行
项目根目录执行：
\$env:PYTHONPATH="."; python tests/test_baseline_strategies.py

## 单元pytest
\$env:PYTHONPATH="."; python -m pytest tests/test_baseline_strategies.py -v
```
