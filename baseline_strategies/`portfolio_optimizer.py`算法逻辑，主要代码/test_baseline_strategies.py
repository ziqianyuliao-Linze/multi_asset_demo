import numpy as np
from portfolio_optimizer import equal_weight, min_variance, risk_parity

if __name__ == "__main__":
    # 手写假协方差矩阵，和任务示例一致
    cov_sim = np.array([
        [0.05, 0.01, 0.02],
        [0.01, 0.001, 0.0005],
        [0.02, 0.0005, 0.03]
    ])
    print("输入模拟协方差矩阵：")
    print(cov_sim)
    print("-"*50)

    w_ew = equal_weight(cov_sim)
    print(f"等权策略输出权重： {np.round(w_ew,4)}")

    w_mv = min_variance(cov_sim)
    print(f"最小方差输出权重： {np.round(w_mv,4)}")

    w_rp, rc_rp = risk_parity(cov_sim)
    print(f"风险平价输出权重： {np.round(w_rp,4)}")
    print(f"风险贡献：{np.round(rc_rp,4)}")
