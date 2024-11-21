import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def chi(k):
    # 设置卡方分布的自由度

    num_samples = 10000  # 模拟10000个样本

    # 生成独立的标准正态分布随机变量
    Z = np.random.randn(num_samples, k)

    # 计算标准正态分布随机变量的平方和
    X = np.sum(Z**2, axis=1)
    print(Z)
    print(X)
    # 绘制卡方分布的直方图
    plt.hist(X, bins=100, density=True, alpha=0.6, color='g', label='Simulated Chi-squared')

    # 绘制卡方分布的理论概率密度函数
    x = np.linspace(0, np.max(X), 1000)
    plt.plot(x, chi2.pdf(x, k), 'r-', label=f'Chi-squared PDF (df={k})')

    # 添加标签和标题
    plt.title(f'Chi-squared Distribution (df={k})')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # 显示图形
    plt.show()

chi(2)