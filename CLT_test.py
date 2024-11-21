import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

# Target Distribution: double gamma distribution
dg = stats.dgamma(a=1)  # 定义双伽马分布
qx = stats.norm(loc=0, scale=2)

# Generate samples for PDF
x = np.linspace(min(dg.ppf(0.001), norm.ppf(0.001)), max(dg.ppf(0.999), norm.ppf(0.999)), 1000)
dg_samples = dg.pdf(x)
norm_samples = qx.pdf(x)

# Find scaling constant K for envelope
K = max(dg_samples / norm_samples)

def rejection_sampling():
    while True:
        # Re-use global parameters from above
        x = np.random.normal(0, 2)  # 正态分布中生成一个随机数
        envelope = K * qx.pdf(x)
        p = np.random.uniform(0, envelope)
        if p < dg.pdf(x):
            return x

means = []
for i in tqdm(range(1000)):
    samples = [rejection_sampling() for x in range(100)]
    mean = sum(samples)/1000
    means.append(mean)  # 修正这一行，将 mean 追加到 means

plt.hist(means, bins=50, density=True, alpha=0.6, color='g')

# 添加x轴的刻度
plt.xticks(np.linspace(min(means), max(means), 50))  # 设置x轴刻度为从最小值到最大值的10个刻度
plt.xlabel('Mean Value')  # 设置x轴标签
plt.ylabel('Density')     # 设置y轴标签
plt.title('Histogram of Sample Means')  # 添加图表标题

plt.show()