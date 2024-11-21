import numpy as np
import matplotlib.pyplot as plt

# 从均值为 0，标准差为 1 的正态分布中生成 10000 个随机数
samples = np.random.normal(loc=0, scale=1, size=10000)

# 绘制随机采样的直方图，bins=50 表示分为 50 个区间
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# 添加理论正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - 0) / 1) ** 2) / (np.sqrt(2 * np.pi) * 1)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Random Sampling from Normal Distribution")
plt.show()