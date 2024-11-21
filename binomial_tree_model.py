import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)


def binomial_tree(mu, sigma, S0, N, T, step):
    # 计算状态价格和概率
    u = np.exp(sigma * np.sqrt(step))  # 上升因子
    d = 1.0 / u  # 下降因子
    p = 0.5 + 0.5 * (mu / sigma) * np.sqrt(step)  # 上升概率

    # 二项树模拟
    up_times = np.zeros((N, len(T)))
    down_times = np.zeros((N, len(T)))
    for idx in range(len(T)):
        up_times[:, idx] = np.random.binomial(T[idx] / step, p, N)  # 记录上升次数
        down_times[:, idx] = T[idx] / step - up_times[:, idx]  # 计算下降次数

    # 计算终端价格
    ST = S0 * u ** up_times * d ** down_times

    # 生成图表
    plt.figure()
    plt.plot(ST[:, 0], color='b', alpha=0.5, label='1 month horizon')
    plt.plot(ST[:, 1], color='r', alpha=0.5, label='1 year horizon')
    plt.xlabel('time step, day')
    plt.ylabel('price')
    plt.title('Binomial-Tree Stock Simulation')
    plt.legend()
    plt.show()

    plt.figure()
    plt.hist(ST[:, 0], color='b', alpha=0.5, label='1 month horizon')
    plt.hist(ST[:, 1], color='r', alpha=0.5, label='1 year horizon')
    plt.xlabel('price')
    plt.ylabel('count')
    plt.title('Binomial-Tree Stock Simulation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 模型参数
    mu = 0.1  # 年化收益率
    sigma = 0.15  # 年化波动率
    S0 = 1  # 初始股票价格

    N = 10000  # 蒙特卡洛模拟次数
    T = [21.0 / 252, 1.0]  # 时间范围（1个月和1年）
    step = 1.0 / 252  # 每个时间步的长度（交易日）

    binomial_tree(mu, sigma, S0, N, T, step)
