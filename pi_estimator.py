import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子，以便结果可重复
#np.random.seed(42)


def pi_est(radius=1, num_iter=10000):
    # 生成随机点
    X = np.random.uniform(-radius, +radius, num_iter)
    Y = np.random.uniform(-radius, +radius, num_iter)

    # 计算每个点到圆心的距离的平方
    R2 = X ** 2 + Y ** 2

    # 判断点是否在圆内
    inside = R2 < radius ** 2
    outside = ~inside


    # 计算正方形的面积和积分的估计
    samples = (2 * radius) * (2 * radius) * inside

    I_hat = np.mean(samples)

    # 估计π的值
    pi_hat = I_hat / radius ** 2
    pi_hat_se = np.std(samples) / np.sqrt(num_iter)

    # 打印结果
    print("pi est: {} +/- {:f}".format(pi_hat, pi_hat_se))

    # 可视化结果
    plt.figure()
    plt.scatter(X[inside], Y[inside], c='b', alpha=0.5, label='Inside Circle')
    plt.scatter(X[outside], Y[outside], c='r', alpha=0.5, label='Outside Circle')
    plt.xlim(-radius * 1.5, radius * 1.5)
    plt.ylim(-radius * 1.5, radius * 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Monte Carlo Estimation of π')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    pi_est()
