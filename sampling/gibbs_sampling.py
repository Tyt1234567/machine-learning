from scipy import stats
import math
import matplotlib.pyplot as plt


class gibbs_sampling_2D_norm:
    def __init__(self, norm_a_mu, norm_a_sigma, norm_b_mu, norm_b_sigma, cov_ab):
        self.norm_a_mu = norm_a_mu
        self.norm_a_sigma = norm_a_sigma
        self.norm_b_mu = norm_b_mu
        self.norm_b_sigma = norm_b_sigma
        self.cov_ab = cov_ab

        # 计算相关系数
        self.cor = self.cov_ab / math.sqrt(self.norm_a_sigma * self.norm_b_sigma)

    def sampling(self, n):
        count = 0
        sample_x = []
        sample_y = []
        x = 0  # 初始的x值

        while count < n:
            # 先对 y 进行采样，基于给定的 x
            y_mean = self.norm_b_mu + self.cor * self.norm_b_sigma / self.norm_a_sigma * (x - self.norm_a_mu)
            y_std = math.sqrt(self.norm_b_sigma ** 2 * (1 - self.cor ** 2))
            y = stats.norm.rvs(y_mean, y_std)

            # 记录采样值
            sample_x.append(x)
            sample_y.append(y)

            # 再对 x 进行采样，基于给定的 y
            x_mean = self.norm_a_mu + self.cor * self.norm_a_sigma / self.norm_b_sigma * (y - self.norm_b_mu)
            x_std = math.sqrt(self.norm_a_sigma ** 2 * (1 - self.cor ** 2))
            x = stats.norm.rvs(x_mean, x_std)

            # 记录采样值
            sample_x.append(x)
            sample_y.append(y)

            count += 2

        # 打印平均值
        print("Mean of X:", sum(sample_x) / len(sample_x))
        print("Mean of Y:", sum(sample_y) / len(sample_y))

        # 绘制采样散点图
        plt.scatter(sample_x, sample_y)
        plt.title("Gibbs Sampling - Bivariate Normal Distribution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

'''
# 示例采样
sampler = gibbs_sampling_2D_norm(1, 2, 1, 1, 1)
sampler.sampling(10000)
'''

from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class gibbs_sampling_3D_norm:
    def __init__(self, norm_mu, norm_sigma, cov_matrix):
        self.norm_mu = norm_mu  # 均值向量 [mu1, mu2, mu3]
        self.norm_sigma = norm_sigma  # 对角方差矩阵的对角元素 [sigma1^2, sigma2^2, sigma3^2]
        self.cov_matrix = cov_matrix  # 协方差矩阵 3x3

        # 计算相关系数矩阵
        self.cor_matrix = self.cov_matrix / np.sqrt(np.outer(self.norm_sigma, self.norm_sigma))

    def conditional_mean_and_variance(self, x, given_idx, given_values):
        """计算给定部分变量下的条件均值和方差"""
        indices = [0, 1, 2]  # 变量索引
        remaining_idx = [i for i in indices if i != given_idx]
        mu_a = self.norm_mu[given_idx]

        covab_colum = self.cov_matrix[given_idx]
        covab = covab_colum[remaining_idx]

        covbb = self.cov_matrix[np.ix_(remaining_idx, remaining_idx)]

        xb_mub = np.array(x)-self.norm_mu[remaining_idx]

        mu_given = mu_a + covab@np.linalg.inv(covbb)@xb_mub.transpose()

        variance_given = self.cov_matrix[given_idx,given_idx] - covab@np.linalg.inv(covbb)@covab.transpose()

        return mu_given, variance_given

    def sampling(self, n):
        count = 0
        sample_x1 = []
        sample_x2 = []
        sample_x3 = []

        # 第一个采样点的x1, x2, x3为0，分别采样
        x1, x2, x3 = 0, 0, 0
        while count < n:
            # 先对 x1 进行采样，给定 x2, x3
            mu_x1, var_x1 = self.conditional_mean_and_variance([x2, x3], 0, np.array([x2, x3]))
            x1 = stats.norm.rvs(mu_x1, math.sqrt(var_x1))
            sample_x1.append(x1)
            sample_x2.append(x2)
            sample_x3.append(x3)

            # 再对 x2 进行采样，给定 x1, x3
            mu_x2, var_x2 = self.conditional_mean_and_variance([x1, x3], 1, np.array([x1, x3]))
            x2 = stats.norm.rvs(mu_x2, math.sqrt(var_x2))
            sample_x1.append(x1)
            sample_x2.append(x2)
            sample_x3.append(x3)

            # 最后对 x3 进行采样，给定 x1, x2
            mu_x3, var_x3 = self.conditional_mean_and_variance([x1, x2], 2, np.array([x1, x2]))
            x3 = stats.norm.rvs(mu_x3, math.sqrt(var_x3))
            sample_x1.append(x1)
            sample_x2.append(x2)
            sample_x3.append(x3)

            count += 3

        print("Mean of X1:", sum(sample_x1) / len(sample_x1))
        print("Mean of X2:", sum(sample_x2) / len(sample_x2))
        print("Mean of X3:", sum(sample_x3) / len(sample_x3))

        # 绘制 3D 散点图
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sample_x1, sample_x2, sample_x3, alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("X3")
        ax.set_title("Gibbs Sampling in 3D Space")

        plt.show()


# 示例采样
mu = np.array([1, 2, 3])  # 均值向量
sigma = np.array([1, 2, 1.5])  # 方差向量
cov_matrix = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 0.5]])  # 协方差矩阵

sampler = gibbs_sampling_3D_norm(mu, sigma, cov_matrix)
sampler.sampling(10000)