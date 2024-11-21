import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform
from tqdm import tqdm
import math

class MH_Sampler:
    def __init__(self):
        self.num_samples = 1000
        self.target_mu = [[3, 3], [-3, 3], [-3, -3], [3, -3]]
        self.target_sigma = [[1.2, 0], [0, 1.2]]  # 协方差矩阵
        self.samples = np.zeros((self.num_samples, 2))
        self.n_accept = 0

    def mixture_pdf(self, x, y):
        pdf_value = 0
        for i in range(4):
            # 使用 scipy 的 multivariate_normal 计算每个正态分布的PDF
            pdf_value += 0.25 * multivariate_normal.pdf([x, y], mean=self.target_mu[i], cov=self.target_sigma)
        return pdf_value

    def sample(self):
        # 从初始位置采样
        x_init = multivariate_normal.rvs([0, 0], [[2, 0], [0, 2]], 1)
        self.samples[0, :] = x_init

        for i in tqdm(range(1, self.num_samples)):
            x_cur = self.samples[i-1, :]  # 当前样本
            x_new = multivariate_normal.rvs(x_cur, [[2, 0], [0, 2]], 1)  # 提议新样本

            # 计算接受率
            rate = self.mixture_pdf(x_new[0], x_new[1]) / self.mixture_pdf(x_cur[0], x_cur[1])
            acceptance_prob = min(1, rate)

            # 根据接受率决定是否接受新样本
            if uniform.rvs() <= acceptance_prob:
                self.n_accept += 1
                self.samples[i, :] = x_new  # 接受新样本
            else:
                self.samples[i, :] = x_cur  # 拒绝新样本，保持当前样本

        print("MH acceptance ratio:", self.n_accept / self.num_samples)
        return self.samples


class KL:
    def __init__(self,samples):
        self.samples = samples
        self.target_mu = [[3, 3], [-3, 3], [-3, -3], [3, -3]]
        self.target_sigma = [[1.2, 0], [0, 1.2]]

    def qx(self,x,y):
        pdf_value = 0
        for i in range(4):
            # 使用 scipy 的 multivariate_normal 计算每个正态分布的PDF
            pdf_value += 0.25 * multivariate_normal.pdf([x, y], mean=self.target_mu[i], cov=self.target_sigma)
        return pdf_value

    def px(self,x,y,a,b):
        #a,b为圆心
        r = math.sqrt((x-a)**2 + (y-b)**2)
        if r<=5:
            return -0.08*r + 0.4
        else:
            return 0

    #求正向KL散度
    def forward_KL(self, a, b):
        KL = 0
        epsilon = 1e-10  # 避免对零取对数
        for sample in self.samples:
            q = self.qx(sample[0], sample[1])
            p = self.px(sample[0], sample[1], a, b)
            if p > 0:  # 避免除以零
                KL += q * math.log(q / p + epsilon)
        return KL

    #使用梯度下降找到最佳a和b
    #计算梯度
    def for_compute_gradients(self, a, b, epsilon=1e-6):
        # 使用有限差分计算梯度
        grad_a = (self.forward_KL(a + epsilon, b) - self.forward_KL(a - epsilon, b)) / (2 * epsilon)
        grad_b = (self.forward_KL(a, b + epsilon) - self.forward_KL(a, b - epsilon)) / (2 * epsilon)
        return grad_a, grad_b

    def for_gradient_descent(self, a_init, b_init, learning_rate=0.1, max_iter=1000, tolerance=1e-6):
        a, b = a_init, b_init
        print(f'KL divergence: {self.forward_KL(a, b)}, loc：({a}),({b})')
        for iteration in range(max_iter):
            grad_a, grad_b = self.for_compute_gradients(a, b)
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b


            print(f"Iteration {iteration}, KL divergence: {self.forward_KL(a, b)}, loc：({a}),({b})")
            self.plot_samples_with_coloring(a,b,iteration+1)


        return a, b

    def rev_KL(self, a, b):
        KL = 0
        epsilon = 1e-10  # 避免对零取对数
        for sample in self.samples:
            q = self.qx(sample[0], sample[1])
            p = self.px(sample[0], sample[1], a, b)
            if p > 0:  # 避免除以零
                KL += p * math.log(p / q + epsilon)
        return KL

    # 使用梯度下降找到最佳a和b
    # 计算梯度
    def rev_compute_gradients(self, a, b, epsilon=1e-6):
        # 使用有限差分计算梯度
        grad_a = (self.rev_KL(a + epsilon, b) - self.rev_KL(a - epsilon, b)) / (2 * epsilon)
        grad_b = (self.rev_KL(a, b + epsilon) - self.rev_KL(a, b - epsilon)) / (2 * epsilon)
        return grad_a, grad_b

    def rev_gradient_descent(self, a_init, b_init, learning_rate=0.1, max_iter=1000, tolerance=1e-6):
        a, b = a_init, b_init
        print(f'KL divergence: {self.rev_KL(a, b)}, loc：({a}),({b})')
        for iteration in range(max_iter):
            grad_a, grad_b = self.for_compute_gradients(a, b)
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b

            print(f"Iteration {iteration}, KL divergence: {self.rev_KL(a, b)}, loc：({a}),({b})")
            self.plot_samples_with_coloring(a, b, iteration + 1)

        return a, b



    def plot_samples_with_coloring(self, a, b, i):
        distances = np.sqrt((self.samples[:, 0] - a) ** 2 + (self.samples[:, 1] - b) ** 2)
        colors = distances / distances.max()  # Normalize distances to [0, 1] for color mapping

        plt.figure(figsize=(8, 8))
        plt.scatter(self.samples[:, 0], self.samples[:, 1], c=colors, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(label='Distance from ({}, {})'.format(a, b))
        plt.scatter([a], [b], color='red', marker='x', s=100, label='Center ({}, {})'.format(a, b))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Samples with Coloring Based on Distance to Center ({}, {})".format(a, b))
        plt.legend()
        plt.savefig(f'reverse_KL_result/Iteration{i}.jpg', dpi=300, bbox_inches='tight')

# 实例化并运行采样
mhg = MH_Sampler()
samples = mhg.sample()

#梯度下降找到最优a,b(正向)
#a,b = KL(samples).for_gradient_descent(0,0)

#梯度下降找到最优a,b(逆向)
a,b = KL(samples).rev_gradient_descent(0,0)




# 可视化结果
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1], label='MH samples', alpha=0.5, s=10)
plt.grid(True)
plt.legend()
plt.title("Metropolis-Hastings Sampling of 2D Gaussian Mixture")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()