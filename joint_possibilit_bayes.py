import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bivariate_normal(x, y, mux, muy, sigma):
    return np.exp(-((x - mux) ** 2 + (y - muy) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

def likelihood(r, x, y):
    true_distance = np.sqrt(x**2 + y**2)
    return norm.pdf(r, loc=true_distance, scale=1)

def posterior(x, y, prior, r):
    return prior * likelihood(r, x, y)

def plot_distribution(X, Y, Z, title):
    plt.contourf(X, Y, Z, cmap='Blues')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

# 设置网格
x_vals = np.linspace(-5, 10, 100)
y_vals = np.linspace(-5, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# 先验分布
mux, muy = 3, 3
sigma = 2
prior = bivariate_normal(X, Y, mux, muy, sigma)
plot_distribution(X, Y, prior, "Prior Distribution")

# 观测值
r = 4
likelihood_vals = likelihood(r, X, Y)
plot_distribution(X, Y, likelihood_vals, "Likelihood (Given r = 4)")

# 后验分布（未归一化）
posterior_vals = posterior(X, Y, prior, r)
plot_distribution(X, Y, posterior_vals, "Posterior Distribution (Unnormalized)")

# 后验分布归一化
posterior_normalized = posterior_vals / np.sum(posterior_vals)
plot_distribution(X, Y, posterior_normalized, "Posterior Distribution (Normalized)")