import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import multivariate_normal

np.random.seed(42)

class ImportanceSampler:
    def __init__(self, k=1.5, mu=0.8, sigma=np.sqrt(1.5), c=3):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.c = c

    def target_pdf(self, x):
        return (x**(self.k - 1)) * np.exp(-x**2 / 2.0)

    def proposal_pdf(self, x):
        return self.c * 1.0 / np.sqrt(2 * np.pi * 1.5) * np.exp(-(x - self.mu)**2 / (2 * self.sigma**2))

    def fx(self, x):
        return 2 * np.sin((np.pi / 1.5) * x)

    def sample(self, num_samples):
        x = multivariate_normal.rvs(self.mu, self.sigma, num_samples)
        idx = np.where(x >= 0)
        x_pos = x[idx]
        isw = self.target_pdf(x_pos) / self.proposal_pdf(x_pos) #权重
        fw = (isw / np.sum(isw)) * self.fx(x_pos)
        f_est = np.sum(fw)

        return isw, f_est

if __name__ == "__main__":
    num_samples = [10, 100, 1000, 10000, 100000, 1000000]
    F_est_iter, IS_weights_var_iter = [], []

    for k in num_samples:
        IS = ImportanceSampler()
        IS_weights, F_est = IS.sample(k)
        IS_weights_var = np.var(IS_weights / np.sum(IS_weights))
        F_est_iter.append(F_est)
        IS_weights_var_iter.append(IS_weights_var)

    # Ground truth (numerical integration)
    k = 1.5
    I_gt, _ = quad(lambda x: 2.0 * np.sin((np.pi / 1.5) * x) * (x**(k - 1)) * np.exp(-x**2 / 2.0), 0, 5)

    # Generate plots
    plt.figure()
    xx = np.linspace(0, 8, 100)
    plt.plot(xx, IS.target_pdf(xx), '-r', label='Target PDF p(x)')
    plt.plot(xx, IS.proposal_pdf(xx), '-b', label='Proposal PDF q(x)')
    plt.plot(xx, IS.fx(xx) * IS.target_pdf(xx), '-k', label='p(x)f(x) Integrand')
    plt.grid(True)
    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Importance Sampling Components")
    plt.show()

    plt.figure()
    plt.hist(IS_weights, label="IS Weights", bins=30)
    plt.grid(True)
    plt.legend()
    plt.title("Importance Weights Histogram")
    plt.show()

    plt.figure()
    plt.semilogx(num_samples, F_est_iter, label="IS Estimate of E[f(x)]")
    plt.semilogx(num_samples, I_gt * np.ones(len(num_samples)), label="Ground Truth")
    plt.grid(True)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel("E[f(x)] Estimate")
    plt.title("IS Estimate of E[f(x)]")
    plt.show()
