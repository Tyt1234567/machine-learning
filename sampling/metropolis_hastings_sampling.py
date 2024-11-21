import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, uniform


class MH_Gauss:
    def __init__(self, dim, K, num_samples, target_mu, target_sigma, target_pi, proposal_mu, proposal_sigma):
        self.dim = dim
        self.K = K
        self.num_samples = num_samples
        self.target_mu = target_mu
        self.target_sigma = target_sigma
        self.target_pi = target_pi
        self.proposal_mu = proposal_mu
        self.proposal_sigma = proposal_sigma
        self.n_accept = 0
        self.mh_samples = np.zeros((num_samples, dim))

    def target_pdf(self, x):
        return sum(pi * multivariate_normal.pdf(x, mu, sigma)
                   for pi, mu, sigma in zip(self.target_pi, self.target_mu.T, self.target_sigma))

    def proposal_pdf(self, x):
        return multivariate_normal.pdf(x, self.proposal_mu, self.proposal_sigma)

    def sample(self):
        x_curr = multivariate_normal.rvs(self.proposal_mu, self.proposal_sigma)
        self.mh_samples[0, :] = x_curr

        for i in range(1, self.num_samples):
            x_new = multivariate_normal.rvs(x_curr, self.proposal_sigma)
            alpha = self.target_pdf(x_new) / self.target_pdf(x_curr)
            r = min(1, alpha)

            if uniform.rvs() <= r:
                self.n_accept += 1
                x_curr = x_new

            self.mh_samples[i, :] = x_curr


        print("MH acceptance ratio:", self.n_accept / self.num_samples)


if __name__ == "__main__":
    dim, K, num_samples = 2, 2, 5000
    target_mu = np.array([[4, -4], [0, 0]])
    target_sigma = np.array([[[2, 1], [1, 1]], [[1, 0], [0, 1]]])
    target_pi = np.array([0.4, 0.6])
    proposal_mu = np.zeros(dim)
    proposal_sigma = 5 * np.eye(dim)

    mhg = MH_Gauss(dim, K, num_samples, target_mu, target_sigma, target_pi, proposal_mu, proposal_sigma)
    mhg.sample()

    plt.figure()
    plt.scatter(mhg.mh_samples[:, 0], mhg.mh_samples[:, 1], label='MH samples')
    plt.grid(True)
    plt.legend()
    plt.title("Metropolis-Hastings Sampling of 2D Gaussian Mixture")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
