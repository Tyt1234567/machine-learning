import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal

np.random.seed(42)
sns.set_style('whitegrid')


class ImageDenoising:
    def __init__(self, img_binary, sigma=2, J=1):
        # Mean-field parameters
        self.sigma = sigma
        self.y = img_binary + self.sigma * np.random.randn(*img_binary.shape)
        self.J = J
        self.rate = 0.5
        self.max_iter = 50
        self.ELBO = np.zeros(self.max_iter)
        self.Hx_mean = np.zeros(self.max_iter)

    def mean_field(self):
        # Mean-Field VI
        print("Running mean-field variational inference...")
        M, N = self.y.shape
        logodds = multivariate_normal.logpdf(self.y .flatten(), mean=+1, cov=self.sigma ** 2) - \
                  multivariate_normal.logpdf(self.y.flatten(), mean=-1, cov=self.sigma ** 2)
        logodds = np.reshape(logodds, (M, N))

        # Initialization
        p1 = sigmoid(logodds)
        mu = 2 * p1 - 1
        a = mu + 0.5 * logodds
        qxp1 = sigmoid(+2 * a)  # q_i(x_i=+1)
        qxm1 = sigmoid(-2 * a)  # q_i(x_i=-1)
        logp1 = np.reshape(multivariate_normal.logpdf(self.y.flatten(), mean=+1, cov=self.sigma ** 2), (M, N))
        logm1 = np.reshape(multivariate_normal.logpdf(self.y.flatten(), mean=-1, cov=self.sigma ** 2), (M, N))

        for i in tqdm(range(self.max_iter)):
            muNew = mu.copy()
            for ix in range(N):
                for iy in range(M):
                    pos = iy + M * ix
                    neighborhood = pos + np.array([-1, 1, -M, M])
                    boundary_idx = [iy != 0, iy != M - 1, ix != 0, ix != N - 1]
                    neighborhood = neighborhood[np.where(boundary_idx)[0]]
                    xx, yy = np.unravel_index(pos, (M, N), order='F')
                    nx, ny = np.unravel_index(neighborhood, (M, N), order='F')

                    Sbar = self.J * np.sum(mu[nx, ny])
                    muNew[xx, yy] = (1 - self.rate) * muNew[xx, yy] + self.rate * np.tanh(Sbar + 0.5 * logodds[xx, yy])

                    self.ELBO[i] += 0.5 * (Sbar * muNew[xx, yy])

            mu = muNew
            a = mu + 0.5 * logodds
            qxp1 = sigmoid(+2 * a)  # q_i(x_i=+1)
            qxm1 = sigmoid(-2 * a)  # q_i(x_i=-1)
            Hx = -qxm1 * np.log(qxm1 + 1e-10) - qxp1 * np.log(qxp1 + 1e-10)  # entropy

            self.ELBO[i] += np.sum(qxp1 * logp1 + qxm1 * logm1) + np.sum(Hx)
            self.Hx_mean[i] = np.mean(Hx)
            self.save_fig(i+1,mu)


        return mu

    def save_fig(self,i,mu):
        plt.figure()
        plt.imshow(mu)
        plt.title(f"After {i+1} Mean-Field Iterations")
        plt.savefig(f'denoise_images/iter{i}.png')



if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = Image.open('raw_img.png').convert('L')
    img = np.array(data, dtype=float)
    img_mean = np.mean(img)
    img_binary = +1 * (img > img_mean) + -1 * (img < img_mean)
    M, N = img_binary.shape

    # Initialize and run model
    mrf = ImageDenoising(img_binary, sigma=2, J=1)
    mu = mrf.mean_field()

    # Generate plots
    plt.figure()
    plt.imshow(mrf.y)
    plt.title("Observed Noisy Image")
    plt.show()

    plt.figure()
    plt.imshow(mu)
    plt.title(f"After {mrf.max_iter} Mean-Field Iterations")
    plt.show()

    plt.figure()
    plt.plot(mrf.Hx_mean, color='b', lw=2.0, label='Avg Entropy')
    plt.title('Variational Inference for Ising Model')
    plt.xlabel('Iterations')
    plt.ylabel('Average Entropy')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    plt.plot(mrf.ELBO, color='b', lw=2.0, label='ELBO')
    plt.title('Variational Inference for Ising Model')
    plt.xlabel('Iterations')
    plt.ylabel('ELBO Objective')
    plt.legend(loc='upper left')
    plt.show()

