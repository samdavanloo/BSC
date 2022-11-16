# %% Library

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import itertools


# own class
from Algorithms import Bregman_SoR

# setting for plot
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "ps.usedistiller": "xpdf"})


class SCSC_SoR(Bregman_SoR):
    def __init__(self,  A, batch_size, x_init, alpha, beta, max_iter, R, lmbda):
        # default setting for x_hat calculation
        k1, k2, tau =158, 1.58, 0.025
        self.alpha = alpha
        super().__init__(A, batch_size, x_init, k1, k2, tau, beta, max_iter, R, lmbda)

    def _projectd_gradient_step(self, x, w):
        # gradient step
        y = x - self.alpha * w
        # projection
        y = y * min(1, self.R/np.linalg.norm(y))
        return y

    def train(self):
        x = self.x_init

        # initial sample

        w = np.random.randn(self.d)
        u = np.random.randn(2)

        # save initial information
        self.x_traj[:, 0] = x
        self.x_hat_traj[:, 0] = self._get_x_hat(x)
        self.val_F_traj[0] = self._get_val_F(x)

        for iter in range(1, self.max_iter):
            x_pre = x
            # update
            x = self._projectd_gradient_step(x, w)

            A_sample = self._sample_A()
            u = self._update_u(A_sample, u, x_pre, x)

            A_sample = self._sample_A()
            v = self._get_grad_g(A_sample, x)
            s = self._get_grad_f(u)
            w = v.T @ s

            # save information
            self.x_traj[:, iter] = x
            self.x_hat_traj[:, iter] = self._get_x_hat(x)
            self.val_F_traj[iter] = self._get_val_F(x)

# define the function for grid search

alpha_grid = np.logspace(-5, -3, num=6)
def GridSearch(args):
    i, = args

    alpha = alpha_grid[i]
    SCSC = SCSC_SoR(A, batch_size, x_init, alpha, beta, max_iter, R, lmbda)
    SCSC.train()
    SCSC.plot(64,1.5, 0.025, avg=True)

    filename = f"Results/Grid_search/SCSC_GridSearch_i{i}.pdf"
    plt.savefig(filename)
    plt.close()


# parameters
d = 50  # dimension of matrix
n = 1000  # number of random matrix
lmbda = 10  # weight of var part
R = 10  # constraint norm(x) <= R
noise_level = 3
Lf = 2*lmbda
Lg = 1
tau = min(0.5, Lf/(Lf+8), 1/Lf) / 2
beta = Lf * tau

batch_size = 500
max_iter = 300

# Generate matrix
np.random.seed(10)
A_avg = np.random.randn(d, d)
A_avg = (A_avg+A_avg.T)/2
noise = np.random.randn(d, d, n)
A = np.tile(A_avg[:, :, np.newaxis], (1, 1, n)) + \
    noise_level * (noise+np.swapaxes(noise, 0, 1)) / \
    2    # make sure A are all symmetric

A_avg = A.mean(axis=2)  # Mean of matrix A based on the generated data

A_avg_norm = np.linalg.norm(A_avg, 2)
A_norm2_avg = np.mean(np.linalg.norm(A, axis=(0, 1), ord=2)
                      ** 2)  # calculate (E[|A_xi|^2])


x_init = np.random.randn(d)
x_init = x_init/np.linalg.norm(x_init)*R  # initial point


# %%
alpha = alpha_grid[3]
beta = 0.5

SCSC = SCSC_SoR(A, batch_size, x_init, alpha, beta, max_iter, R, lmbda)
SCSC.train()

# SCSC.plot(158, 1.58, 0.025, avg=True)

# %%

# %% Grid Search

if __name__ == '__main__':

    grid_list = [range(6)]
    args = [p for p in itertools.product(*grid_list)]
    with Pool(8) as pool:
        # prepare arguments
        
        # issue multiple tasks each with multiple arguments
        pool.map(GridSearch, args)
        pool.close()
        pool.join()
