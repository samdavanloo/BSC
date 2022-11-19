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


class NASA_SoR(Bregman_SoR):
    def __init__(self,  A, batch_size, x_init, tau, beta, a, b, max_iter, R, lmbda, k1, k2, tau_Breg, beta_Breg):
        # default setting for x_hat calculation

        super().__init__(A, batch_size, x_init, k1, k2,
                         tau_Breg, beta_Breg, max_iter, R, lmbda)

        self.tau_NASA = tau
        self.beta_NASA = beta
        self.a = a
        self.b = b

    def _projectd_gradient_step(self, x, w):
        # gradient step
        y = x - 1 / self.beta_NASA * w
        # projection
        y = y * min(1, self.R/np.linalg.norm(y))
        return y

    def _update_u(self, A_sample, u, x):
        # update inner function value estimator
        g = self._get_val_g(A_sample, x)
        u = (1 - self.b * self.tau_NASA) * u + self.b * self.tau_NASA * g
        return u

    def train(self):
        x = self.x_init

        # initial sample
        A_sample = self._sample_A()
        u = self._get_val_g(A_sample, x)

        A_sample = self._sample_A()
        v = self._get_grad_g(A_sample, x)
        s = self._get_grad_f(u)
        w = v.T @ s

        grad_Fdet = self._get_grad_Fdet(x)
        x_hat = self._solve_Breg_sub(x, grad_Fdet)

        # save initial information
        self.x_traj[:, 0] = x
        self.x_hat_traj[:, 0] = x_hat
        self.val_F_traj[0] = self._get_val_F(x)
        self.grad_Fdet_traj[:, 0] = grad_Fdet
        self.grad_F_traj[:, 0] = w

        for iter in range(1, self.max_iter):
            x_pre = x
            # update

            y = self._projectd_gradient_step(x, w)
            if iter == 1:
                x = y
            else:
                x = x_pre + self.tau_NASA * (y - x_pre)

            s = self._get_grad_f(u)
            A_sample = self._sample_A()
            J = self._get_grad_g(A_sample, x)
            w = (1 - self.a * self.tau_NASA) * w + \
                self.a * self.tau_NASA * J.T @ s

            u = self._update_u(A_sample, u, x)

            grad_Fdet = self._get_grad_Fdet(x)
            x_hat = self._solve_Breg_sub(x, grad_Fdet)

            # save information
            self.x_traj[:, iter] = x
            self.x_hat_traj[:, iter] = x_hat
            self.val_F_traj[iter] = self._get_val_F(x)
            self.grad_Fdet_traj[:, iter] = grad_Fdet
            self.grad_F_traj[:, iter] = w


# parameters
d = 50  # dimension of matrix
n = 5000  # number of random matrix
lmbda = 10  # weight of var part
R = 10  # constraint norm(x) <= R
noise_level = 0.5
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




# define the function for grid search
tau_grid = np.logspace(-3, 0, num=6)
beta_grid = np.logspace(-4, 2, num=6)


def GridSearch(args):
    i, j = args
    batch_size = 100
    max_iter = 300
    tau = tau_grid[i]
    beta = beta_grid[j]
    a = 0.5/tau
    b = 0.5/tau
    k1, k2, tau_Breg, beta_Breg =10, 1.58e-3, 0.025,0.5
    
    NASA = NASA_SoR(A, batch_size, x_init, tau, beta, a, b,
                    max_iter, R, lmbda, k1, k2, tau_Breg, beta_Breg)
    NASA.train()
    NASA.plot(k1, k2, tau_Breg, avg=False)

    filename = f"Results/Grid_search/NASA_GridSearch_i{i}_j{j}.pdf"
    plt.savefig(filename)
    plt.close()


# %%
if "get_ipython" in dir():
    tau = 0.063  # tau_grid[3]
    beta = 158.5  # beta_grid[1]
    a = 0.5/tau
    b = 0.5/tau
    tau = tau_grid[3]
    beta = beta_grid[4]#2
    batch_size = 100
    max_iter = 300
    k1, k2, tau_Breg, beta_Breg = 63.1, 1.58, 0.025, 0.5
    NASA = NASA_SoR(A, batch_size, x_init, tau, beta, a, b,
                    max_iter, R, lmbda, k1, k2, tau_Breg, beta_Breg)
    NASA.train()
    NASA.calculate_measure_avg()
    NASA.plot(k1, k2, tau_Breg, avg=True)



# %% Grid Search

if __name__ == '__main__' and "get_ipython" not in dir():

    grid_list = [range(6), range(6)]
    args = [p for p in itertools.product(*grid_list)]
    with Pool(8) as pool:
        # prepare arguments

        # issue multiple tasks each with multiple arguments
        pool.map(GridSearch, args)
        pool.close()
        pool.join()
