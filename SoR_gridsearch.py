"""
Used to find best c1, c2 combination for specific dataset(seed 10)
"""

# import sklearn
# from sklearn import datasets  # used to generate PSD matrix
import numpy as np
import matplotlib.pyplot as plt
from method_Bregman import *
# %% Parameters
d = 50  # dimension of matrix
n = 1000  # number of random matrix
lmbda = 1  # weight of var part
R = 10  # constraint norm(x) <= R
noiselevel = 1

iter_max = 100
n_batch = 50

# %% Generate matrix
np.random.seed(10)

#A= np.random.randn(d, d)
#A = (A  + A.T)/2


A_avg = np.random.randn(d, d)
A_avg = (A_avg+A_avg.T)/2
#A_avg = sklearn.datasets.make_spd_matrix(d)
noise = np.random.randn(d, d, n)
A = np.tile(A_avg[:, :, np.newaxis], (1, 1, n)) + \
    1*(noise+np.swapaxes(noise, 0, 1))/2


A_avg = A.mean(axis=2)  # Mean of matrix A based on the generated data
D, V = np.linalg.eig(A_avg)  # used to get the eigenvalue

A_norm = np.linalg.norm(A_avg, 2)

# %%

Lf = 2*lmbda
Lg = 3 * A_norm**2

Cf = np.sqrt(lmbda**2 * R**4 * A_norm**2 + 2*lmbda*R**2 * A_norm)
Cg = np.sqrt(A_norm**2 * R**2 + A_norm**4 * R**6)


tau = min(0.5, Lf/(Lf+8), 1/Lf) / 2
beta = Lf * tau

x_init = np.random.randn(d)
x_init = x_init/np.linalg.norm(x_init)*R  # initial point
w = np.zeros(d)
u = np.zeros(2)

# coefs in generating function h(x); coef of norm(x)^2
c1 = (Cg**2 * Lf + Cf * Lg * A_norm)
c2 = 3 * (Cf * Lg * A_norm**2)


l1 = np.logspace(-4, 2, num=9)
l2 = np.logspace(-4, 2, num=9)

for i in range(len(l1)):
    for j in range(len(l2)):
        c1 = l1[i] + l2[j]*A_norm
        c2 = l2[j] * 3 * A_norm**2

        f_BG, x_BG, x_history = S_breg(A, lmbda, R, tau, beta,
                                       n_batch, c1, c2, iter_max, x_init)

        x_norm2_BG = x_history[:, 1:] - x_history[:, 0:-1]
        x_norm2_BG = np.sum(x_norm2_BG*x_norm2_BG, axis=0)/2

        fig = plt.figure()
        plt.subplot(311)
        plt.plot(f_BG-min(f_BG))
        plt.ylabel('f-min(f)')
        plt.yscale("log")
        plt.subplot(312)
        plt.plot(f_BG)
        plt.ylabel('f')
        plt.subplot(313)

        plt.plot(x_norm2_BG)
        plt.ylabel('norm(x-x_pre)')
        plt.yscale("log")

        fig.suptitle('d=%i, n=%i, lambda=%i, R=%i, tau_k=%.2f, n_batch=%i,\n c1=%.2f(%i), c2=%.2f(%i)' %
                     (d, n,  lmbda, R, tau, n_batch, c1, i, c2, j))
        plt.show()
        print('finish ' + str(i) + '_' + str(j))
