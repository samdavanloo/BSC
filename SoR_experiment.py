#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:06:46 2021

@author: yin_liu
"""

# import sklearn
# from sklearn import datasets  # used to generate PSD matrix
import numpy as np
import matplotlib.pyplot as plt
from method_Bregman import *



plt.rcParams.update({
    "text.usetex": False})



# %% Parameters
d = 50  # dimension of matrix
n = 1000  # number of random matrix
lmbda = 1  # weight of var part
R = 10  # constraint norm(x) <= R
noiselevel = 3

oracles = 30000
n_batch = 1

iter_max = oracles//n_batch

exp_number=20
# %% Generate matrix
np.random.seed(10)

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
i = 8
j = 2
l1 = l1[i]
l2 = l2[j]

c1 = l1 + l2*A_norm
c2 = l2 * 3 * A_norm**2

for repeat in range(exp_number):
    f_BG, x_BG, x_history, x_hat_history = SoR_breg(A, lmbda, R, tau, beta,
                                   n_batch, c1, c2, iter_max, x_init, track_f=0)

    # x_norm2_BG = x_BG[:, 1:] - x_BG[:, 0:-1]
    # x_norm2_BG = np.sqrt(np.sum(x_norm2_BG*x_norm2_BG, axis=0))

    x_norm2_BG = x_history[:, 1:] - x_history[:, 0:-1]
    x_norm2_BG = np.sum(x_norm2_BG*x_norm2_BG, axis=0)/2

    x_norm2_avg_BG = np.zeros(iter_max-1)
    for iter in range(0, iter_max-1):
        x_norm2_avg_BG[iter] = np.mean(x_norm2_BG[:iter+1])

    x_norm4_BG = np.zeros(iter_max-1)
    x_norm4_avg_BG = np.zeros(iter_max-1)
    for iter in range(0, iter_max-1):
        y = x_history[:, iter+1]
        x = x_history[:, iter]
        x_norm4_BG[iter] = np.linalg.norm(
            y)**4 / 4 - np.linalg.norm(x)**4 / 4 - np.linalg.norm(x)**2 * x@(y-x)
        x_norm4_avg_BG[iter] = np.mean(x_norm4_BG[:iter+1])

    x_hat_norm2_BG = x_hat_history[:, 1:] - x_history[:, 0:-1]
    x_hat_norm2_BG = np.sum(x_hat_norm2_BG*x_hat_norm2_BG, axis=0)/2
    
    x_hat_norm2_avg_BG = np.zeros(iter_max-1)
    for iter in range(0, iter_max-1):
        x_hat_norm2_avg_BG[iter] = np.mean(x_hat_norm2_BG[:iter+1])
    
    x_hat_norm4_BG = np.zeros(iter_max-1)
    x_hat_norm4_avg_BG = np.zeros(iter_max-1)
    for iter in range(0, iter_max-1):
        y = x_hat_history[:, iter+1]
        x = x_history[:, iter]
        x_hat_norm4_BG[iter] = np.linalg.norm(
            y)**4 / 4 - np.linalg.norm(x)**4 / 4 - np.linalg.norm(x)**2 * x@(y-x)
        x_hat_norm4_avg_BG[iter] = np.mean(x_hat_norm4_BG[:iter+1])
        
      


    fig = plt.figure()

    #plt.plot(f_BG-min(f_BG))
  #  plt.ylabel('f-min(f)')
  #  plt.yscale("log")
    # plt.subplot(412)
    # plt.plot(f_BG)
    # plt.ylabel('f')
    # plt.subplot(413)
    
    plt.subplot(121)
    plt.plot(x_norm2_avg_BG)
    plt.plot(x_hat_norm2_avg_BG)
    plt.ylabel('norm(x-x_pre)')
    plt.yscale("log")
    plt.xlabel('norm2')
    
    plt.subplot(122)
    plt.plot(x_norm4_avg_BG,label='x')
    plt.plot(x_hat_norm4_avg_BG,label='x_hat')
    plt.ylabel('norm(x-x_pre)')
    plt.yscale("log")
    plt.xlabel('x_norm4')

    plt.legend()
    fig.suptitle('sample %i, d=%i, n=%i, lambda=%i, R=%i, tau_k=%.2f, n_batch=%i,\n c1=%.2f(%i), c2=%.2f(%i)' %
                 (repeat, d, n,  lmbda, R, tau, n_batch, c1, i, c2, j))
    plt.show()
    print('finish ' + str(repeat))
    filename = 'Result/SoR_batch_' + \
        str(int(n_batch)) + 'repeat_' + str(int(repeat)) + '.npz'
    np.savez(filename, f_BG=f_BG, x_BG=x_BG, x_history=x_history, x_hat_history = x_hat_history,
              x_norm4_avg_BG=x_norm4_avg_BG, x_norm2_avg_BG=x_norm2_avg_BG, x_hat_norm2_avg_BG = x_hat_norm2_avg_BG, x_hat_norm4_avg_BG = x_hat_norm4_avg_BG )
