#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:22:33 2021

@author: yin_liu

Solve composition problem min_x D_KL(E[b],E[A]x)
where b in R^m++, A in R^mxn+, x in R^n+

"""
import numpy as np

import matplotlib.pyplot as plt

from method_Bregman import *


m = 50
n = 30
N = 30000

tau = 30
lmbda = 40

batch_grad = 10
batch_val = batch_grad**2


oracles = 30000
iter_max = oracles//batch_grad










exp_num=20
# %% Generate data
np.random.seed(10)
A_avg = 2 * np.random.rand(m, n) + 5
A = np.tile(A_avg[:, :, np.newaxis], (1, 1, N)) + np.random.randn(m, n, N)
A[A < 0] = 0  # make sure all elements are non-negative
A_avg = A.mean(axis=2)  # recalculate A based on random samples

x_true = np.random.rand(n)+1
b = A_avg @ x_true
b = np.tile(b[:, np.newaxis], (1, N)) + np.random.poisson(lam=10,
                                                          size=[m, N])  # each mxn matrix multiply x

b_avg = b.mean(axis=1)
Lf = b_avg.sum()

# initial point
x_init = np.random.rand(n)


for repeat in range(exp_num):

    x_history, x_tilde_history,oracle_innervalue,oracle_innergrad = RoS_breg(
        A, b, tau, lmbda, batch_grad, batch_val, iter_max, x_init)

    Dh1 = np.zeros(iter_max-1)
    Dh2 = np.zeros(iter_max-1)
    Dh1_avg = np.zeros(iter_max-1)
    Dh2_avg = np.zeros(iter_max-1)
    
    Dh1_tilde = np.zeros(iter_max-1)
    Dh2_tilde = np.zeros(iter_max-1)
    Dh1_tilde_avg = np.zeros(iter_max-1)
    Dh2_tilde_avg = np.zeros(iter_max-1)


    for iter in range(iter_max-1):
        x1 = x_history[:, iter]
        x2 = x_history[:, iter+1]
        Dh2[iter] = - np.sum(np.log(A_avg @ x2)) + \
            np.sum(np.log(A_avg @ x1)) + A_avg.T @ (1/(A_avg@x1)) @ (x2-x1)
        Dh1[iter] = 1/2 * np.linalg.norm(x2-x1)**2

        Dh1_avg[iter] = np.mean(Dh1[0:iter+1])
        Dh2_avg[iter] = np.mean(Dh2[0:iter+1])
        
        
        x1 = x_history[:, iter]
        x2 = x_tilde_history[:, iter+1]
        Dh2_tilde[iter] = - np.sum(np.log(A_avg @ x2)) + \
            np.sum(np.log(A_avg @ x1)) + A_avg.T @ (1/(A_avg@x1)) @ (x2-x1)
        Dh1_tilde[iter] = 1/2 * np.linalg.norm(x2-x1)**2

        Dh1_tilde_avg[iter] = np.mean(Dh1_tilde[0:iter+1])
        Dh2_tilde_avg[iter] = np.mean(Dh2_tilde[0:iter+1])        

    # %% Plot

    plt.subplot(221)
    plt.plot(Dh1_avg)
    plt.yscale("log")
    plt.ylabel('D_hg')

    plt.subplot(222)
    plt.plot(Dh1_tilde_avg)
    plt.ylabel('D_hg')
    plt.yscale("log")

    plt.subplot(223)
    plt.plot(Dh2_avg)
    plt.ylabel('D_hf_tilde')
    plt.yscale("log")
    
    plt.subplot(224)
    plt.plot(Dh2_tilde_avg)
    plt.ylabel('D_hf_tilde')
    plt.yscale("log")
    plt.suptitle('repeat %i tau = %.2f, lmbda = %.2f , batch_grad = %i' %
                 (repeat, tau,  lmbda,  batch_grad))
    plt.show()
    print('finish ' + str(repeat))
    filename = 'Result/RoS_batch_' + \
        str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'
    np.savez(filename,   x_history=x_history, x_tilde_history = x_tilde_history,
             Dh1=Dh1, Dh2=Dh2, Dh1_avg=Dh1_avg, Dh2_avg=Dh2_avg,
             Dh1_tilde=Dh1_tilde, Dh2_tilde=Dh2_tilde, Dh1_tilde_avg=Dh1_tilde_avg,
             Dh2_tilde_avg=Dh2_tilde_avg, oracle_innervalue = oracle_innervalue,
             oracle_innergrad = oracle_innergrad)
