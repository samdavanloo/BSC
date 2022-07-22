#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:22:33 2021

@author: yin_liu

Solve composition problem min_x D_KL(E[b],E[A]x)
where b in R^m++, A in R^mxn+, x in R^n+
find best tau and lmbda

"""
import numpy as np
import cvxpy as cp  # solver used for the subproblem

import matplotlib.pyplot as plt

iter_max = 200

m = 50
n = 30
N = 5000

tau = 1
lmbda = 1000

batch_grad = 10
batch_val = batch_grad**2


# %% Generate data
np.random.seed(10)
A_avg = 2 * np.random.rand(m, n)
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


l1 = np.logspace(0, 3, num=7)
l2 = np.logspace(0, 4, num=7)

for grid_1 in range(len(l1)):
    for grid_2 in range(len(l2)):
        tau = l1[grid_1]
        lmbda = l2[grid_2]
        try:
            # %% Algorithm
            f = np.zeros(iter_max)
            x_history = np.zeros([n, iter_max])
            x_out = np.zeros([n, iter_max])

            grad_norm = np.zeros(iter_max)

            x = x_init

            for iter in range(iter_max):

                # u update
                if batch_val > 1:

                    i = np.random.randint(0, N, batch_val)
                    u = np.einsum('ijk,j->ik', A[:, :, i], x)
                    u = np.mean(u, 1)
                else:
                    i = np.random.randint(N)
                    u = A[:, :, i]@x

                # w update
                if batch_grad > 1:
                    i = np.random.randint(0, N, batch_grad)
                    v = np.mean(A[:, :, i], 2)

                    i = np.random.randint(0, N, batch_grad)
                    s = 1 - np.mean(b[:, i])/u

                    w = v.T @ s
                else:
                    i = np.random.randint(N)
                    v = A[:, :, i]

                    i = np.random.randint(N)
                    s = 1 - b[:, i]/u

                    w = v.T @ s
                # solve the subproblem
                y = cp.Variable(n)
                constraints = [y >= 0]

                grad_hf = - 1/u
                hf_linear = - cp.sum(cp.log(u + v@(y-x)))

                sub_loss = (tau * w - Lf * v.T @ grad_hf) @ y + \
                    Lf * hf_linear + lmbda / 2 * cp.sum_squares(y-x)

                obj = cp.Minimize(sub_loss)
                prob = cp.Problem(obj, constraints)
                prob.solve()  # return the result of subproblem

                # save the result
                x = y.value

                x_history[:, iter] = x
                x_out[:, iter] = np.mean(x_history[:, 0:iter+1], 1)

                f[iter] = np.sum(
                    b_avg * np.log(b_avg/(A_avg @ x_out[:, iter])) + A_avg @ x_out[:, iter] - b_avg)

            Dh1 = np.zeros(iter_max)
            Dh2 = np.zeros(iter_max)
            Dh1_avg = np.zeros(iter_max-1)
            Dh2_avg = np.zeros(iter_max-1)
            for iter in range(iter_max-1):
                x1 = x_history[:, iter]
                x2 = x_history[:, iter+1]
                Dh2[iter] = - np.sum(np.log(A_avg @ x2)) + np.sum(
                    np.log(A_avg @ x1)) + 1/(A_avg@x1) @ (A_avg @ (x2-x1))
                Dh1[iter] = 1/2 * np.linalg.norm(x2-x1)**2

                Dh1_avg[iter] = np.mean(Dh1[0:iter+1])
                Dh2_avg[iter] = np.mean(Dh2[0:iter+1])
            # %% Plot
            fig = plt.figure()
            plt.figure(1)
            plt.subplot(311)
            plt.plot(f)
            plt.ylabel('f(iter)')
            plt.yscale("log")
            plt.subplot(312)
            plt.plot(Dh1_avg)
            plt.yscale("log")
            plt.subplot(313)
            plt.plot(Dh2_avg)
            plt.ylabel('||grad||')
            plt.yscale("log")
            fig.suptitle('tau = %.2f (%i), lmbda = %.2f (%i), batch = %i' %
                         (tau, grid_1, lmbda, grid_2, batch_grad))
            plt.show()
        except:
            pass
