#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:29:55 2021

@author: yin_liu
"""
# %%

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "ps.usedistiller": "xpdf"})


oracles = 30000
n_batch = 100

iter_max = oracles//n_batch

exp_num = 20


n100_x_norm2 = np.zeros([iter_max-1, exp_num])
n100_x_norm4 = np.zeros([iter_max-1, exp_num])
n100_x_hat_norm2 = np.zeros([iter_max-1, exp_num])
n100_x_hat_norm4 = np.zeros([iter_max-1, exp_num])
for repeat in range(exp_num):
    filename = 'Result/SoR_batch_' + \
        str(int(n_batch)) + 'repeat_' + str(int(repeat)) + '.npz'
    npzfile = np.load(filename)

    n100_x_norm2[:, repeat] = npzfile['x_norm2_avg_BG']
    n100_x_norm4[:, repeat] = npzfile['x_norm4_avg_BG']
    n100_x_hat_norm2[:, repeat] = npzfile['x_hat_norm2_avg_BG']
    n100_x_hat_norm4[:, repeat] = npzfile['x_hat_norm4_avg_BG']


n100_norm2_mean = np.mean(n100_x_norm2, 1)
n100_norm4_mean = np.mean(n100_x_norm4, 1)
n100_hat_norm2_mean = np.mean(n100_x_hat_norm2, 1)
n100_hat_norm4_mean = np.mean(n100_x_hat_norm4, 1)

# %%
n_batch = 10

iter_max = oracles//n_batch

n10_x_norm2 = np.zeros([iter_max-1, exp_num])
n10_x_norm4 = np.zeros([iter_max-1, exp_num])
n10_x_hat_norm2 = np.zeros([iter_max-1, exp_num])
n10_x_hat_norm4 = np.zeros([iter_max-1, exp_num])
for repeat in range(exp_num):
    filename = 'Result/SoR_batch_' + \
        str(int(n_batch)) + 'repeat_' + str(int(repeat)) + '.npz'
    npzfile = np.load(filename)

    n10_x_norm2[:, repeat] = npzfile['x_norm2_avg_BG']
    n10_x_norm4[:, repeat] = npzfile['x_norm4_avg_BG']
    n10_x_hat_norm2[:, repeat] = npzfile['x_hat_norm2_avg_BG']
    n10_x_hat_norm4[:, repeat] = npzfile['x_hat_norm4_avg_BG']


n10_norm2_mean = np.mean(n10_x_norm2, 1)
n10_norm4_mean = np.mean(n10_x_norm4, 1)
n10_hat_norm2_mean = np.mean(n10_x_hat_norm2, 1)
n10_hat_norm4_mean = np.mean(n10_x_hat_norm4, 1)


# %%
n_batch = 1

iter_max = oracles//n_batch

n1_x_norm2 = np.zeros([iter_max-1, exp_num])
n1_x_norm4 = np.zeros([iter_max-1, exp_num])
n1_x_hat_norm2 = np.zeros([iter_max-1, exp_num])
n1_x_hat_norm4 = np.zeros([iter_max-1, exp_num])
for repeat in range(exp_num):
    filename = 'Result/SoR_batch_' + \
        str(int(n_batch)) + 'repeat_' + str(int(repeat)) + '.npz'
    npzfile = np.load(filename)

    n1_x_norm2[:, repeat] = npzfile['x_norm2_avg_BG']
    n1_x_norm4[:, repeat] = npzfile['x_norm4_avg_BG']
    n1_x_hat_norm2[:, repeat] = npzfile['x_hat_norm2_avg_BG']
    n1_x_hat_norm4[:, repeat] = npzfile['x_hat_norm4_avg_BG']


n1_norm2_mean = np.mean(n1_x_norm2, 1)
n1_norm4_mean = np.mean(n1_x_norm4, 1)
n1_hat_norm2_mean = np.mean(n1_x_hat_norm2, 1)
n1_hat_norm4_mean = np.mean(n1_x_hat_norm4, 1)


# %% plot based on iteration number （x)
xrange = range(299)

fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(xrange, n100_norm2_mean, color='C0', label=r"$n=100$")
ax1.plot(xrange, n10_norm2_mean[xrange], color='C1', label=r"$n=10$")
ax1.plot(xrange, n1_norm2_mean[xrange], color='C2', label=r"$n=1$")

for i in range(exp_num):
    ax1.fill_between(
        xrange, n100_x_norm2[xrange, i], n100_norm2_mean[xrange], color='C0', alpha=0.07)
    ax1.fill_between(
        xrange, n10_x_norm2[xrange, i], n10_norm2_mean[xrange], color='C1', alpha=0.07)
    ax1.fill_between(
        xrange, n1_x_norm2[xrange, i], n1_norm2_mean[xrange], color='C2', alpha=0.07)

ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/SoR_norm2_iteration.pdf', bbox_inches='tight')


fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(xrange, n100_norm4_mean[xrange], color='C0')
ax2.plot(xrange, n10_norm4_mean[xrange], color='C1')
ax2.plot(xrange, n1_norm4_mean[xrange], color='C2')

for i in range(exp_num):
    ax2.fill_between(
        xrange, n100_x_norm4[xrange, i], n100_norm4_mean[xrange], color='C0', alpha=0.07)
    ax2.fill_between(
        xrange, n10_x_norm4[xrange, i], n10_norm4_mean[xrange], color='C1', alpha=0.07)
    ax2.fill_between(
        xrange, n1_x_norm4[xrange, i], n1_norm4_mean[xrange], color='C2', alpha=0.07)


ax2.set_yscale("log")
ax2.set_xlabel('iteration')

ax2.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/SoR_norm4_iteration.pdf', bbox_inches='tight')


# %% plot based on oracles


fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(range(0, 100*299, 100), n100_norm2_mean,
         color='C0', label=r"$n= 100$")
ax1.plot(range(0, 10*2999, 10), n10_norm2_mean, color='C1', label=r"$n= 10$")
ax1.plot(range(0, 1*29999, 10),
         n1_norm2_mean[range(0, 1*29999, 10)], color='C2', label=r"$n= 1$")

for i in range(exp_num):
    ax1.fill_between(range(0, 100*299, 100),
                     n100_x_norm2[:, i], n100_norm2_mean, color='C0', alpha=0.07)
    ax1.fill_between(range(0, 10*2999, 10),
                     n10_x_norm2[:, i], n10_norm2_mean, color='C1', alpha=0.07)
    ax1.fill_between(range(0, 1*29999, 30), n1_x_norm2[range(
        0, 1*29999, 30), i], n1_norm2_mean[range(0, 1*29999, 30)], color='C2', alpha=0.07)

ax1.set_xlabel('\# of gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/SoR_norm2_grad_oracle.pdf', bbox_inches='tight')


##
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(range(0, 100*299, 100), n100_norm4_mean,
         color='C0', label="batch 100")
ax2.plot(range(0, 10*2999, 10), n10_norm4_mean, color='C1', label="batch 10")
ax2.plot(range(0, 1*29999, 1), n1_norm4_mean, color='C2', label="batch 1")

for i in range(exp_num):
    ax2.fill_between(range(0, 100*299, 100),
                     n100_x_norm4[:, i], n100_norm4_mean, color='C0', alpha=0.07)
    ax2.fill_between(range(0, 10*2999, 10),
                     n10_x_norm4[:, i], n10_norm4_mean, color='C1', alpha=0.07)
    ax2.fill_between(range(0, 1*29999, 30), n1_x_norm4[range(
        0, 1*29999, 30), i], n1_norm4_mean[range(0, 1*29999, 30)], color='C2', alpha=0.07)


ax2.set_yscale("log")
ax2.set_xlabel('\# of gradient oracles')

ax2.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/SoR_norm4_grad_oracle.pdf', bbox_inches='tight')


########## based on x_hat #######################################################


# %% plot based on iteration number
xrange = range(299)

fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(xrange, n100_hat_norm2_mean, color='C0', marker='s', markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=100$")
ax1.plot(xrange, n10_hat_norm2_mean[xrange], marker='v', markevery=30,
         color='C1', label=r"$|\mathcal{B}_{\nabla}|=10$")
ax1.plot(xrange, n1_hat_norm2_mean[xrange], marker='o', markevery=30,
         color='C2', label=r"$|\mathcal{B}_{\nabla}|=1$")

for i in range(exp_num):
    ax1.fill_between(
        xrange, n100_x_hat_norm2[xrange, i], n100_hat_norm2_mean[xrange], color='C0', alpha=0.07)
    ax1.fill_between(
        xrange, n10_x_hat_norm2[xrange, i], n10_hat_norm2_mean[xrange], color='C1', alpha=0.07)
    ax1.fill_between(
        xrange, n1_x_hat_norm2[xrange, i], n1_hat_norm2_mean[xrange], color='C2', alpha=0.07)

ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(\hat{x}^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm2_iteration.pdf', bbox_inches='tight')


fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(xrange, n100_hat_norm4_mean[xrange],
         color='C0', marker='s', markevery=30, label=r"$|\mathcal{B}_{\nabla}|=100$")
ax2.plot(xrange, n10_hat_norm4_mean[xrange],
         color='C1', marker='v', markevery=30, label=r"$|\mathcal{B}_{\nabla}|=10$")
ax2.plot(xrange, n1_hat_norm4_mean[xrange],
         color='C2', marker='o', markevery=30, label=r"$|\mathcal{B}_{\nabla}|=1$")

for i in range(exp_num):
    ax2.fill_between(
        xrange, n100_x_hat_norm4[xrange, i], n100_hat_norm4_mean[xrange], color='C0', alpha=0.07)
    ax2.fill_between(
        xrange, n10_x_hat_norm4[xrange, i], n10_hat_norm4_mean[xrange], color='C1', alpha=0.07)
    ax2.fill_between(
        xrange, n1_x_hat_norm4[xrange, i], n1_hat_norm4_mean[xrange], color='C2', alpha=0.07)


ax2.set_yscale("log")
ax2.set_xlabel('iteration')

ax2.set_ylabel(r"$E[D_{h_2}(\hat{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm4_iteration.pdf', bbox_inches='tight')


# %% plot based on oracles


fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(range(0, 100*299, 100), n100_hat_norm2_mean,
         color='C0', marker='s', markevery=30, label=r"$|\mathcal{B}_{\nabla}|= 100$")
ax1.plot(range(0, 10*2999, 10), n10_hat_norm2_mean, color='C1',
         marker='v', markevery=300, label=r"$|\mathcal{B}_{\nabla}|= 10$")
ax1.plot(range(0, 1*29999, 1),
         n1_hat_norm2_mean, color='C2', marker='o', markevery=3000, label=r"$|\mathcal{B}_{\nabla}|= 1$")

for i in range(exp_num):
    ax1.fill_between(range(0, 100*299, 100),
                     n100_x_hat_norm2[:, i], n100_hat_norm2_mean, color='C0', alpha=0.07)
    ax1.fill_between(range(0, 10*2999, 10),
                     n10_x_hat_norm2[:, i], n10_hat_norm2_mean, color='C1', alpha=0.07)
    ax1.fill_between(range(0, 1*29999, 30), n1_x_hat_norm2[range(
        0, 1*29999, 30), i], n1_hat_norm2_mean[range(0, 1*29999, 30)], color='C2', alpha=0.07)

ax1.set_xlabel('\# of gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(\hat{x}^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm2_grad_oracle.pdf', bbox_inches='tight')


##
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(range(0, 100*299, 100), n100_hat_norm4_mean,
         color='C0', marker='s', markevery=30, label=r"$|\mathcal{B}_{\nabla}|= 100$")
ax2.plot(range(0, 10*2999, 10), n10_hat_norm4_mean, color='C1',
         marker='v', markevery=300, label=r"$|\mathcal{B}_{\nabla}|= 10$")
ax2.plot(range(0, 1*29999, 1), n1_hat_norm4_mean, color='C2',
         marker='o', markevery=3000, label=r"$|\mathcal{B}_{\nabla}|= 1$")

for i in range(exp_num):
    ax2.fill_between(range(0, 100*299, 100),
                     n100_x_hat_norm4[:, i], n100_hat_norm4_mean, color='C0', alpha=0.07)
    ax2.fill_between(range(0, 10*2999, 10),
                     n10_x_hat_norm4[:, i], n10_hat_norm4_mean, color='C1', alpha=0.07)
    ax2.fill_between(range(0, 1*29999, 30), n1_x_hat_norm4[range(
        0, 1*29999, 30), i], n1_hat_norm4_mean[range(0, 1*29999, 30)], color='C2', alpha=0.07)


ax2.set_yscale("log")
ax2.set_xlabel('\# of gradient oracles')

ax2.set_ylabel(r"$E[D_{h_2}(\hat{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm4_grad_oracle.pdf', bbox_inches='tight')


# %% Plot k1*h1+k2*h2 iteration number

k1 = 100
k2 = 0.89

n100_hat_norm_mean = k1*n100_hat_norm2_mean + k2*n100_hat_norm4_mean
n10_hat_norm_mean = k1*n10_hat_norm2_mean + k2*n10_hat_norm4_mean
n1_hat_norm_mean = k1*n1_hat_norm2_mean + k2*n1_hat_norm4_mean

n100_x_hat_norm = k1*n100_x_hat_norm2+k2*n100_x_hat_norm4
n10_x_hat_norm = k1*n10_x_hat_norm2+k2*n10_x_hat_norm4
n1_x_hat_norm = k1*n1_x_hat_norm2+k2*n1_x_hat_norm4

xrange = range(299)

fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(xrange, n100_hat_norm_mean, color='C0', marker='s', markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=100$")
ax1.plot(xrange, n10_hat_norm_mean[xrange], marker='v', markevery=30,
         color='C1', label=r"$|\mathcal{B}_{\nabla}|=10$")
ax1.plot(xrange, n1_hat_norm_mean[xrange], marker='o', markevery=30,
         color='C2', label=r"$|\mathcal{B}_{\nabla}|=1$")

for i in range(exp_num):
    ax1.fill_between(
        xrange, n100_x_hat_norm[xrange, i], n100_hat_norm_mean[xrange], color='C0', alpha=0.07)
    ax1.fill_between(
        xrange, n10_x_hat_norm[xrange, i], n10_hat_norm_mean[xrange], color='C1', alpha=0.07)
    ax1.fill_between(
        xrange, n1_x_hat_norm[xrange, i], n1_hat_norm_mean[xrange], color='C2', alpha=0.07)

ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h}(\hat{x}^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm_iteration.pdf', bbox_inches='tight')

# %% Plot k1*h1+k2*h2 oracles
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(range(0, 100*299, 100), n100_hat_norm_mean,
         color='C0', marker='s', markevery=30, label=r"$|\mathcal{B}_{\nabla}|= 100$")
ax2.plot(range(0, 10*2999, 10), n10_hat_norm_mean, color='C1',
         marker='v', markevery=300, label=r"$|\mathcal{B}_{\nabla}|= 10$")
ax2.plot(range(0, 1*29999, 1), n1_hat_norm_mean, color='C2',
         marker='o', markevery=3000, label=r"$|\mathcal{B}_{\nabla}|= 1$")

for i in range(exp_num):
    ax2.fill_between(range(0, 100*299, 100),
                     n100_x_hat_norm[:, i], n100_hat_norm_mean, color='C0', alpha=0.07)
    ax2.fill_between(range(0, 10*2999, 10),
                     n10_x_hat_norm[:, i], n10_hat_norm_mean, color='C1', alpha=0.07)
    ax2.fill_between(range(0, 1*29999, 30), n1_x_hat_norm[range(
        0, 1*29999, 30), i], n1_hat_norm_mean[range(0, 1*29999, 30)], color='C2', alpha=0.07)


ax2.set_yscale("log")
ax2.set_xlabel('\# of gradient oracles')

ax2.set_ylabel(r"$E[D_{h}(\hat{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/SoR_hat_norm_grad_oracle.pdf', bbox_inches='tight')

# %%
