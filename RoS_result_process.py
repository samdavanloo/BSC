#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 19:16:45 2021

@author: yin_liu
plot the result for RoS/RoR method in the paper
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

# n100
batch_grad = 100

iter_max = oracles//batch_grad

n100_Dh1 = np.zeros([iter_max-1, 20])
n100_Dh2 = np.zeros([iter_max-1, 20])
n100_Dh1_avg = np.zeros([iter_max-1, 20])
n100_Dh2_avg = np.zeros([iter_max-1, 20])

n100_Dh1_tilde = np.zeros([iter_max-1, 20])
n100_Dh2_tilde = np.zeros([iter_max-1, 20])
n100_Dh1_tilde_avg = np.zeros([iter_max-1, 20])
n100_Dh2_tilde_avg = np.zeros([iter_max-1, 20])

for repeat in range(20):

    filename = 'Result/RoS_batch_' + \
        str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'

    npzfile = np.load(filename)
    n100_Dh1[:, repeat] = npzfile['Dh1']
    n100_Dh2[:, repeat] = npzfile['Dh2']
    n100_Dh1_tilde[:, repeat] = npzfile['Dh1_tilde']
    n100_Dh2_tilde[:, repeat] = npzfile['Dh2_tilde']

    n100_Dh1_avg[:, repeat] = npzfile['Dh1_avg']
    n100_Dh2_avg[:, repeat] = npzfile['Dh2_avg']
    n100_Dh1_tilde_avg[:, repeat] = npzfile['Dh1_tilde_avg']
    n100_Dh2_tilde_avg[:, repeat] = npzfile['Dh2_tilde_avg']


n100_oracles = npzfile['oracle_innervalue']
n100_oracles = n100_oracles[:-1]
n100_oracles_grad = npzfile['oracle_innergrad']
n100_oracles_grad = n100_oracles_grad[:-1]


repeat = 0
filename = 'Result/RoS_SP_batch_' + \
    str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'


npzfile = np.load(filename)
temp = npzfile['Dh1']
xlen = len(temp)
n100_Dh1_SP = np.zeros([xlen, 20])
n100_Dh2_SP = np.zeros([xlen, 20])
n100_Dh1_SP_avg = np.zeros([xlen, 20])
n100_Dh2_SP_avg = np.zeros([xlen, 20])
n100_Dh1_SP_tilde = np.zeros([xlen, 20])
n100_Dh2_SP_tilde = np.zeros([xlen, 20])
n100_Dh1_SP_tilde_avg = np.zeros([xlen, 20])
n100_Dh2_SP_tilde_avg = np.zeros([xlen, 20])


for repeat in range(20):

    filename = 'Result/RoS_SP_batch_' + \
        str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'

    npzfile = np.load(filename)
    n100_Dh1_SP[:, repeat] = npzfile['Dh1']
    n100_Dh2_SP[:, repeat] = npzfile['Dh2']
    n100_Dh1_SP_tilde[:, repeat] = npzfile['Dh1_tilde']
    n100_Dh2_SP_tilde[:, repeat] = npzfile['Dh2_tilde']

    n100_Dh1_SP_avg[:, repeat] = npzfile['Dh1_avg']
    n100_Dh2_SP_avg[:, repeat] = npzfile['Dh2_avg']
    n100_Dh1_SP_tilde_avg[:, repeat] = npzfile['Dh1_tilde_avg']
    n100_Dh2_SP_tilde_avg[:, repeat] = npzfile['Dh2_tilde_avg']

n100_SP_oracles = npzfile['oracle_innervalue']
n100_SP_oracles = n100_SP_oracles[:-1]
n100_SP_oracles_grad = npzfile['oracle_innergrad']
n100_SP_oracles_grad = n100_SP_oracles_grad[:-1]


# n20, still use n10 name
batch_grad = 20

iter_max = oracles//batch_grad

n10_Dh1 = np.zeros([iter_max-1, 20])
n10_Dh2 = np.zeros([iter_max-1, 20])
n10_Dh1_avg = np.zeros([iter_max-1, 20])
n10_Dh2_avg = np.zeros([iter_max-1, 20])

n10_Dh1_tilde = np.zeros([iter_max-1, 20])
n10_Dh2_tilde = np.zeros([iter_max-1, 20])
n10_Dh1_tilde_avg = np.zeros([iter_max-1, 20])
n10_Dh2_tilde_avg = np.zeros([iter_max-1, 20])

for repeat in range(20):

    filename = 'Result/RoS_batch_' + \
        str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'

    npzfile = np.load(filename)
    n10_Dh1[:, repeat] = npzfile['Dh1']
    n10_Dh2[:, repeat] = npzfile['Dh2']
    n10_Dh1_tilde[:, repeat] = npzfile['Dh1_tilde']
    n10_Dh2_tilde[:, repeat] = npzfile['Dh2_tilde']

    n10_Dh1_avg[:, repeat] = npzfile['Dh1_avg']
    n10_Dh2_avg[:, repeat] = npzfile['Dh2_avg']
    n10_Dh1_tilde_avg[:, repeat] = npzfile['Dh1_tilde_avg']
    n10_Dh2_tilde_avg[:, repeat] = npzfile['Dh2_tilde_avg']

n10_oracles = npzfile['oracle_innervalue']
n10_oracles = n10_oracles[:-1]
n10_oracles_grad = npzfile['oracle_innergrad']
n10_oracles_grad = n10_oracles_grad[:-1]

repeat = 0
filename = 'Result/RoS_SP_batch_' + \
    str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'
npzfile = np.load(filename)
temp = npzfile['Dh1']
xlen = len(temp)


n10_Dh1_SP = np.zeros([xlen, 20])
n10_Dh2_SP = np.zeros([xlen, 20])
n10_Dh1_SP_avg = np.zeros([xlen, 20])
n10_Dh2_SP_avg = np.zeros([xlen, 20])
n10_Dh1_SP_tilde = np.zeros([xlen, 20])
n10_Dh2_SP_tilde = np.zeros([xlen, 20])
n10_Dh1_SP_tilde_avg = np.zeros([xlen, 20])
n10_Dh2_SP_tilde_avg = np.zeros([xlen, 20])


n10_SP_oracles = npzfile['oracle_innervalue']
n10_SP_oracles = n10_SP_oracles[:-1]
n10_SP_oracles_grad = npzfile['oracle_innergrad']
n10_SP_oracles_grad = n10_SP_oracles_grad[:-1]


for repeat in range(20):

    filename = 'Result/RoS_SP_batch_' + \
        str(int(batch_grad)) + 'repeat_' + str(int(repeat)) + '.npz'

    npzfile = np.load(filename)
    n10_Dh1_SP[:, repeat] = npzfile['Dh1']
    n10_Dh2_SP[:, repeat] = npzfile['Dh2']
    n10_Dh1_SP_tilde[:, repeat] = npzfile['Dh1_tilde']
    n10_Dh2_SP_tilde[:, repeat] = npzfile['Dh2_tilde']

    n10_Dh1_SP_avg[:, repeat] = npzfile['Dh1_avg']
    n10_Dh2_SP_avg[:, repeat] = npzfile['Dh2_avg']
    n10_Dh1_SP_tilde_avg[:, repeat] = npzfile['Dh1_tilde_avg']
    n10_Dh2_SP_tilde_avg[:, repeat] = npzfile['Dh2_tilde_avg']


# %% Plot by iteration (based on x)
fig, ax1 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh1_avg, 0))

ax1.plot(xrange, np.mean(n100_Dh1_avg, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(xrange, np.mean(n100_Dh1_SP_avg[xrange], 1), color='C0', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(xrange, np.mean(n10_Dh1_avg[xrange], 1), color='C1', marker="o",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(xrange, np.mean(n10_Dh1_SP_avg[xrange], 1), color='C1', marker="s", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(xrange, n100_Dh1_avg[xrange, i], np.mean(
        n100_Dh1_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh1_avg[xrange, i], np.mean(
        n10_Dh1_avg[xrange, :], 1), color='C2', alpha=0.08)
    ax1.fill_between(xrange, n100_Dh1_SP_avg[xrange, i], np.mean(
        n100_Dh1_SP_avg[xrange, :], 1), color='C1', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh1_SP_avg[xrange, i], np.mean(
        n10_Dh1_SP_avg[xrange, :], 1), color='C3', alpha=0.08)


ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_iteration.pdf', bbox_inches='tight')


# %% Plot by iteration (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh1_tilde_avg, 0))

ax2.plot(xrange, np.mean(n100_Dh1_tilde_avg, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(xrange, np.mean(n100_Dh1_SP_tilde_avg[xrange, :], 1), color='C0', marker="s", markevery=50,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(xrange, np.mean(n10_Dh1_tilde_avg[xrange, :], 1), linestyle='dashed', color='C1', marker="o",
         markevery=30, markerfacecolor='none', label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")

ax2.plot(xrange, np.mean(n10_Dh1_SP_tilde_avg[xrange, :], 1), linestyle='dashed', color='C1', marker="s", markevery=70, markerfacecolor='none',
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(xrange, n100_Dh1_tilde_avg[xrange, i], np.mean(
        n100_Dh1_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh1_tilde_avg[xrange, i], np.mean(
        n10_Dh1_tilde_avg[xrange, :], 1), color='C2', alpha=0.08)
    ax2.fill_between(xrange, n100_Dh1_SP_tilde_avg[xrange, i], np.mean(
        n100_Dh1_SP_tilde_avg[xrange, :], 1), color='C1', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh1_SP_tilde_avg[xrange, i], np.mean(
        n10_Dh1_SP_tilde_avg[xrange, :], 1), color='C3', alpha=0.08)


ax2.set_xlabel('iteration')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[\|\tilde{x}^{R+1}-x^R\|^2]$")
ax2.grid(True, alpha=0.5)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/RoS_h1_tilde_iteration.pdf', bbox_inches='tight')


# %% Plot by g oracles (x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles, np.mean(n100_Dh1_avg, 1), color='C0', marker="o",
         markevery=5, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles, np.mean(n100_Dh1_SP_avg, 1), color='C0', marker="s", markevery=40,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(n10_oracles, np.mean(n10_Dh1_avg, 1), linestyle='dashed', color='C1', marker="o",
         markevery=100, markerfacecolor='none', label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles, np.mean(n10_Dh1_SP_avg, 1), linestyle='dashed', color='C1', marker="s", markevery=500, markerfacecolor='none',
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles, n100_Dh1_avg[:, i], np.mean(
        n100_Dh1_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_oracles, n10_Dh1_avg[:, i], np.mean(
        n10_Dh1_avg, 1), color='C1', alpha=0.08)
    ax1.fill_between(n100_SP_oracles, n100_Dh1_SP_avg[:, i], np.mean(
        n100_Dh1_SP_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_SP_oracles, n10_Dh1_SP_avg[:, i], np.mean(
        n10_Dh1_SP_avg, 1), color='C1', alpha=0.08)


ax1.set_xlabel('\# of inner function value oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 200000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_oracle.pdf', bbox_inches='tight')


# %% Plot by g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles, np.mean(n100_Dh1_tilde_avg, 1), color='C0', marker="o",
         markevery=5, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles, np.mean(n100_Dh1_SP_tilde_avg, 1), color='C0', marker="s", markevery=60,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles, np.mean(n10_Dh1_tilde_avg, 1), linestyle='dashed', color='C1', marker="o",
         markevery=150, markerfacecolor='none', label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles, np.mean(n10_Dh1_SP_tilde_avg, 1), linestyle='dashed', color='C1', marker="s", markevery=500, markerfacecolor='none',
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles, n100_Dh1_tilde_avg[:, i], np.mean(
        n100_Dh1_tilde_avg, 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(n10_oracles, n10_Dh1_tilde_avg[:, i], np.mean(
        n10_Dh1_tilde_avg, 1), color='C1', edgecolor='none', alpha=0.08)
    ax2.fill_between(n100_SP_oracles, n100_Dh1_SP_tilde_avg[:, i], np.mean(
        n100_Dh1_SP_tilde_avg, 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(n10_SP_oracles, n10_Dh1_SP_tilde_avg[:, i], np.mean(
        n10_Dh1_SP_tilde_avg, 1), color='C1', edgecolor='none', alpha=0.08)


ax2.set_xlabel('\# of inner function value oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[\|\tilde{x}^{R+1}-x^R\|^2]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 200000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/RoS_h1_tilde_oracle.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles_grad, np.mean(n100_Dh1_avg, 1), color='C0', marker="o",
         markevery=100, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles_grad, np.mean(n100_Dh1_SP_avg, 1), color='C0', marker="s", markevery=100,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(n10_oracles_grad, np.mean(n10_Dh1_avg, 1), color='C1', marker="o",
         markevery=200, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles_grad, np.mean(n10_Dh1_SP_avg, 1), color='C1', marker="s", markevery=500,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles_grad, n100_Dh1_avg[:, i], np.mean(
        n100_Dh1_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_oracles_grad, n10_Dh1_avg[:, i], np.mean(
        n10_Dh1_avg, 1), color='C1', alpha=0.08)
    ax1.fill_between(n100_SP_oracles_grad, n100_Dh1_SP_avg[:, i], np.mean(
        n100_Dh1_SP_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_SP_oracles_grad, n10_Dh1_SP_avg[:, i], np.mean(
        n10_Dh1_SP_avg, 1), color='C1', alpha=0.08)


ax1.set_xlabel('\# of inner function gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 30000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_oracl_grad.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles_grad, np.mean(n100_Dh1_tilde_avg, 1), color='C0', marker="o",
         markevery=100, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles_grad, np.mean(n100_Dh1_SP_tilde_avg, 1), color='C0', marker="s", markevery=100,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles_grad, np.mean(n10_Dh1_tilde_avg, 1), linestyle='dashed', color='C1', marker="o", markevery=400,
         markerfacecolor='none', label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles_grad, np.mean(n10_Dh1_SP_tilde_avg, 1), linestyle='dashed', color='C1', marker="s", markevery=600, markerfacecolor='none',
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles_grad, n100_Dh1_tilde_avg[:, i], np.mean(
        n100_Dh1_tilde_avg, 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(n10_oracles_grad, n10_Dh1_tilde_avg[:, i], np.mean(
        n10_Dh1_tilde_avg, 1), color='C1', edgecolor='none', alpha=0.08)
    ax2.fill_between(n100_SP_oracles_grad, n100_Dh1_SP_tilde_avg[:, i], np.mean(
        n100_Dh1_SP_tilde_avg, 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(n10_SP_oracles_grad, n10_Dh1_SP_tilde_avg[:, i], np.mean(
        n10_Dh1_SP_tilde_avg, 1), color='C1', edgecolor='none', alpha=0.08)


ax2.set_xlabel('\# of inner function gradient oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[\|\tilde{x}^{R+1}-x^R\|^2]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 30000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/RoS_h1_tilde_oracle_grad.pdf', bbox_inches='tight')

# %% Plot by iteration, noise (based on x)
fig, ax1 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh1, 0))

ax1.plot(xrange, np.mean(n100_Dh1, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(xrange, np.mean(n100_Dh1_SP[xrange], 1), color='C1', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(xrange, np.mean(n10_Dh1[xrange], 1), color='C2', marker="o",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(xrange, np.mean(n10_Dh1_SP[xrange], 1), color='C3', marker="s", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(xrange, n100_Dh1[xrange, i], np.mean(
        n100_Dh1, 1), color='C0', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh1[xrange, i], np.mean(
        n10_Dh1[xrange, :], 1), color='C2', alpha=0.08)
    ax1.fill_between(xrange, n100_Dh1_SP[xrange, i], np.mean(
        n100_Dh1_SP[xrange, :], 1), color='C1', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh1_SP[xrange, i], np.mean(
        n10_Dh1_SP[xrange, :], 1), color='C3', alpha=0.08)


ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')


plt.savefig('Figures/RoS_h1_iteration_noise.pdf', bbox_inches='tight')


# %% Plot by iteration, noise (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh1_tilde, 0))

ax2.plot(xrange, np.mean(n100_Dh1_tilde, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(xrange, np.mean(n100_Dh1_SP_tilde[xrange, :], 1), color='C0', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax2.plot(xrange, np.mean(n10_Dh1_tilde[xrange, :], 1), linestyle='dashed', color='C1', marker="o", markevery=30,
         markerfacecolor='none', label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")

ax2.plot(xrange, np.mean(n10_Dh1_SP_tilde[xrange, :], 1), linestyle='dashed', color='C1', marker="s", markevery=30, markerfacecolor='none',
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(xrange, n100_Dh1_tilde[xrange, i], np.mean(
        n100_Dh1_tilde, 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh1_tilde[xrange, i], np.mean(
        n10_Dh1_tilde[xrange, :], 1), color='C1', edgecolor='none', alpha=0.08)
    ax2.fill_between(xrange, n100_Dh1_SP_tilde[xrange, i], np.mean(
        n100_Dh1_SP_tilde[xrange, :], 1), color='C0', edgecolor='none', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh1_SP_tilde[xrange, i], np.mean(
        n10_Dh1_SP_tilde[xrange, :], 1), color='C1', edgecolor='none', alpha=0.08)


ax2.set_xlabel('iteration')
ax2.set_yscale("log")
ax2.set_ylabel(r"$\|\tilde{x}^{R+1}-x^R\|^2$")
ax2.grid(True, alpha=0.5)


ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_tilde_iteration_noise.pdf', bbox_inches='tight')


# %% Plot by g oracles noise(x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles, np.mean(n100_Dh1, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles, np.mean(n100_Dh1_SP, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(n10_oracles, np.mean(n10_Dh1, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles, np.mean(n10_Dh1_SP, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles, n100_Dh1[:, i], np.mean(
        n100_Dh1, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_oracles, n10_Dh1[:, i], np.mean(
        n10_Dh1, 1), color='C2', alpha=0.08)
    ax1.fill_between(n100_SP_oracles, n100_Dh1_SP[:, i], np.mean(
        n100_Dh1_SP, 1), color='C1', alpha=0.08)
    ax1.fill_between(n10_SP_oracles, n10_Dh1_SP[:, i], np.mean(
        n10_Dh1_SP, 1), color='C3', alpha=0.08)


ax1.set_xlabel('\# of inner function value oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 60000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_oracle_noise.pdf', bbox_inches='tight')


# %% Plot by g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles, np.mean(n100_Dh1_tilde, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles, np.mean(n100_Dh1_SP_tilde, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles, np.mean(n10_Dh1_tilde, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles, np.mean(n10_Dh1_SP_tilde, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles, n100_Dh1_tilde[:, i], np.mean(
        n100_Dh1_tilde, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_oracles, n10_Dh1_tilde[:, i], np.mean(
        n10_Dh1_tilde, 1), color='C2', alpha=0.08)
    ax2.fill_between(n100_SP_oracles, n100_Dh1_SP_tilde[:, i], np.mean(
        n100_Dh1_SP_tilde, 1), color='C1', alpha=0.08)
    ax2.fill_between(n10_SP_oracles, n10_Dh1_SP_tilde[:, i], np.mean(
        n10_Dh1_SP_tilde, 1), color='C3', alpha=0.08)


ax2.set_xlabel('\# of inner function value oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[\|\tilde{x}^{R+1},x^R\|^2]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 60000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_tilde_oracle_noise.pdf', bbox_inches='tight')


# %% Plot by grad g oracles noise(x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles_grad, np.mean(n100_Dh1, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles_grad, np.mean(n100_Dh1_SP, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(n10_oracles_grad, np.mean(n10_Dh1, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles_grad, np.mean(n10_Dh1_SP, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles_grad, n100_Dh1[:, i], np.mean(
        n100_Dh1, 1), color='C0', alpha=0.05)
    ax1.fill_between(n10_oracles_grad, n10_Dh1[:, i], np.mean(
        n10_Dh1, 1), color='C2', alpha=0.05)
    ax1.fill_between(n100_SP_oracles_grad, n100_Dh1_SP[:, i], np.mean(
        n100_Dh1_SP, 1), color='C1', alpha=0.05)
    ax1.fill_between(n10_SP_oracles_grad, n10_Dh1_SP[:, i], np.mean(
        n10_Dh1_SP, 1), color='C3', alpha=0.05)


ax1.set_xlabel('\# of inner function gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_1}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 10000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_oracle_grad_noise.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles_grad, np.mean(n100_Dh1_tilde, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles_grad, np.mean(n100_Dh1_SP_tilde, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles_grad, np.mean(n10_Dh1_tilde, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles_grad, np.mean(n10_Dh1_SP_tilde, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles_grad, n100_Dh1_tilde[:, i], np.mean(
        n100_Dh1_tilde, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_oracles_grad, n10_Dh1_tilde[:, i], np.mean(
        n10_Dh1_tilde, 1), color='C2', alpha=0.08)
    ax2.fill_between(n100_SP_oracles_grad, n100_Dh1_SP_tilde[:, i], np.mean(
        n100_Dh1_SP_tilde, 1), color='C1', alpha=0.08)
    ax2.fill_between(n10_SP_oracles_grad, n10_Dh1_SP_tilde[:, i], np.mean(
        n10_Dh1_SP_tilde, 1), color='C3', alpha=0.08)


ax2.set_xlabel('\# of inner function gradient oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[\|\tilde{x}^{R+1},x^R\|^2]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 10000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h1_tilde_oracle_grad_noise.pdf', bbox_inches='tight')


###################################################################################
# h2
# %% Plot by iteration (based on x)
fig, ax1 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh2_avg, 0))

ax1.plot(xrange, np.mean(n100_Dh2_avg, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(xrange, np.mean(n100_Dh2_SP_avg[xrange], 1), color='C0', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(xrange, np.mean(n10_Dh2_avg[xrange], 1), color='C1', marker="o",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(xrange, np.mean(n10_Dh2_SP_avg[xrange], 1), color='C1', marker="s", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(xrange, n100_Dh2_avg[xrange, i], np.mean(
        n100_Dh2_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh2_avg[xrange, i], np.mean(
        n10_Dh2_avg[xrange, :], 1), color='C2', alpha=0.08)
    ax1.fill_between(xrange, n100_Dh2_SP_avg[xrange, i], np.mean(
        n100_Dh2_SP_avg[xrange, :], 1), color='C1', alpha=0.08)
    ax1.fill_between(xrange, n10_Dh2_SP_avg[xrange, i], np.mean(
        n10_Dh2_SP_avg[xrange, :], 1), color='C3', alpha=0.08)


ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_iteration.pdf', bbox_inches='tight')


# %% Plot by iteration (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh2_tilde_avg, 0))

ax2.plot(xrange, np.mean(n100_Dh2_tilde_avg, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(xrange, np.mean(n100_Dh2_SP_tilde_avg[xrange], 1), color='C0', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(xrange, np.mean(n10_Dh2_tilde_avg[xrange], 1), color='C1', marker="o",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")

ax2.plot(xrange, np.mean(n10_Dh2_SP_tilde_avg[xrange], 1), color='C1', marker="s", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(xrange, n100_Dh2_tilde_avg[xrange, i], np.mean(
        n100_Dh2_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh2_tilde_avg[xrange, i], np.mean(
        n10_Dh2_tilde_avg[xrange, :], 1), color='C2', alpha=0.08)
    ax2.fill_between(xrange, n100_Dh2_SP_tilde_avg[xrange, i], np.mean(
        n100_Dh2_SP_tilde_avg[xrange, :], 1), color='C1', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh2_SP_tilde_avg[xrange, i], np.mean(
        n10_Dh2_SP_tilde_avg[xrange, :], 1), color='C3', alpha=0.08)


ax2.set_xlabel('iteration')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)


ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_tilde_iteration.pdf', bbox_inches='tight')


# %% Plot by g oracles (x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles, np.mean(n100_Dh2_avg, 1), color='C0', marker="o",
         markevery=5, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles, np.mean(n100_Dh2_SP_avg, 1), color='C0', marker="s", markevery=40,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(n10_oracles, np.mean(n10_Dh2_avg, 1), color='C1', marker="o",
         markevery=100, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles, np.mean(n10_Dh2_SP_avg, 1), color='C1', marker="s", markevery=500,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles, n100_Dh2_avg[:, i], np.mean(
        n100_Dh2_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_oracles, n10_Dh2_avg[:, i], np.mean(
        n10_Dh2_avg, 1), color='C1', alpha=0.08)
    ax1.fill_between(n100_SP_oracles, n100_Dh2_SP_avg[:, i], np.mean(
        n100_Dh2_SP_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_SP_oracles, n10_Dh2_SP_avg[:, i], np.mean(
        n10_Dh2_SP_avg, 1), color='C1', alpha=0.08)


ax1.set_xlabel('\# of inner function value oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 200000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_oracle.pdf', bbox_inches='tight')


# %% Plot by g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles, np.mean(n100_Dh2_tilde_avg, 1), color='C0', marker="o",
         markevery=5, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles, np.mean(n100_Dh2_SP_tilde_avg, 1), color='C0', marker="s", markevery=60,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles, np.mean(n10_Dh2_tilde_avg, 1), color='C1', marker="o",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles, np.mean(n10_Dh2_SP_tilde_avg, 1), color='C1', marker="s", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles, n100_Dh2_tilde_avg[:, i], np.mean(
        n100_Dh2_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_oracles, n10_Dh2_tilde_avg[:, i], np.mean(
        n10_Dh2_tilde_avg, 1), color='C1', alpha=0.08)
    ax2.fill_between(n100_SP_oracles, n100_Dh2_SP_tilde_avg[:, i], np.mean(
        n100_Dh2_SP_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_SP_oracles, n10_Dh2_SP_tilde_avg[:, i], np.mean(
        n10_Dh2_SP_tilde_avg, 1), color='C1', alpha=0.08)


ax2.set_xlabel('\# of inner function value oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 200000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/RoS_h2_tilde_oracle.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles_grad, np.mean(n100_Dh2_avg, 1), color='C0', marker="o",
         markevery=100, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles_grad, np.mean(n100_Dh2_SP_avg, 1), color='C0', marker="s", markevery=100,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax1.plot(n10_oracles_grad, np.mean(n10_Dh2_avg, 1), color='C1', marker="o",
         markevery=200, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles_grad, np.mean(n10_Dh2_SP_avg, 1), color='C1', marker="s", markevery=500,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles_grad, n100_Dh2_avg[:, i], np.mean(
        n100_Dh2_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_oracles_grad, n10_Dh2_avg[:, i], np.mean(
        n10_Dh2_avg, 1), color='C1', alpha=0.08)
    ax1.fill_between(n100_SP_oracles_grad, n100_Dh2_SP_avg[:, i], np.mean(
        n100_Dh2_SP_avg, 1), color='C0', alpha=0.08)
    ax1.fill_between(n10_SP_oracles_grad, n10_Dh2_SP_avg[:, i], np.mean(
        n10_Dh2_SP_avg, 1), color='C1', alpha=0.08)


ax1.set_xlabel('\# of inner function gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 30000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_oracl_grad.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles_grad, np.mean(n100_Dh2_tilde_avg, 1), color='C0', marker="o",
         markevery=100, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles_grad, np.mean(n100_Dh2_SP_tilde_avg, 1), color='C0', marker="s", markevery=100,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles_grad, np.mean(n10_Dh2_tilde_avg, 1), color='C1', marker="o",
         markevery=200, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles_grad, np.mean(n10_Dh2_SP_tilde_avg, 1), color='C1', marker="s", markevery=500,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles_grad, n100_Dh2_tilde_avg[:, i], np.mean(
        n100_Dh2_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_oracles_grad, n10_Dh2_tilde_avg[:, i], np.mean(
        n10_Dh2_tilde_avg, 1), color='C1', alpha=0.08)
    ax2.fill_between(n100_SP_oracles_grad, n100_Dh2_SP_tilde_avg[:, i], np.mean(
        n100_Dh2_SP_tilde_avg, 1), color='C0', alpha=0.08)
    ax2.fill_between(n10_SP_oracles_grad, n10_Dh2_SP_tilde_avg[:, i], np.mean(
        n10_Dh2_SP_tilde_avg, 1), color='C1', alpha=0.08)


ax2.set_xlabel('\# of inner function value oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 30000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')


plt.savefig('Figures/RoS_h2_tilde_oracle_grad.pdf', bbox_inches='tight')

# %% Plot by iteration, noise (based on x)
fig, ax1 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh2, 0))

ax1.plot(xrange, np.mean(n100_Dh2, 1), color='C0', marker="o",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(xrange, np.mean(n100_Dh2_SP[xrange], 1), color='C1', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(xrange, np.mean(n10_Dh2[xrange], 1), color='C2', marker="o",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(xrange, np.mean(n10_Dh2_SP[xrange], 1), color='C3', marker="s", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(xrange, n100_Dh2[xrange, i], np.mean(
        n100_Dh2, 1), color='C0', alpha=0.02)
    ax1.fill_between(xrange, n10_Dh2[xrange, i], np.mean(
        n10_Dh2[xrange, :], 1), color='C2', alpha=0.02)
    ax1.fill_between(xrange, n100_Dh2_SP[xrange, i], np.mean(
        n100_Dh2_SP[xrange, :], 1), color='C1', alpha=0.02)
    ax1.fill_between(xrange, n10_Dh2_SP[xrange, i], np.mean(
        n10_Dh2_SP[xrange, :], 1), color='C3', alpha=0.02)


ax1.set_xlabel('iteration')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)

ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')


plt.savefig('Figures/RoS_h2_iteration_noise.pdf', bbox_inches='tight')


# %% Plot by iteration, noise (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))
xrange = range(np.size(n100_Dh2_tilde, 0))

ax2.plot(xrange, np.mean(n100_Dh2_tilde, 1), color='C0', marker="P",
         markevery=20, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(xrange, np.mean(n100_Dh2_SP_tilde[xrange], 1), color='C1', marker="s", markevery=35,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax2.plot(xrange, np.mean(n10_Dh2_tilde[xrange], 1), color='C2', marker="v",
         markevery=30, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")

ax2.plot(xrange, np.mean(n10_Dh2_SP_tilde[xrange], 1), color='C3', marker="o", markevery=30,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(xrange, n100_Dh2_tilde[xrange, i], np.mean(
        n100_Dh2_tilde, 1), color='C0', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh2_tilde[xrange, i], np.mean(
        n10_Dh2_tilde[xrange, :], 1), color='C2', alpha=0.08)
    ax2.fill_between(xrange, n100_Dh2_SP_tilde[xrange, i], np.mean(
        n100_Dh2_SP_tilde[xrange, :], 1), color='C1', alpha=0.08)
    ax2.fill_between(xrange, n10_Dh2_SP_tilde[xrange, i], np.mean(
        n10_Dh2_SP_tilde[xrange, :], 1), color='C3', alpha=0.08)


ax2.set_xlabel('iteration')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)


ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_tilde_iteration_noise.pdf', bbox_inches='tight')


# %% Plot by g oracles noise(x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles, np.mean(n100_Dh2, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles, np.mean(n100_Dh2_SP, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(n10_oracles, np.mean(n10_Dh2, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles, np.mean(n10_Dh2_SP, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles, n100_Dh2[:, i], np.mean(
        n100_Dh2, 1), color='C0', alpha=0.02)
    ax1.fill_between(n10_oracles, n10_Dh2[:, i], np.mean(
        n10_Dh2, 1), color='C2', alpha=0.02)
    ax1.fill_between(n100_SP_oracles, n100_Dh2_SP[:, i], np.mean(
        n100_Dh2_SP, 1), color='C1', alpha=0.02)
    ax1.fill_between(n10_SP_oracles, n10_Dh2_SP[:, i], np.mean(
        n10_Dh2_SP, 1), color='C3', alpha=0.02)


ax1.set_xlabel('\# of inner function value oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 60000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_oracle_noise.pdf', bbox_inches='tight')


# %% Plot by g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles, np.mean(n100_Dh2_tilde, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles, np.mean(n100_Dh2_SP_tilde, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles, np.mean(n10_Dh2_tilde, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles, np.mean(n10_Dh2_SP_tilde, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles, n100_Dh2_tilde[:, i], np.mean(
        n100_Dh2_tilde, 1), color='C0', alpha=0.02)
    ax2.fill_between(n10_oracles, n10_Dh2_tilde[:, i], np.mean(
        n10_Dh2_tilde, 1), color='C2', alpha=0.02)
    ax2.fill_between(n100_SP_oracles, n100_Dh2_SP_tilde[:, i], np.mean(
        n100_Dh2_SP_tilde, 1), color='C1', alpha=0.02)
    ax2.fill_between(n10_SP_oracles, n10_Dh2_SP_tilde[:, i], np.mean(
        n10_Dh2_SP_tilde, 1), color='C3', alpha=0.02)


ax2.set_xlabel('\# of inner function value oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 60000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_tilde_oracle_noise.pdf', bbox_inches='tight')


# %% Plot by grad g oracles noise(x)
fig, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(n100_oracles_grad, np.mean(n100_Dh2, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax1.plot(n100_SP_oracles_grad, np.mean(n100_Dh2_SP, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$")
ax1.plot(n10_oracles_grad, np.mean(n10_Dh2, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax1.plot(n10_SP_oracles_grad, np.mean(n10_Dh2_SP, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax1.fill_between(n100_oracles_grad, n100_Dh2[:, i], np.mean(
        n100_Dh2, 1), color='C0', alpha=0.02)
    ax1.fill_between(n10_oracles_grad, n10_Dh2[:, i], np.mean(
        n10_Dh2, 1), color='C2', alpha=0.02)
    ax1.fill_between(n100_SP_oracles_grad, n100_Dh2_SP[:, i], np.mean(
        n100_Dh2_SP, 1), color='C1', alpha=0.02)
    ax1.fill_between(n10_SP_oracles_grad, n10_Dh2_SP[:, i], np.mean(
        n10_Dh2_SP, 1), color='C3', alpha=0.02)


ax1.set_xlabel('\# of inner function gradient oracles')
ax1.set_yscale("log")
ax1.set_ylabel(r"$E[D_{h_2}(x^{R+1},x^R)]$")
ax1.grid(True, alpha=0.5)
ax1.set_xlim(0, 10000)
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
ax1.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_oracle_grad_noise.pdf', bbox_inches='tight')


# %% Plot by grad g oracles (x_tilde)
fig, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(n100_oracles_grad, np.mean(n100_Dh2_tilde, 1), color='C0', marker="P",
         markevery=1, label=r"$|\mathcal{B}_{\nabla}|=100,|\mathcal{B}_g|=10^4$")
ax2.plot(n100_SP_oracles_grad, np.mean(n100_Dh2_SP_tilde, 1), color='C1', marker="s", markevery=20,
         label=r"$|\mathcal{B}_{\nabla}|=100, |\mathcal{B}_{g}| = 10^4,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=10,\ \  |\mathcal{S}_g| = 100$", alpha=0.9)
ax2.plot(n10_oracles_grad, np.mean(n10_Dh2_tilde, 1), color='C2', marker="v",
         markevery=300, label=r"$|\mathcal{B}_{\nabla}|=20,|\mathcal{B}_g|=400$")
ax2.plot(n10_SP_oracles_grad, np.mean(n10_Dh2_SP_tilde, 1), color='C3', marker="o", markevery=1000,
         label=r"$|\mathcal{B}_{\nabla}|=20, |\mathcal{B}_{g}| = 400,$" + "\n" + r"$ |\mathcal{S}_{\nabla}|=5,\ \  |\mathcal{S}_g| = 25$")

for i in range(20):
    ax2.fill_between(n100_oracles_grad, n100_Dh2_tilde[:, i], np.mean(
        n100_Dh2_tilde, 1), color='C0', alpha=0.02)
    ax2.fill_between(n10_oracles_grad, n10_Dh2_tilde[:, i], np.mean(
        n10_Dh2_tilde, 1), color='C2', alpha=0.02)
    ax2.fill_between(n100_SP_oracles_grad, n100_Dh2_SP_tilde[:, i], np.mean(
        n100_Dh2_SP_tilde, 1), color='C1', alpha=0.02)
    ax2.fill_between(n10_SP_oracles_grad, n10_Dh2_SP_tilde[:, i], np.mean(
        n10_Dh2_SP_tilde, 1), color='C3', alpha=0.02)


ax2.set_xlabel('\# of inner function gradient oracles')
ax2.set_yscale("log")
ax2.set_ylabel(r"$E[D_{h_2}(\tilde{x}^{R+1},x^R)]$")
ax2.grid(True, alpha=0.5)
ax2.set_xlim(0, 10000)
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.legend(loc='upper right')

plt.savefig('Figures/RoS_h2_tilde_oracle_grad_noise.pdf', bbox_inches='tight')
