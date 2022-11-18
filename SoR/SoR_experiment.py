""" Experiment of SoR compositional problem

Generate results for Bregman, NASA, SCSC algorithms

Requirements file: Algorithms.py

Author: Liu, Yin
Email: liu.6630 at osu(dot)edu
Created with Python 3.10.6

"""


# %% Initialize

import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing.pool import Pool
import itertools

import os
import time  # used to generate random seeds
# own class
from Algorithms import Bregman_SoR, NASA_SoR, SCSC_SoR

# plot setting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "ps.usedistiller": "xpdf"})

# Parameters
d = 50  # dimension of matrix
n = 1000  # number of random matrix
lmbda = 10  # weight of var part
R = 10  # constraint norm(x) <= R
noise_level = 3
Lf = 2*lmbda
Lg = 1
tau = min(0.5, Lf/(Lf+8), 1/Lf) / 2
beta = Lf * tau

# Generate data
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


# %% Repeat algorithm and save the result

oracles = 60000  # total oracles to samples
batch_list = [300, 100, 50]

number_experiment = 20  # number of repeated experiment

k1, k2 = 63.1, 1.58
alpha = 1.58e-4
tau_NASA, beta_NASA = 0.063, 158.5
a = b = 0.5/tau_NASA




def task(args):
    # create different random seed for multiprocess
    seed = (os.getpid() * int(time.time())) % 123456789
    np.random.seed(seed)
    i, idx_exp = args
    batch_size = batch_list[i]
    max_iter = oracles // batch_size

    BG = Bregman_SoR(A, batch_size, x_init, k1, k2,
                     tau, beta, max_iter, R, lmbda)
    BG.train()
    BG.calculate_Dh_avg()
    # save class
    with open(f'Results/Experiments/Breg_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(BG, file)

    SCSC = SCSC_SoR(A, batch_size, x_init, alpha, beta, max_iter, R, lmbda)
    SCSC.train()
    SCSC.calculate_Dh_avg()
    with open(f'Results/Experiments/SCSC_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(SCSC, file)

    NASA = NASA = NASA_SoR(A, batch_size, x_init, tau_NASA,
                           beta_NASA, a, b, max_iter, R, lmbda)
    NASA.train()
    NASA.calculate_Dh_avg()
    with open(f'Results/Experiments/NASA_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(NASA, file)

    print(
        f"finish batch {batch_size}, example No.{idx_exp},seed {seed}", flush=True)


if __name__ == '__main__'and "get_ipython" not in dir():
    list = [range(len(batch_list)), range(number_experiment)]
    args = [p for p in itertools.product(*list)]
    with Pool(8) as pool:
        

        # issue multiple tasks each with multiple arguments
        pool.imap(task, args)
        pool.close()
        pool.join()


# # %% Plot the result

# Dh1_avg_Breg = np.zeros([len(batch_list), number_experiment, oracles])
# Dh2_avg_Breg = np.zeros([len(batch_list), number_experiment, oracles])

# Dh1_avg_SCSC = np.zeros([len(batch_list), number_experiment, oracles])
# Dh2_avg_SCSC = np.zeros([len(batch_list), number_experiment, oracles])

# # read data
# for idx_batch in range(len(batch_list)):
#     max_iter = oracles // batch_list[idx_batch]
#     for idx_exp in range(number_experiment):

#         with open(f'Results/Experiments/Breg_batch{batch_list[idx_batch]}_exp{idx_exp}.pickle', 'rb') as file:
#             Breg_exp = pickle.load(file)
#         Dh1_avg_Breg[idx_batch, idx_exp, 0:max_iter] = Breg_exp.Dh1_x_hat_avg
#         Dh2_avg_Breg[idx_batch, idx_exp, 0:max_iter] = Breg_exp.Dh2_x_hat_avg

#         with open(f'Results/Experiments/SCSC_batch{batch_list[idx_batch]}_exp{idx_exp}.pickle', 'rb') as file:
#             SCSC_exp = pickle.load(file)
#         Dh1_avg_SCSC[idx_batch, idx_exp, 0:max_iter] = SCSC_exp.Dh1_x_hat_avg
#         Dh2_avg_SCSC[idx_batch, idx_exp, 0:max_iter] = SCSC_exp.Dh2_x_hat_avg


# # %%
# def plot_line(data, ax, kwargs_mean, kwargs_fill):
#     # plot each experiment and their average
#     # array shape n x L, where n is the number of experiment, L is the length of one experiment
#     # kwargs_mean: settings for average line
#     # kwargs_fill: settings for each line

#     data_mean = np.mean(data, axis=0)

#     ax.plot(data_mean, **kwargs_mean)
#     for idx in range(data.shape[0]):
#         ax.fill_between(range(data.shape[1]),
#                         data[idx, :], data_mean, **kwargs_fill)


# k1, k2, tau = 100, 21.5, 0.025

# fig, ax = plt.subplots(figsize=(4, 3))
# # plot_line(k1 * Dh1_avg_Breg[0, :, 0:300] / tau**2, ax,
# #           line_color='C0', line_label=r"$|\mathcal{B}_{\nabla}|= 100$")
# kwargs_batch100_mean = {
#     'color': 'C0', 'label': r"$|\mathcal{B}_{\nabla}|= 100$", 'marker': 'o', 'markevery': 50}
# kwargs_batch100_fill = {'color': 'C0', 'alpha': 0.05}
# kwargs_batch10_mean = {
#     'color': 'C1', 'label': r"$|\mathcal{B}_{\nabla}|= 10$", 'marker': 'o', 'markevery': 50}
# kwargs_batch10_fill = {'color': 'C1', 'alpha': 0.05}
# kwargs_batch1_mean = {
#     'color': 'C2', 'label': r"$|\mathcal{B}_{\nabla}|= 1$", 'marker': 'o', 'markevery': 50}
# kwargs_batch1_fill = {'color': 'C2', 'alpha': 0.05}

# plot_line(k1 * Dh1_avg_Breg[0, :, 0:300] /
#           tau**2, ax, kwargs_batch100_mean, kwargs_batch100_fill)


# plot_line(k1 * Dh1_avg_Breg[1, :, 0:300] /
#           tau**2, ax, kwargs_batch10_mean, kwargs_batch10_fill)

# plot_line(k1 * Dh1_avg_Breg[2, :, 0:300] /
#           tau**2, ax, kwargs_batch1_mean, kwargs_batch1_fill)


# plot_line(k1 * Dh1_avg_SCSC[0, :, 0:300] /
#           tau**2, ax, {**kwargs_batch100_mean, 'linestyle': ':'}, kwargs_batch100_fill)


# plot_line(k1 * Dh1_avg_SCSC[1, :, 0:300] /
#           tau**2, ax, kwargs_batch10_mean, kwargs_batch10_fill)

# plot_line(k1 * Dh1_avg_SCSC[2, :, 0:300] /
#           tau**2, ax, kwargs_batch1_mean, kwargs_batch1_fill)
# # plot_line(k1 * Dh1_avg_SCSC[0, :, 0:300] / tau**2, ax,
# #           line_color='C2', line_label=r"$|\mathcal{B}_{\nabla}|= 100$")
# # plot_line(k1 * Dh1_avg_SCSC[1, :, 0:300] / tau**2, ax,
# #           line_color='C3', line_label=r"$|\mathcal{B}_{\nabla}|= 10$")
# ax.legend()
# ax.set_yscale('log')
# # %%

