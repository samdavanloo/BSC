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

oracles = 300*300  # total oracles to samples
batch_list = [300, 100]  # , 50]

number_experiment = 20  # number of repeated experiment

k1, k2, beta_Breg = 63.1, 1.58, 0.5
tau_Breg = 0.025
alpha = 1.58e-4
tau_NASA, beta_NASA = 0.063, 158.5
a = b = beta_Breg/tau_NASA


def task_compare_algs(args):
    # create different random seed for multiprocess
    folder = f'Results/exp_algs_compare/'
    seed = (os.getpid() * int(time.time())) % 123456789
    np.random.seed(seed)
    i, idx_exp = args
    batch_size = batch_list[i]
    max_iter = oracles // batch_size

    BG = Bregman_SoR(A, batch_size, x_init, k1, k2,
                     tau, beta, max_iter, R, lmbda)
    BG.train()
    BG.calculate_measure_avg()
    # save class
    with open(folder + f'Breg_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(BG, file)

    SCSC = SCSC_SoR(A, batch_size, x_init, alpha, beta,
                    max_iter, R, lmbda, k1, k2, tau_Breg)
    SCSC.train()
    SCSC.calculate_measure_avg()
    with open(folder + f'SCSC_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(SCSC, file)

    NASA = NASA_SoR(A, batch_size, x_init, tau, beta, a, b,
                    max_iter, R, lmbda, k1, k2, tau_Breg, beta_Breg)
    NASA.train()
    NASA.calculate_measure_avg()
    with open(folder + f'NASA_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(NASA, file)
    print(
        f"finish batch {batch_size}, example No.{idx_exp},seed {seed}", flush=True)

def task_compare_batch(args):
    batch_list = [200,100,50]
    folder = f'Results/exp_batch/'
    seed = (os.getpid() * int(time.time())) % 123456789
    np.random.seed(seed)
    i, idx_exp = args
    batch_size = batch_list[i]
    max_iter = oracles // batch_size

    BG = Bregman_SoR(A, batch_size, x_init, k1, k2,
                     tau, beta, max_iter, R, lmbda)
    BG.train()
    BG.calculate_measure_avg()
    # save class
    with open(folder + f'Breg_batch{batch_size}_exp{idx_exp}.pickle', 'wb') as file:
        pickle.dump(BG, file)

if __name__ == '__main__' and "get_ipython" not in dir():
    batch_list = [100,50,10,1]
    list = [range(len(batch_list)), range(20)]
    args = [p for p in itertools.product(*list)]
    with Pool(8) as pool:

        # issue multiple tasks each with multiple arguments
        pool.imap(task_compare_algs, args)
        pool.close()
        pool.join()
