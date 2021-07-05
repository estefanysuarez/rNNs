# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:50:52 2019

@author: Estefany Suarez
"""

import random

import matplotlib.colors as mcolors
import numpy as np

from scipy.signal import sweep_poly


def generate_IOData(task, **kwargs):

    if task == 'mem_cap':
        x, y = io_mem_cap(**kwargs)

    elif task == 'pttn_recog':
        x, y = io_pttn_recog(**kwargs)
        pass

    return x, y


def io_mem_cap(time_len=1000, **kwargs):

    input_train = np.random.uniform(-1, 1, (time_len))[:, np.newaxis]
    input_test  = np.random.uniform(-1, 1, (time_len))[:, np.newaxis]

    return (input_train, input_test), (input_train.copy(), input_test.copy())


def io_pttn_recog(n_input_nodes=10, input_gain=3, n_patterns=10, n_repeats=100, pttn_len=50, **kwargs):
    """
        Generates noisy, random sinusoidal patterns with variable frequency
    """

    coeffs = [np.random.uniform(-2, 2, size=4) for _ in range(n_patterns)]
    t = np.linspace(0, 10, pttn_len)

    patterns = []
    labels = []
    for _ in range(n_repeats):
        for label in range(n_patterns):
            coeff = coeffs[label]
            poly = np.poly1d([coeff[0], coeff[1], coeff[2], coeff[3]])
            w = input_gain*sweep_poly(t, poly)
            w += np.random.normal(0, 0.1, len(w))

            labels.append(label)
            patterns.append(w)

    data = list(zip(patterns, labels))

    # split training/test sets
    train_frac = 0.5
    n_train_samples = int(train_frac*n_repeats)*n_patterns

    train_data = data[:n_train_samples]
    test_data = data[n_train_samples:]

    # shuffle data
    random.shuffle(train_data)
    random.shuffle(test_data)

    train_patterns, train_labels = zip(*train_data)
    test_patterns, test_labels = zip(*test_data)

    x_train = []
    y_train = []
    for pattern, label in zip(train_patterns, train_labels):
        new_label = -1*np.ones((len(pattern), n_patterns), dtype=np.int16)
        new_label[:, label] = 1

        x_train.append(pattern[:,np.newaxis])
        y_train.append(new_label)

    x_test = []
    y_test = []
    for pattern, label in zip(test_patterns, test_labels):
        new_label = -1*np.ones((len(pattern), n_patterns), dtype=np.int16)
        new_label[:, label] = 1

        x_test.append(pattern[:,np.newaxis])
        y_test.append(new_label)

    # concatenate data
    x_train = np.vstack(x_train)
    y_train = np.vstack(y_train)

    x_test = np.vstack(x_test)
    y_test = np.vstack(y_test)

    return (x_train, x_test), (y_train, y_test)


def io_memory(tau, batch_size, seq_len, input_size=1):
    """
        This method is used by the RNN/Lyapunov Exponents
    """


    n_iters = int(seq_len/batch_size)
    seq = np.random.uniform(-1, 1, (seq_len+tau,input_size))

    x = np.stack(np.split(seq.copy()[tau:,:], n_iters))
    y = np.stack(np.split(seq.copy()[:-tau], n_iters))

    return x, y
