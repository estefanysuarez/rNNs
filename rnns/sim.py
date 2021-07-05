# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:44:16 2021

@author: Estefany Suarez
"""

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from . import rnns
from . import utils


def sim_reservoir(w_ih, w_hh, x, **kwargs):
    """
        Simulates the dynamics of the network provided some inputs.

        Parameters
        ----------
        w_ih: (N, N_inputs) numpy.ndarray
            Input connectivity matrix
            N_inputs: number of external input nodes
            N: number of nodes in the network

        w_hh : (N, N) numpy.ndarray
            Network connectivity matrix
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to target (source) nodes.

        x : (t, N_inputs) numpy.ndarray
            External input signal
            t : number ot time steps
            N_inputs : number of external input nodes


        Returns
        -------
        x_train, x_test: (t, N) numpy.darray
            Reservoir states for training and test data
            t : number ot time steps
            N : number of nodes in the network

    """
    x_train, x_test = x

    # add one dimension corresponding to bacth
    x_train = x_train[np.newaxis, :,:]
    x_test = x_test[np.newaxis, :,:]

    reservoir_params = {
                        'input_size':w_ih.shape[-1],
                        'hidden_size':w_hh.shape[0],
                        # 'w_ih':w_ih,
                        # 'w_hh':w_hh,
                        }

    reservoir = rnns.Reservoir(**reservoir_params)
    reservoir.to(reservoir.device)

    reservoir.set_input_weights(w_ih)
    reservoir.set_rnn_weights(w_hh)

    # Forward pass
    x_train = torch.tensor(x_train, dtype=torch.float, device=reservoir.device)
    x_test  = torch.tensor(x_test, dtype=torch.float, device=reservoir.device)

    y_train = reservoir(x_train)
    y_test  = reservoir(x_test)

    return y_train.detach().numpy(), y_test.detach().numpy()


def run_multiple_sim(w_ih, w_hh, inputs, alphas, **kwargs):
    """
        Simulates the dynamics of the network for a range of alpha values.

        Parameters
        ----------
        w_ih: (N, N_inputs) numpy.ndarray
            Input connectivity matrix
            N_inputs: number of external input nodes
            N: number of nodes in the network

        w_hh : (N, N) numpy.ndarray
            Network connectivity matrix
            N: number of nodes in the network. If w is directed, then rows
            (columns) should correspond to target (source) nodes.

        inputs : (t, N_inputs) numpy.ndarray
            External input signal
            t : number ot time steps
            N_inputs : number of external input nodes

        alphas : list
            List of alpha values to scale the connectivity matrix
            (equivalent to the spectral radii)

        Returns
        -------
        states_train, states_test : list
            List of reservoir states for training and test data

    """

    states_train = []
    states_test  = []
    for alpha in tqdm(alphas):
        y_train, y_test = sim_reservoir(w_ih,
                                        alpha*w_hh.copy(),
                                        inputs,
                                        **kwargs
                                        )

        states_train.append(utils.unbatch(y_train))
        states_test.append(utils.unbatch(y_test))

    return states_train, states_test
