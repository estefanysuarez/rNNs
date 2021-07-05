# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd

from . import tasks

#%% --------------------------------------------------------------------------------------------------------------------
# BASIC CODING MODE (V1)
# ----------------------------------------------------------------------------------------------------------------------
def coding(task, reservoir_states, target, alphas, readout_nodes=None, **kwargs):

    # get performa nce (R) across task parameters and alpha values
    performance, capacity = tasks.run_task(task=task,
                                           X=reservoir_states,
                                           Y=target,
                                           readout_nodes=readout_nodes,
                                           **kwargs)

    df_res = pd.DataFrame(data=np.column_stack((alphas, performance, capacity)),
                          columns=['alpha', 'performance', 'capacity'])

    df_res['n_nodes'] = len(readout_nodes)

    return df_res


def encoder(task, reservoir_states, target, readout_modules=None, **kwargs):
    """
        Given the reservoir_states of the network and the target signal
        for a given task, this method returns the encoding capacity for a given
        set of readout_modules. If readout_modules is None, then it will return the
        encoding capacity of all the nodes in reservoir_states.
    """

    if readout_modules is None:
        df_encoding = coding(task=task,
                             reservoir_states=reservoir_states,
                             target=target,
                             **kwargs
                             )

    else:
        module_ids = np.unique(readout_modules)

        encoding = []
        for module in module_ids:
            print(f'--------------------------- Module : {module} ------------------------------')

            # get set of output nodes
            readout_nodes = np.where(readout_modules == module)[0]

            # create temporal dataframe
            df_module = coding(task=task,
                               reservoir_states=reservoir_states,
                               target=target,
                               readout_nodes=readout_nodes,
                               **kwargs
                               )

            df_module['module'] = module

            #get encoding scores
            encoding.append(df_module)

        df_encoding = pd.concat(encoding)

    return df_encoding
