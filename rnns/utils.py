# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:44:16 2021

@author: Estefany Suarez
"""

import numpy as np

def unbatch(x):
    """
        Transforms (batch, seq_len, features)
        to (batch*seq_len, features)
    """
    return np.concatenate(x, axis=0)


def batch(x, batch_size):

    pass
