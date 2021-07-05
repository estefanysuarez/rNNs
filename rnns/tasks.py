# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:40:33 2019

@author: Estefany Suarez
"""

import numpy as np
import pandas as pd
import scipy as sp
import mdp

from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing


def default_task_params(task):
    if task ==  'mem_cap':
        return np.arange(1,16)


def check_xy_dims(x,y):
    """
        Check that X,Y have the right dimensions
    """
    x_train, x_test = x
    y_train, y_test = y

    if not ((x_train.squeeze().ndim == 2) and (x_test.ndim == 2)):
        x_train = x_train.squeeze()[:, np.newaxis]
        x_test  = x_test.squeeze()[:, np.newaxis]
    else:
        x_train = x_train.squeeze()
        x_test  = x_test.squeeze()

    y_train = y_train.squeeze()
    y_test  = y_test.squeeze()

    return x_train, x_test, y_train, y_test


def memory_capacity_task(x, y, delays=None, t_on=0, **kwargs):
    """
        Delayed signal reconstruction
    """
    # get training and test sets
    x_train, x_test, y_train, y_test = check_xy_dims(x,y)

    x_train = x_train.squeeze()[t_on:,:]
    y_train = y_train.squeeze()[t_on:]

    x_test  = x_test.squeeze()[t_on:,:]
    y_test  = y_test.squeeze()[t_on:]

    if delays is None: delays = default_task_params('mem_cap')

    res = []
    for tau in delays:
        model = LinearRegression(fit_intercept=False, normalize=False).fit(x_train[tau:], y_train[:-tau])
        y_pred =  model.predict(x_test[tau:])

        with np.errstate(divide='ignore', invalid='ignore'):
            perf = np.abs(np.corrcoef(y_test[:-tau], y_pred)[0][1])

        # save results
        res.append(perf)

    return np.array(res), delays


def pattern_recognition_task(x, y, pttn_lens, **kwargs):
    """
        Temporal pattern recognition
    """

    # get training and test sets
    x_train, x_test, y_train, y_test = check_xy_dims(x,y)

    # split samples
    sections = [np.sum(pttn_lens[:idx]) for idx in range(1, len(pttn_lens))]
    y_test = np.split(y_test, sections, axis=0)

    ridge_multi_regr = MultiOutputRegressor(Ridge(fit_intercept=True, normalize=False, solver='auto', alpha=0.0))
    y_pred = np.split(ridge_multi_regr.fit(x_train, y_train).predict(x_test), sections, axis=0)

    y_test_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_test])
    y_pred_mean = sp.array([sp.argmax(mdp.numx.atleast_2d(mdp.numx.mean(sample, axis=0))) for sample in y_pred])

    with np.errstate(divide='ignore', invalid='ignore'):
        cm = metrics.confusion_matrix(y_test_mean, y_pred_mean)
        perf = np.diagonal(cm)/np.sum(cm, axis=1)

    return cm


def get_scores_per_alpha(task, performance, task_params, thres=0.9, normalize=False):
    """
        This method returns the parameters at which the best performance across
        different alpha values occurs.
    """

    # estimate capacity across task params per alpha value
    if (task == 'mem_cap') or (task == 'nonlin_cap'):

        # estimate capacity across task params per alpha value
        if normalize:
            cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]-np.min(task_params))/(np.max(task_params)-np.min(task_params)) if (perf>thres).any() else 0 for perf in performance] #performance normalized in [0,1] range
        else:
            cap_per_alpha = [(task_params[perf<thres][np.argmax(perf[perf<thres])]) if (perf>thres).any() else 0 for perf in performance]

        perf_per_alpha = np.array([np.nansum(perf) for perf in performance])

    elif task == 'fcn_app':

        param_tau, param_omega = task_params

        # estimate capacity across task params per alpha value
        cap_per_alpha = []
        for perf in performance:
            idx_tau_below_thrd, idx_omega_below_thrd = np.where(perf == np.max(perf[perf<thres]))

            if normalize:
                tmp_cap_param_tau   = (param_tau[idx_tau_below_thrd[0]]-np.min(param_tau))/(np.max(param_tau)-np.min(param_tau)) #performance normalized in [0,1] range
                tmp_cap_param_omega = (param_omega[idx_omega_below_thrd[0]]-np.min(param_omega))/(np.max(param_omega)-np.min(param_omega)) #performance normalized in [0,1] range

            else:
                tmp_cap_param_tau   = param_tau[idx_tau_below_thrd[0]]
                tmp_cap_param_omega = param_omega[idx_omega_below_thrd[0]]

            cap_per_alpha.append(tmp_cap_param_tau+tmp_cap_param_omega)

        perf_per_alpha = np.array([np.sum(perf) for perf in performance])

    elif task == 'pttn_recog':

        # estimate capacity across task params per alpha value. There is no capacity for the pattern recognition task
        cm_norm = [np.diagonal(perf)/np.sum(perf, axis=1) for perf in performance]

        cap_per_alpha  = np.array([len(np.where(cm > 0.55)[0]) for cm in cm_norm])

        perf_per_alpha = [np.sum(np.diagonal(perf))/np.sum(perf) for perf in performance]

    return perf_per_alpha, cap_per_alpha


def run_task(task, X, Y, readout_nodes=None, **kwargs):
    """
        Performs a task for multiple reservoir states, and returns the consensus
        performance and capacity (across task parameters) for each reservoir
        state.
    """

    res = []
    for i, x in enumerate(X):

        if readout_nodes is not None:
            x = x.squeeze()[:, :, readout_nodes]
        else:
            x = x.squeeze()
        y = Y.squeeze()

        if task == 'mem_cap':
            perf, task_params = memory_capacity_task(x, y, **kwargs)

        elif task == 'pttn_recog':
            perf = pattern_recognition_task(x, y, **kwargs)
            task_params = None

        res.append(perf) # across task parameters

    performance, capacity = get_scores_per_alpha(task=task,
                                                 performance=res,
                                                 task_params=task_params,
                                                 )

    return performance, capacity
