import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import random

from time import time

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

import os, shutil, sys
from pathlib import Path
import json
import subprocess

from scipy.stats import anderson
from sksurv.metrics import brier_score, integrated_brier_score, cumulative_dynamic_auc

sys.path.insert(0, "../src")
from package import gndr_utils as utils


def format_ipcw_data(results_dict):
    y_train = results_dict['y_train'].flatten()
    delta_train = results_dict['delta_train'].flatten().astype(bool)
    y_test = results_dict['y_test'].flatten()
    delta_test = results_dict['delta_test'].flatten().astype(bool)
    
    ts_grid = results_dict['ts_grid'].flatten()
    S_ts_test = results_dict["S_ts_test"]
    
    # Create structured arrays required by scikit-survival
    dt = np.dtype([('event', 'bool'), ('time', 'float')])
    train_surv = np.empty( len(y_train), dtype = dt )
    train_surv['event'], train_surv['time'] = delta_train, y_train
    test_surv = np.empty(len(y_test), dtype=dt)
    test_surv['event'], test_surv['time'] = delta_test, y_test

    # The bounds must be strictly within the TEST set's follow-up limits
    min_time = np.min( y_test[delta_test] ) + 1e-5
    max_time = np.max( y_test ) - 1e-5
    
    # Get the grid of values only on the regions where the times are adequate
    safe_mask = (ts_grid > min_time) & (ts_grid < max_time)
    eval_times = ts_grid[safe_mask]
    eval_surv_probs = S_ts_test[safe_mask, :].T
    if(eval_surv_probs.shape[0] == 1):
        eval_surv_probs = np.repeat(eval_surv_probs, y_test.shape[0], axis = 0)

    return train_surv, test_surv, eval_surv_probs, eval_times

def calculate_ad_statistic(results_dict):
    resid_model_test = utils.compute_randomized_residuals_censoring( results_dict["S_test"], results_dict["delta_test"] )
    # Normalize the AD statistic according to the test set size
    model_ad = anderson(resid_model_test).statistic / len(results_dict["S_test"])
    return model_ad

def calculate_ipcw_brier_score(results_dict):
    """
        Obtains the IPCW Brier Score curve along the time grid used for the model's results.
    """
    train_surv, test_surv, eval_surv_probs, eval_times = format_ipcw_data(results_dict)
    # Calculate Time-Dependent Brier Score
    times, bs_times = brier_score(train_surv, test_surv, eval_surv_probs, eval_times)
    return times, bs_times

def calculate_integrated_ipcw_brier_score(results_dict):
    """
        Obtains the Integrated IPCW Brier Score curve along the time grid used for the model's results.
    """
    train_surv, test_surv, eval_surv_probs, eval_times = format_ipcw_data(results_dict)
    # Calculate Time-Dependent Brier Score
    ibs = integrated_brier_score(train_surv, test_surv, eval_surv_probs, eval_times)
    return ibs

def calculate_unos_auc(results_dict):
    """
        Obtains the Uno's Cumulative/Dynamic Time-Dependent AUC.
    """
    train_surv, test_surv, eval_surv_probs, eval_times = format_ipcw_data(results_dict)
    # Transform survival probabilities into risk scores for concordance
    risk_scores = 1.0 - eval_surv_probs
    # Calculate Uno's AUC
    auc, mean_auc = cumulative_dynamic_auc(train_surv, test_surv, risk_scores, eval_times)
    return eval_times, auc, mean_auc
