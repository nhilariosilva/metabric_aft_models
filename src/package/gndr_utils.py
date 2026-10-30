from IPython.display import Image, display

import warnings
import time

import pandas as pd
import numpy as np

import pickle

import matplotlib as mtl
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import plotly.graph_objects as go
import lifelines
from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter

import os

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from keras import optimizers, initializers

from tensorflow.keras.callbacks import Callback
from tqdm.keras import TqdmCallback
from tqdm import tqdm

import scipy.stats as stats
from scipy.stats import norm, t
from scipy.special import gamma

import thetaflow as thf


def plot_worm(residuals, ax = None, title = "Worm Plot"):
    '''
        Generates a worm plot (detrended Q-Q plot) with 95% confidence intervals.
    '''
    if(ax is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
    
    # Remove NaNs if any
    res = residuals[~np.isnan(residuals)]
    n = len(res)
    
    # Sort residuals
    r_sorted = np.sort(res)
    
    # Calculate empirical probabilities and theoretical standard normal quantiles
    p = (np.arange(1, n + 1) - 0.5) / n
    z = stats.norm.ppf(p)
    
    # Calculate the deviations (empirical - theoretical)
    deviation = r_sorted - z
    
    # Fit a cubic polynomial to the deviations to highlight the "worm" trend
    coeffs = np.polyfit(z, deviation, 3)
    trend = np.polyval(coeffs, z)
    
    # Calculate 95% pointwise confidence intervals
    se = np.sqrt(p * (1 - p) / n) / stats.norm.pdf(z)
    upper_ci = 1.96 * se
    lower_ci = -1.96 * se
    
    # Plotting
    ax.scatter(z, deviation, alpha = 0.5, color = 'black', s = 10)
    ax.plot(z, trend, color = 'red', linewidth = 2, label = 'Fitted Trend')
    ax.plot(z, upper_ci, color = 'blue', linestyle = '--', linewidth = 1.5, label = '95% CI')
    ax.plot(z, lower_ci, color = 'blue', linestyle = '--', linewidth = 1.5)
    
    ax.axhline(0, color = 'gray', linestyle = '-', linewidth = 1)
    ax.set_title(title)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Deviation")
    ax.set_ylim([-1.5, 1.5]) # Standard limits for worm plots
    ax.grid(True, alpha = 0.3)

def plot_qq(residuals, ax = None, title = "Q-Q Plot"):
    '''
        Standard Q-Q plot against the normal distribution.
    '''
    if(ax is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
        
    stats.probplot(residuals, dist = "norm", plot = ax)
    ax.set_title(title)
    ax.get_lines()[0].set_markerfacecolor('black')
    ax.get_lines()[0].set_markeredgecolor('black')
    ax.get_lines()[0].set_alpha(0.5)
    ax.get_lines()[0].set_markersize(4)
    ax.get_lines()[1].set_color('red') # The theoretical line

def plot_model_convergence(thf_model, ax1 = None, ax2 = None, ax3 = None, ax4 = None, figsize = None):
    '''
        Receives a thetaflow model after training as input and build convergence loss plots
    '''
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    has_validation = hasattr(thf_model, "val_loss_history") and thf_model.val_loss_history is not None
    has_finetuning = hasattr(thf_model, "loss_history_finetune") and thf_model.loss_history_finetune is not None
    
    if(ax1 is None or ax2 is None):
        if(has_validation and has_finetuning):
            if(ax3 is None or ax4 is None):
                if(figsize is None):
                    figsize = (14,12)
                fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = figsize)
                ax1 = ax[0,0]
                ax2 = ax[0,1]
                ax3 = ax[1,0]
                ax4 = ax[1,1]
        else:
            if(figsize is None):
                figsize = (14,6)
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = figsize)
            ax1 = ax[0]
            ax2 = ax[1]

    loss_idx = np.arange( thf_model.last_epoch + 1 )

    ax1.plot(loss_idx, thf_model.loss_history.numpy()[loss_idx], label = "Train loss", color = colors[0])
    if(has_validation):
        ax1.axvline( thf_model.best_metric_epoch, color = colors[1], label = "Minimal validation loss", linestyle = "dashed" )
    else:
        ax1.axvline( thf_model.best_metric_epoch, color = colors[0], label = "Minimal training loss", linestyle = "dashed" )
    ax1.set_title("Training loss - Training")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss Value")
    ax1.ticklabel_format(style = 'plain', axis = 'y')

    # If there is validation ax2 always represents the validation loss during training
    if(has_validation):
        ax2.plot(loss_idx, thf_model.val_loss_history.numpy()[loss_idx], label = "Validation loss", color = colors[1])
        ax2.axvline( thf_model.best_metric_epoch, color = colors[1], label = "Minimal validation loss", linestyle = "dashed" )
        ax2.ticklabel_format(style = 'plain', axis = 'y')
        ax2.set_title("Validation loss - Training")
    
    if(has_finetuning):
        # If there is fine-tuning, but no validation, plot the finetuning in the second axis
        if(not has_validation):
            finetuning_loss_idx = np.arange( thf_model.last_epoch_finetune + 1 )
            ax2.plot(finetuning_loss_idx, thf_model.loss_history_finetune.numpy()[finetuning_loss_idx], label = "Train loss", color = colors[0])
            ax2.axvline( thf_model.best_metric_epoch_finetune, color = colors[0], label = "Minimal training loss", linestyle = "dashed" )
            ax2.set_title("Training loss - Fine-tuning")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Loss Value")
            ax2.ticklabel_format(useOffset = False, style = 'plain', axis = 'y')
        # If there are both fine-tuning and validation, do it on the third axis
        else:
            finetuning_loss_idx = np.arange( thf_model.last_epoch_finetune + 1 )
            ax3.plot(finetuning_loss_idx, thf_model.loss_history_finetune.numpy()[finetuning_loss_idx], label = "Train loss", color = colors[0])
            ax3.axvline( thf_model.best_metric_epoch_finetune, color = colors[0], label = "Minimal training loss", linestyle = "dashed" )
            ax3.set_title("Training loss - Fine-tuning")
            ax3.set_xlabel("Epochs")
            ax3.set_ylabel("Loss Value")
            ax3.ticklabel_format(style = 'plain', axis = 'y')

            ax4.plot(finetuning_loss_idx, thf_model.val_loss_history_finetune.numpy()[finetuning_loss_idx], label = "Validation loss", color = colors[1])
            ax4.axvline( thf_model.best_metric_epoch_finetune, color = colors[0], label = "Minimal training loss", linestyle = "dashed" )
            ax4.set_title("Validation loss - Fine-tuning")
            ax4.set_xlabel("Epochs")
            ax4.set_ylabel("Loss Value")
            ax4.ticklabel_format(style = 'plain', axis = 'y')

    # Gather all active axes into a list
    all_axes = [ax for ax in [ax1, ax2, ax3, ax4] if ax is not None]
    
    # Get the figure object safely (handles cases where axes were passed in)
    fig = all_axes[0].get_figure()

    # Collect handles and labels from every axis
    handles, labels = [], []
    for ax in all_axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Remove duplicates by creating a dictionary (maps label -> handle)
    unique_legend = dict(zip(labels, handles))

    # Define your perfect left-to-right order
    desired_order = [
        "Train loss",                 # Solid Blue
        "Minimal training loss",      # Dashed Blue (if no validation)
        "Minimal fine-tuning loss",   # Dashed Blue (if fine-tuning)
        "Val loss",                   # Solid Orange
        "Validation loss",            # Solid Orange
        "Minimal validation loss"     # Dashed Orange
    ]

    # Filter out any labels that aren't in this specific run, keeping the strict order
    ordered_labels = [label for label in desired_order if label in unique_legend]
    ordered_handles = [unique_legend[label] for label in ordered_labels]

    # Place the single ordered legend on the figure
    fig.legend(
        ordered_handles, 
        ordered_labels, 
        loc='lower center',          
        bbox_to_anchor=(0.5, 1.0),   
        ncol=len(ordered_labels),     
        borderaxespad=1.0            
    )

    # Adjust layout so the plots don't overlap with your new top legend
    plt.tight_layout()
    fig.subplots_adjust(top=0.96)  # Shrink the top margin slightly to make room


def average_kaplan_meier(ts_grid,
                         S_ts_train, S_ts_test,
                         y_train, delta_train, y_test, delta_test,
                         show_individual = False, ax1 = None, ax2 = None):

    S_avg_train = np.mean(S_ts_train, axis = 1)
    S_avg_test = np.mean(S_ts_test, axis = 1)
    
    if(ax1 is None or ax2 is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
        ax1 = ax[0]
        ax2 = ax[1]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if(show_individual):
        for j in range(S_ts_train.shape[1]):
            ax1.plot(ts_grid, S_ts_train[:,j], color = "black", alpha = 0.2, linewidth = 0.8)
        
        for j in range(S_ts_test.shape[1]):
            ax2.plot(ts_grid, S_ts_test[:,j], color = "black", alpha = 0.2, linewidth = 0.8)
    
    km = lifelines.KaplanMeierFitter()
    
    km.fit(y_train, delta_train)
    km.plot(ax = ax1, ci_show = False, show_censors = False, label = "Kaplan-Meier Train", color = colors[0])
    ax1.plot(ts_grid.flatten(), S_avg_train, color = "red", label = "Average survival curve")
    
    km.fit(y_test, delta_test)
    km.plot(ax = ax2, ci_show = False, show_censors = False, label = "Kaplan-Meier Test", color = colors[0])
    ax2.plot(ts_grid.flatten(), S_avg_test, color = "red", label = "Average survival curve")
    
    ax1.set_ylim(0,1.05)
    ax1.set_title("Training set")
    ax2.set_ylim(0,1.05)
    ax2.set_title("Test set")

def median_kaplan_meier(ts_grid,
                        S_ts_train, S_ts_test,
                        y_train, delta_train, y_test, delta_test,
                        show_individual = False, ax1 = None, ax2 = None):

    S_avg_train = np.median(S_ts_train, axis = 1)
    S_avg_test = np.median(S_ts_test, axis = 1)
    
    if(ax1 is None or ax2 is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
        ax1 = ax[0]
        ax2 = ax[1]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if(show_individual):
        for j in range(S_ts_train.shape[1]):
            ax1.plot(ts_grid, S_ts_train[:,j], color = "black", alpha = 0.2, linewidth = 0.8)
        
        for j in range(S_ts_test.shape[1]):
            ax2.plot(ts_grid, S_ts_test[:,j], color = "black", alpha = 0.2, linewidth = 0.8)
    
    km = lifelines.KaplanMeierFitter()
    
    km.fit(y_train, delta_train)
    km.plot(ax = ax1, ci_show = False, show_censors = False, label = "Kaplan-Meier Train", color = colors[0])
    ax1.plot(ts_grid.flatten(), S_avg_train, color = "red", label = "Average survival curve")
    
    km.fit(y_test, delta_test)
    km.plot(ax = ax2, ci_show = False, show_censors = False, label = "Kaplan-Meier Test", color = colors[0])
    ax2.plot(ts_grid.flatten(), S_avg_test, color = "red", label = "Average survival curve")
    
    ax1.set_ylim(0,1.05)
    ax1.set_title("Training set")
    ax2.set_ylim(0,1.05)
    ax2.set_title("Test set")

def optimal_split_risk_group_test(y_train, delta_train, y_test, delta_test,
                                  risk_score_train, risk_score_test, plot = False):
    # Find the risk threshold (expected lifetime) that best separates groups of patients
    threshold_grid = np.linspace(0.05, 0.95, 100)
    
    p_values_logrank = []
    
    for threshold in threshold_grid:       
        # Threshold on the negative risk scale
        # risk_threshold = np.quantile(risk_score_train, 1 - np.mean(delta_train))
        risk_threshold = np.quantile(risk_score_train, threshold)
        
        low_hazard_train = (risk_score_train < risk_threshold)
        high_hazard_train = (risk_score_train >= risk_threshold)
        low_hazard_test = (risk_score_test < risk_threshold)
        high_hazard_test = (risk_score_test >= risk_threshold)
    
        res = logrank_test(y_test[low_hazard_test], y_test[high_hazard_test], delta_test[low_hazard_test], delta_test[high_hazard_test])
        
        p_values_logrank.append( res.p_value )

    if(plot):
        plt.plot(threshold_grid, p_values_logrank)
    
    i_min = np.argmin( p_values_logrank )
    print("Threshold that best separates test groups: {}".format( threshold_grid[ i_min ] ))
    print("Minimal logrank value: {}".format( p_values_logrank[ i_min ] ))

    return threshold_grid[ i_min ]
    
def split_risk_groups(ts_grid,
                      S_ts_train, S_ts_test,
                      y_train, delta_train, y_test, delta_test,
                      risk_score_train, risk_score_test,
                      threshold_quantile, show_individual = False, ax1 = None, ax2 = None):
    # Threshold on the negative risk scale
    risk_threshold = np.quantile(risk_score_train, threshold_quantile)
    
    low_hazard_train = (risk_score_train < risk_threshold)
    high_hazard_train = (risk_score_train >= risk_threshold)
    low_hazard_test = (risk_score_test < risk_threshold)
    high_hazard_test = (risk_score_test >= risk_threshold)

    print("High hazard (Train): {} ({:.2f}%)".format(np.sum(high_hazard_train), np.mean(high_hazard_train*100)))
    print("Low hazard (Train): {} ({:.2f}%)".format(np.sum(low_hazard_train), np.mean(low_hazard_train)*100))
    print("High hazard (Test): {} ({:.2f}%)".format(np.sum(high_hazard_test), np.mean(high_hazard_test)*100))
    print("Low hazard (Test): {} ({:.2f}%)".format(np.sum(low_hazard_test), np.mean(low_hazard_test)*100))
    
    S_low_hazard_avg_train = np.mean(S_ts_train[:, low_hazard_train], axis = 1)
    S_high_hazard_avg_train = np.mean(S_ts_train[:, high_hazard_train], axis = 1)
    S_low_hazard_avg_test = np.mean(S_ts_test[:, low_hazard_test], axis = 1)
    S_high_hazard_avg_test = np.mean(S_ts_test[:, high_hazard_test], axis = 1)

    if(ax1 is None or ax2 is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
        ax1 = ax[0]
        ax2 = ax[1]
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if(show_individual):
        for j in range(S_ts_train.shape[0]):
            if(low_hazard_train[j]):
                ax1.plot(ts_grid, S_ts_train[:,j], color = colors[0], alpha = 0.2)
            else:
                ax1.plot(ts_grid, S_ts_train[:,j], color = colors[1], alpha = 0.2)
        
        for j in range(S_ts_test.shape[0]):
            if(low_hazard_test[j]):
                ax2.plot(ts_grid, S_ts_test[:,j], color = colors[0], alpha = 0.2)
            else:
                ax2.plot(ts_grid, S_ts_test[:,j], color = colors[1], alpha = 0.2)
    
    res_logrank = logrank_test(y_test[low_hazard_test], y_test[high_hazard_test], delta_test[low_hazard_test], delta_test[high_hazard_test])
    print("Logrank test:\nTest statistic: {}\np-value: {}".format(res_logrank.test_statistic, res_logrank.p_value))
    
    km = lifelines.KaplanMeierFitter()
    
    km.fit(y_train[low_hazard_train], delta_train[low_hazard_train])
    km.plot(ax = ax1, ci_show = False, show_censors = False, label = "Group 1 (lower risk)", color = colors[0])
    ax1.plot(ts_grid.flatten(), S_low_hazard_avg_train, color = colors[0])
    
    km.fit(y_train[high_hazard_train], delta_train[high_hazard_train])
    km.plot(ax = ax1, ci_show = False, show_censors = False, label = "Group 2 (higher risk)", color = colors[1])
    ax1.plot(ts_grid.flatten(), S_high_hazard_avg_train, color = colors[1])
    
    ax1.set_ylim(0,1.05)
    ax1.set_title("Training set")
    
    km.fit(y_test[low_hazard_test], delta_test[low_hazard_test])
    km.plot(ax = ax2, ci_show = False, show_censors = False, label = "Group 1 (lower risk)", color = colors[0])
    ax2.plot(ts_grid.flatten(), S_low_hazard_avg_test, color = colors[0])
    
    km.fit(y_test[high_hazard_test], delta_test[high_hazard_test])
    km.plot(ax = ax2, ci_show = False, show_censors = False, label = "Group 2 (higher risk)", color = colors[1])
    ax2.plot(ts_grid.flatten(), S_high_hazard_avg_test, color = colors[1])
    
    ax2.set_ylim(0,1.05)
    ax2.set_title("Test set")
    
    plt.show()

    
def compute_randomized_residuals_censoring(survival_y, delta, seed = 42):
    if(seed is not None):
        np.random.seed(seed)

    cdf_y = 1.0 - survival_y
    u = np.zeros_like(cdf_y)
    
    delta = delta.flatten()
    event_mask = (delta == 1)
    cens_mask = (delta == 0)
    
    u[event_mask] = cdf_y[event_mask]
    n_censored = np.sum(cens_mask)
    if( n_censored > 0 ):
        u[cens_mask] = np.random.uniform(low = cdf_y[cens_mask], 
                                         high = 1.0 - 1e-7, 
                                         size = n_censored)
    u = np.clip(u, 1e-7, 1.0 - 1e-7)
    return norm.ppf(u)

def plot_cox_snell(cs_residuals, delta, ax = None, title = "Cox-Snell Plot", label_fit = None):
    if(ax is None):
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,6))
    
    naf = NelsonAalenFitter()
    naf.fit(cs_residuals, event_observed = delta)

    if(label_fit is not None):
        ax.plot(naf.cumulative_hazard_.index, naf.cumulative_hazard_.values, drawstyle='steps-post', label = label_fit)
    else:
        ax.plot(naf.cumulative_hazard_.index, naf.cumulative_hazard_.values, drawstyle='steps-post')
    
    ax.plot([0, cs_residuals.max()], [0, cs_residuals.max()], 'r--')
    
    ax.set_title(title)
    ax.set_xlabel('Cox-Snell Residuals')
    ax.set_ylabel('Cumulative Hazard')
    if(label_fit is not None):
        ax.legend()
    
    