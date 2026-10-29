import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import types

from time import time
from tqdm import tqdm

from scipy.special import comb, loggamma, lambertw
from scipy.stats import multinomial, expon

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import optimizers, initializers, regularizers, layers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)

import os, shutil
from pathlib import Path
import json
import subprocess

import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

import sys
sys.path.insert(0, "../src")
from package import cure_aft
from package import gndr_utils as utils
from package import evaluation_metrics as evaluations

import thetaflow as thf

def load_cv_train_test(fold):
    df_train = pd.read_csv("../data/01_mlready/Cross Validation/{}_fold_train.csv".format(fold))
    df_test = pd.read_csv("../data/01_mlready/Cross Validation/{}_fold_test.csv".format(fold))

    # Lifetimes and censoring indicators
    y_train = df_train["time"] / 12
    delta_train = df_train["delta"]
    # Data for the neural network interpreter
    X_train_nn = df_train.iloc[:, 10:]
    # For all parameters from the AFT model, consider the same set of features as predictors
    z_train_mu = df_train.iloc[:, 3:10]
    # z_train_mu.insert(0, "const", 1) # Can't have an intercept in mu since the neural net component already accounts for it
    z_train_sigma = df_train.iloc[:, 5:10]
    z_train_sigma.insert(0, "const", 1)
    z_train_p = df_train.iloc[:, 3:10]
    z_train_p.insert(0, "const", 1)
    
    # Lifetimes and censoring indicators
    y_test = df_test["time"] / 12
    delta_test = df_test["delta"]
    # Data for the neural network interpreter
    X_test_nn = df_test.iloc[:, 10:]
    # For all parameters from the AFT model, consider the same set of features as predictors
    z_test_mu = df_test.iloc[:, 3:10]
    # z_test_mu.insert(0, "const", 1) # Can't have an intercept in mu since the neural net component already accounts for it
    z_test_sigma = df_test.iloc[:, 5:10]
    z_test_sigma.insert(0, "const", 1)
    z_test_p = df_test.iloc[:, 3:10]
    z_test_p.insert(0, "const", 1)

    return y_train, delta_train, X_train_nn, z_train_mu, z_train_sigma, z_train_p, \
    y_test, delta_test, X_test_nn, z_test_mu, z_test_sigma, z_test_p

def fit_kfolds_structure(structure, distribution):
    for i in tqdm(range(1,5+1)):
        y_train, delta_train, X_train_nn, z_train_mu, z_train_sigma, z_train_p, \
        y_test, delta_test, X_test_nn, z_test_mu, z_test_sigma, z_test_p = load_cv_train_test(i)

        if(structure == 1):
            # Keep all intercepts as the model is fully linear
            data_train = [z_train_mu, z_train_sigma, z_train_p, y_train, delta_train]
            data_test = [z_test_mu, z_test_sigma, z_test_p, y_test, delta_test]

            with tf.device("/CPU:0"):
                model_parameters, model_loss, model_survival_cure = cure_aft.build_cure_aft_structure1(distribution, z_train_mu, z_train_sigma, z_train_p)
                model = cure_aft.train_model( model_parameters, model_loss, None, None, None, None, data_train, verbose = False )
        elif(structure == 2):
            # Remove the intercept column for the mu parameter
            data_train = [z_train_mu.iloc[:,1:], z_train_sigma, z_train_p, y_train, delta_train]
            data_test = [z_test_mu.iloc[:,1:], z_test_sigma, z_test_p, y_test, delta_test]

            with tf.device("/CPU:0"):
                model_parameters, model_loss, model_survival_cure = cure_aft.build_cure_aft_structure2(distribution, z_train_mu.iloc[:,1:], z_train_sigma, z_train_p)
                model = cure_aft.train_model( model_parameters, model_loss, cure_aft.neural_network1, cure_aft.network_call, cure_aft.network_call_nolast, X_train_nn, data_train, verbose = False )
        elif(structure == 3):
            # Remove the intercept column for the p parameter
            data_train = [z_train_mu, z_train_sigma, z_train_p.iloc[:,1:], y_train, delta_train]
            data_test = [z_test_mu, z_test_sigma, z_test_p.iloc[:,1:], y_test, delta_test]

            with tf.device("/CPU:0"):
                model_parameters, model_loss, model_survival_cure = cure_aft.build_cure_aft_structure3(distribution, z_train_mu, z_train_sigma, z_train_p.iloc[:,1:])
                model = cure_aft.train_model( model_parameters, model_loss, cure_aft.neural_network1, cure_aft.network_call, cure_aft.network_call_nolast, X_train_nn, data_train, verbose = False )
        elif(structure == 4):
            # Remove the intercept column for the mu and p parameters
            data_train = [z_train_mu.iloc[:,1:], z_train_sigma, z_train_p.iloc[:,1:], y_train, delta_train]
            data_test = [z_test_mu.iloc[:,1:], z_test_sigma, z_test_p.iloc[:,1:], y_test, delta_test]

            with tf.device("/CPU:0"):
                model_parameters, model_loss, model_survival_cure = cure_aft.build_cure_aft_structure4(distribution, z_train_mu.iloc[:,1:], z_train_sigma, z_train_p.iloc[:,1:])
                model = cure_aft.train_model( model_parameters, model_loss, cure_aft.neural_network2, cure_aft.network_call, cure_aft.network_call_nolast, X_train_nn, data_train, verbose = False )
        else:
            raise("Please, provide a valid model structure (1, 2, 3 or 4)")

        model.save_model("../model_weights/Cross Validation/{}_model{}_fold_{}".format(distribution, structure, i))

if(__name__ == "__main__"):
    distributions = ["wb", "ln", "ll"]
    for i in range(1, 4+1):
        for dist in distributions:
            print("Fitting 5 folds for Structure {} - Distribution {}".format(i, dist))
            fit_kfolds_structure(i, dist)

