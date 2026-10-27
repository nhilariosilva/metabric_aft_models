import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import random
import types

from time import time

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

import thetaflow as thf

def build_cure_aft_structure1(aft_distribution, z_mu, z_sigma, z_p):

    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )
    
    aft_model_parameters = {
        "mu_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_mu.shape[1], "init": None},
        "sigma_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_sigma.shape[1], "init": None},
        "p_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_p.shape[1], "init": None}
    }
    
    # Weibull distribution - Gumbel formulation
    if(aft_distribution == "wb"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.exp(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) + z - tf.math.exp(z)
            
    # Log-normal distribution - Normal formulation
    elif(aft_distribution == "ln"):
        def S1(z, k = None):
            norm_cdf = ( 1 + tf.math.erf(z / tf.math.sqrt(2.0)) ) / 2.0
            return 1 - norm_cdf
        def log_f1(t, z, sigma, k = None):
            c = tf.cast(0.5 * tf.math.log(2.0 * np.pi), tf.float32)
            return -tf.math.log(t) - tf.math.log(sigma) - z**2 / 2 - c
            
    # Log-logistic distribution - Logistic formulation
    elif(aft_distribution == "ll"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.softplus(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) - 2.0 * tf.math.softplus(-z) - z
            
    # Generalized-Gamma distribution - Log-Gamma formulation
    elif(aft_distribution == "gg"):
        # Consider an extra global shape parameter for the Generalized Gamma distribution
        aft_model_parameters["k"] = {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None}

        def S1(z, k = None):
            return 2
        def log_f1(t, z, sigma, k = None):
            return 2
    else:
        raise Exception("Please, specify model in implemented family of AFT models. ('wb', 'ln', 'll', 'gg')")

    def loglikelihood_loss(model, nn_output, data):
        X, z_mu_data, z_sigma_data, z_p_data, y, delta = data

        k = None
        if(aft_distribution == "gg"):
            k = model.get_variable("k")
        
        mu_beta = model.get_variable("mu_beta")[:,None]
        mu = tf.linalg.matmul( z_mu_data, mu_beta )

        sigma_beta = model.get_variable("sigma_beta")[:,None]
        sigma = tf.math.exp( tf.linalg.matmul( z_sigma_data, sigma_beta ) )
        
        eps = tf.constant(1.0e-6, dtype = tf.float32)
        
        p_beta = model.get_variable("p_beta")[:,None]
        logit_p = tf.linalg.matmul( z_p_data, p_beta )
        logit_p = tf.clip_by_value(logit_p, -80.0, 80.0)
        p = tf.clip_by_value( tf.math.sigmoid( logit_p ), eps, 1.0-eps )
        log_p = tf.math.log(p)
        exp_mlogit_p = tf.math.exp(-logit_p)

        y_safe = y + eps
        z_y = (tf.math.log(y_safe) - mu) / sigma
        S1_y = S1(z_y, k = k)
        log_f1_y = log_f1(y_safe, z_y, sigma, k = k)
        
        log_Sp = log_p + tf.math.log1p(exp_mlogit_p * S1_y)
        log_fp = tf.math.log( 1-p ) + log_f1_y

        loglik_terms = delta * log_fp + (1.0 - delta) * log_Sp
        neg_loglik = -tf.reduce_sum(loglik_terms)
        return neg_loglik

    def get_survival_cure(model,
                          y_train, delta_train, z_train_mu, z_train_sigma, z_train_p,
                          y_test, delta_test, z_test_mu, z_test_sigma, z_test_p,
                          ngrid = 100):

        y_train = tf.cast(y_train, tf.float32)
        delta_train = tf.cast(delta_train, tf.float32)
        z_train_mu = tf.cast(z_train_mu, tf.float32)
        z_train_sigma = tf.cast(z_train_sigma, tf.float32)
        z_train_p = tf.cast(z_train_p, tf.float32)
        
        y_test = tf.cast(y_test, tf.float32)
        delta_test = tf.cast(delta_test, tf.float32)
        z_test_mu = tf.cast(z_test_mu, tf.float32)
        z_test_sigma = tf.cast(z_test_sigma, tf.float32)
        z_test_p = tf.cast(z_test_p, tf.float32)

        k = None
        if(aft_distribution == "gg"):
            k = model.predict("k")

        p_beta = model.predict("p_beta")[:,None]
        logit_p_train = tf.linalg.matmul( z_train_p, p_beta )
        p_train = tf.math.sigmoid( logit_p_train )
        logit_p_test = tf.linalg.matmul( z_test_p, p_beta )
        p_test = tf.math.sigmoid( logit_p_test )
    
        sigma_beta = model.predict("sigma_beta")[:,None]
        sigma_train = tf.math.exp( tf.linalg.matmul( z_train_sigma, sigma_beta ) )
        sigma_test = tf.math.exp( tf.linalg.matmul( z_test_sigma, sigma_beta ) )
    
        mu_beta = model.predict("mu_beta")[:,None]
    
        mu_train = tf.linalg.matmul( z_train_mu, mu_beta )
        mu_test = tf.linalg.matmul( z_test_mu, mu_beta )

        p_train = p_train.numpy().flatten()
        p_test = p_test.numpy().flatten()
        mu_train = mu_train.numpy().flatten()
        mu_test = mu_test.numpy().flatten()
        sigma_train = sigma_train.numpy().flatten()
        sigma_test = sigma_test.numpy().flatten()
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

        eps = tf.constant(1.0e-6, dtype = tf.float32)
        y_train_safe = y_train + eps
        z_y_train = (tf.math.log(y_train_safe) - mu_train) / sigma_train
        S1_y_train = S1(z_y_train, k = k)
        z_ts_train = (tf.math.log(ts_grid) - mu_train) / sigma_train
        S1_ts_train = S1(z_ts_train, k = k)

        y_test_safe = y_test + eps
        z_y_test = (tf.math.log(y_test_safe) - mu_test) / sigma_test
        S1_y_test = S1(z_y_test, k = k)
        z_ts_test = (tf.math.log(ts_grid) - mu_test) / sigma_test
        S1_ts_test = S1(z_ts_test, k = k)
    
        S_ts_train = p_train + (1-p_train) * S1_ts_train
        S_ts_test = p_test + (1-p_test) * S1_ts_test
        S_train = p_train + (1-p_train) * S1_y_train
        S_test = p_test + (1-p_test) * S1_y_test

        S_train = np.clip(S_train, 1e-15, 1.0)
        S_test = np.clip(S_test, 1e-15, 1.0)
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        if(hasattr(y_train, "to_numpy")):
            y_train = y_train.to_numpy()
        if(hasattr(delta_train, "to_numpy")):
            delta_train = delta_train.to_numpy()
        if(hasattr(y_test, "to_numpy")):
            y_test = y_test.to_numpy()
        if(hasattr(delta_test, "to_numpy")):
            delta_test = delta_test.to_numpy()
            
        return {
            "ts_grid": ts_grid,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "mu_train": mu_train,
            "mu_test": mu_test,
            "sigma_train": sigma_train,
            "sigma_test": sigma_test,
            "p_train": p_train,
            "p_test": p_test,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test
        }

    return aft_model_parameters, loglikelihood_loss, get_survival_cure


def build_cure_aft_structure2(aft_distribution, z_mu, z_sigma, z_p):

    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )
    
    aft_model_parameters = {
        "mu_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_mu.shape[1], "init": None},
        "mu_nn": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None},
        "sigma_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_sigma.shape[1], "init": None},
        "p_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_p.shape[1], "init": None}
    }
    
    # Weibull distribution - Gumbel formulation
    if(aft_distribution == "wb"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.exp(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) + z - tf.math.exp(z)
            
    # Log-normal distribution - Normal formulation
    elif(aft_distribution == "ln"):
        def S1(z, k = None):
            norm_cdf = ( 1 + tf.math.erf(z / tf.math.sqrt(2.0)) ) / 2.0
            return 1 - norm_cdf
        def log_f1(t, z, sigma, k = None):
            c = tf.cast(0.5 * tf.math.log(2.0 * np.pi), tf.float32)
            return -tf.math.log(t) - tf.math.log(sigma) - z**2 / 2 - c
            
    # Log-logistic distribution - Logistic formulation
    elif(aft_distribution == "ll"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.softplus(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) - 2.0 * tf.math.softplus(-z) - z
            
    # Generalized-Gamma distribution - Log-Gamma formulation
    elif(aft_distribution == "gg"):
        # Consider an extra global shape parameter for the Generalized Gamma distribution
        aft_model_parameters["k"] = {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None}

        def S1(z, k = None):
            return 2
        def log_f1(t, z, sigma, k = None):
            return 2
    else:
        raise Exception("Please, specify model in implemented family of AFT models. ('wb', 'ln', 'll', 'gg')")

    def loglikelihood_loss(model, nn_output, data):
        X, z_mu_data, z_sigma_data, z_p_data, y, delta = data

        k = None
        if(aft_distribution == "gg"):
            k = model.get_variable("k")
        
        mu_beta = model.get_variable("mu_beta")[:,None]
        mu_nn = model.get_variable("mu_nn", nn_output)
        mu = tf.linalg.matmul( z_mu_data, mu_beta ) + mu_nn

        sigma_beta = model.get_variable("sigma_beta")[:,None]
        sigma = tf.math.exp( tf.linalg.matmul( z_sigma_data, sigma_beta ) )
        
        eps = tf.constant(1.0e-6, dtype = tf.float32)
        
        p_beta = model.get_variable("p_beta")[:,None]
        logit_p = tf.linalg.matmul( z_p_data, p_beta )
        logit_p = tf.clip_by_value(logit_p, -80.0, 80.0)
        p = tf.clip_by_value( tf.math.sigmoid( logit_p ), eps, 1.0-eps )
        log_p = tf.math.log(p)
        exp_mlogit_p = tf.math.exp(-logit_p)

        y_safe = y + eps
        z_y = (tf.math.log(y_safe) - mu) / sigma
        S1_y = S1(z_y, k = k)
        log_f1_y = log_f1(y_safe, z_y, sigma, k = k)
        
        log_Sp = log_p + tf.math.log1p(exp_mlogit_p * S1_y)
        log_fp = tf.math.log( 1-p ) + log_f1_y

        loglik_terms = delta * log_fp + (1.0 - delta) * log_Sp
        neg_loglik = -tf.reduce_sum(loglik_terms)
        return neg_loglik

    def get_survival_cure(model,
                          y_train, delta_train, X_train, z_train_mu, z_train_sigma, z_train_p,
                          y_test, delta_test, X_test, z_test_mu, z_test_sigma, z_test_p,
                          ngrid = 100):

        y_train = tf.cast(y_train, tf.float32)
        delta_train = tf.cast(delta_train, tf.float32)
        X_train = tf.cast(X_train, tf.float32)
        z_train_mu = tf.cast(z_train_mu, tf.float32)
        z_train_sigma = tf.cast(z_train_sigma, tf.float32)
        z_train_p = tf.cast(z_train_p, tf.float32)
        
        y_test = tf.cast(y_test, tf.float32)
        delta_test = tf.cast(delta_test, tf.float32)
        X_test = tf.cast(X_test, tf.float32)
        z_test_mu = tf.cast(z_test_mu, tf.float32)
        z_test_sigma = tf.cast(z_test_sigma, tf.float32)
        z_test_p = tf.cast(z_test_p, tf.float32)
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        k = None
        if(aft_distribution == "gg"):
            k = model.predict("k")

        p_beta = model.predict("p_beta")[:,None]
        logit_p_train = tf.linalg.matmul( z_train_p, p_beta )
        p_train = tf.math.sigmoid( logit_p_train )
        logit_p_test = tf.linalg.matmul( z_test_p, p_beta )
        p_test = tf.math.sigmoid( logit_p_test )
    
        sigma_beta = model.predict("sigma_beta")[:,None]
        sigma_train = tf.math.exp( tf.linalg.matmul( z_train_sigma, sigma_beta ) )
        sigma_test = tf.math.exp( tf.linalg.matmul( z_test_sigma, sigma_beta ) )
    
        mu_beta = model.predict("mu_beta")[:,None]
    
        mu_nn_train = pred_train["mu_nn"]
        mu_train = tf.linalg.matmul( z_train_mu, mu_beta ) + mu_nn_train
        mu_nn_test = pred_test["mu_nn"]
        mu_test = tf.linalg.matmul( z_test_mu, mu_beta ) + mu_nn_test

        p_train = p_train.numpy().flatten()
        p_test = p_test.numpy().flatten()
        mu_train = mu_train.numpy().flatten()
        mu_test = mu_test.numpy().flatten()
        sigma_train = sigma_train.numpy().flatten()
        sigma_test = sigma_test.numpy().flatten()
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

        eps = tf.constant(1.0e-6, dtype = tf.float32)
        y_train_safe = y_train + eps
        z_y_train = (tf.math.log(y_train_safe) - mu_train) / sigma_train
        S1_y_train = S1(z_y_train, k = k)
        z_ts_train = (tf.math.log(ts_grid) - mu_train) / sigma_train
        S1_ts_train = S1(z_ts_train, k = k)

        y_test_safe = y_test + eps
        z_y_test = (tf.math.log(y_test_safe) - mu_test) / sigma_test
        S1_y_test = S1(z_y_test, k = k)
        z_ts_test = (tf.math.log(ts_grid) - mu_test) / sigma_test
        S1_ts_test = S1(z_ts_test, k = k)
    
        S_ts_train = p_train + (1-p_train) * S1_ts_train
        S_ts_test = p_test + (1-p_test) * S1_ts_test
        S_train = p_train + (1-p_train) * S1_y_train
        S_test = p_test + (1-p_test) * S1_y_test

        S_train = np.clip(S_train, 1e-15, 1.0)
        S_test = np.clip(S_test, 1e-15, 1.0)
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        if(hasattr(y_train, "to_numpy")):
            y_train = y_train.to_numpy()
        if(hasattr(delta_train, "to_numpy")):
            delta_train = delta_train.to_numpy()
        if(hasattr(y_test, "to_numpy")):
            y_test = y_test.to_numpy()
        if(hasattr(delta_test, "to_numpy")):
            delta_test = delta_test.to_numpy()
            
        return {
            "ts_grid": ts_grid,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "mu_train": mu_train,
            "mu_test": mu_test,
            "sigma_train": sigma_train,
            "sigma_test": sigma_test,
            "p_train": p_train,
            "p_test": p_test,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test
        }

    return aft_model_parameters, loglikelihood_loss, get_survival_cure


def build_cure_aft_structure3(aft_distribution, z_mu, z_sigma, z_p):

    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )
    
    aft_model_parameters = {
        "mu_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_mu.shape[1], "init": None},
        "sigma_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_sigma.shape[1], "init": None},
        "p_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_p.shape[1], "init": None},
        "p_nn": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None}
    }
    
    # Weibull distribution - Gumbel formulation
    if(aft_distribution == "wb"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.exp(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) + z - tf.math.exp(z)
            
    # Log-normal distribution - Normal formulation
    elif(aft_distribution == "ln"):
        def S1(z, k = None):
            norm_cdf = ( 1 + tf.math.erf(z / tf.math.sqrt(2.0)) ) / 2.0
            return 1 - norm_cdf
        def log_f1(t, z, sigma, k = None):
            c = tf.cast(0.5 * tf.math.log(2.0 * np.pi), tf.float32)
            return -tf.math.log(t) - tf.math.log(sigma) - z**2 / 2 - c
            
    # Log-logistic distribution - Logistic formulation
    elif(aft_distribution == "ll"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.softplus(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) - 2.0 * tf.math.softplus(-z) - z
            
    # Generalized-Gamma distribution - Log-Gamma formulation
    elif(aft_distribution == "gg"):
        # Consider an extra global shape parameter for the Generalized Gamma distribution
        aft_model_parameters["k"] = {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None}

        def S1(z, k = None):
            return 2
        def log_f1(t, z, sigma, k = None):
            return 2
    else:
        raise Exception("Please, specify model in implemented family of AFT models. ('wb', 'ln', 'll', 'gg')")

    def loglikelihood_loss(model, nn_output, data):
        X, z_mu_data, z_sigma_data, z_p_data, y, delta = data

        k = None
        if(aft_distribution == "gg"):
            k = model.get_variable("k")
        
        mu_beta = model.get_variable("mu_beta")[:,None]
        mu = tf.linalg.matmul( z_mu_data, mu_beta )

        sigma_beta = model.get_variable("sigma_beta")[:,None]
        sigma = tf.math.exp( tf.linalg.matmul( z_sigma_data, sigma_beta ) )
        
        eps = tf.constant(1.0e-6, dtype = tf.float32)
        
        p_beta = model.get_variable("p_beta")[:,None]
        p_nn = model.get_variable("p_nn", nn_output)
        logit_p = tf.linalg.matmul( z_p_data, p_beta ) + p_nn
        logit_p = tf.clip_by_value(logit_p, -80.0, 80.0)
        p = tf.clip_by_value( tf.math.sigmoid( logit_p ), eps, 1.0-eps )
        log_p = tf.math.log(p)
        exp_mlogit_p = tf.math.exp(-logit_p)

        y_safe = y + eps
        z_y = (tf.math.log(y_safe) - mu) / sigma
        S1_y = S1(z_y, k = k)
        log_f1_y = log_f1(y_safe, z_y, sigma, k = k)
        
        log_Sp = log_p + tf.math.log1p(exp_mlogit_p * S1_y)
        log_fp = tf.math.log( 1-p ) + log_f1_y

        loglik_terms = delta * log_fp + (1.0 - delta) * log_Sp
        neg_loglik = -tf.reduce_sum(loglik_terms)
        return neg_loglik

    def get_survival_cure(model,
                          y_train, delta_train, X_train, z_train_mu, z_train_sigma, z_train_p,
                          y_test, delta_test, X_test, z_test_mu, z_test_sigma, z_test_p,
                          ngrid = 100):

        y_train = tf.cast(y_train, tf.float32)
        delta_train = tf.cast(delta_train, tf.float32)
        X_train = tf.cast(X_train, tf.float32)
        z_train_mu = tf.cast(z_train_mu, tf.float32)
        z_train_sigma = tf.cast(z_train_sigma, tf.float32)
        z_train_p = tf.cast(z_train_p, tf.float32)
        
        y_test = tf.cast(y_test, tf.float32)
        delta_test = tf.cast(delta_test, tf.float32)
        X_test = tf.cast(X_test, tf.float32)
        z_test_mu = tf.cast(z_test_mu, tf.float32)
        z_test_sigma = tf.cast(z_test_sigma, tf.float32)
        z_test_p = tf.cast(z_test_p, tf.float32)
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        k = None
        if(aft_distribution == "gg"):
            k = model.predict("k")

        p_beta = model.predict("p_beta")[:,None]
    
        p_nn_train = pred_train["p_nn"]

        logit_p_train = tf.linalg.matmul( z_train_p, p_beta ) + p_nn_train
        p_train = tf.math.sigmoid( logit_p_train )
        p_nn_test = pred_test["p_nn"]
        logit_p_test = tf.linalg.matmul( z_test_p, p_beta ) + p_nn_test
        p_test = tf.math.sigmoid( logit_p_test )
    
        sigma_beta = model.predict("sigma_beta")[:,None]
        sigma_train = tf.math.exp( tf.linalg.matmul( z_train_sigma, sigma_beta ) )
        sigma_test = tf.math.exp( tf.linalg.matmul( z_test_sigma, sigma_beta ) )
    
        mu_beta = model.predict("mu_beta")[:,None]
    
        mu_train = tf.linalg.matmul( z_train_mu, mu_beta )
        mu_test = tf.linalg.matmul( z_test_mu, mu_beta )

        p_train = p_train.numpy().flatten()
        p_test = p_test.numpy().flatten()
        mu_train = mu_train.numpy().flatten()
        mu_test = mu_test.numpy().flatten()
        sigma_train = sigma_train.numpy().flatten()
        sigma_test = sigma_test.numpy().flatten()
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

        eps = tf.constant(1.0e-6, dtype = tf.float32)
        y_train_safe = y_train + eps
        z_y_train = (tf.math.log(y_train_safe) - mu_train) / sigma_train
        S1_y_train = S1(z_y_train, k = k)
        z_ts_train = (tf.math.log(ts_grid) - mu_train) / sigma_train
        S1_ts_train = S1(z_ts_train, k = k)

        y_test_safe = y_test + eps
        z_y_test = (tf.math.log(y_test_safe) - mu_test) / sigma_test
        S1_y_test = S1(z_y_test, k = k)
        z_ts_test = (tf.math.log(ts_grid) - mu_test) / sigma_test
        S1_ts_test = S1(z_ts_test, k = k)
    
        S_ts_train = p_train + (1-p_train) * S1_ts_train
        S_ts_test = p_test + (1-p_test) * S1_ts_test
        S_train = p_train + (1-p_train) * S1_y_train
        S_test = p_test + (1-p_test) * S1_y_test

        S_train = np.clip(S_train, 1e-15, 1.0)
        S_test = np.clip(S_test, 1e-15, 1.0)
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        if(hasattr(y_train, "to_numpy")):
            y_train = y_train.to_numpy()
        if(hasattr(delta_train, "to_numpy")):
            delta_train = delta_train.to_numpy()
        if(hasattr(y_test, "to_numpy")):
            y_test = y_test.to_numpy()
        if(hasattr(delta_test, "to_numpy")):
            delta_test = delta_test.to_numpy()
            
        return {
            "ts_grid": ts_grid,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "mu_train": mu_train,
            "mu_test": mu_test,
            "sigma_train": sigma_train,
            "sigma_test": sigma_test,
            "p_train": p_train,
            "p_test": p_test,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test
        }

    return aft_model_parameters, loglikelihood_loss, get_survival_cure




def build_cure_aft_structure4(aft_distribution, z_mu, z_sigma, z_p):

    def logit(u):
        return -( tf.math.log(1-u) - tf.math.log(u) )
    
    aft_model_parameters = {
        "mu_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_mu.shape[1], "init": None},
        "mu_nn": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None},
        "sigma_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_sigma.shape[1], "init": None},
        "p_beta": {"link": tf.identity, "link_inv": tf.identity, "par_type": "independent", "shape": z_p.shape[1], "init": None},
        "p_nn": {"link": tf.identity, "link_inv": tf.identity, "par_type": "nn", "shape": 1, "init": None}
    }
    
    # Weibull distribution - Gumbel formulation
    if(aft_distribution == "wb"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.exp(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) + z - tf.math.exp(z)
            
    # Log-normal distribution - Normal formulation
    elif(aft_distribution == "ln"):
        def S1(z, k = None):
            norm_cdf = ( 1 + tf.math.erf(z / tf.math.sqrt(2.0)) ) / 2.0
            return 1 - norm_cdf
        def log_f1(t, z, sigma, k = None):
            c = tf.cast(0.5 * tf.math.log(2.0 * np.pi), tf.float32)
            return -tf.math.log(t) - tf.math.log(sigma) - z**2 / 2 - c
            
    # Log-logistic distribution - Logistic formulation
    elif(aft_distribution == "ll"):
        def S1(z, k = None):
            return tf.math.exp( -tf.math.softplus(z) )
        def log_f1(t, z, sigma, k = None):
            return -tf.math.log(t) - tf.math.log(sigma) - 2.0 * tf.math.softplus(-z) - z
            
    # Generalized-Gamma distribution - Log-Gamma formulation
    elif(aft_distribution == "gg"):
        # Consider an extra global shape parameter for the Generalized Gamma distribution
        aft_model_parameters["k"] = {"link": tf.math.exp, "link_inv": tf.math.log, "par_type": "independent", "shape": 1, "init": None}

        def S1(z, k = None):
            return 2
        def log_f1(t, z, sigma, k = None):
            return 2
    else:
        raise Exception("Please, specify model in implemented family of AFT models. ('wb', 'ln', 'll', 'gg')")

    def loglikelihood_loss(model, nn_output, data):
        X, z_mu_data, z_sigma_data, z_p_data, y, delta = data

        k = None
        if(aft_distribution == "gg"):
            k = model.get_variable("k")
        
        mu_beta = model.get_variable("mu_beta")[:,None]
        mu_nn = model.get_variable("mu_nn", nn_output)
        mu = tf.linalg.matmul( z_mu_data, mu_beta ) + mu_nn

        sigma_beta = model.get_variable("sigma_beta")[:,None]
        sigma = tf.math.exp( tf.linalg.matmul( z_sigma_data, sigma_beta ) )
        
        eps = tf.constant(1.0e-6, dtype = tf.float32)
        
        p_beta = model.get_variable("p_beta")[:,None]
        p_nn = model.get_variable("p_nn", nn_output)
        logit_p = tf.linalg.matmul( z_p_data, p_beta ) + p_nn
        logit_p = tf.clip_by_value(logit_p, -80.0, 80.0)
        p = tf.clip_by_value( tf.math.sigmoid( logit_p ), eps, 1.0-eps )
        log_p = tf.math.log(p)
        exp_mlogit_p = tf.math.exp(-logit_p)

        y_safe = y + eps
        z_y = (tf.math.log(y_safe) - mu) / sigma
        S1_y = S1(z_y, k = k)
        log_f1_y = log_f1(y_safe, z_y, sigma, k = k)
        
        log_Sp = log_p + tf.math.log1p(exp_mlogit_p * S1_y)
        log_fp = tf.math.log( 1-p ) + log_f1_y

        loglik_terms = delta * log_fp + (1.0 - delta) * log_Sp
        neg_loglik = -tf.reduce_sum(loglik_terms)
        return neg_loglik

    def get_survival_cure(model,
                          y_train, delta_train, X_train, z_train_mu, z_train_sigma, z_train_p,
                          y_test, delta_test, X_test, z_test_mu, z_test_sigma, z_test_p,
                          ngrid = 100):

        y_train = tf.cast(y_train, tf.float32)
        delta_train = tf.cast(delta_train, tf.float32)
        X_train = tf.cast(X_train, tf.float32)
        z_train_mu = tf.cast(z_train_mu, tf.float32)
        z_train_sigma = tf.cast(z_train_sigma, tf.float32)
        z_train_p = tf.cast(z_train_p, tf.float32)
        
        y_test = tf.cast(y_test, tf.float32)
        delta_test = tf.cast(delta_test, tf.float32)
        X_test = tf.cast(X_test, tf.float32)
        z_test_mu = tf.cast(z_test_mu, tf.float32)
        z_test_sigma = tf.cast(z_test_sigma, tf.float32)
        z_test_p = tf.cast(z_test_p, tf.float32)
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        k = None
        if(aft_distribution == "gg"):
            k = model.predict("k")

        p_beta = model.predict("p_beta")[:,None]
    
        p_nn_train = pred_train["p_nn"]

        logit_p_train = tf.linalg.matmul( z_train_p, p_beta ) + p_nn_train
        p_train = tf.math.sigmoid( logit_p_train )
        p_nn_test = pred_test["p_nn"]
        logit_p_test = tf.linalg.matmul( z_test_p, p_beta ) + p_nn_test
        p_test = tf.math.sigmoid( logit_p_test )
    
        sigma_beta = model.predict("sigma_beta")[:,None]
        sigma_train = tf.math.exp( tf.linalg.matmul( z_train_sigma, sigma_beta ) )
        sigma_test = tf.math.exp( tf.linalg.matmul( z_test_sigma, sigma_beta ) )
    
        mu_beta = model.predict("mu_beta")[:,None]
    
        mu_nn_train = pred_train["mu_nn"]
        mu_train = tf.linalg.matmul( z_train_mu, mu_beta ) + mu_nn_train
        mu_nn_test = pred_test["mu_nn"]
        mu_test = tf.linalg.matmul( z_test_mu, mu_beta ) + mu_nn_test

        p_train = p_train.numpy().flatten()
        p_test = p_test.numpy().flatten()
        mu_train = mu_train.numpy().flatten()
        mu_test = mu_test.numpy().flatten()
        sigma_train = sigma_train.numpy().flatten()
        sigma_test = sigma_test.numpy().flatten()
    
        ts_grid = np.linspace(0.0001 , np.max(np.concatenate([y_train, y_test])), ngrid)[:,None]

        eps = tf.constant(1.0e-6, dtype = tf.float32)
        y_train_safe = y_train + eps
        z_y_train = (tf.math.log(y_train_safe) - mu_train) / sigma_train
        S1_y_train = S1(z_y_train, k = k)
        z_ts_train = (tf.math.log(ts_grid) - mu_train) / sigma_train
        S1_ts_train = S1(z_ts_train, k = k)

        y_test_safe = y_test + eps
        z_y_test = (tf.math.log(y_test_safe) - mu_test) / sigma_test
        S1_y_test = S1(z_y_test, k = k)
        z_ts_test = (tf.math.log(ts_grid) - mu_test) / sigma_test
        S1_ts_test = S1(z_ts_test, k = k)
    
        S_ts_train = p_train + (1-p_train) * S1_ts_train
        S_ts_test = p_test + (1-p_test) * S1_ts_test
        S_train = p_train + (1-p_train) * S1_y_train
        S_test = p_test + (1-p_test) * S1_y_test

        S_train = np.clip(S_train, 1e-15, 1.0)
        S_test = np.clip(S_test, 1e-15, 1.0)
        H_train = -np.log( S_train )
        H_test = -np.log( S_test )
    
        if(hasattr(y_train, "to_numpy")):
            y_train = y_train.to_numpy()
        if(hasattr(delta_train, "to_numpy")):
            delta_train = delta_train.to_numpy()
        if(hasattr(y_test, "to_numpy")):
            y_test = y_test.to_numpy()
        if(hasattr(delta_test, "to_numpy")):
            delta_test = delta_test.to_numpy()
            
        return {
            "ts_grid": ts_grid,
            "y_train": y_train,
            "y_test": y_test,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "mu_train": mu_train,
            "mu_test": mu_test,
            "sigma_train": sigma_train,
            "sigma_test": sigma_test,
            "p_train": p_train,
            "p_test": p_test,
            "S_ts_train": S_ts_train,
            "S_ts_test": S_ts_test,
            "S_train": S_train,
            "S_test": S_test,
            "H_train": H_train,
            "H_test": H_test
        }

    return aft_model_parameters, loglikelihood_loss, get_survival_cure





