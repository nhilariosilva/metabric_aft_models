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

def build_cure_aft(aft_distribution):

    # Weibull distribution - Gumbel formulation
    if(aft_distribution == "wb"):
        def S1(t, mu, sigma, p):
            
            
    # Log-normal distribution - Normal formulation
    elif(aft_distribution == "ln"):
        pass
    # Log-logistic distribution - Logistic formulation
    elif(aft_distribution == "ll"):
        pass
    # Generalized-Gamma distribution - Log-Gamma formulation
    elif(aft_distribution == "gg"):
        pass
    else:
        raise Exception("Please, specify model in implemented family of AFT models. ('wb', 'ln', 'll', 'gg')")

    




