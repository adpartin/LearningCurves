""" 
This code generates CV splits and/or train/test splits of a dataset.
TODO: Add plots of the splits (e.g. drug, cell line, reponse distributions).
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 0


# Utils
from cv_splitter import cv_splitter, plot_ytr_yvl_dist

            
def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    if fit is not None:
        (mu, sigma) = stats.norm.fit(x)
        fit = stats.norm
        label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
    else:
        label = None
    
    alpha = 0.6
    fig, ax = plt.subplots()
#     sns.distplot(x, bins=bins, kde=True, fit=fit, 
#                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
#                  kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
#                  fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
#                           'label': label})
    sns.distplot(x, bins=bins, kde=False, fit=fit, 
                 hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
    plt.grid(True)
    if label is not None: plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')



