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


def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))              
        

def scale_ticks_params(tick_scale='linear'):
    """ Helper function for learning cureve plots.
    Args:
        tick_scale : available values are [linear, log2, log10]
    """
    if tick_scale == 'linear':
        base = None
        label_scale = 'Linear Scale'
    else:
        if tick_scale == 'log2':
            base = 2
            label_scale = 'Log2 Scale'
        elif tick_scale == 'log10':
            base = 10
            label_scale = 'Log10 Scale'
        else:
            raise ValueError('The specified tick scale is not supported.')
    return base, label_scale        

            
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


    
def plot_runtime(rt:pd.DataFrame, outdir:Path=None, figsize=(7,5),
        xtick_scale:str='linear', ytick_scale:str='linear'):
    """ Plot training time vs shard size. """
    fontsize = 13
    fig, ax = plt.subplots(figsize=figsize)
    for f in rt['fold'].unique():
        d = rt[rt['fold']==f]
        ax.plot(d['tr_sz'], d['time'], '.--', label='fold'+str(f))
       
    # Set axes scale and labels
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)

    ax.set_xlabel(f'Train Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ax.set_ylabel(f'Training Time (minutes) ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)

    ax.set_title('Runtime')
    #ax.set_xlabel(f'Training Size', fontsize=fontsize)
    #ax.set_ylabel(f'Training Time (minutes)', fontsize=fontsize)
    ax.legend(loc='best', frameon=True, fontsize=fontsize)
    ax.grid(True)

    # Save fig
    if outdir is not None: plt.savefig(outdir/'runtime.png', bbox_inches='tight')    
