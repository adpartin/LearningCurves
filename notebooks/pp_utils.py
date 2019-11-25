""" Util funcs for post-processing. """
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd
from glob import glob

import matplotlib
import matplotlib.pyplot as plt

filepath = Path(__file__).resolve().parent

# Make all python scripts available in the path
sys.path.append('filepath/../')

import lrn_crv_plot


def get_xy(scores=None, path:str=None, metric_name:str='mean_absolute_error', tr_set:str='te', shard_min_idx:int=0, cv_folds:int=1):
    """ Extract x and y (tr size and score) for the plot.
    Args:
        df : df that contains all the scores
        path : as an alternative to the actual df, allow to pass a full path to lrn_crv_scores.csv
        tr_set : 'tr', 'vl', 'te'
    Returns:
        x : vector of tr size 
        y : vector of scores
    """
    if scores is not None:
        scores = scores.copy()
    elif path is not None:
        dpath = Path(path)/'lrn_crv_scores.csv'
        if dpath.exists():
            scores = pd.read_csv( dpath )
        else:
            return (None, None)
        # scores = pd.read_csv( Path(path)/'lrn_crv_scores.csv' )
        
    df = scores[scores['metric']==metric_name].reset_index(drop=True)

    fold_col_names = [c for c in df.columns if 'fold' in c]
    cols = ['tr_size'] + fold_col_names[:cv_folds]
    dd = df.loc[df['set']==tr_set, cols].sort_values('tr_size').reset_index(drop=True)

    x = dd['tr_size'].values
    # y = dd.iloc[:,1:].mean(axis=1).values # mean over folds
    y = dd.iloc[:,1:].median(axis=1).values # median over folds
    x, y = x[shard_min_idx:], y[shard_min_idx:]
    return x, y


def plot_lc_multi_runs(runs:list, labels:list=None, metric_name:str='mean_absolute_error', tr_set:str='te', cv_folds:int=1,
                       shard_min_idx:int=0, xtick_scale:str='log2', ytick_scale:str='log2', figsize:tuple=(6, 4.5)):
    """ Plot LCs from multiple runs of the same graph.
    Args:
        runs : list of dirs where each dir contains lrn_crv_scores.csv
        labels : list of labels (strings) for the corresponding runs
        tr_set : 'tr', 'vl', 'te'
    """
    for i, r in enumerate(runs):        
        x, y = get_xy(path=Path(r), metric_name=metric_name, cv_folds=cv_folds, tr_set=tr_set, shard_min_idx=shard_min_idx)
        if x is None: continue
        
        plot_kwargs = {'x': x, 'y': y, 'metric_name': metric_name, 'figsize': figsize,
                       'xtick_scale': ytick_scale, 'ytick_scale': ytick_scale,
                       'marker': '.', 'alpha': 0.7, 'title': 'Learning Curves'}
        
        if i == 0:
            ax = None
            
        if labels is None:
            ax = lrn_crv_plot.plot_lrn_crv_new(ax=ax, **plot_kwargs)
        else:
            ax = lrn_crv_plot.plot_lrn_crv_new(ax=ax, **plot_kwargs, label=labels[i])
            
    return ax
        
        
def fit_lc_multi_runs(runs:list, labels:list=None, metric_name:str='mean_absolute_error', tr_set='te', cv_folds:int=1,
                      shard_min_idx:int=0, xtick_scale:str='log2', ytick_scale:str='log2', figsize:tuple=(6, 4.5)):
    """ Plot LCs from multiple runs of the same graph.
    Args:
        runs : list of dirs where each dir contains lrn_crv_scores.csv
        labels : list of labels (strings) for the corresponding runs
        tr_set : 'tr', 'vl', 'te'
    """
    for i, r in enumerate(runs):
        x, y = get_xy(path=Path(r), metric_name=metric_name, cv_folds=cv_folds, tr_set=tr_set, shard_min_idx=shard_min_idx)
        
        plot_kwargs = {'x': x, 'y': y, 'metric_name': metric_name, 'figsize': figsize,
                       'xtick_scale': ytick_scale, 'ytick_scale': ytick_scale}

        if i == 0:
            ax = None
            
        if labels is None:
            ax, fit_prms, rmse = lrn_crv_plot.plot_lrn_crv_power_law(ax=ax, **plot_kwargs)
        else:
            ax, fit_prms, rmse = lrn_crv_plot.plot_lrn_crv_power_law(ax=ax, **plot_kwargs, label=labels[i])
            
    return ax

            
