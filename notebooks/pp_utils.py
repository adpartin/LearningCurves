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
        scores : df that contains all the scores (data from lrn_crv_scores.csv)
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
            # print(dpath)
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
                       shard_min_idx:int=0, xtick_scale:str='log2', ytick_scale:str='log2', figsize:tuple=(6, 4.5), shadow=False):
    """ Plot LCs from multiple runs of the same graph.
    Args:
        runs : list of dirs where each dir contains lrn_crv_scores.csv
        labels : list of labels (strings) for the corresponding runs
        tr_set : 'tr', 'vl', 'te'
    """
    y_agg = []
    ax = None
    for i, r in enumerate(runs):
        # print(r)
        x, y = get_xy(path=Path(r), metric_name=metric_name, cv_folds=cv_folds, tr_set=tr_set, shard_min_idx=shard_min_idx)
        if x is None:
            continue
        
        plot_kwargs = {'x': x, 'y': y, 'metric_name': metric_name, 'figsize': figsize,
                       'xtick_scale': xtick_scale, 'ytick_scale': ytick_scale,
                       'ls': '', 'marker': '.', 'alpha': 0.5, 'color': 'b',
                       #'title': 'Learning Curves'
                      }        
        # print(i)
        if len(y_agg) == 0:
            y_agg = y
        else:
            y_agg += y

        if not shadow:
            # Plot data from each loaded run
            if labels is None:
                ax = lrn_crv_plot.plot_lrn_crv_new(ax=ax, **plot_kwargs)
            else:
                ax = lrn_crv_plot.plot_lrn_crv_new(ax=ax, **plot_kwargs, label=labels[i])
    
    if shadow:
        # Instead of plotting data from each run, plot mean/median and shadow
        # TODO: this doens't work at this point!
        scores_mean = np.mean(y_agg)
        scores_std  = np.std( y_agg)
        # ax.plot(tr_shards, scores_mean, '.-', color=color, alpha=0.5, label=f'{phase} Score')
        # ax.fill_between(tr_shards, scores_mean - scores_std, scores_mean + scores_std, alpha=0.1, color=color)            
        ax = lrn_crv_plot.plot_lrn_crv_new(**plot_kwargs)
        ax.fill_between(x, scores_mean - scores_std, scores_mean + scores_std, alpha=0.1)
        # ax.set_xscale('log', basex=2)
        # ax.set_yscale('log', basey=2)
    
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

            
def plot_all_from_hp_df(hp, metric_name='mean_absolute_error', marker='.', color=None, alpha=0.5, title=None,
                        label='Data', figsize=(6, 4.5), xtick_scale='log2', ytick_scale='log2', fontsize=12):
    """ Plot all data points from ps-HPO run. """
    legend_fontsize = fontsize - 2
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(hp['tr_size'], hp[metric_name], '.', marker=marker, alpha=alpha, color=color, label=label+f' ({hp.shape[0]})');
    
    basex, xlabel_scale = lrn_crv_plot.scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = lrn_crv_plot.scale_ticks_params(tick_scale=ytick_scale)
    
    ax.set_xlabel(f'Training Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)    
    
    ylabel = lrn_crv_plot.capitalize_metric(metric_name)
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)       
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Results from all HP runs (total: {})'.format(hp.shape[0]))
        
    ax.legend(frameon=True, fontsize=legend_fontsize, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)        
    return ax


def parse_and_agg_scores_for_run(run_path, hp=[], missing_runs=[], scores_fname='scores.csv'):    
    """ This func takes a dir of a single run of learning curves, extracts and aggregates
    scores for each training set size into a single structure. """
    # Choose the smallest fold out of available folds
    if len(sorted(run_path.glob('**/cv0*sz*')))==0:
        return hp, missing_runs
    
    fold = np.unique( sorted([str(r).split('/cv')[-1].split('_')[0] for r in sorted(run_path.glob('**/cv0_sz*'))]) )[0]
    sz_dirs = sorted( run_path.glob(f'**/cv{fold}*sz*') )
    
    lrn_crv_scores = []
    
    for sz_dir in sz_dirs:
        path_scores = sz_dir / scores_fname
        print(path_scores)
        
        if path_scores.exists(): # check if scores exist (some trainings may not have completed)
            scr = pd.read_csv( path_scores )
            tr_size = int( str(sz_dir).split('sz')[-1] )
            # aa = scr.loc[ scr['set']=='te', ['metric', 'fold1'] ].reset_index(drop=True) # Note! Gets only the first fold!
            fold_col_name = sorted( [c for c in scr.columns if 'fold' in c] )[0]
            aa = scr.loc[ scr['set']=='te', ['metric', fold_col_name] ].reset_index(drop=True) # Note! Gets only the first fold!
            aa = {aa.loc[i, 'metric']: aa.loc[i, fold_col_name] for i in range(aa.shape[0])}
            aa['tr_size'] = tr_size
            
            # Agg scores from each sz to master table
            lrn_crv_scores.append(scr)

            # Read and parse args
            # path_args = sz_dir / 'model_args.txt'
            path_args = sz_dir / 'args.txt'
            if path_args.exists():
                args = parse_args_file(path_args)
                aa.update(args) # combine scores with args
            else:
                # If (some reason) the args file was not found, results are useless.
                # Don't record the results but just continue.
                continue

            # If keras model, get the early stop epoch
            if (sz_dir/'krs_history.csv').exists():
                h = pd.read_csv( sz_dir/'krs_history.csv' )
                aa['epoch_stop'] = h['epoch'].max()        

            # Append results to the global list
            hp.append(aa)
        else:
            missing_runs.append(str(sz_dir))
            print(f'{scores_fname} does not exist.')
            print('continue')
            
    lrn_crv_scores = pd.concat(lrn_crv_scores, axis=0)
    lrn_crv_scores = lrn_crv_scores.sort_values(['set', 'metric', 'tr_size']).reset_index(drop=True)
            
    return hp, missing_runs, lrn_crv_scores