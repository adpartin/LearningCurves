"""
This script contains functions that help with meta-analysis of the different runs.
Need to pass the main hpo dir as input arg (e.g., /vol/ml/apartin/projects/LearningCurves/hpo_runs/lc_ctrp_hpo).

It dumps:
hpo_summary.csv: summary of all runs
ps_hpo_best.csv: per-subset best runs to generate the learning curve
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

# File path
filepath = Path(__file__).resolve().parent
# utils_path = os.path.abspath(os.path.join('../'))
# sys.path.append(utils_path)


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate summary table from HPO run executed on HPC (parallel).')
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to the main HPO dir (run on Theta/Summit) (default: None).')
    # parser.add_argument('--scr_fname', default=None, type=str, help='File name of the scores file (default: scores_tmp.csv).')
    args = parser.parse_args(args)
    return args


def verify_dirpath(dirpath):
    """ Verify the dirpath exists and contain the dataset. """
    if dirpath is None:
        sys.exit('Program terminated. You must specify a path to a data via the input argument -dp.')

    dirpath = Path(dirpath)
    assert dirpath.exists(), 'The specified dirpath {dirpath} (via argument -dp) was not found.'
    return dirpath


def top_n_runs(df, tr_size=None, n=10, sort_by_metric='mean_absolute_error', plot=False):
    """ Return top runs based on sort_by_metric. """
    if tr_size is not None:
        df = df[df['tr_size']==tr_size].reset_index(drop=True)
    if sort_by_metric == 'r2':
        df = df.sort_values(sort_by_metric, ascending=False).iloc[:n, :]
    else:
        df = df.sort_values(sort_by_metric, ascending=True).iloc[:n, :]
    
    if plot: display(df.transpose())
    return df.reset_index(drop=True)


def parse_args_file(args_path):
    """ ... """
    with open(args_path, 'r') as f:
        args = {}
        for l in f:
            k, v = l.rstrip('\n').split(': ')
            args[k] = v
    return args


def parse_and_agg_scores(run_path, args, scores_fname='scores_tmp.csv', hp=[]):
    """ ... """
    for tr_dir in sorted( run_path.glob('cv1_sz*') ):
        scores_path = tr_dir / scores_fname
        
        if scores_path.exists(): # check if scores exist (some trainings may not have completed)
            scr = pd.read_csv( scores_path )
            tr_size = int( str(tr_dir).split('sz')[-1] )
            aa = scr.loc[ scr['set']=='te', ['metric', 'fold1'] ].reset_index(drop=True) # Note! Gets only the first fold!
            aa = {aa.loc[i, 'metric']: aa.loc[i, 'fold1'] for i in range(aa.shape[0])}
            aa['tr_size'] = tr_size

            aa.update(args) # combine scores with args

            # If keras model, get the early stop epoch
            if (tr_dir/'krs_history.csv').exists():
                h = pd.read_csv( tr_dir/'krs_history.csv' )
                aa['epoch_stop'] = h['epoch'].max()        

            # Append results to the global list
            hp.append(aa)
            
    return hp


def get_df_from_upf_runs(hpo_dir):
    """ Aggregate results from all HPO runs into a dataframe. """
    all_runs = sorted(glob(str(hpo_dir/'run'/'id_*'))) # each run starts with "id_"
    # all_runs = sorted(glob(str(hpo_dir/'run'/'run*'))) # each run starts with "id_"
    print('Total runs {}'.format( len(all_runs) ))
    
    hp = []
    for i, r in enumerate(all_runs):
        # run_path = Path( glob(str(Path(r)/'output'/'*'))[0] )
        run_path = Path( glob(str(Path(r)/'output'/'*'/'*'))[0] )

        # Parse args from the current run into dict
        args = parse_args_file( run_path/'args.txt' )

        # Parse scores, and agg
        hp = parse_and_agg_scores(run_path, args, hp=hp)

        if (i+1)%50 == 0: print('Done with {}'.format(i+1))

    hp = pd.DataFrame(hp)

    # Cast columns to specific types
    cast_types = {'batch_size': int, 'lr': float, 'opt': str, 'dr_rate': float, 
                  'mean_absolute_error': float, 'mean_squared_error': float, 'median_absolute_error': float, 'r2': float}
    hp = hp.astype({k: cast_types[k] for k in cast_types.keys() if k in hp.columns})
    return hp


def run(args):
    hpo_dir = verify_dirpath(args['dirpath'])
    scores_fname = 'scores_tmp.csv'
    
    hp = get_df_from_upf_runs(hpo_dir=hpo_dir)
    
    hp.to_csv(hpo_dir/'hpo_all.csv', index=False)
    
    # Get best PS-HPO
    met = 'mean_absolute_error'
    ps_best = hp.sort_values(met, ascending=True).drop_duplicates(['tr_size']).sort_values('tr_size').reset_index(drop=True)
    ps_best.to_csv(hpo_dir/'hpo_ps_best.csv', index=False)
    
    # Subset of args
    prms = ['batch_size', 'batchnorm', 'cell_fea', 'clr_base_lr', 'clr_gamma', 'clr_max_lr', 'clr_mode',
            'cv_method', 'dirpath', 'dr_rate', 'drug_fea', 'epoch_stop', 'epochs', 'framework', 'gbm_leaves',
            'gbm_lr', 'gbm_max_depth', 'gbm_trees', 'id', 'lr', 'mean_absolute_error', 'median_absolute_error',
            'model_name', 'mse', 'opt', 'r2', 'rmse', 'run_id', 'scaler', 'seed', 'target_name',  'tr_size']
    prms = set(ps_best.columns).intersection(set(prms))
    ps_best[sorted(prms)].to_csv(hpo_dir/'hpo_ps_best_small.csv', index=False)
    
    # Make all python scripts available in the path
    sys.path.append('../')
    from lrn_crv_plot import capitalize_metric
    
    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(hp['tr_size'], hp[met], '.', alpha=0.5);
    plt.xlabel('Train Size (Log2)'); plt.ylabel(capitalize_metric(met) + ' (Log2)');
    plt.grid(True);
    ax.set_title('Results from Grid Search')
    ax.set_xscale('log', basex=2); ax.set_yscale('log', basey=2)
    plt.savefig(hpo_dir/'all_point_log.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(hp['tr_size'], hp[met], '.', alpha=0.5);
    plt.xlabel('Train Size (Linear)'); plt.ylabel(capitalize_metric(met) + ' (Linear)');
    plt.grid(True);
    ax.set_title('Results from Grid Search')
    plt.savefig(hpo_dir/'all_point.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(ps_best['tr_size'], ps_best[met], 'o--', alpha=0.7);
    plt.xlabel('Train Size (Log2)'); plt.ylabel(capitalize_metric(met) + ' (Log2)');
    plt.grid(True);
    ax.set_title('Results from Grid Search')
    ax.set_xscale('log', basex=2); ax.set_yscale('log', basey=2)
    plt.savefig(hpo_dir/'ps_point_log.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(ps_best['tr_size'], ps_best[met], 'o--', alpha=0.7);
    plt.xlabel('Train Size (Linear)'); plt.ylabel(capitalize_metric(met) + ' (Linear)');
    plt.grid(True);
    ax.set_title('Results from Grid Search')
    plt.savefig(hpo_dir/'ps_point.png', bbox_inches='tight')


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])
