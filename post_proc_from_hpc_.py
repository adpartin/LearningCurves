"""
This script contains functions that help with meta-analysis of the different runs.
Need to pass the main hpo dir as input arg (e.g., /vol/ml/apartin/projects/LearningCurves/trn.gdsc/hpo_lgbm/trn.gdsc/split_on_none).
Note! The diff beteen this script and post_proc_from_hpc.py is the probably the func get_df_from_upf_runs

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


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate summary table from serial HPO run of learning curves.')
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to the main HPO dir (not Theta/Summit) (default: None).')
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


def parse_args_file(path_args):
    """ ... """
    with open(path_args, 'r') as f:
        args = {}
        for l in f:
            k, v = l.rstrip('\n').split(': ')
            args[k] = v
    return args


# def parse_and_agg_scores(run_path, args, scores_fname='scores.csv', hp=[]):
def parse_and_agg_scores(run_path, hp=[], missing_runs=[], scores_fname='scores.csv'):    
    """ ... """
    # Choose the smallest fold out of available folds
    fold = np.unique( sorted([str(r).split('/cv')[-1].split('_')[0] for r in sorted(run_path.glob('**/cv0_sz*'))]) )[0]
    sz_dirs = sorted( run_path.glob(f'**/cv{fold}_sz*') )
    
    for sz_dir in sz_dirs:
        path_scores = sz_dir / scores_fname
        
        if path_scores.exists(): # check if scores exist (some trainings may not have completed)
            scr = pd.read_csv( path_scores )
            tr_size = int( str(sz_dir).split('sz')[-1] )
            # aa = scr.loc[ scr['set']=='te', ['metric', 'fold1'] ].reset_index(drop=True) # Note! Gets only the first fold!
            fold_col_name = sorted( [c for c in scr.columns if 'fold' in c] )[0]
            aa = scr.loc[ scr['set']=='te', ['metric', fold_col_name] ].reset_index(drop=True) # Note! Gets only the first fold!
            aa = {aa.loc[i, 'metric']: aa.loc[i, fold_col_name] for i in range(aa.shape[0])}
            aa['tr_size'] = tr_size

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
            
    return hp, missing_runs


def get_df_from_upf_runs(hpo_dir, scores_fname='scores.csv'):
    """ Aggregate results from all HPO runs into a dataframe. """
    prfx_dir = 'run_'  # each run starts with this prefix "id_"
    all_runs = sorted(glob(str(hpo_dir/f'{prfx_dir}*'))) 
    print('Total runs {}'.format( len(all_runs) ))
    
    hp = []
    missing_runs = []
    t0 = time()
    for i, r in enumerate(all_runs):
        # run_path = Path( glob(str(Path(r)/'output'/'*'))[0] )
        # run_path = Path( glob(str(Path(r)/'output'/'*'/'*'))[0] )
        run_path = Path(r)
        print(str(run_path))

        # Parse scores, and agg
        hp, missing_runs = parse_and_agg_scores(run_path, hp=hp, missing_runs=missing_runs, scores_fname=scores_fname)

        if (i+1)%50 == 0: print('Done with {}'.format(i+1))

    print( 'Runtime: {:.1f} min'.format( (time()-t0)/60 ) )
    hp = pd.DataFrame(hp)
    
    # Dump missing runs into text file
    with open(hpo_dir/'missing_runs.txt', 'w') as f:
        for line in missing_runs:
            f.write(str(line)+'\n')

    # Cast columns to specific types
    cast_types = {'batch_size': int, 'lr': float, 'opt': str, 'dr_rate': float, 
                  'mean_absolute_error': float, 'mean_squared_error': float, 'median_absolute_error': float, 'r2': float}
    hp = hp.astype({k: cast_types[k] for k in cast_types.keys() if k in hp.columns})
    return hp


# def my_lambda(df_row, hps_dct):
#     """ Return True if a run used hps that appear in hps_dct. """
#     cols = list( set(df_row.index).intersection(hps_dct.keys()) )
#     return np.array( [df_row[c]==hps_dct[c] for c in cols] ).all()


# def extract_runs_by_specific_hps(hp_df, hps_dct):
#     """ Take df where each row contains parameters of a single run, and extract all runs (rows)
#     that share the same parameters specified by hps_dct.
#     """
#     hp_df = hp_df[ hp_df.apply(lambda df_row: my_lambda(df_row, hps_dct), axis=1 ) ].sort_values('tr_size').reset_index(drop=True)
#     return hp_df


def run(args):
    hpo_dir = verify_dirpath(args['dirpath'])
    # scores_fname = 'scores_tmp.csv'
    # scores_fname = 'lrn_crv_scores.csv'
    scores_fname = 'scores.csv'
    
    hp = get_df_from_upf_runs(hpo_dir=hpo_dir, scores_fname=scores_fname)
    
    hp.to_csv(hpo_dir/'hpo_all.csv', index=False)
    
    # Get best PS-HPO
    met = 'mean_absolute_error'
    ps_best = hp.sort_values(met, ascending=True).drop_duplicates(['tr_size']).sort_values('tr_size').reset_index(drop=True)
    ps_best.to_csv(hpo_dir/'hpo_ps_best.csv', index=False)
    
#     # Get best LS-HPO
#     met = 'mean_absolute_error'
#     ls_best = top_n_runs(hp, tr_size=hp['tr_size'].max(), n=1, sort_by_metric=met, plot=False)
    
#     hp_cols = [c for c in dflt_hp_dct.keys()]
#     ls_hp_best_dct = {c: best[c].values[0] for c in hp_cols}
#     ls_best = extract_runs_by_specific_hps(hp, ls_hp_best_dct) # hp_ls_best    
    
#     ps_best = hp.sort_values(met, ascending=True).drop_duplicates(['tr_size']).sort_values('tr_size').reset_index(drop=True)
#     ps_best.to_csv(hpo_dir/'hpo_ls_best.csv', index=False)    
    
    # Subset of args
    """
    prms = ['batch_size', 'batchnorm', 'cell_fea', 'clr_base_lr', 'clr_gamma', 'clr_max_lr', 'clr_mode',
            'cv_method', 'dirpath', 'dr_rate', 'drug_fea', 'epoch_stop', 'epochs', 'framework', 'gbm_leaves',
            'gbm_lr', 'gbm_max_depth', 'gbm_trees', 'lr', 'mean_absolute_error', 'median_absolute_error',
            'model_name', 'mse', 'opt', 'r2', 'rmse', 'scaler', 'seed', 'target_name',  'tr_size',
            'id', 'run_id']
    """
    # prms = set(ps_best.columns).intersection(set(prms))
    # ps_best[sorted(prms)].to_csv(hpo_dir/'hpo_ps_best.csv', index=False)
    
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
    
    print('Done with run().')


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])