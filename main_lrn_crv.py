""" 
This script generates learning curves.
It launches a script (e.g. trn_lrn_crv.py) that train ML model(s) on various training set sizes.
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
from time import time
from pprint import pprint, pformat
from collections import OrderedDict
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# File path
filepath = Path(__file__).resolve().parent


# Utils
from classlogger import Logger
from lrn_crv import LearningCurve
    
        
def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate learning curves.")

    # Input data
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')

    # Select target to predict
    parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC', 'AUC1'], help='Name of target variable (default: AUC).')

    # Select feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell line features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')
    # parser.add_argument('-of', '--other_fea', default=[], choices=[],
    #         help='Other feature types (derived from cell lines and drugs). E.g.: cancer type, etc).') # ['cell_labels', 'drug_labels', 'ctype', 'csite', 'rna_clusters']

    # Data split methods
    parser.add_argument('-cvm', '--cv_method', default='simple', type=str, choices=['simple', 'group'], help='CV split method (default: simple).')
    parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    parser.add_argument('-cvf_arr', '--cv_folds_arr', nargs='+', type=int, default=None, help='The specific folds in the cross-val run (default: None).')
    
    # ML models
    # parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    parser.add_argument('-ml', '--model_name', default='lgb_reg', type=str,
                        choices=['lgb_reg', 'rf_reg', 'nn_reg0', 'nn_reg1', 'nn_reg_layer_less', 'nn_reg_layer_more',
                                 'nn_reg_neuron_less', 'nn_reg_neuron_more'], help='ML model (default: lgb_reg).')

    # LightGBM params
    parser.add_argument('--gbm_trees', default=100, type=int, help='Number of trees (default: 100).')
    parser.add_argument('--gbm_max_depth', default=-1, type=int, help='Maximum tree depth for base learners (default: -1).')
    # parser.add_argument('--gbm_iters', default=None, type=int, help='Limit number of iterations in the prediction (default: None).')
    parser.add_argument('--gbm_lr', default=0.1, type=float, help='Boosting learning rate (default: 0.1).')
    parser.add_argument('--gbm_leaves', default=31, type=int, help='Maximum tree leaves for base learners (default: 31).')
    
    # NN hyper_params
    parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'],
            help='Feature normalization method (stnd, minmax, rbst) (default: stnd).')
    parser.add_argument('--batchnorm', action='store_true', help='Whether to use batch normalization (default: False).')
    # parser.add_argument('--initializer', default='he', type=str, choices=['he', 'glorot'], help='Keras initializer name (default: he).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')
    parser.add_argument('--lr', default='0.001', type=float, help='Learning rate of adaptive optimizers (default: 0.001).')

    parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # Learning curve
    parser.add_argument('--shard_step_scale', default='log2', type=str, choices=['log2', 'log', 'log10', 'linear'],
            help='Scale of progressive sampling of shards (log2, log, log10, linear) (default: log2).')
    parser.add_argument('--min_shard', default=128, type=int, help='The lower bound for the shard sizes (default: 128).')
    parser.add_argument('--max_shard', default=None, type=int, help='The upper bound for the shard sizes (default: None).')
    parser.add_argument('--n_shards', default=None, type=int, help='Number of shards (used only when shard_step_scale is `linear` (default: None).')
    parser.add_argument('--shards_arr', nargs='+', type=int, default=None, help='List of the actual shards in the learning curve plot (default: None).')
    
    # Other
    parser.add_argument('--n_jobs', default=4, type=int, help='Default: 4.')
    parser.add_argument('--seed', default=0, type=int, help='Default: 0.')

    # Parse args
    args = parser.parse_args(args)
    return args
        

def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [('cvf'+str(args['cv_folds']))] + args['cell_fea'] + args['drug_fea'] + [args['target_name']] 
    if args['clr_mode'] is not None: l = [args['clr_mode']] + l
    if 'nn' in args['model_name']: l = [args['opt']] + l
                
    name_sffx = '.'.join( [src] + [args['model_name']] + l )
    outdir = Path(outdir) / (name_sffx + '_' + t)
    #outdir = Path(outdir) / name_sffx
    os.makedirs(outdir)
    #os.makedirs(outdir, exist_ok=True)
    return outdir


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))
    

def get_file(fpath):
    return pd.read_csv(fpath, header=None).squeeze().values if fpath.is_file() else None


def read_data_file(fpath, file_format='csv'):
    fpath = Path(fpath)
    if fpath.is_file():
        if file_format=='csv':
            df = pd.read_csv( fpath )
        elif file_format=='parquet':
            df = pd.read_parquet( fpath )
        else:
            raise ValueError('file format is not supported.')
    return df    
    
    
def scale_fea(xdata, scaler_name='stnd', dtype=np.float32):
    """ Returns the scaled dataframe. """
    if scaler_name is not None:
        if scaler_name == 'stnd':
            scaler = StandardScaler()
        elif scaler_name == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_name == 'rbst':
            scaler = RobustScaler()
    
    cols = xdata.columns
    return pd.DataFrame( scaler.fit_transform(xdata), columns=cols, dtype=dtype )    
    
    
def run(args):
    dirpath = Path(args['dirpath'])
    assert dirpath.exists(), 'You must specify the dirpath.'

    target_name = args['target_name']
    cv_folds = args['cv_folds']
    cv_folds_arr = args['cv_folds_arr']

    # Features 
    cell_fea = args['cell_fea']
    drug_fea = args['drug_fea']
    # other_fea = args['other_fea']
    # fea_list = cell_fea + drug_fea + other_fea    
    fea_list = cell_fea + drug_fea

    # NN params
    epochs = args['epochs']
    batch_size = args['batch_size']
    dr_rate = args['dr_rate']
    batchnorm = args['batchnorm']

    # Optimizer
    opt_name = args['opt']
    lr = args['lr']
    clr_keras_kwargs = {'mode': args['clr_mode'], 'base_lr': args['clr_base_lr'],
                        'max_lr': args['clr_max_lr'], 'gamma': args['clr_gamma']}

    # Learning curve
    shard_step_scale = args['shard_step_scale']
    min_shard = args['min_shard']
    max_shard = args['max_shard']
    n_shards = args['n_shards']
    shards_arr = args['shards_arr']

    # Other params
    # framework = args['framework']
    model_name = args['model_name']
    n_jobs = args['n_jobs']

    # ML type ('reg' or 'cls')
    if 'reg' in model_name:
        mltype = 'reg'
    elif 'cls' in model_name:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")

    # Define metrics
    # metrics = {'r2': 'r2',
    #            'neg_mean_absolute_error': 'neg_mean_absolute_error', #sklearn.metrics.neg_mean_absolute_error,
    #            'neg_median_absolute_error': 'neg_median_absolute_error', #sklearn.metrics.neg_median_absolute_error,
    #            'neg_mean_squared_error': 'neg_mean_squared_error', #sklearn.metrics.neg_mean_squared_error,
    #            'reg_auroc_score': utils.reg_auroc_score}
    
    
    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    xdata = read_data_file( dirpath/'xdata.parquet', 'parquet' )
    meta  = read_data_file( dirpath/'meta.parquet', 'parquet' )
    ydata = meta[[target_name]]

    tr_id = read_data_file( dirpath/f'{cv_folds}fold_tr_id.csv' )
    vl_id = read_data_file( dirpath/f'{cv_folds}fold_vl_id.csv' )
    te_id = read_data_file( dirpath/f'{cv_folds}fold_te_id.csv' )

    src = dirpath.name.split('_')[0]


    # -----------------------------------------------
    #       Create outdir and logger
    # -----------------------------------------------
    outdir = Path( str(dirpath).split('_')[0] + '_trn' )
    run_outdir = create_outdir(outdir, args, src)
    lg = Logger(run_outdir/'logfile.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    dump_dict(args, outpath=run_outdir/'args.txt')     
    
    
    # -----------------------------------------------
    #       Data preprocessing
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features
    
    
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
    if model_name == 'lgb_reg':
        framework = 'lightgbm'
        init_kwargs = {'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
                       'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
                       'n_jobs': args['n_jobs'], 'random_state': args['seed']}
        fit_kwargs = {'verbose': False}
    elif model_name == 'rf_reg':
        framework = 'sklearn'
        init_kwargs = {'n_jobs': args['n_jobs'], 'random_state': args['seed']}
        fit_kwargs = {}
    elif model_name == 'nn_reg0' or 'nn_reg1' or 'nn_reg_layer_less' or 'nn_reg_layer_more' or 'nn_reg_neuron_less' or 'nn_reg_neuron_more':
        framework = 'keras'
        init_kwargs = {'input_dim': xdata.shape[1], 'dr_rate': dr_rate, 'opt_name': opt_name, 'lr': lr, 'batchnorm': batchnorm, 'logger': lg.logger}
        fit_kwargs = {'batch_size': batch_size, 'epochs': epochs, 'verbose': 1}  # 'validation_split': 0.1


    # -----------------------------------------------
    #      Learning curve 
    # -----------------------------------------------
    lg.logger.info('\n\n{}'.format('-' * 50))
    lg.logger.info(f'Learning curves {src} ...')
    lg.logger.info('-' * 50)

    lrn_crv_init_kwargs = { 'cv': None, 'cv_lists': (tr_id, vl_id, te_id), 'cv_folds_arr': cv_folds_arr,
            'shard_step_scale': shard_step_scale, 'n_shards': n_shards, 'min_shard': min_shard, 'max_shard': max_shard,
            'shards_arr': shards_arr, 'args': args, 'logger': lg.logger, 'outdir': run_outdir}

    lrn_crv_trn_kwargs = { 'framework': framework, 'mltype': mltype, 'model_name': model_name,
            'init_kwargs': init_kwargs, 'fit_kwargs': fit_kwargs, 'clr_keras_kwargs': clr_keras_kwargs,
            'n_jobs': n_jobs, 'random_state': args['seed'] }

    t0 = time()
    lc = LearningCurve( X=xdata, Y=ydata, **lrn_crv_init_kwargs )
    lrn_crv_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
    lg.logger.info('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )


    # -------------------------------------------------
    # Learning curve (sklearn method)
    # Problem! cannot log multiple metrics.
    # -------------------------------------------------
    """
    lg.logger.info('\nStart learning curve (sklearn method) ...')
    # Define params
    metric_name = 'neg_mean_absolute_error'
    base = 10
    train_sizes_frac = np.logspace(0.0, 1.0, lc_ticks, endpoint=True, base=base)/base

    # Run learning curve
    t0 = time()
    lrn_curve_scores = learning_curve(
        estimator=model.model, X=xdata, y=ydata,
        train_sizes=train_sizes_frac, cv=cv, groups=groups,
        scoring=metric_name,
        n_jobs=n_jobs, exploit_incremental_learning=False,
        random_state=SEED, verbose=1, shuffle=False)
    lg.logger.info('Runtime: {:.1f} mins'.format( (time()-t0)/60) )

    # Dump results
    # lrn_curve_scores = utils.cv_scores_to_df(lrn_curve_scores, decimals=3, calc_stats=False) # this func won't work
    # lrn_curve_scores.to_csv(os.path.join(run_outdir, 'lrn_curve_scores_auto.csv'), index=False)

    # Plot learning curves
    lrn_crv.plt_learning_curve(rslt=lrn_curve_scores, metric_name=metric_name,
        title='Learning curve (target: {}, data: {})'.format(target_name, tr_sources_name),
        path=os.path.join(run_outdir, 'auto_learning_curve_' + target_name + '_' + metric_name + '.png'))
    """
    
    lg.kill_logger()
    del xdata, ydata
    
    print('Done.')


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    

if __name__ == '__main__':
    main(sys.argv[1:])


