""" 
Run grid-hpo for the largest shard.
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
t0 = time()


# File path
filepath = Path(__file__).resolve().parent

# Utils
from classlogger import Logger
# from lrn_crv import LearningCurve
# from plots import plot_hist, plot_runtime
# import lrn_crv_plot
    
    
def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate learning curves.")

    # Input data
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')

    # Global outdir
    parser.add_argument('-gout', '--global_outdir', default=None, type=str, help='Gloabl outdir. In this path another dir is created for the run (default: None).')

    # Select target to predict
    parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC', 'AUC1'], help='Name of target variable (default: AUC).')

    # Select feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell line features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')

    # Data split methods
    parser.add_argument('-cvf', '--cv_folds', default=1, type=str, help='Number cross-val folds (default: 1).')
    parser.add_argument('-cvf_arr', '--cv_folds_arr', nargs='+', type=int, default=None, help='The specific folds in the cross-val run (default: None).')
    
    # ML models
    # parser.add_argument('-frm', '--framework', default='lightgbm', type=str, choices=['keras', 'lightgbm', 'sklearn'], help='ML framework (default: lightgbm).')
    parser.add_argument('-ml', '--model_name', default='lgb_reg', type=str,
                        choices=['lgb_reg', 'rf_reg', 'nn_reg', 'nn_reg0', 'nn_reg1', 'nn_reg_layer_less', 'nn_reg_layer_more',
                                 'nn_reg_neuron_less', 'nn_reg_neuron_more'], help='ML model (default: lgb_reg).')

    # LightGBM params
    parser.add_argument('--gbm_trees', default=100, type=int, help='Number of trees (default: 100).')
    parser.add_argument('--gbm_max_depth', default=-1, type=int, help='Maximum tree depth for base learners (default: -1).')
    parser.add_argument('--gbm_lr', default=0.1, type=float, help='Boosting learning rate (default: 0.1).')
    parser.add_argument('--gbm_leaves', default=31, type=int, help='Maximum tree leaves for base learners (default: 31).')
    
    # Random Forest params
    parser.add_argument('--rf_trees', default=100, type=int, help='Number of trees (default: 100).')   
    
    # NN hyper_params
    parser.add_argument('-ep', '--epochs', default=200, type=int, help='Number of epochs (default: 200).')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size (default: 32).')
    parser.add_argument('--dr_rate', default=0.2, type=float, help='Dropout rate (default: 0.2).')
    parser.add_argument('-sc', '--scaler', default='stnd', type=str, choices=['stnd', 'minmax', 'rbst'],
            help='Feature normalization method (stnd, minmax, rbst) (default: stnd).')
    parser.add_argument('--batchnorm', action='store_true', help='Whether to use batch normalization (default: False).')
    # parser.add_argument('--initializer', default='he', type=str, choices=['he', 'glorot'], help='Keras initializer name (default: he).')

    parser.add_argument('--opt', default='sgd', type=str, choices=['sgd', 'adam'], help='Optimizer name (default: sgd).')
    parser.add_argument('--lr', default='0.0001', type=float, help='Learning rate of adaptive optimizers (default: 0.001).')

    parser.add_argument('--clr_mode', default=None, type=str, choices=['trng1', 'trng2', 'exp'], help='CLR mode (default: trng1).')
    parser.add_argument('--clr_base_lr', type=float, default=1e-4, help='Base lr for cycle lr.')
    parser.add_argument('--clr_max_lr', type=float, default=1e-3, help='Max lr for cycle lr.')
    parser.add_argument('--clr_gamma', type=float, default=0.999994, help='Gamma parameter for learning cycle LR.')

    # HPO metric
    # parser.add_argument('--hpo_metric', default='mean_absolute_error', type=str, choices=['mean_absolute_error'],
    #         help='Metric for HPO evaluation. Required for UPF workflow on Theta HPC (default: mean_absolute_error).')

    # Learning curve
    # parser.add_argument('--shard_step_scale', default='log2', type=str, choices=['log2', 'log', 'log10', 'linear'],
    #         help='Scale of progressive sampling of shards (log2, log, log10, linear) (default: log2).')
    # parser.add_argument('--min_shard', default=128, type=int, help='The lower bound for the shard sizes (default: 128).')
    # parser.add_argument('--max_shard', default=None, type=int, help='The upper bound for the shard sizes (default: None).')
    # parser.add_argument('--n_shards', default=None, type=int, help='Number of shards (used only when shard_step_scale is `linear` (default: None).')
    # parser.add_argument('--shards_arr', nargs='+', type=int, default=None, help='List of the actual shards in the learning curve plot (default: None).')
    
    # HPs file
    parser.add_argument('--hp_file', default=None, type=str, help='File containing hyperparameters for training (default: None).')
    
    # Other
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 4.')
    parser.add_argument('--seed', default=0, type=int, help='Default: 0.')

    # Parse args
    args = parser.parse_args(args)
    return args
        

def verify_dirpath(dirpath):
    """ Verify the dirpath exists and contain the dataset. """
    if dirpath is None:
        sys.exit('Program terminated. You must specify a path to a data via the input argument -dp.')

    dirpath = Path(dirpath)
    assert dirpath.exists(), 'The specified dirpath was not found: {dirpath}.'
    return dirpath
    
    
def create_outdir(outdir, args, src):
    t = datetime.now()
    t = [t.year, '-', t.month, '-', t.day, '_', 'h', t.hour, '-', 'm', t.minute]
    t = ''.join([str(i) for i in t])
    
    l = [args['model_name']] + [('cvf'+str(args['cv_folds']))] + args['cell_fea'] + args['drug_fea'] + [args['target_name']] 
    if args['clr_mode'] is not None: l = [args['clr_mode']] + l
    if 'nn' in args['model_name']: l = [args['opt']] + l
                
    fname = '.'.join( [src] + l ) + '_' + t
    # outdir = Path( src + '_trn' ) / ('split_on_' + args['split_on']) / fname
    # outdir = Path( 'trn.' + src) / ('split_on_' + args['split_on']) / fname
    outdir = outdir / Path( 'trn.' + src) / ('split_on_' + args['split_on']) / fname
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
    """ Returns the scaled dataframe of features. """
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
    dirpath = verify_dirpath(args['dirpath'])

    # Global outdir
    # OUTDIR = filepath/'./' if args['global_outdir'] is None else Path(args['global_outdir'])
    if args['global_outdir'] is None:
        OUTDIR = filepath/'./'
    else:
        OUTDIR = Path(args['global_outdir']).absolute()
    

    clr_keras_kwargs = {'mode': args['clr_mode'], 'base_lr': args['clr_base_lr'],
                        'max_lr': args['clr_max_lr'], 'gamma': args['clr_gamma']}

    # ML type ('reg' or 'cls')
    if 'reg' in args['model_name']:
        mltype = 'reg'
    elif 'cls' in args['model_name']:
        mltype = 'cls'
    else:
        raise ValueError("model_name must contain 'reg' or 'cls'.")

    # Find out which metadata field was used for hard split (cell, drug, or none)
    f = [f for f in dirpath.glob('*args.txt')][0]
    with open(f) as f: lines = f.readlines()
    def find_arg(lines, arg):
        return [l.split(':')[-1].strip() for l in lines if arg+':' in l][0]
    args['split_on'] = find_arg(lines, 'split_on').lower()
    args['split_seed'] = find_arg(lines, 'seed').lower()


    # -----------------------------------------------
    #       Load data and pre-proc
    # -----------------------------------------------
    # TODO: allow to pass the original dataframe and then split into x, y, m
    xdata = read_data_file( dirpath/'xdata.parquet', 'parquet' )
    meta  = read_data_file( dirpath/'meta.parquet', 'parquet' )
    ydata = meta[[ args['target_name'] ]]

    tr_id = read_data_file( dirpath/'{}fold_tr_id.csv'.format(args['cv_folds']) )
    vl_id = read_data_file( dirpath/'{}fold_vl_id.csv'.format(args['cv_folds']) )
    te_id = read_data_file( dirpath/'{}fold_te_id.csv'.format(args['cv_folds']) )

    # src = str(dirpath.parent).split('/')[-1].split('.')[0]
    src = str(dirpath.parent).split('/')[-1].split('.')[1]


    # -----------------------------------------------
    #       Create outdir and logger
    # -----------------------------------------------
    outdir = filepath/f'hpo_grid_{src}'
    os.makedirs(outdir, exist_ok=True)
    
    
    # -----------------------------------------------
    #       Data preprocessing
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features
    
    
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------

    if args['model_name'] == 'lgb_reg':
        args['framework'] = 'lightgbm'
    elif args['model_name'] == 'rf_reg':
        args['framework'] = 'sklearn'
    elif 'nn_' in args['model_name']:
        args['framework'] = 'keras'

    def get_model_kwargs(args):
        # TODO: consider creating a per-model file that lists the init and fit parametrs!
        # This may require CANDLE functionality in terms of specifying input args.
        if args['framework'] == 'lightgbm':
            # gbm_args = ['gbm_trees', 'gbm_max_depth', 'gbm_lr', 'gbm_leaves', 'n_jobs', 'seed']
            model_init_kwargs = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
                                  'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
                                  'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
            model_fit_kwargs = {'verbose': False}

        elif args['framework'] == 'sklearn':
            # rf_args = ['rf_trees', 'n_jobs', 'seed']
            model_init_kwargs = { 'n_estimators': args['rf_trees'], 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
            model_fit_kwargs = {}

        elif args['framework'] == 'keras':
            # nn_args = ['dr_rate', 'opt_name', 'lr', 'batchnorm', 'batch_size']
            model_init_kwargs = { 'input_dim': xdata.shape[1], 'dr_rate': args['dr_rate'],
                                  'opt_name': args['opt'], 'lr': args['lr'], 'batchnorm': args['batchnorm']}
            model_fit_kwargs = { 'batch_size': args['batch_size'], 'epochs': args['epochs'], 'verbose': 1 }        
        
        return model_init_kwargs, model_fit_kwargs
    #====================================
    import ml_models
    import math
    import json
    
    def get_data_by_id(idx):
        x_data = xdata.loc[idx, :].reset_index(drop=True)
        y_data = np.squeeze(ydata.loc[idx, :]).reset_index(drop=True)
        m_data = meta.loc[idx, :].reset_index(drop=True)
        return x_data, y_data, m_data

    def trn_lgbm_model(model, xtr, ytr, fit_kwargs, eval_set=None):
        """ Train and save LigthGBM model. """
        fit_kwargs = fit_kwargs
        fit_kwargs['eval_set'] = eval_set
        fit_kwargs['early_stopping_rounds'] = 10
        model.fit(xtr, ytr, **fit_kwargs)
        return model

    # Defaults used to run PS-HPO
    # GBM_LEAVES = [31, 10, 50, 100]
    # GBM_LR = [0.1, 0.01, 0.001]
    # GBM_MAX_DEPTH = [-1, 5, 10, 20]
    # GBM_TREES = [100, 1000, 2000]

    # CTRP
    GBM_LEAVES = [50, 100, 150]
    GBM_LR = [0.1, 0.05]
    GBM_MAX_DEPTH = [20, 30, 40]
    GBM_TREES = [1000, 1500, 2000]

    # NCI60
    # GBM_LEAVES = [50, 100, 150]
    # GBM_LR = [0.1, 0.01, 0.001]
    # GBM_MAX_DEPTH = [10, 30, 50]
    # GBM_TREES = [1000, 2000, 3000, 4000]

    # CCLE
    # GBM_LEAVES = [30, 50, 70]
    # GBM_LR = [0.005, 0.01, 0.05]
    # GBM_MAX_DEPTH = [-1, 30, 50, 70]
    # GBM_TREES = [1500, 2000, 3000, 4000]

    # GDSC
    # GBM_LEAVES = [70, 100, 150, 200]
    # GBM_LR = [0.3, 0.1, 0.05]
    # GBM_MAX_DEPTH = [7, 10, 15]
    # GBM_TREES = [1500, 2000, 2500, 3000, 4000]

    # GBM_LEAVES = [100, 150]
    # GBM_LR = [0.3, 0.1]
    # GBM_MAX_DEPTH = [10]
    # GBM_TREES = [2500]

    prm_grid = {'num_leaves': GBM_LEAVES, 'max_depth': GBM_MAX_DEPTH,
                'learning_rate': GBM_LR, 'n_estimators': GBM_TREES}

    from sklearn.model_selection import ParameterGrid
    sets = list(ParameterGrid(prm_grid))
    print('Total HP sets', len(sets))

    xtr, ytr, mtr = get_data_by_id( tr_id.values.reshape(-1,).tolist() ) # samples from xtr are sequentially sampled for TRAIN
    xvl, yvl, mvl = get_data_by_id( vl_id.values.reshape(-1,).tolist() ) # fixed set of VAL samples for the current CV split
    xte, yte, mte = get_data_by_id( te_id.values.reshape(-1,).tolist() ) # fixed set of TEST samples for the current CV split

    scores = {}
    records = []
    times = []

    # with open(Path(outdir)/'upf-lc.txt', 'w') as f:
    #     for item in sets:
    #         f.write(json.dumps(item)+'\n')

    f = open(Path(outdir)/'upf-lc.txt', 'w')

    for i, init_kwargs in enumerate( sets ):
        pprint(init_kwargs)
        # if 2**init_kwargs['max_depth'] <= init_kwargs['num_leaves']:
            # continue
        t0_run = time()
        init_kwargs.update({'n_jobs': args['n_jobs'], 'random_state': 0})
        estimator = ml_models.get_model(args['model_name'], init_kwargs=init_kwargs)
        model = estimator.model
        model = trn_lgbm_model(model, xtr=xtr, ytr=ytr, fit_kwargs={'verbose': False}, eval_set=(xvl, yvl))

        # Pred on test set
        y_pred = np.squeeze(model.predict(xte))
        y_true = np.squeeze(yte)

        # Calc scores
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['mse'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['rmse'] = math.sqrt( sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred) )

        print('MAE: {:.3f}'.format( scores['mean_absolute_error'] ))

        record = init_kwargs.copy()
        record.update(scores)
        records.append( record )

        f.write(json.dumps(record)+'\n')
        
        t = ( time()-t0_run )/60
        times.append( t )
        # lg.logger.info(f'Set {i} time: {t} mins')
        print(f'Set {i} time: {t} mins')

    df = pd.DataFrame(records)
    df.to_csv(outdir/'hpo_runs.csv', index=False)

    f.close()

    if (time()-t0)//3600 > 0:
        # lg.logger.info('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
        print('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    else:
        # lg.logger.info('Runtime: {:.1f} min'.format( (time()-t0)/60) )
        print('Runtime: {:.1f} min'.format( (time()-t0)/60) )
        
    # lg.kill_logger()
    print('Done.')
    del xdata, ydata


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])


