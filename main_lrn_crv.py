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
from plots import plot_hist, plot_runtime
import lrn_crv_plot
    
    
# Default settings
OUTDIR = filepath / './'    
    
        
def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate learning curves.")

    # Input data
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')

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
    parser.add_argument('--hpo_metric', default='mean_absolute_error', type=str, choices=['mean_absolute_error'],
            help='Metric for HPO evaluation. Required for UPF workflow on Theta HPC (default: mean_absolute_error).')

    # Learning curve
    parser.add_argument('--shard_step_scale', default='log2', type=str, choices=['log2', 'log', 'log10', 'linear'],
            help='Scale of progressive sampling of shards (log2, log, log10, linear) (default: log2).')
    parser.add_argument('--min_shard', default=128, type=int, help='The lower bound for the shard sizes (default: 128).')
    parser.add_argument('--max_shard', default=None, type=int, help='The upper bound for the shard sizes (default: None).')
    parser.add_argument('--n_shards', default=None, type=int, help='Number of shards (used only when shard_step_scale is `linear` (default: None).')
    parser.add_argument('--shards_arr', nargs='+', type=int, default=None, help='List of the actual shards in the learning curve plot (default: None).')
    
    # HPs file
    parser.add_argument('--hp_file', default=None, type=str, help='File containing hyperparameters for training (default: None).')
    
    # Other
    parser.add_argument('--n_jobs', default=4, type=int, help='Default: 4.')
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
    outdir = Path( 'trn.' + src) / ('split_on_' + args['split_on']) / fname
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
    outdir = create_outdir(OUTDIR, args, src)
    args['outdir'] = outdir
    lg = Logger(outdir/'logfile.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')
    
    
    # -----------------------------------------------
    #       Data preprocessing
    # -----------------------------------------------
    xdata = scale_fea(xdata=xdata, scaler_name=args['scaler'])  # scale features
    
    
    # -----------------------------------------------
    #      ML model configs
    # -----------------------------------------------
#     # TODO: consider creating a per-model file that lists the init and fit parametrs!
#     # This will probably require the CANDLE functionality in terms of specifying input args.
#     if args['model_name'] == 'lgb_reg':
#         args['framework'] = 'lightgbm'
#         init_kwargs = { 'n_estimators': args['gbm_trees'], 'max_depth': args['gbm_max_depth'],
#                         'learning_rate': args['gbm_lr'], 'num_leaves': args['gbm_leaves'],
#                         'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
#         fit_kwargs = {'verbose': False}

#     elif args['model_name'] == 'rf_reg':
#         args['framework'] = 'sklearn'
#         init_kwargs = { 'n_estimators': args['rf_trees'], 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }
#         fit_kwargs = {}

#     elif args['model_name'] == 'nn_reg0' or 'nn_reg1' or 'nn_reg_layer_less' or 'nn_reg_layer_more' or 'nn_reg_neuron_less' or 'nn_reg_neuron_more':
#         args['framework'] = 'keras'
#         init_kwargs = { 'input_dim': xdata.shape[1], 'dr_rate': args['dr_rate'], 'opt_name': args['opt'],
#                         'lr': args['lr'], 'batchnorm': args['batchnorm'], 'logger': lg.logger }
#         fit_kwargs = { 'batch_size': args['batch_size'], 'epochs': args['epochs'], 'verbose': 1 }
        
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
    
    lrn_crv_init_kwargs = { 'cv': None, 'cv_lists': (tr_id, vl_id, te_id), 'cv_folds_arr': args['cv_folds_arr'],
                            'shard_step_scale': args['shard_step_scale'], 'n_shards': args['n_shards'],
                            'min_shard': args['min_shard'], 'max_shard': args['max_shard'], 'outdir': args['outdir'],
                            'shards_arr': args['shards_arr'], 'args': args, 'logger': lg.logger } 
                    
    lrn_crv_trn_kwargs = { 'framework': args['framework'], 'mltype': mltype, 'model_name': args['model_name'],
                           'clr_keras_kwargs': clr_keras_kwargs, 'n_jobs': args['n_jobs'], 'random_state': args['seed'] }        
    
    lc = LearningCurve( X=xdata, Y=ydata, meta=meta, **lrn_crv_init_kwargs )        

    if args['hp_file'] is None:
        # The regular workflow where all subsets are trained with the same HPs
        model_init_kwargs, model_fit_kwargs = get_model_kwargs(args)
        lrn_crv_trn_kwargs['init_kwargs'] = model_init_kwargs
        lrn_crv_trn_kwargs['fit_kwargs'] = model_fit_kwargs
        
        lrn_crv_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
    else:
        # The workflow follows PS-HPO where we a the set HPs per subset.
        # In this case we need to call the trn_learning_curve() method for
        # every subset with appropriate HPs. We'll need to update shards_arr
        # for every run of trn_learning_curve().
        fpath = verify_dirpath(args['hp_file'])
        hp = pd.read_csv(fpath)
        hp.to_csv(args['outdir']/'hpo_ps.csv', index=False)

        # Params to update based on framework
        if args['framework'] == 'lightgbm':
            prm_names = ['gbm_trees', 'gbm_max_depth', 'gbm_lr', 'gbm_leaves']
        elif args['framework'] == 'sklearn':
            prm_names = ['rf_trees']
        elif args['framework'] == 'keras':
            prm_names = ['dr_rate', 'opt', 'lr', 'batchnorm', 'batch_size']

        # Params of interest
        df_print = hp[ prm_names + ['tr_size', 'mean_absolute_error'] ]
        lg.logger.info( df_print )

        # Find the intersect btw available and requested tr sizes
        tr_sizes = list( set(lc.tr_shards).intersection(set(hp['tr_size'].unique())) )
        lg.logger.info('\nIntersect btw available and requested tr sizes: {}'.format( tr_sizes ))

        lrn_crv_scores = []
        for sz in tr_sizes:
            prm = hp[hp['tr_size']==sz]
            lrn_crv_init_kwargs['shards_arr'] = [sz]
            lc.tr_shards = [sz] 
            
            # Update model_init and model_fit params
            prm = prm.to_dict(orient='records')[0]  # unroll single-raw df into dict
                
            # Update args
            lg.logger.info('\nUpdate args for tr size {}'.format(sz))
            lg.logger.info( df_print[ df_print['tr_size']==sz ] )
            for n in prm_names:
                lg.logger.info('{}: set to {}'.format(n, prm[n]))
                args[n] = prm[n]

            model_init_kwargs, model_fit_kwargs = get_model_kwargs(args)
            lrn_crv_trn_kwargs['init_kwargs'] = model_init_kwargs
            lrn_crv_trn_kwargs['fit_kwargs'] = model_fit_kwargs

            # model_init_kwargs = {prm[k] for k in model_init_kwargs.keys()}
            # model_fit_kwargs = {prm[k] for k in model_fit_kwargs.keys()}

            per_subset_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
            lrn_crv_scores.append( per_subset_scores )

            # Print scores
            # print( per_subset_scores[ per_subset_scores['set']=='te' ] )

        # Concat per-subset scores 
        lrn_crv_scores = pd.concat(lrn_crv_scores, axis=0)

        # Save tr, vl, te separently
        lrn_crv_scores[ lrn_crv_scores['set']=='tr' ].to_csv( args['outdir']/'tr_lrn_crv_scores.csv', index=False) 
        lrn_crv_scores[ lrn_crv_scores['set']=='vl' ].to_csv( args['outdir']/'vl_lrn_crv_scores.csv', index=False) 
        lrn_crv_scores[ lrn_crv_scores['set']=='te' ].to_csv( args['outdir']/'te_lrn_crv_scores.csv', index=False) 

    # Dump all scores
    lrn_crv_scores.to_csv( args['outdir']/'lrn_crv_scores.csv', index=False)

    # Load results and plot
    # lrn_crv_plot.plot_runtime( runtime_df, outdir=outdir, xtick_scale='log2', ytick_scale='log2' )
    lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=args['outdir'] )
    lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=args['outdir'], xtick_scale='log2', ytick_scale='log2' )
    
    #====================================

    # Dump args
    dump_dict(args, outpath=outdir/'args.txt')        
        
        

    # -----------------------------------------------
    #      Learning curve 
    # -----------------------------------------------
    """
    lg.logger.info('\n\n{}'.format('-'*50))
    lg.logger.info(f'Learning Curves {src} ...')
    lg.logger.info('-'*50)

    lrn_crv_init_kwargs = { 'cv': None, 'cv_lists': (tr_id, vl_id, te_id), 'cv_folds_arr': args['cv_folds_arr'],
            'shard_step_scale': args['shard_step_scale'], 'n_shards': args['n_shards'], 'min_shard': args['min_shard'], 'max_shard': args['max_shard'],
            'shards_arr': args['shards_arr'], 'outdir': args['outdir'], 'args': args, 'logger': lg.logger}

    lrn_crv_trn_kwargs = { 'framework': args['framework'], 'mltype': mltype, 'model_name': args['model_name'],
            'init_kwargs': model_init_kwargs, 'fit_kwargs': model_fit_kwargs, 'clr_keras_kwargs': clr_keras_kwargs,
            'n_jobs': args['n_jobs'], 'random_state': args['seed'] }

    t0 = time()
    lc = LearningCurve( X=xdata, Y=ydata, meta=meta, **lrn_crv_init_kwargs )
    lrn_crv_scores = lc.trn_learning_curve( **lrn_crv_trn_kwargs )
    
    # Load results and plot
    # lrn_crv_plot.plot_runtime( runtime_df, outdir=outdir, xtick_scale='log2', ytick_scale='log2' )
    # lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=outdir )
    # lrn_crv_plot.plot_lrn_crv_all_metrics( lrn_crv_scores, outdir=outdir, xtick_scale='log2', ytick_scale='log2' )
    
    lg.logger.info('Runtime: {:.1f} hrs'.format( (time()-t0)/3600) )
    """


    # -------------------------------------------------
    # Learning curve (sklearn method)
    # Main problem! cannot log multiple metrics.
    # -------------------------------------------------
    """
    lg.logger.info('\nStart learning curve (sklearn method) ...')
    # Define params
    metric_name = 'neg_mean_absolute_error'
    base = 10
    train_sizes_frac = np.logspace(0.0, 1.0, lc_ticks, endpoint=True, base=base)/base

    # Run learning curve
    lrn_curve_scores = learning_curve(
        estimator=model.model, X=xdata, y=ydata,
        train_sizes=train_sizes_frac, cv=cv, groups=groups,
        scoring=metric_name,
        n_jobs=n_jobs, exploit_incremental_learning=False,
        random_state=SEED, verbose=1, shuffle=False)
    """
    
    lg.kill_logger()
    del xdata, ydata

    # This is required for HPO via UPF workflow on Theta HPC
    return lrn_crv_scores[(lrn_crv_scores['metric'] == args['hpo_metric']) & (lrn_crv_scores['set'] == 'te')].values[0][3]


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])


