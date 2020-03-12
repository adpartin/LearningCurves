""" 
This code generates CV splits and/or train/test splits of a dataset.
TODO: Add plots of the splits (e.g. drug, cell line, reponse distributions).
TODO: Hard split doesn't work as intended due to weird funcionality of sklearn.model_selection.GroupShuffleSplit.
      This sklearn function splits groups based on group count and not based on sample count.
      Consider to look into that function and modify it.
      # https://github.com/scikit-learn/scikit-learn/issues/13369
      # https://github.com/scikit-learn/scikit-learn/issues/9193
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from time import time
import argparse
from pprint import pprint, pformat
from glob import glob

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder


# File path
filepath = Path(__file__).resolve().parent


# Utils
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist
from plots import plot_hist


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save dataset splits.')

    # Input data
    ##! parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to dir that contains the data file (default: None).')
    parser.add_argument('-dp', '--datapath', required=True, default=None, type=str, help='Full path to the data.')

    # Feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')    

    # Data split methods
    # parser.add_argument('--te_method', default='simple', choices=['simple', 'group'], help='Test split method (default: None).')
    # parser.add_argument('--cv_method', default='simple', choices=['simple', 'group'], help='Cross-val split method (default: simple).')
    parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    # parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    parser.add_argument('--split_on', type=str, default=None, choices=['cell', 'drug'], help='Specify how to make a hard split. (default: None).')
    parser.add_argument('--n_splits', type=int, default=100, help='Number of splits to generate (default: 100).')  ##! new

    # Other
    # parser.add_argument('--seed', type=int, default=0, help='Seed number (Default: 0)')
    parser.add_argument('--n_jobs', default=4,  type=int, help='Default: 4.')

    # Parse args and run
    args = parser.parse_args(args)
    return args


def verify_path(path):
    """ Verify the dirpath exists and contain the dataset. """
    if path is None:
        sys.exit('Program terminated. You must specify a correct path.')

    path = Path(path)
    assert path.exists(), 'The specified path {path} was not found.'
    return path


def load_data( datapath ):
    datapath = verify_path( datapath ) 
    if str(datapath).split('.')[-1]=='parquet':
        data = pd.read_parquet( datapath ) 
    elif str(datapath).split('.')[-1]=='hdf5':
        data = pd.read_hdf5( datapath ) 
    return data
    

def create_outdir(dirpath, args):
    if args['split_on'] is None:
        outdir = dirpath/'data_splits_seed{}'.format(args['seed'])
    else:
        outdir = dirpath/'data_splits_{}_seed{}'.format(args['split_on'], args['seed'])
    os.makedirs(outdir, exist_ok=True)
    return outdir


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))


def get_print_func(logger):
    """ Returns the python 'print' function if logger is None. Othersiwe, returns logger.info. """
    return print if logger is None else logger.info
    
    
def cnt_fea(df, fea_sep='_', verbose=True, logger=None):
    """ Count the number of features per feature type. """
    print_fn = get_print_fn(logger)

    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    
    if verbose:
        # logger.info(pformat(dct))
        print_fn( pformat(dct) )
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]
    
            
def print_intersect_on_var(df, tr_id, vl_id, te_id, grp_col='CELL', logger=None):
    """ Print intersection between train, val, and test datasets with respect
    to grp_col column if provided. df is usually metadata.
    """
    if grp_col in df.columns:
        tr_grp_unq = set(df.loc[tr_id, grp_col])
        vl_grp_unq = set(df.loc[vl_id, grp_col])
        te_grp_unq = set(df.loc[te_id, grp_col])
        print_fn = get_print_func(logger)
    
        print_fn(f'\tTotal intersects on {grp_col} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tTotal intersects on {grp_col} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}')
        print_fn(f'\tUnique {grp_col} in tr: {len(tr_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in vl: {len(vl_grp_unq)}')
        print_fn(f'\tUnique {grp_col} in te: {len(te_grp_unq)}')    



def run(args):
    t0 = time()
    te_size = split_size( args['te_size'] )
    fea_list = args['cell_fea'] + args['drug_fea']

    # Hard split
    split_on = None if args['split_on'] is None else args['split_on'].upper()
    cv_method = 'simple' if split_on is None else 'group'
    te_method = cv_method 

    # TODO: this needs to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    # -----------------------------------------------
    #       Create outdir
    # -----------------------------------------------
    ##! outdir = create_outdir(dirpath, args)
    datapath = Path(args['datapath']).absolute().parent
    sufx = 'none' if split_on is None else split_on
    out_splits = datapath / f'split_on_{sufx}'
    out_figs = out_splits / 'out_figs'
    os.makedirs(out_splits, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)
    args['out_splits'] = str(out_splits)

    # -----------------------------------------------
    #       Create logger
    # -----------------------------------------------
    ##! lg = Logger(outdir/'data_splitter_logfile.log')
    lg = Logger(out_splits/'data_splitter_logfile.log')
    print_fn = get_print_func(lg.logger)
    print_fn(f'File path: {filepath}')
    print_fn(f'\n{pformat(args)}')
    ##! dump_dict(args, outpath=outdir/'data_splitter_args.txt') # dump args.
    dump_dict(args, outpath=out_splits/'data_splitter_args.txt') # dump args.
    
    # -----------------------------------------------
    #       Load data
    # -----------------------------------------------
    ##! commented below!
    # lg.logger.info('\nLoad master dataset.')
    # files = list(dirpath.glob('./*.parquet'))
    # if len(files) > 0:
    #     data = pd.read_parquet( files[0] ) # TODO: assumes that there is only one data file
    # lg.logger.info('data.shape {}'.format(data.shape))

    print_fn('\nLoad master dataset.')
    data = load_data( args['datapath'] )
    print_fn('data.shape {}'.format(data.shape))
    # if args['datapath'].split('.')[-1]=='parquet':
    #     data = pd.read_parquet( args['datapath'] ) 
    # if args['datapath'].split('.')[-1]=='hdf5':
    #     data = pd.read_hdf5( args['datapath'] ) 
    # print_fn('data.shape {}'.format(data.shape))

    # Split features and traget, and dump to file
    ##! commented below!
    # lg.logger.info('\nSplit features and meta.')
    # xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='_')
    # meta = data.drop(columns=xdata.columns)
    # xdata.to_parquet( outdir/'xdata.parquet' )
    # meta.to_parquet( outdir/'meta.parquet' )
    
    # lg.logger.info('Total DD: {}'.format( len([c for c in xdata.columns if 'DD_' in c]) ))
    # lg.logger.info('Total GE: {}'.format( len([c for c in xdata.columns if 'GE_' in c]) ))
    # if 'CELL' in meta.columns:
    #     lg.logger.info('Unique cells: {}'.format( meta['CELL'].nunique() ))
    # if 'DRUG' in meta.columns:
    #     lg.logger.info('Unique drugs: {}'.format( meta['DRUG'].nunique() ))
    # # cnt_fea(df, fea_sep='_', verbose=True, logger=lg.logger)
    
    # if 'AUC' in meta.columns:
    #     plot_hist(meta['AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_all.png')
    
    print_fn('Total DD: {}'.format( len([c for c in data.columns if 'DD_' in c]) ))
    print_fn('Total GE: {}'.format( len([c for c in data.columns if 'GE_' in c]) ))
    if 'CELL' in data.columns:
        print_fn('Unique cells: {}'.format( data['CELL'].nunique() ))
    if 'DRUG' in data.columns:
        print_fn('Unique drugs: {}'.format( data['DRUG'].nunique() ))
    # cnt_fea(df, fea_sep='_', verbose=True, logger=lg.logger)
    
    if 'AUC' in data.columns:
        plot_hist(data['AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/'AUC_hist_all.png')
     
    # -----------------------------------------------
    #       Generate Hold-Out split (train/val/test)
    # -----------------------------------------------
    """ First, we split the data into train and test. The remaining of train set is further
    splitted into train and validation.
    TODO: consider to put this in separate func.
    """
    print_fn('\n{}'.format('-'*50))
    print_fn('Split into hold-out train/val/test')
    print_fn('{}'.format('-'*50))

    n_splits = args['n_splits']
    for seed in range( n_splits ):
        digits = len(str(n_splits))
        seed_str = f"{seed}".zfill(digits)
        output = '1fold_s' + seed_str 

        # Note that we don't shuffle the original dataset, but rather
        # create a vector array of representative indices.
        np.random.seed( seed )
        idx_vec = np.random.permutation( data.shape[0] )
        
        # Create splitter that splits the full dataset into tr and te
        te_folds = int(1/te_size)
        te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds, test_size=None,
                                  mltype=mltype, shuffle=False, random_state=seed)
        
        te_grp = None if split_on is None else data[split_on].values[idx_vec]
        if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
        
        # Split tr into tr and te
        tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
        tr_id = idx_vec[tr_id] # adjust the indices! we'll split the remaining tr into te and vl
        te_id = idx_vec[te_id] # adjust the indices!

        # Update a vector array that excludes the test indices
        idx_vec_ = tr_id; del tr_id

        # Define vl_size while considering the new full size of the available samples
        vl_size = te_size / (1 - te_size)
        cv_folds = int(1/vl_size)

        # Create splitter that splits tr into tr and vl
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=None,
                         mltype=mltype, shuffle=False, random_state=seed)    
        
        cv_grp = None if split_on is None else data[split_on].values[idx_vec_]
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
        
        # Split tr into tr and vl
        tr_id, vl_id = next(cv.split(idx_vec_, groups=cv_grp))
        tr_id = idx_vec_[tr_id] # adjust the indices!
        vl_id = idx_vec_[vl_id] # adjust the indices!
        
        # Dump tr, vl, te indices
        np.savetxt(out_splits/f'{output}_tr_id.csv', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(out_splits/f'{output}_vl_id.csv', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        np.savetxt(out_splits/f'{output}_te_id.csv', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
        
        # Check that indices do not overlap
        assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
        assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
        assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and vl'
        
        print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
        print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
        print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))
        
        # Confirm that group splits are correct (no intersect)
        grp_col = 'CELL' if split_on is None else split_on
        print_intersect_on_var(data, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=grp_col)

        if 'AUC' in data.columns:
            plot_hist(data.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_train.png')
            plot_hist(data.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_val.png')
            plot_hist(data.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_test.png')
                
            # plot_ytr_yvl_dist(ytr=data.loc[tr_id, 'AUC'], yvl=data.loc[vl_id, 'AUC'],
            #                   title='ytr_yvl_dist', outpath=out_figs/f'{output}_ytr_yvl_dist.png')    

    ##! Commented below!
    ## Note that we don't shuffle the original dataset, but rather we create a vector array of
    ## representative indices.
    #np.random.seed( args['seed'] )
    #idx_vec = np.random.permutation( data.shape[0] )
    
    ## Create splitter that splits the full dataset into tr and te
    #te_folds = int(1/te_size)
    #te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds, test_size=None,
    #                          mltype=mltype, shuffle=False, random_state=args['seed'])
    
    #te_grp = None if split_on is None else meta[split_on].values[idx_vec]
    #if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
    
    ## Split tr into tr and te
    #tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
    #tr_id = idx_vec[tr_id] # adjust the indices! we'll split the remaining tr into te and vl
    #te_id = idx_vec[te_id] # adjust the indices!

    ## Update a vector array that excludes the test indices
    #idx_vec_ = tr_id; del tr_id

    ## Define vl_size while considering the new full size of the available samples
    #vl_size = te_size / (1 - te_size)
    #cv_folds = int(1/vl_size)

    ## Create splitter that splits tr into tr and vl
    #cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=None,
    #                 mltype=mltype, shuffle=False, random_state=args['seed'])    
    
    #cv_grp = None if split_on is None else meta[split_on].values[idx_vec_]
    #if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
    ## Split tr into tr and vl
    #tr_id, vl_id = next(cv.split(idx_vec_, groups=cv_grp))
    #tr_id = idx_vec_[tr_id] # adjust the indices!
    #vl_id = idx_vec_[vl_id] # adjust the indices!
    
    ## Dump tr, vl, te indices
    #np.savetxt(outdir/'1fold_tr_id.csv', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    #np.savetxt(outdir/'1fold_vl_id.csv', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    #np.savetxt(outdir/'1fold_te_id.csv', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    
    ## Check that indices do not overlap
    #assert len( set(tr_id).intersection(set(vl_id)) ) == 0, 'Overlapping indices btw tr and vl'
    #assert len( set(tr_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and te'
    #assert len( set(vl_id).intersection(set(te_id)) ) == 0, 'Overlapping indices btw tr and vl'
    
    #print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
    #print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
    #print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))
    
    ## Confirm that group splits are correct (no intersection)
    #grp_col = 'CELL' if split_on is None else split_on
    #print_intersection_on_var(meta, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=grp_col, logger=lg.logger)

    ###! commented below!
    ## if 'AUC' in meta.columns:
    ##     plot_hist(meta.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_train.png')
    ##     plot_hist(meta.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_val.png')
    ##     plot_hist(meta.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_test.png')
            
    ##     plot_ytr_yvl_dist(ytr=meta.loc[tr_id, 'AUC'], yvl=meta.loc[vl_id, 'AUC'],
    ##                       title='ytr_yvl_dist', outpath=outdir/'ytr_yvl_dist.png')    
    
    #if 'AUC' in data.columns:
    #    plot_hist(data.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_train.png')
    #    plot_hist(data.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_val.png')
    #    plot_hist(data.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=out_figs/f'{output}_AUC_hist_test.png')
            
    #    plot_ytr_yvl_dist(ytr=data.loc[tr_id, 'AUC'], yvl=data.loc[vl_id, 'AUC'],
    #                      title='ytr_yvl_dist', outpath=out_figs/f'{output}_ytr_yvl_dist.png')    
            
    # -----------------------------------------------
    #       Generate CV splits (new)
    # -----------------------------------------------
    """ K-fold CV split is applied with multiple values of k. For each set of splits k, the dataset is divided
    into k splits, where each split results in train and val samples. In this process, we take the train samples,
    and divide them into a smaller subset of train samples and test samples.
    """
    """
    lg.logger.info('\n{}'.format('-'*50))
    lg.logger.info(f"Split into multiple sets k-fold splits (multiple k's)")
    lg.logger.info('{}'.format('-'*50))
    cv_folds_list = [5, 7, 10, 15, 20]

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\n----- {cv_folds}-fold splits -----')

        # Create CV splitter
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=None,
                         mltype=mltype, shuffle=False, random_state=args['seed'])

        cv_grp = None if split_on is None else meta[split_on].values[idx_vec]
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds, vl_folds, te_folds = {}, {}, {}
        
        # Start CV iters (this for loop generates the tr and vl splits)
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            lg.logger.info(f'\nFold {fold+1}')
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            # -----------------
            # Store vl ids
            vl_folds[fold] = vl_id.tolist()

            # Update te_size to the new full size of available samples
            te_size_ = len(vl_id)/len(idx_vec) / (1 - len(vl_id)/len(idx_vec))
            te_folds_split = int(1/te_size_)

            # Create splitter that splits tr into tr and te
            te_splitter = cv_splitter(cv_method=te_method, cv_folds=te_folds_split, test_size=None,
                                      mltype=mltype, shuffle=False, random_state=args['seed'])

            # Update the index array
            idx_vec_ = tr_id; del tr_id

            te_grp = None if split_on is None else meta[split_on].values[idx_vec_]
            if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)

            # Split tr into tr and te
            tr_id, te_id = next(te_splitter.split(idx_vec_, groups=te_grp))
            tr_id = idx_vec_[tr_id] # adjust the indices!
            te_id = idx_vec_[te_id] # adjust the indices!

            # Store tr and te ids
            tr_folds[fold] = tr_id.tolist()
            te_folds[fold] = te_id.tolist()
            # -----------------

            lg.logger.info('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/xdata.shape[0] ))
            lg.logger.info('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/xdata.shape[0] ))
            lg.logger.info('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/xdata.shape[0] ))

            # Confirm that group splits are correct (no intersection)
            grp_col = 'CELL' if split_on is None else split_on
            print_intersection_on_var(meta, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=grp_col, logger=lg.logger)

        # Convet to df
        # from_dict takes too long  -->  faster described here: stackoverflow.com/questions/19736080/
        # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
        # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
        tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
        vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))
        te_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in te_folds.items() ]))

        # Dump
        tr_folds.to_csv( outdir/f'{cv_folds}fold_tr_id.csv', index=False )
        vl_folds.to_csv( outdir/f'{cv_folds}fold_vl_id.csv', index=False )
        te_folds.to_csv( outdir/f'{cv_folds}fold_te_id.csv', index=False )
    """
        
    lg.kill_logger()
    print_fn('Runtime: {:.1f} min'.format( (time()-t0)/60) )
    print('Done.')
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])

