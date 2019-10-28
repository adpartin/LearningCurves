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
import platform
from pathlib import Path
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

SEED = 0


# File path
filepath = Path(__file__).resolve().parent


# Utils
from classlogger import Logger
from cv_splitter import cv_splitter, plot_ytr_yvl_dist
from plotting import plot_hist


def parse_args(args):
    parser = argparse.ArgumentParser(description='Generate and save dataset splits.')

    # Input data
    parser.add_argument('--dirpath', default=None, type=str, help='Full path to dir that contains the data file (default: None).')

    # Feature types
    parser.add_argument('-cf', '--cell_fea', nargs='+', default=['GE'], choices=['GE'], help='Cell features (default: GE).')
    parser.add_argument('-df', '--drug_fea', nargs='+', default=['DD'], choices=['DD'], help='Drug features (default: DD).')    

    # Data split methods
    # parser.add_argument('--te_method', default='simple', choices=['simple', 'group'], help='Test split method (default: None).')
    # parser.add_argument('--cv_method', default='simple', choices=['simple', 'group'], help='Cross-val split method (default: simple).')
    # parser.add_argument('--te_size', type=float, default=0.1, help='Test size split ratio (default: 0.1).')
    parser.add_argument('--vl_size', type=float, default=0.1, help='Val size split ratio for single split (default: 0.1).')

    parser.add_argument('--split_on', type=str, default=None, choices=[], help='Specify how to make a hard split. (default: None).')

    # Define n_jobs
    parser.add_argument('--n_jobs', default=4,  type=int, help='Default: 4.')

    # Parse args and run
    args = parser.parse_args(args)
    return args


def split_size(x):
    """ Split size can be float (0, 1) or int (casts value as needed). """
    assert x > 0, 'Split size must be greater than 0.'
    return int(x) if x > 1.0 else x


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))

            
# def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
#     """ Plot hist of a 1-D array x. """
#     if fit is not None:
#         (mu, sigma) = stats.norm.fit(x)
#         fit = stats.norm
#         label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
#     else:
#         label = None
    
#     alpha = 0.6
#     fig, ax = plt.subplots()
# #     sns.distplot(x, bins=bins, kde=True, fit=fit, 
# #                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
# #                  kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
# #                  fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
# #                           'label': label})
#     sns.distplot(x, bins=bins, kde=False, fit=fit, 
#                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
#     plt.grid(True)
#     if label is not None: plt.legend()
#     plt.title(var_name + ' hist')
#     plt.savefig(path, bbox_inches='tight')

    
def cnt_fea(df, fea_sep='_', verbose=True, logger=None):
    """ Count the number of features per feature type. """
    dct = {}
    unq_prfx = df.columns.map(lambda x: x.split(fea_sep)[0]).unique() # unique feature prefixes
    for prfx in unq_prfx:
        fea_type_cols = [c for c in df.columns if (c.split(fea_sep)[0]) in prfx] # all fea names of specific type
        dct[prfx] = len(fea_type_cols)
    
    if verbose and logger is not None:
        logger.info(pformat(dct))
    elif verbose:
        pprint(dct)
    return dct


def extract_subset_fea(df, fea_list, fea_sep='_'):
    """ Extract features based feature prefix name. """
    fea = [c for c in df.columns if (c.split(fea_sep)[0]) in fea_list]
    return df[fea]    
    
            
def run(args):
    dirpath = Path(args['dirpath'])

    # Data splits
    # te_method = args['te_method']
    # cv_method = args['cv_method']
    # te_size = split_size(args['te_size'])
    vl_size = split_size(args['vl_size'])

    # Features 
    cell_fea = args['cell_fea']
    drug_fea = args['drug_fea']
    fea_list = cell_fea + drug_fea
    
    # Other params
    n_jobs = args['n_jobs']

    # Hard split
    # grp_by_col = None
    split_on = args['split_on'] if args['split_on'] is None else args['split_on'].upper()
    cv_method = 'simple' if split_on is None else 'group'
    te_method = cv_method 

    # TODO: this needs to be improved
    mltype = 'reg'  # required for the splits (stratify in case of classification)
    
    
    # -----------------------------------------------
    #       Create outdir and logger
    # -----------------------------------------------
    outdir = Path( str(dirpath) + '_splits' )
    os.makedirs(outdir, exist_ok=True)
    
    lg = Logger(outdir/'splitter.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')

    # Dump args to file
    dump_dict(args, outpath=outdir/'args.txt')

    
    # -----------------------------------------------
    #       Load and break data
    # -----------------------------------------------
    lg.logger.info('\nLoad master dataset.')
    files = list(dirpath.glob('**/*.parquet'))
    if len(files) > 0: data = pd.read_parquet( files[0], engine='auto', columns=None ) # TODO: assumes that there is only one data file
    lg.logger.info('data.shape {}'.format(data.shape))

    # Split features and traget, and dump to file
    lg.logger.info('\nSplit features and meta.')
    xdata = extract_subset_fea(data, fea_list=fea_list, fea_sep='_')
    meta = data.drop(columns=xdata.columns)
    xdata.to_parquet( outdir/'xdata.parquet' )
    meta.to_parquet( outdir/'meta.parquet' )
    
    lg.logger.info('Total DD: {}'.format( len([c for c in xdata.columns if 'DD_' in c]) ))
    lg.logger.info('Total GE: {}'.format( len([c for c in xdata.columns if 'GE_' in c]) ))
    lg.logger.info('Unique cells: {}'.format( meta['CELL'].nunique() ))
    lg.logger.info('Unique drugs: {}'.format( meta['DRUG'].nunique() ))
    # cnt_fea(df, fea_sep='_', verbose=True, logger=lg.logger)

    plot_hist(meta['AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_all.png')
    

    # -----------------------------------------------
    #       Generate CV splits (new)
    # -----------------------------------------------
    """
    np.random.seed(SEED)
    idx_vec = np.random.permutation(xdata.shape[0])

    cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        # Convert vl_size into int
        # vl_size_int = int(vl_size*len(idx_vec))

        # Create CV splitter
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        # Command meta[split_on].values[idx_vec] returns the vector meta[split_on].values
        # in an order specified by idx_vec
        # For example:
        # aa = meta[split_on][:3]
        # print(aa.values)
        # print(aa.values[[0,2,1]])
        # m = meta[split_on]
        cv_grp = meta[split_on].values[idx_vec] if split_on is not None else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {} 
        vl_folds = {} 
        te_folds = {}
        
        # Start CV iters
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            lg.logger.info(f'\nFold {fold+1}')
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!
            # t = meta.loc[tr_id, split_on]
            # v = meta.loc[vl_id, split_on]
            # print(len(vl_id)/len(idx_vec))
            
            # -----------------
            # Store tr ids
            tr_folds[fold] = tr_id.tolist()

            # Create splitter that splits vl into vl and te (splits by half)
            te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=0.5,
                                      mltype=mltype, shuffle=False, random_state=SEED)

            # Update the index array
            idx_vec_ = vl_id; del vl_id

            te_grp = meta[split_on].values[idx_vec_] if split_on is not None else None
            if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)

            # Split vl set into vl and te
            vl_id, te_id = next(te_splitter.split(idx_vec_, groups=te_grp))
            vl_id = idx_vec_[vl_id] # adjust the indices!
            te_id = idx_vec_[te_id] # adjust the indices!
            # v = meta.loc[vl_id, split_on]
            # e = meta.loc[te_id, split_on]

            # Store vl and te ids
            vl_folds[fold] = vl_id.tolist()
            te_folds[fold] = te_id.tolist()
            # -----------------

            lg.logger.info('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/xdata.shape[0] ))
            lg.logger.info('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/xdata.shape[0] ))
            lg.logger.info('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/xdata.shape[0] ))

            # Confirm that group splits are correct
            if split_on is not None:
                tr_grp_unq = set(meta.loc[tr_id, split_on])
                vl_grp_unq = set(meta.loc[vl_id, split_on])
                te_grp_unq = set(meta.loc[te_id, split_on])
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in te: {len(te_grp_unq)}.')

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
        
        # Plot target dist only for the 1-fold case
        # TODO: consider to plot dist for all k-fold where k>1
        if cv_folds==1 and fold==0:
            plot_hist(meta.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_train.png')
            plot_hist(meta.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_val.png')
            plot_hist(meta.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_test.png')
            
            plot_ytr_yvl_dist(ytr=meta.loc[tr_id, 'AUC'], yvl=meta.loc[vl_id, 'AUC'],
                              title='ytr_yvl_dist', outpath=outdir/'ytr_yvl_dist.png')
            
            # pd.Series(meta.loc[tr_id, 'AUC'].values, name='ytr').to_csv(outdir/'ytr.csv')
            # pd.Series(meta.loc[vl_id, 'AUC'].values, name='yvl').to_csv(outdir/'yvl.csv')
            # pd.Series(meta.loc[te_id, 'AUC'].values, name='yte').to_csv(outdir/'yte.csv')
    """


    # -----------------------------------------------
    #       Generate CV splits (new)
    # -----------------------------------------------
    np.random.seed(SEED)
    idx_vec = np.random.permutation(xdata.shape[0])

    cv_folds_list = [1, 5, 7, 10, 15, 20]
    lg.logger.info(f'\nStart CV splits ...')

    for cv_folds in cv_folds_list:
        lg.logger.info(f'\nCV folds: {cv_folds}')

        # Create CV splitter
        cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
                         mltype=mltype, shuffle=False, random_state=SEED)

        cv_grp = meta[split_on].values[idx_vec] if split_on is not None else None
        if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
    
        tr_folds = {} 
        vl_folds = {} 
        te_folds = {}
        
        # Start CV iters (this for loop generates the tr and vl splits)
        for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
            lg.logger.info(f'\nFold {fold}')
            tr_id = idx_vec[tr_id] # adjust the indices!
            vl_id = idx_vec[vl_id] # adjust the indices!

            # -----------------
            # Store vl ids
            vl_folds[fold] = vl_id.tolist()

            # Update te_size to the new full size of available samples
            if cv_folds == 1:
                te_size_ = vl_size / (1 - vl_size)
            else:
                te_size_ = len(vl_id)/len(idx_vec) / (1 - len(vl_id)/len(idx_vec))

            # Create splitter that splits tr into tr and te
            te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size_,
                                      mltype=mltype, shuffle=False, random_state=SEED)

            # Update the index array
            idx_vec_ = tr_id; del tr_id

            te_grp = meta[split_on].values[idx_vec_] if split_on is not None else None
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

            # Confirm that group splits are correct
            if split_on is not None:
                tr_grp_unq = set(meta.loc[tr_id, split_on])
                vl_grp_unq = set(meta.loc[vl_id, split_on])
                te_grp_unq = set(meta.loc[te_id, split_on])
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tTotal group ({split_on}) intersec btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}.')
                lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
                lg.logger.info(f'\tUnique cell lines in te: {len(te_grp_unq)}.')

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
        
        # Plot target dist only for the 1-fold case
        # TODO: consider to plot dist for all k-fold where k>1
        if cv_folds==1 and fold==0:
            plot_hist(meta.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_train.png')
            plot_hist(meta.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_val.png')
            plot_hist(meta.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_test.png')
            
            plot_ytr_yvl_dist(ytr=meta.loc[tr_id, 'AUC'], yvl=meta.loc[vl_id, 'AUC'],
                              title='ytr_yvl_dist', outpath=outdir/'ytr_yvl_dist.png')
            
            # pd.Series(meta.loc[tr_id, 'AUC'].values, name='ytr').to_csv(outdir/'ytr.csv')
            # pd.Series(meta.loc[vl_id, 'AUC'].values, name='yvl').to_csv(outdir/'yvl.csv')
            # pd.Series(meta.loc[te_id, 'AUC'].values, name='yte').to_csv(outdir/'yte.csv')



#     # -----------------------------------------------
#     #       Train-test split
#     # -----------------------------------------------
#     np.random.seed(SEED)
#     idx_vec = np.random.permutation(xdata.shape[0])
# 
#     if te_method is not None:
#         lg.logger.info('\nSplit train/test.')
#         te_splitter = cv_splitter(cv_method=te_method, cv_folds=1, test_size=te_size,
#                                   mltype=mltype, shuffle=False, random_state=SEED)
# 
#         te_grp = meta[grp_by_col].values[idx_vec] if te_method=='group' else None
#         if is_string_dtype(te_grp): te_grp = LabelEncoder().fit_transform(te_grp)
#    
#         # Split train/test
#         tr_id, te_id = next(te_splitter.split(idx_vec, groups=te_grp))
#         tr_id = idx_vec[tr_id] # adjust the indices!
#         te_id = idx_vec[te_id] # adjust the indices!
# 
#         pd.Series(tr_id).to_csv( outdir/f'tr_id.csv', index=False, header=[0] )
#         pd.Series(te_id).to_csv( outdir/f'te_id.csv', index=False, header=[0] )
#         
#         lg.logger.info('Train: {:.1f}'.format( len(tr_id)/xdata.shape[0] ))
#         lg.logger.info('Test:  {:.1f}'.format( len(te_id)/xdata.shape[0] ))
#         
#         # Update the master idx vector for the CV splits
#         idx_vec = tr_id
# 
#         # Plot dist of responses (TODO: this can be done to all response metrics)
#         # plot_ytr_yvl_dist(ytr=tr_ydata.values, yvl=te_ydata.values,
#         #         title='tr and te', outpath=run_outdir/'tr_te_resp_dist.png')
# 
#         # Confirm that group splits are correct
#         if te_method=='group' and grp_by_col is not None:
#             tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
#             te_grp_unq = set(meta.loc[te_id, grp_by_col])
#             lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}.')
#             lg.logger.info(f'\tA few intersections : {list(tr_grp_unq.intersection(te_grp_unq))[:3]}.')
# 
#         # Update vl_size to effective vl_size
#         vl_size = vl_size * xdata.shape[0]/len(tr_id)
#         
#         # Plot hist te
#         pd.Series(meta.loc[te_id, 'AUC'].values, name='yte').to_csv(outdir/'yte.csv')
#         plot_hist(meta.loc[te_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_test.png')
# 
#         del tr_id, te_id
# 
# 
#     # -----------------------------------------------
#     #       Generate CV splits
#     # -----------------------------------------------
#     cv_folds_list = [1, 5, 7, 10, 15, 20, 25]
#     lg.logger.info(f'\nStart CV splits ...')
#     
#     for cv_folds in cv_folds_list:
#         lg.logger.info(f'\nCV folds: {cv_folds}')
# 
#         cv = cv_splitter(cv_method=cv_method, cv_folds=cv_folds, test_size=vl_size,
#                          mltype=mltype, shuffle=False, random_state=SEED)
# 
#         cv_grp = meta[grp_by_col].values[idx_vec] if cv_method=='group' else None
#         if is_string_dtype(cv_grp): cv_grp = LabelEncoder().fit_transform(cv_grp)
#     
#         tr_folds = {}
#         vl_folds = {}
# 
#         # Start CV iters
#         for fold, (tr_id, vl_id) in enumerate(cv.split(idx_vec, groups=cv_grp)):
#             tr_id = idx_vec[tr_id] # adjust the indices!
#             vl_id = idx_vec[vl_id] # adjust the indices!
# 
#             tr_folds[fold] = tr_id.tolist()
#             vl_folds[fold] = vl_id.tolist()
# 
#             # Confirm that group splits are correct
#             if cv_method=='group' and grp_by_col is not None:
#                 tr_grp_unq = set(meta.loc[tr_id, grp_by_col])
#                 vl_grp_unq = set(meta.loc[vl_id, grp_by_col])
#                 lg.logger.info(f'\tTotal group ({grp_by_col}) intersections btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}.')
#                 lg.logger.info(f'\tUnique cell lines in tr: {len(tr_grp_unq)}.')
#                 lg.logger.info(f'\tUnique cell lines in vl: {len(vl_grp_unq)}.')
#         
#         # Convet to df
#         # from_dict takes too long  -->  faster described here: stackoverflow.com/questions/19736080/
#         # tr_folds = pd.DataFrame.from_dict(tr_folds, orient='index').T 
#         # vl_folds = pd.DataFrame.from_dict(vl_folds, orient='index').T
#         tr_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in tr_folds.items() ]))
#         vl_folds = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in vl_folds.items() ]))
# 
#         # Dump
#         tr_folds.to_csv( outdir/f'{cv_folds}fold_tr_id.csv', index=False )
#         vl_folds.to_csv( outdir/f'{cv_folds}fold_vl_id.csv', index=False )
#         
#         # Plot target dist only for the 1-fold case
#         if cv_folds==1 and fold==0:
#             plot_hist(meta.loc[tr_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_train.png')
#             plot_hist(meta.loc[vl_id, 'AUC'], var_name='AUC', fit=None, bins=100, path=outdir/'AUC_hist_val.png')
#             
#             plot_ytr_yvl_dist(ytr=meta.loc[tr_id, 'AUC'], yvl=meta.loc[vl_id, 'AUC'],
#                               title='ytr_yvl_dist', outpath=outdir/'ytr_yvl_dist.png')
#             
#             pd.Series(meta.loc[tr_id, 'AUC'].values, name='ytr').to_csv(outdir/'ytr.csv')
#             pd.Series(meta.loc[vl_id, 'AUC'].values, name='yvl').to_csv(outdir/'yvl.csv')
 
    lg.kill_logger()
    print('Done.')
    
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    ret = run(args)
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
