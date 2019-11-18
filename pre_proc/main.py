"""
Train multiple drug response predictors for 3 types of datasets. The datasets
differ by the batch-effect removal appraoch applied to the RNA-Seq data:
1) Raw RNA-Seq data
2) RNA-Seq normalized using Combat
3) RNA-Seq normalized using source scaling
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

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler

# File path
filepath = Path(__file__).resolve().parent
utils_path = os.path.abspath(os.path.join('../'))
sys.path.append(utils_path)

# Utils
import ml_models
from build_tidy_data import load_rsp, load_rna, load_dsc
from cv_splitter import cv_splitter
from gen_data_splits import extract_subset_fea, dump_dict, print_intersection_on_var, get_print_func
from lrn_crv import calc_preds, calc_scores


def merge_all(rsp, rna, dsc):
    # # Merge with cmeta
    # print('\nMerge response (rsp) and cell metadata (cmeta) ...')
    # data = pd.merge(rsp, cmeta, on='CELL', how='left') # left join to keep all rsp values
    # print(f'data.shape  {data.shape}\n')
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index())
    # del rsp

    # # Merge with dmeta
    # print('\nMerge with drug metadata (dmeta) ...')
    # data = pd.merge(data, dmeta, on='DRUG', how='left') # left join to keep all rsp values
    # print(f'data.shape  {data.shape}\n')
    # print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    data = rsp.copy()

    # Merge with rna
    print('\nMerge with expression (rna) ...')
    data = pd.merge(data, rna, on='CELL', how='inner') # inner join to keep samples that have rna
    print(f'data.shape {data.shape}\n')
    print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    # del rna

    # Merge with dsc
    print('\nMerge with descriptors (dsc) ...')
    data = pd.merge(data, dsc, on='DRUG', how='inner') # inner join to keep samples that have dsc
    print(f'data.shape {data.shape}\n')
    print(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    # del dsc
    
    return data


def get_splits(data, te_size=0.1, split_on=None, mltype='reg', seed=0, logger=None, verbose=False):
    cv_method = 'simple' if split_on is None else 'group'
    te_method = cv_method 

    # Note that we don't shuffle the original dataset, but rather we create a vector array of
    # representative indices.
    np.random.seed(seed)
    idx_vec = np.random.permutation(data.shape[0])
    
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
    # np.savetxt(outdir/'1fold_tr_id.csv', tr_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    # np.savetxt(outdir/'1fold_vl_id.csv', vl_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    # np.savetxt(outdir/'1fold_te_id.csv', te_id.reshape(-1,1), fmt='%d', delimiter='', newline='\n')
    print_fn = get_print_func(logger)
    print_fn('Train samples {} ({:.2f}%)'.format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
    print_fn('Val   samples {} ({:.2f}%)'.format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
    print_fn('Test  samples {} ({:.2f}%)'.format( len(te_id), 100*len(te_id)/data.shape[0] ))
    
    # Confirm that group splits are correct (no intersection)
    grp_col = 'CELL' if split_on is None else split_on
    print_intersection_on_var(data, tr_id=tr_id, vl_id=vl_id, te_id=te_id, grp_col=grp_col, logger=None)
    
    
    df_tr = data.loc[tr_id].reset_index(drop=True)
    df_vl = data.loc[vl_id].reset_index(drop=True)
    df_te = data.loc[te_id].reset_index(drop=True)
    return df_tr, df_vl, df_te


def get_xym(df, target_name, fea_list=['GE', 'DD']):
    """" Get features, target, and meta. """
    x = extract_subset_fea(df, fea_list=['GE', 'DD'])
    y = df[target_name]
    m = df.drop(columns=x.columns)
    return x, y, m


def create_outdir(outdir, args):    
    outdir = outdir/'data_splits_{}_{}'.format(str(args['split_on']).lower(), args['rna_norm'])
    #os.makedirs(outdir)
    os.makedirs(outdir, exist_ok=True)
    return outdir


RSP_FILENAME = 'combined_single_response_agg'  # reposne data
DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)
DATADIR = filepath/'../data'
OUTDIR = filepath / './'  


def parse_args(args):
    parser = argparse.ArgumentParser(description="Test affect of feature normalization on drug response prediction.")
    
    parser.add_argument('--rna_norm', type=str, choices=['raw', 'combat', 'src'], default='raw', help='Default: raw.')
    parser.add_argument('--n_models', default=10, type=int, help='Number of models to train (Default: 10).')
    parser.add_argument('-t', '--target_name', default='AUC', type=str, choices=['AUC'], help='Name of target variable (default: AUC).')
    parser.add_argument('--split_on', type=str, default=None, choices=['cell', 'drug'], help='Specify how to make a hard split. (default: None).')
    parser.add_argument('--n_jobs', default=8, type=int, help='Default: 4.')
    parser.add_argument('--src', nargs='+', default=['ccle', 'nci60'], choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).')

    args = parser.parse_args(args)
    return args


def run(args):
    t0_run = time()
    fea_list = ['GE', 'DD']
    model_name = 'lgb_reg'
    mltype = 'reg'

    rna_norm = args['rna_norm']
    target_name = args['target_name']
    split_on = None if args['split_on'] is None else args['split_on'].upper()
    n_models = args['n_models']
    n_jobs = args['n_jobs']
    src = args['src'] # ['ccle', 'nci60']
    
    # Load rna
    t0 = time()
    rna = pd.read_csv(f'rna_{rna_norm}.csv')
    rsp = load_rsp(filename=DATADIR/RSP_FILENAME, src=src)
    dsc = load_dsc(filename=DATADIR/DSC_FILENAME, plot=False)
    print(rsp.shape)
    print(rsp.SOURCE.value_counts())
    print(dsc.shape)
    df = merge_all(rsp, dsc=dsc, rna=rna)
    runtime = (time() - t0)/60
    print('\nData prep time: {:.2f} min'.format(runtime))
    
    # Get subset of data
    ccle_bool = df['SOURCE']=='ccle'
    df_ccle = df[ccle_bool]
    df_nci60 = df[~ccle_bool]
    df_nci60 = df_nci60.sample(n=df_ccle.shape[0], random_state=0)
    df = pd.concat([df_ccle, df_nci60], axis=0).reset_index(drop=True)
    
    outdir = create_outdir(OUTDIR, args)
    dump_dict(args, outpath=outdir/'args.txt')
    
    scores_all = []
    for seed in range(n_models):
        print(f'\nSeed {seed}')
        # Split data
        df_tr, df_vl, df_te = get_splits(data=df, te_size=0.1, split_on=split_on, mltype=mltype, seed=seed, verbose=False)
        xtr, ytr, mtr = get_xym(df=df_tr, target_name=target_name, fea_list=fea_list)
        xvl, yvl, mvl = get_xym(df=df_vl, target_name=target_name, fea_list=fea_list)
        xte, yte, mte = get_xym(df=df_te, target_name=target_name, fea_list=fea_list)

        # Scale features
        scaler = StandardScaler() # RobustScaler
        col_names = xtr.columns
        xtr = pd.DataFrame( scaler.fit_transform(xtr), columns=col_names, dtype=np.float32 )    
        xvl = pd.DataFrame( scaler.transform(xvl), columns=col_names, dtype=np.float32 )    
        xte = pd.DataFrame( scaler.transform(xte), columns=col_names, dtype=np.float32 )    

        # Train model
        init_kwargs = {'n_estimators': 100, 'n_jobs': n_jobs, 'random_state': seed}
        estimator = ml_models.get_model(model_name, init_kwargs=init_kwargs)
        model = estimator.model

        fit_kwargs = {}
        fit_kwargs['verbose'] = False
        fit_kwargs['eval_set'] = (xvl, yvl)
        fit_kwargs['early_stopping_rounds'] = 10
        # t0 = time()
        model.fit(xtr, ytr, **fit_kwargs)
        # runtime = (time() - t0)/60
        # print('Train time: {:.2f} min'.format(runtime))

        # Agg scores
        y_pred, y_true = calc_preds(model, xte, yte, mltype=mltype)
        scores = calc_scores(y_true, y_pred, mltype=mltype)
        scores['seed'] = seed
        scores_all.append(scores)
        
    df_scores = pd.DataFrame(scores_all)
    df_scores.to_csv(outdir/'scores_{}.csv'.format(rna_norm), index=False)
    
    runtime = (time() - t0_run)/60
    print('Program runtime: {:.2f} min'.format(runtime))

        
    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])    
