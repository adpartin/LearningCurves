"""
This "new" version of the code uses a different dataframe for descriptors:
'pan_drugs_dragon7_descriptors.tsv' instead of 'combined_pubchem_dragon7_descriptors.tsv'
"""
from __future__ import division, print_function

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from pprint import pprint, pformat

import sklearn
import numpy as np
import pandas as pd

# github.com/mtg/sms-tools/issues/36
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
t0 = time()


# File path
filepath = Path(__file__).resolve().parent


# Utils
from classlogger import Logger
from impute import impute_values


# Default settings
DATADIR = filepath / './data'
OUTDIR = filepath / './'
RSP_FILENAME = 'combined_single_response_agg'  # reposne data
# RSP_FILENAME_CHEM = 'chempartner_single_response_agg'  # reposne data
DSC_FILENAME = 'pan_drugs_dragon7_descriptors.tsv'  # drug descriptors data (new)
# DSC_FILENAME = 'combined_pubchem_dragon7_descriptors.tsv'  # drug descriptors data (old)
DRUG_META_FILENAME = 'drug_info'
CELL_META_FILENAME = 'combined_cancer_types'
# CELL_META_FILENAME = 'combined_metadata_2018May.txt'


# Settings
na_values = ['na', '-', '']
fea_prfx_dct = {'rna': 'GE_', 'cnv': 'CNV_', 'snp': 'SNP_',
                'dsc': 'DD_', 'fng': 'FNG_'}

prfx_dtypes = {'rna': np.float32, 'cnv': np.int8, 'snp': np.int8,
               'dsc': np.float32, 'fng': np.int8}



def create_basename(args):
    """ Name to characterize the data. Can be used for dir name and file name. """
    ls = args['drug_fea'] + args['cell_fea'] + [args['rna_norm']] # + ['seed'+str(args['seed'])]
    if args['src'] is None:
        # name = '.'.join( ['tidy'] + ls )
        name = '.'.join( ls )
    else:
        # name = '.'.join( ['tidy'] + args['src'] + ls )
        src_names = '_'.join( args['src'] )
        name = '.'.join( [src_names] + ls )
    return name


def create_outdir(outdir, args):
    """ Creates output dir. """
    basename = create_basename(args)
    outdir = Path(outdir)/basename
    os.makedirs(outdir)
    return outdir


def dump_dict(dct, outpath='./dict.txt'):
    """ Dump dict into file. """
    with open( Path(outpath), 'w' ) as file:
        for k in sorted(dct.keys()):
            file.write('{}: {}\n'.format(k, dct[k]))
            

def groupby_src_and_print(df, logger):        
    if logger:
        logger.info( df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index() )
    else:
        df.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index()
            
    
def load_rsp(filename, src=None, keep_bad=False, logger=None):
    """ Load drug response data. """
    if logger: logger.info(f'\nLoad response from ... {DATADIR / RSP_FILENAME}')
    rsp = pd.read_table(DATADIR/RSP_FILENAME, sep='\t', na_values=na_values, warn_bad_lines=True)
    rsp.drop(columns='STUDY', inplace=True) # gives error when saves in 'parquet' format
    # print(rsp.dtypes)

    # Drop bad samples
    if keep_bad is False:
        # Yitan
        # lg.logger.info('\n\nDrop bad samples ...')
        # id_drop = (rsp['AUC'] == 0) & (rsp['EC50se'] == 0) & (rsp['R2fit'] == 0)
        # rsp = rsp.loc[~id_drop,:]
        # lg.logger.info(f'Dropped {sum(id_drop)} rsp data points.')
        # lg.logger.info(f'rsp.shape {rsp.shape}')
        # (ap)
        # TODO: check this (may require a more rigorous filtering)
        if logger: logger.info('\nDrop samples with low R2fit ...')
        id_drop = rsp['R2fit'] <= 0
        rsp = rsp.loc[~id_drop,:]
        if logger: logger.info(f'Dropped {sum(id_drop)} rsp data points.')

    if logger: logger.info(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, logger=logger)        
        
    if logger: logger.info('\nExtract specific sources.')
    rsp['SOURCE'] = rsp['SOURCE'].apply(lambda x: x.lower())
    rsp.replace([np.inf, -np.inf], value=np.nan, inplace=True) # Replace -inf and inf with nan

    if src is not None:
        rsp = rsp[rsp['SOURCE'].isin(src)].reset_index(drop=True)

    if logger: logger.info(f'rsp.shape {rsp.shape}')
    groupby_src_and_print(rsp, logger=logger)
    return rsp


def load_rna(datadir, rna_norm, logger=None, keep_cells_only=True, float_type=np.float32, impute=True):  
    """ Load RNA-Seq data. """
    if rna_norm == 'raw':
        fname = 'combined_rnaseq_data_lincs1000'
    else:
        fname = f'combined_rnaseq_data_lincs1000_{rna_norm}'
        
    if logger: logger.info('\nLoad RNA-Seq ... {datadir / fname}')
    rna = pd.read_csv(Path(datadir)/fname, sep='\t', low_memory=False, na_values=na_values, warn_bad_lines=True)
    rna = rna.astype(dtype={c: float_type for c in rna.columns[1:]})  # Cast features
    rna = rna.rename(columns={c: fea_prfx_dct['rna']+c for c in rna.columns[1:] if fea_prfx_dct['rna'] not in c}) # prefix rna gene names
    rna.rename(columns={'Sample': 'CELL'}, inplace=True) # rename cell col name
    # rna = rna.set_index(['CELL'])
    
    cell_sources = ['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60']
    rna = rna.loc[rna['CELL'].map(lambda s: s.split('.')[0].lower() in cell_sources), :].reset_index(drop=True)

    # Impute missing values
    if impute:
        if logger: logger.info('Impute NA values ...')
        rna = impute_values(rna, logger=logger)

    if logger: logger.info(f'rna.shape {rna.shape}')
    return rna
        

def load_dsc(filename, logger=None, dropna_th=0.4, float_type=np.float32, impute=True, plot=True):  
    """ Load drug descriptors. """
    if logger: logger.info(f'\nLoad drug descriptors ... {DATADIR / filename}')
    dsc = pd.read_csv(DATADIR/filename, sep='\t', low_memory=False, na_values=na_values, warn_bad_lines=True)
    dsc = dsc.astype(dtype={c: float_type for c in dsc.columns[1:]})  # Cast features
    dsc = dsc.rename(columns={c: fea_prfx_dct['dsc']+c for c in dsc.columns[1:] if fea_prfx_dct['dsc'] not in c}) # prefix drug desc names
    dsc.rename(columns={'NAME': 'DRUG'}, inplace=True)

    # ------------------
    # Filter descriptors
    # ------------------
    # dsc.nunique(dropna=True).value_counts()
    # dsc.nunique(dropna=True).sort_values()
    if logger: logger.info('Drop descriptors with too many NA values ...')
    if plot: plot_dsc_na_dist(dsc=dsc, savepath=Path(args['outdir'])/'dsc_hist_ratio_of_na.png')
    dsc = dropna(dsc, axis=1, th=dropna_th)
    if logger: logger.info(f'dsc.shape {dsc.shape}')
    # dsc.isna().sum().sort_values(ascending=False)

    # There are descriptors where there is a single unique value excluding NA (drop those)
    if logger: logger.info('Drop descriptors that have a single unique value (excluding NAs) ...')
    col_idx = dsc.nunique(dropna=True).values==1
    dsc = dsc.iloc[:, ~col_idx]
    if logger: logger.info(f'dsc.shape {dsc.shape}')

    # There are still lots of descriptors which have only a few unique values
    # we can categorize those values. e.g.: 564 descriptors have only 2 unique vals,
    # and 154 descriptors have only 3 unique vals, etc.
    # todo: use utility code from p1h_alex/utils/data_preproc.py that transform those
    # features into categorical and also applies an appropriate imputation.
    # dsc.nunique(dropna=true).value_counts()[:10]
    # dsc.nunique(dropna=true).value_counts().sort_index()[:10]

    # Impute missing values
    if impute:
        if logger: logger.info('Impute NA values ...')
        dsc = impute_values(data=dsc, logger=logger)

    if logger: logger.info(f'dsc.shape {dsc.shape}')
    return dsc


def plot_dsc_na_dist(dsc, savepath=None):
    """ Plot distbirution of na values in drug descriptors. """
    fig, ax = plt.subplots()
    sns.distplot(dsc.isna().sum(axis=0)/dsc.shape[0], bins=100, kde=False, hist_kws={'alpha': 0.7})
    plt.xlabel('Ratio of total NA values in a descriptor to the total drug count')
    plt.ylabel('Total # of descriptors with the specified NA ratio')
    plt.title('Histogram of descriptors based on ratio of NA values')
    plt.grid(True)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight') # dpi=200
    else:
        plt.savefig('dsc_hist_ratio_of_na.png', bbox_inches='tight') # dpi=200
        

def dropna(df, axis=0, th=0.4):
    """ Drop rows or cols based on the ratio of NA values along the axis.
    Args:
        th (float) : if the ratio of NA values along the axis is larger that th, then drop all the values
        axis (int) : 0 to drop rows; 1 to drop cols
    """
    df = df.copy()
    axis = 0 if axis==1 else 1
    col_idx = df.isna().sum(axis=axis)/df.shape[axis] <= th
    df = df.iloc[:, col_idx.values]
    return df        
        
        
def plot_rsp_dists(rsp, rsp_cols, savepath=None):
    """ Plot distributions of all response variables.
    Args:
        rsp : df of response values
        rsp_cols : list of col names
        savepath : full path to save the image
    """
    ncols = 4
    nrows = int(np.ceil(len(rsp_cols)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, figsize=(10,10))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(rsp_cols):
            fig.delaxes(ax) # delete un-used ax
        else:
            target_name = rsp_cols[i]
            x = rsp[target_name].copy()
            x = x[~x.isna()].values
            sns.distplot(x, bins=100, kde=True, ax=ax, label=target_name, # fit=norm, 
                        kde_kws={'color': 'k', 'lw': 0.4, 'alpha': 0.8},
                        hist_kws={'color': 'b', 'lw': 0.4, 'alpha': 0.5})
            ax.tick_params(axis='both', which='major', labelsize=7)
            txt = ax.yaxis.get_offset_text(); txt.set_size(7) # adjust exponent fontsize in xticks
            txt = ax.xaxis.get_offset_text(); txt.set_size(7)
            ax.legend(fontsize=5, loc='best')
            ax.grid(True)

    # plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight') # dpi=200
    else:
        plt.savefig('rsp_dists.png', bbox_inches='tight')
        
        
def parse_args(args):
    parser = argparse.ArgumentParser(description='Create tidy data.')
    parser.add_argument('--drug_fea', type=str, nargs='+', choices=['dsc', 'fng'], default=['dsc'], help='Default: [dsc].')
    parser.add_argument('--cell_fea', type=str, nargs='+', choices=['rna', 'cnv'], default=['rna'], help='Default: [rna].')
    parser.add_argument('--rna_norm', type=str, choices=['raw', 'combat', 'source_scale'], default='raw', help='Default: raw.')

    parser.add_argument('--keep_bad', action='store_true', default=False, help='Default: False')
    parser.add_argument('--dropna_th', type=float, default=0.4, help='Default: 0.4')

    parser.add_argument('--src', nargs='+', default=None, choices=['ccle', 'gcsi', 'gdsc', 'ctrp', 'nci60'],
                        help='Data sources to extract (default: None).')

    args = parser.parse_args(args)
    return args


def run(args):
    file_format = 'parquet'

    # Response columns
    rsp_cols = ['AUC', 'AUC1', 'EC50', 'EC50se', 'R2fit', 'Einf', 'IC50', 'HS', 'AAC1', 'DSS1']

    # Analysis of fibro samples are implemented in ccle_fibroblast.py and ccle_preproc.R
    fibro_names = ['CCLE.HS229T', 'CCLE.HS739T', 'CCLE.HS840T', 'CCLE.HS895T', 'CCLE.RKN',
                   'CTRP.Hs-895-T', 'CTRP.RKN', 'GDSC.RKN', 'gCSI.RKN']

    # Prefix to add to feature names based on feature types
    # fea_prfx_dict = {'rna': 'cell_rna.', 'cnv': 'cell_cnv.',
    #                  'dsc': 'drug_dsc.', 'fng': 'drug_fng.',
    #                  'clb': 'cell_lbl.', 'dlb': 'drug_lbl.'}
#     fea_prfx_dct = {'rna': 'GE_', 'cnv': 'CNV_', 'snp': 'SNP_',
#                     'dsc': 'DD_', 'fng': 'FNG_'}

#     prfx_dtypes = {'rna': np.float32, 'cnv': np.int8, 'snp': np.int8,
#                    'dsc': np.float32, 'fng': np.int8}


    
    # -----------------------------------------------
    #     Create outdir and logger
    # -----------------------------------------------
    outdir = create_outdir(OUTDIR, args)
    args['outdir'] = str(outdir)
    lg = Logger(outdir/'create_tidy_logfile.log')
    lg.logger.info(f'File path: {filepath}')
    lg.logger.info(f'\n{pformat(args)}')
    dump_dict(args, outpath=outdir/'create_tidy_args.txt') # dump args


    # -----------------------------------------------
    #     Load response data, and features
    # -----------------------------------------------
    rsp = load_rsp(RSP_FILENAME, src=args['src'], keep_bad=args['keep_bad'], logger=lg.logger)
    rna = load_rna(DATADIR, rna_norm=args['rna_norm'], logger=lg.logger, float_type=prfx_dtypes['rna'])
    dsc = load_dsc(DSC_FILENAME, dropna_th=args['dropna_th'], logger=lg.logger, float_type=prfx_dtypes['dsc'])


    # -----------------------------------------------
    #     Load cell and drug meta
    # -----------------------------------------------
    cmeta = pd.read_csv(DATADIR/CELL_META_FILENAME, sep='\t', header=None, names=['CELL', 'CANCER_TYPE'])
    # cmeta = pd.read_csv(DATADIR/'combined_metadata_2018May.txt', sep='\t').rename(columns={'sample_name': 'CELL', 'core_str': 'CELL_CORE'})
    # cmeta = cmeta[['CELL', 'CELL_CORE', 'tumor_type_from_data_src']]

    dmeta = pd.read_csv(DATADIR/DRUG_META_FILENAME, sep='\t')
    dmeta.rename(columns={'ID': 'DRUG', 'NAME': 'DRUG_NAME', 'CLEAN_NAME': 'DRUG_CLEAN_NAME'}, inplace=True)
    # TODO: What's going on with CTRP and GDSC? Why counts are not consistent across the fields??
    # dmeta.insert(loc=1, column='SOURCE', value=dmeta['DRUG'].map(lambda x: x.split('.')[0].lower()))
    # print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'DRUG_NAME': 'nunique', 'DRUG_CLEAN_NAME': 'nunique', 'PUBCHEM': 'nunique'}).reset_index())
    # print(dmeta.groupby('SOURCE').agg({'DRUG': 'nunique', 'DRUG_NAME': 'unique', 'DRUG_CLEAN_NAME': 'unique'}).reset_index())


    # -----------------------------------------------
    #     Merge data
    # -----------------------------------------------
    lg.logger.info('\n{}'.format('-' * 40))
    lg.logger.info('... Start merging response with other dataframes ...')
    lg.logger.info('-' * 40)

    # Merge with cmeta
    lg.logger.info('\nMerge response (rsp) and cell metadata (cmeta) ...')
    data = pd.merge(rsp, cmeta, on='CELL', how='left') # left join to keep all rsp values
    lg.logger.info(f'data.shape  {data.shape}\n')
    lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique'}).reset_index())
    del rsp

    # Merge with dmeta
    lg.logger.info('\nMerge with drug metadata (dmeta) ...')
    data = pd.merge(data, dmeta, on='DRUG', how='left') # left join to keep all rsp values
    lg.logger.info(f'data.shape  {data.shape}\n')
    lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())

    # Merge with rna
    lg.logger.info('\nMerge with expression (rna) ...')
    data = pd.merge(data, rna, on='CELL', how='inner') # inner join to keep samples that have rna
    lg.logger.info(f'data.shape {data.shape}\n')
    lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    del rna

    # Merge with dsc
    lg.logger.info('\nMerge with descriptors (dsc) ...')
    data = pd.merge(data, dsc, on='DRUG', how='inner') # inner join to keep samples that have dsc
    lg.logger.info(f'data.shape {data.shape}\n')
    lg.logger.info(data.groupby('SOURCE').agg({'CELL': 'nunique', 'DRUG': 'nunique'}).reset_index())
    del dsc

    # Memory usage
    lg.logger.info('\nTidy dataframe: {:.3f} GB'.format(sys.getsizeof(data)/1e9))
    for fea_name, fea_prfx in fea_prfx_dct.items():
        cols = [c for c in data.columns if fea_prfx in c]
        tmp = data[cols]
        mem = 0 if tmp.shape[1]==0 else sys.getsizeof(tmp)/1e9
        lg.logger.info('Memory occupied by {} features: {} ({:.2f} GB)'.format(fea_name, len(cols), mem))


    lg.logger.info('\nRuntime: {:.1f} mins'.format( (time()-t0)/60) )


    # -----------------------------------------------
    #   Plot rsp distributions
    # -----------------------------------------------
    # Plot distributions of target variables
    plot_rsp_dists(data, rsp_cols=rsp_cols, savepath=outdir/'rsp_dists.png')

    # Plot distribution of a single target
    # target_name = 'EC50se'
    # fig, ax = plt.subplots()
    # x = rsp[target_name].copy()
    # x = x[~x.isna()].values
    # sns.distplot(x, bins=100, ax=ax)
    # plt.savefig(os.path.join(OUTDIR, target_name+'.png'), bbox_inches='tight') 


    # -----------------------------------------------
    #   Finally save data
    # -----------------------------------------------
    # Save data
    lg.logger.info(f'\nSave tidy dataframe ({file_format}) ...')
    fname = create_basename(args)
    t0 = time()
    if file_format == 'parquet':
        fpath = outdir/('data.'+fname+'.parquet')
        data.to_parquet(fpath)
    else:
        fpath = outdir/('data.'+fname+'.csv')
        data.to_csv(fpath, sep='\t')
    lg.logger.info('Save time: {:.1f} mins'.format( (time()-t0)/60) )

    # Load data
    lg.logger.info(f'\nLoad tidy dataframe ({file_format}) ...')
    t0 = time()
    if file_format == 'parquet':
        data_fromfile = pd.read_parquet(fpath)
    else:
        data_fromfile = pd.read_table(fpath, sep='\t')
    lg.logger.info('Load time: {:.1f} mins'.format( (time()-t0)/60) )

    # Check that the saved data is the same as original one
    lg.logger.info(f'\nLoaded dataframe is same as original: {data.equals(data_fromfile)}')

    lg.logger.info('\n{}'.format('-'*90))
    lg.logger.info(f'Tidy data filepath:\n{os.path.abspath(fpath)}')
    lg.logger.info('-'*90)
    lg.kill_logger()

    
def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
