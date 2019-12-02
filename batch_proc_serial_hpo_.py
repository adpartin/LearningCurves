"""
A batch prcoessing code that runs HP search by calling main_lrn_crv.py
for different sets of HPs.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from glob import glob
import itertools
import json

# File path
filepath = Path(__file__).resolve().parent
from classlogger import Logger
import main_lrn_crv


def parse_args(args):
    parser = argparse.ArgumentParser(description="Serial HPO learning curve.")
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to data and splits (default: None).')
    parser.add_argument('-gout', '--global_outdir', type=str, required=True, help='Global outdir.')
    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args(args)
    return args, other_args


def verify_dirpath(dirpath):
    """ Verify the dirpath exists and contain the dataset. """
    if dirpath is None:
        sys.exit('Program terminated. You must specify a path to a data via the input argument -dp.')

    dirpath = Path(dirpath)
    assert dirpath.exists(), 'The specified dirpath was not found: {dirpath}.'
    return dirpath


def assign_keys_to_values(keys, values):
    return dict(zip(keys, values))


def dict_to_argv_list(dct):
    l = []
    for k, v in dct.items():
        l.append( '--'+str(k) )
        l.append( str(v) )
    return l


def run(args, other_args):
    t0 = time()
    # dirpath = verify_dirpath(args['dirpath'])
    global_outdir = args['global_outdir']
    args = dict_to_argv_list(args) + other_args

    # Global args
    CV_FOLDS = 1
    N_JOBS = 16
    # global_args= {"dirpath": DIRPATH, "n_jobs": N_JOBS, "cv_folds": CV_FOLDS}
    global_args= {'n_jobs': N_JOBS, 'cv_folds': CV_FOLDS}
    
    # Const args
    MODEL_NAME = 'lgb_reg'
    const_args = {'model_name': MODEL_NAME}

    # HPs
    # GBM_LEAVES = [31, 10, 50, 100, 500, 1000]
    # GBM_MAX_DEPTH = [-1, 5, 10, 20]
    # GBM_LR = [0.1, 0.01, 0.001]
    # GBM_TREES = [100, 1000, 2000]
    GBM_LEAVES = [31, 10, 50, 100]
    GBM_LR = [0.1, 0.01, 0.001]
    GBM_MAX_DEPTH = [-1, 5, 10, 20]
    GBM_TREES = [100, 1000, 2000]
    
    # CCLE
    """
    GBM_LEAVES = [31, 10, 50]
    GBM_LR = [0.1, 0.01]
    GBM_MAX_DEPTH = [-1, 5, 10]
    GBM_TREES = [100, 1000, 2000]
    """
    
    # GBM_LEAVES = [31]
    # GBM_MAX_DEPTH = [-1]
    # GBM_LR = [0.1]
    # GBM_TREES = [100, 2]


    arg_value_list = [ GBM_LEAVES,   GBM_MAX_DEPTH,   GBM_LR,   GBM_TREES ]
    arg_name_list =  [ "gbm_leaves", "gbm_max_depth", "gbm_lr", "gbm_trees" ]

    # Generate HPs sets
    combs = list(itertools.product( *arg_value_list ))
    digits = len(str(len(combs)))
    print('Total runs: {}'.format(len(combs)))
    
    plans = []
    list_of_arg_sets = []
    for i, c in enumerate(combs):
        item = assign_keys_to_values(keys=arg_name_list, values=c)
        # item['id'] = "run" + f"{i}".zfill(digits)
        if 'shards_arr' in item:
            item['shards_arr'] = [item['shards_arr']] # this must come after the definition of the key 'id'
        item.update(const_args)
        item.update(global_args)
        plans.append(json.dumps(item))

        arg_list = args + dict_to_argv_list(item)
        list_of_arg_sets.append(arg_list)

    with open(Path(global_outdir)/'upf-lc.txt', 'w') as f:
        for item in plans:
            f.write(item+'\n')

    for arg_list in list_of_arg_sets:
        main_lrn_crv.main( arg_list )

    runtime = time() - t0
    print('Runtime: {} hrs.'.format(runtime/3600))
    # lg.logger.info('Runtime: {} hrs.'.format(runtime/360)


def main(args):
    args, other_args = parse_args(args)
    args = vars(args)
    run(args, other_args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])

# python batch_proc_hpo_serial.py -gout trn.gdsc/hpo_lgbm/


