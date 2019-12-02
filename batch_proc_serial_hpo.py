"""
A batch prcoessing code that calls main_lrn_crv.py with different sets of input arguments.
The sets of input arguments can be :
    - specified in file!
    - generated using plangen_hpo.py
This pipeline is primarily used for HPO via grid-search.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from glob import glob
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import ParameterGrid

filepath = Path(__file__).resolve().parent
import main_lrn_crv


def parse_args(args):
    parser = argparse.ArgumentParser(description="Serial HPO learning curve.")
    parser.add_argument('--json_plan', type=str, required=True, help='File that contains all input args in dicts to run HPO (default: None).')
    parser.add_argument('-dp', '--dirpath', type=str, required=True, help='Full path to data and splits (default: None).')
    parser.add_argument('-gout', '--global_outdir', type=str, required=True, help='Global outdir.')
    # args = parser.parse_args(args)
    args, other_args = parser.parse_known_args(args)
    return args, other_args


def read_json(fname):
    with open(fname) as f:
        # return(json.load(f))
        return [json.loads(line) for line in f]    
    
    
def dict_to_argv_list(dct):
    l = []
    for k, v in dct.items():
        l.append( '--'+str(k) )
        l.append( str(v) )
    return l    
    

def run(args):
    t0 = time()
    
    # Const args
    MODEL_NAME = 'nn_reg0'
    # MODEL_NAME = 'nn_reg_attn'
    EPOCHS = 1200
    N_JOBS = 8
    SHARDS_ARR = 88415 # 4000
    # SHARDS_ARR = '1000 2000' # 88415
    other_args = {'model_name': MODEL_NAME,
                  'n_jobs': N_JOBS,
                  'dirpath': args['dirpath'],
                  'global_outdir': args['global_outdir'],
                  'epochs': EPOCHS,
                  'shards_arr': SHARDS_ARR,
                 }

    arg_sets = read_json( args['json_plan'] )

    print('Total HP sets', len(arg_sets))
    digits = len(str(len(arg_sets)))
    
    agg_runs = []
    for i, item in enumerate(arg_sets):
        item['run_outdir'] = arg_sets[i]['id'] # str('run_' + f'{i}'.zfill(digits))
        item.update(other_args)
        arg_list = dict_to_argv_list( item )
        mae = main_lrn_crv.main( arg_list )
        # mae = main_lrn_crv.run( hp_set )
        item['mae'] = mae

        agg_runs.append(item)
        
    df = pd.DataFrame(agg_runs)
    df.to_csv(Path(args['global_outdir']) / 'all_hpo_runs.csv', index=False)

    runtime = time() - t0
    print('Runtime: {} hrs.'.format(runtime/3600))
    print('Done.')


def main(args):
    args, other_args = parse_args(args)
    args = vars(args)
    if len(other_args)>0:
        args.update( vars(other_args) )
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])

# python batch_proc_lrn_crv.py 
