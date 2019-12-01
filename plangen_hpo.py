import os
import json
import itertools
import argparse
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description='Generate Learning Curves.')
parser.add_argument('--frm', type=str, choices=['lgbm', 'keras'], help='Framework.')
# parser.add_argument('-ml', '--model_name', type=str, help='Model name.')
# parser.add_argument('-dp', '--dirpath', type=str, help='')
args = parser.parse_args()

frm = args.frm
# DIRPATH = "/projects/CSC249ADOA01/apartin/LearningCurves_/LearningCurves/data.ctrp.dsc.rna.raw/split_on_cell/data_splits_cell_seed00"
# Shards array arg is very tricky to setup!
SHARDS = []

# GDSC
# SHARDS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 88416] # GDSC
# SHARDS = [1024, 4096, 16384, 65536]  # GDSC job 1
# SHARDS = [2048, 8192, 32768, 88416]  # GDSC job 2
# SHARDS = [88416] # GDSC

# CTRP
# SHARDS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 230441] # GDSC
# SHARDS = [1024, 4096, 16384, 65536, 230441]  # GDSC job 1
# SHARDS = [2048, 8192, 32768, 131072]  # GDSC job 2
# SHARDS = [230441]

# CTRP
# SHARDS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 88416] # GDSC
# SHARDS = [256, 1024, 4096, 16384, 65536] # GDSC
# SHARDS = [512, 2048, 8192, 32768, 88416] # GDSC
# SHARDS = [4096]

# CV_FOLDS = 1
# N_JOBS = 8
# global_args = {'dirpath': args.dirpath, 'shards_arr': SHARDS, 'model_name': args.model_name}

# LGBM-specific settings
# if 'lgb_' in args.model_name:
if frm == 'lgbm':
    # frm = 'lgbm'
    # GDSC
    GBM_LEAVES = [31, 10, 50, 100, 200]
    GBM_LR = [0.1, 0.15, 0.05, 0.01] # , 0.005
    GBM_MAX_DEPTH = [-1, 5, 10, 20, 30]
    GBM_TREES = [100, 1000, 2000, 3000] # 4000

    prm_grid = {'gbm_leaves': GBM_LEAVES, 'gbm_lr': GBM_LR,
                'gbm_max_depth': GBM_MAX_DEPTH, 'gbm_trees': GBM_TREES}

    # if len(SHARDS) > 0: prm_grid.update({'shards_arr': SHARDS})
    """
    if len(SHARDS) > 0:
        arg_val_list  = [ GBM_LEAVES,   GBM_MAX_DEPTH,   GBM_LR,   GBM_TREES,   SHARDS ]
        arg_name_list = [ "gbm_leaves", "gbm_max_depth", "gbm_lr", "gbm_trees", "shards_arr" ]
    else:    
        arg_val_list  = [ GBM_LEAVES,   GBM_MAX_DEPTH,   GBM_LR,   GBM_TREES ]
        arg_name_list = [ "gbm_leaves", "gbm_max_depth", "gbm_lr", "gbm_trees" ]
    """

# Keras-specific settings
# elif 'nn_' in args.model_name:
elif frm == 'keras':
    # frm = 'keras'
    # EPOCHS = 1200    
    BATCH = [32, 64, 128]
    # BATCH = [32, 64]
    BATCHNORM = ['True', 'False']
    # BATCHNORM = ['False']    
    DR_RATE = [0.15, 0.2, 0.25, 0.3, 3.5, 0.4]        
    # DR_RATE = [2.5, 3.5]
    LR = [0.0001, 0.001, 0.01]
    # LR = [0.0001, 0.001]
    OPT = ['adam', 'sgd']
    # OPT = ['adam']
    
    prm_grid = OrderedDict(
        {'opt': OPT, 'dr_rate': DR_RATE, 'batch_size': BATCH,
         'lr': LR, 'batchnorm': BATCHNORM})

    # if len(SHARDS) > 0: prm_grid.update({'shards_arr': SHARDS})
    """
    if len(SHARDS) > 0:
        arg_val_list  = [ OPT,   LR,   DR_RATE,   BATCH,        BATCHNORM,   SHARDS ]
        arg_name_list = [ "opt", "lr", "dr_rate", "batch_size", "batchnorm", "shards_arr" ]
    else:
        arg_val_list  = [ OPTS,  LR,   DR_RATES,  BATCHS,       BATCHNORM ]
        arg_name_list = [ "opt", "lr", "dr_rate", "batch_size", "batchnorm" ]
    """

# combs = list(itertools.product( *arg_val_list )) # all args combinations
sets = list(itertools.product( *(prm_grid.values()) )) # all args combinations
def assign_keys_to_values(keys, values):
    return dict(zip(keys, values))

# sets = list(ParameterGrid(prm_grid))

def append_record(record, fout):
    with open(fout, 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)

print('Total HP sets', len(sets))
digits = len(str(len(sets)))        
        
runs = []
# for i, item in enumerate(sets):
for i, values in enumerate(sets):
    # item = assign_keys_to_values(keys=arg_name_list, values=c)
    item = assign_keys_to_values(keys=prm_grid.keys(), values=values)
    
    item['id'] = 'run_' + f'{i}'.zfill(digits)
    if 'shards_arr' in item:
        item['shards_arr'] = [item['shards_arr']] # this must come after the definition of the key 'id'
    # item.update(const_args)
    # item.update(global_args)
    runs.append(json.dumps(item))
    # append_record(item, fout=f'upf-lc-{frm}.txt')

total_runs = i + 1
# print(f'Framework: {args.frm}')
print(f'Total runs: {total_runs} (PROCS={total_runs+1}).')


fname = f'upf_{frm}.txt'

# https://stackoverflow.com/questions/21058935/python-json-loads-shows-valueerror-extra-data
with open(fname, 'w') as outfile:
    for item in runs:
        outfile.write(item+'\n')

with open(fname) as f:
    my_list = [json.loads(line) for line in f]
    
print('Done.')

