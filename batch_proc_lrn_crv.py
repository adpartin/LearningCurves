"""
A batch prcoessing code that calls main_lrn_crv.py with the same set of parameters
but different data_splits_seed#.
The use need to specify the dir to where the data_splits_seed* dirs are located.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time
from glob import glob

# File path
filepath = Path(__file__).resolve().parent
import main_lrn_crv

def main(args):
    t0 = time()
    dirpath = args[0]

    # glob all files data_splits_seed
    # dirs = Path(dirpath).glob('data_splits_seed*')
    dirs = glob(os.path.join(dirpath, 'data_splits_*'))
    digits = len(str(len(dirs)))
    
    tot_to_process = 8
    for i, dpath in enumerate(dirs):
        if i+1 > tot_to_process:
            break
        r_name = 'run_' + f'{i}'.zfill(digits)
        main_lrn_crv.main([ '--dirpath', str(dpath), '--seed', str(i), '--run_outdir', r_name, *args[1:] ])

    runtime = time() - t0
    print('Runtime: {} hrs.'.format(runtime/3600))
    print('Done.')

main(sys.argv[1:])

# python batch_proc_lrn_crv.py 
