"""
A batch prcoessing code that calls main_lrn_crv.py with the same set of parameters
but different data_splits_seed#
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
    dirs = glob(os.path.join(dirpath, 'data_splits_seed*'))
    for dpath in dirs:
        main_lrn_crv.main([ '--dirpath', str(dpath) ])

    runtime = time() - t0
    print('Runtime: {} hrs.'.format(runtime/360)
    print('Done.')

main(sys.argv[1:])

# python batch_proc_lrn_crv.py 
