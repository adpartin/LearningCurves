"""
This is a batch prcoessing code that calls main_lrn_crv.py with the same set of parameters
but different data_splits_seed#
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import argparse
from time import time

# File path
filepath = Path(__file__).resolve().parent
import gen_data_splits

def main(args):
    n_runs = 20
    for seed in range(n_runs):
        # Note that argparse requires to cast argument from sys.argv to str
        # https://stackoverflow.com/questions/49578928/typeerror-int-object-is-not-subscriptable-when-i-try-to-pass-three-arguments
        gen_data_splits.main([ '--seed', str(seed), *args ]) 

    print('Done.')

main(sys.argv[1:])

# python batch_proc_gen_data_splits.py -dp data.gdsc.dsc.rna.raw/
