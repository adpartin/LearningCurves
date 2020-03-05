"""
This code takes the main path to the entire experiment and aggregates the scores
into a single file.
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
from sklearn.preprocessing import LabelEncoder

# File path
filepath = Path(__file__).resolve().parent


def parse_args(args):
    parser = argparse.ArgumentParser(description='Aggregate all results of a single experiment.')
    parser.add_argument('-dp', '--dirpath', default=None, type=str, help='Full path to the learning curve experiment (default: None).')
    args = parser.parse_args(args)
    return args


def run(args):
    # Get all run dirs
    base_dir = Path(args['dirpath'])
    runs = glob(str(base_dir/'*'))

    scores = []
    for i, r in enumerate(runs):
        dpath = Path(r)/'lrn_crv_scores.csv'
        if not dpath.exists(): continue
        print(r)

        scr_tmp = pd.read_csv( dpath )
        scr_tmp.rename(columns={'fold0': 'run'+str(i+1)}, inplace=True)
        if len(scores)==0:
            scores = scr_tmp
        else:
            scores = scores.merge(scr_tmp, on=['metric', 'tr_size', 'set'])

    run_col_names = [c for c in scores.columns if 'run' in c]

    scores_mean   = scores[run_col_names].mean(axis=1)
    scores_median = scores[run_col_names].median(axis=1)
    scores_std    = scores[run_col_names].std(axis=1)

    scores.insert(loc=3, column='mean', value=scores_mean)
    scores.insert(loc=3, column='median', value=scores_median)
    scores.insert(loc=3, column='std', value=scores_std)
    print(len(np.unique(scores.tr_size)))
    print('Training set sizes:', np.unique(scores.tr_size))

    save = True
    if save:
        scores.to_csv(base_dir/'all_seed_runs_scores.csv', index=False)


def main(args):
    args = parse_args(args)
    args = vars(args)
    run(args)
    print('Done.')
    

if __name__ == '__main__':
    main(sys.argv[1:])
