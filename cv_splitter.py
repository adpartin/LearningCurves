""" Util functions to split data into train/val. """
from __future__ import print_function, division

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pandas.api.types import is_string_dtype
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def cv_splitter(cv_method: str='simple', cv_folds: int=1, test_size: float=0.2,
                mltype: str='reg', shuffle: bool=False, random_state=None):
    """ Creates a cross-validation splitter.
    Args:
        cv_method: 'simple', 'stratify' (only for classification), 'groups' (only for regression)
        cv_folds: number of cv folds
        test_size: fraction of test set size (used only if cv_folds=1)
        mltype: 'reg', 'cls'
    """
    # Classification
    if mltype == 'cls':
        if cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
            
        elif cv_method == 'stratify':
            if cv_folds == 1:
                cv = StratifiedShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)

    # Regression
    elif mltype == 'reg':
        # Regression
        if cv_method == 'group':
            if cv_folds == 1:
                raise ValueError('GroupShuffleSplit splits groups based on group count and not based on sample count.')
                # https://github.com/scikit-learn/scikit-learn/issues/13369
                # https://github.com/scikit-learn/scikit-learn/issues/9193
                cv = GroupShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = GroupKFold(n_splits=cv_folds)
            
        elif cv_method == 'simple':
            if cv_folds == 1:
                cv = ShuffleSplit(n_splits=cv_folds, test_size=test_size, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=random_state)
    return cv


def plot_ytr_yvl_dist(ytr, yvl, title=None, outpath='.'):
    """ Plot distributions of response data of train and val sets. """
    fig, ax = plt.subplots()
    plt.hist(ytr, bins=100, label='ytr', color='b', alpha=0.5)
    plt.hist(yvl, bins=100, label='yvl', color='r', alpha=0.5)
    if title is None: title = ''
    plt.title(title)
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    if outpath is None:
        plt.savefig(Path(outpath)/'ytr_yvl_dist.png', bbox_inches='tight')
    else:
        plt.savefig(outpath, bbox_inches='tight')

