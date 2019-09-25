""" Data imputation """
import os
import sys
import logging
import numpy as np
import pandas as pd
import re

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, explained_variance_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# github.com/pandas-dev/pandas/blob/v0.24.2/pandas/core/dtypes/common.py
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype #, ensure_categorical, ensure_float


def get_num_and_cat_cols(df):
    """ Returns 2 dataframes. One with numerical cols and the other with non-numerical cols. """
    cat_cols = [x for x in df.columns if is_string_dtype(df[x]) is True]
    cat_df = df[cat_cols]
    num_df = df.drop(columns=cat_cols)
    # num_df.reset_index(drop=True, inplace=True)
    # cat_df.reset_index(drop=True, inplace=True)
    return num_df, cat_df


def impute_values(data, logger=None):
    """ Impute missing values.
    Args:
        data : tidy dataset (df contains multiple cols including features, meta, and target)
        logger : logging object
    TODO: consider more advanced imputation methods:
    - www.rdocumentation.org/packages/Amelia/versions/1.7.4/topics/amelia
    - try regressor (impute continuous features) or classifier (impute discrete features)
    """
    from sklearn.impute import SimpleImputer, MissingIndicator
    data = data.copy()

    # Total number of cols to impute
    n_impute_cols = sum(data.isna().sum(axis=0) > 0)
    if logger is not None:
        logger.info('Cols with missing values (before impute): {}'.format(n_impute_cols))
    else:
        print('Cols with missing values (before impute): {}'.format(n_impute_cols))

    if n_impute_cols > 0:
        # Split numerical from other features (only numerical will be imputed;
        # the other features can be cell and drug labels)
        num_data, non_num_data = get_num_and_cat_cols(data)

        # Proceed with numerical featues
        colnames = num_data.columns
        idxnames = num_data.index

        # Impute missing values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=1)
        dtypes_dict = num_data.dtypes # keep the original dtypes
        num_data_imputed = pd.DataFrame(imputer.fit_transform(num_data), columns=colnames, index=idxnames)
        num_data_imputed = num_data_imputed.astype(dtypes_dict) # cast back to the original data type
        
        if logger is not None:
            logger.info('Cols with missing values (after impute): {}'.format( sum(num_data_imputed.isna().sum(axis=0) > 0)) )
        else:
            print('Cols with missing values (after impute): {}'.format( sum(num_data_imputed.isna().sum(axis=0) > 0)) )

        # Concat features (xdata_imputed) and other cols (other_data)
        data = pd.concat([non_num_data, num_data_imputed], axis=1)
        
    return data
