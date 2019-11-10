import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from pandas.api.types import is_string_dtype, is_numeric_dtype

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ComBat imports
import patsy  # for combat
from combat import *


def scale_rna(df, fea_start_id, per_source=False):
    """ Scale df values and return updated df. """
    df = df.copy()

    if per_source:
        sources = df['CELL'].map(lambda x: x.split('.')[0].lower()).unique().tolist()
        for i, source in enumerate(sources):
            print('Scaling {}'.format(source))
            # source_vec = df['Sample'].map(lambda x: x.split('.')[0].lower())
            source_vec = df['CELL'].map(lambda x: x.split('.')[0].lower())
            source_idx_bool = source_vec.str.startswith(source)

            # data_values = df.loc[source_idx_bool, df.columns[fea_start_id:].values].values
            fea = df.loc[source_idx_bool, df.columns[fea_start_id:].values]
            fea_scaled = StandardScaler().fit_transform(fea)
            df.loc[source_idx_bool, fea_start_id:] = fea_scaled
    else:
        if is_numeric_dtype(df.iloc[:, fea_start_id:]):
            df.iloc[:, fea_start_id:] = StandardScaler().fit_transform(df.iloc[:, fea_start_id:])
        # data_values = df.iloc[:, 1:].values
        # values_scaled = StandardScaler().fit_transform(data_values)
        # df.iloc[:, 1:] = values_scaled

    return df


def map_rna_values_to_cell_lines(rna, cl_mapping):
    """
    Use mapping to copy ccle and nci60 gene expression to gdsc, gcsi, and ctrp.
    Args:
        rna : df that contains only datasets that have their original rna expression data
        cl_mapping : contains cell line mappings that are used to generate new samples for which
                     we have dose response but don't have rna expression
    Returns:
        df : merged df that contains the original and the mapped samples
    """
    df = rna.copy()
    
    # Check that only the allowed datasets are passed
    datasets_to_keep = ['ccle', 'nci60', 'gdc', 'ncipdm']
    bl = df['CELL'].map(lambda x: True if x.split('.')[0].lower() in datasets_to_keep else False)
    assert all(bl), 'Only these datasets are allowed: {}'.format(datasets_to_keep)
    
    # Merge in order to copy rna values
    cells = cl_mapping.merge(df, left_on='from_cell', right_on='CELL', how='inner')
    
    # Drop and rename columns
    cells = cells.drop(columns=['from_cell', 'CELL']).rename(columns={'to_cell': 'CELL'})

    # Concat 'df' (df that contains original rna profiles) and 'cells' replicated values
    df = pd.concat([df, cells], axis=0).sort_values('CELL').reset_index(drop=True)
    return df


# TODO: Are these still used ???
def combat_ap(rna, meta, sample_col_name:str, batch_col_name:str):
    """
    sample_col_name : name of the column that contains the rna samples
    batch_col_name : name of the column that contains the batch values
    """
    rna_fea, pheno, _, _ = py_df_to_R_df(data=rna, meta=meta, sample_col_name=sample_col_name)
    # dat.columns.name = None
    # pheno.index.name = pheno.columns.name
    # pheno.columns.name = None

    mod = patsy.dmatrix("~1", data=pheno, return_type="dataframe")
    ebat = combat(data = rna_fea,
                  batch = pheno[batch_col_name],  # pheno['batch']
                  model = mod)

    df_rna_be = R_df_to_py_df(ebat, sample_col_name=sample_col_name)
    return df_rna_be


def R_df_to_py_df(data, sample_col_name):
    """ This is applied to the output of combat.py """
    return data.T.reset_index().rename(columns={'index': sample_col_name})


def py_df_to_R_df(data, meta, sample_col_name, filename=None, to_save=False, to_scale=False, var_thres=None):
    """ Convert python dataframe to R dataframe (transpose). """
    # data, meta = update_df_and_meta(data.copy(), meta.copy(), on='Sample')

    # Remove low var columns (this is required for SVA)
    if var_thres is not None:
        data, low_var_col_names = rmv_low_var_genes(data, var_thres=var_thres, per_source=True, verbose=True)

    # Scale dataset
    # todo: Scaling provides worse results in terms of kNN(??)
    # if to_scale:
    #     dat = scale_rna(dat, per_source=False)

    # Transpose df for processing in R
    data_r = data.set_index(sample_col_name, drop=True)
    data_r = data_r.T

    # This is required for R
    meta_r = meta.set_index(sample_col_name, drop=True)
    del meta_r.index.name
    meta_r.columns.name = sample_col_name

    if to_save:
        print('Data shape to save:', data_r.shape)
        file_format = '.txt'
        foldername = 'save'
        verify_folder(foldername)

        if filename is not None:
            data_r.to_csv(os.path.join(foldername, filename + '_dat_R' + file_format), sep='\t')
            meta_r.to_csv(os.path.join(foldername, filename + '_pheno_R' + file_format), sep='\t')
        else:
            data_r.to_csv(os.path.join(foldername, 'dat_R' + file_format), sep='\t')
            meta_r.to_csv(os.path.join(foldername, 'pheno_R' + file_format), sep='\t')

    return data_r, meta_r, data, meta



def plot_pca(df, components=[1, 2], figsize=(8, 6),
             color_vector=None, marker_vector=None,
             scale=False, grid=False, title=None, verbose=True):
    """
    Apply PCA to input df.
    Args:
        color_vector : each element corresponds to a row in df. The unique elements will be colored
            with a different color.
        marker_vector : each element corresponds to a row in df. The unique elements will be marked
            with a different marker.
    Returns:
        pca_obj : object of sklearn.decomposition.PCA()
        pca : pca matrix
        fig : PCA plot figure handle

    https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    """
    if color_vector is not None:
        assert len(df) == len(color_vector), 'len(df) and len(color_vector) must be the same size.'
        n_colors = len(np.unique(color_vector))
        colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))

    if marker_vector is not None:
        assert len(df) == len(marker_vector), 'len(df) and len(marker_vector) shuold be the same size.'
        all_markers = ('o', 'v', 's', 'p', '^', '<', '>', '8', '*', 'h', 'H', 'D', 'd', 'P', 'X')
        markers = all_markers[:len(np.unique(marker_vector))]

    df = df.copy()

    # PCA
    if scale:
        X = StandardScaler().fit_transform(df.values)
    else:
        X = df.values

    n_components = max(components)
    pca_obj = PCA(n_components=n_components)
    pca = pca_obj.fit_transform(X)
    pc0 = components[0] - 1
    pc1 = components[1] - 1

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)

    if (color_vector is not None) and (marker_vector is not None):
        for i, marker in enumerate(np.unique(marker_vector)):
            for color in np.unique(color_vector):
                # print(i, 'marker:', marker, 'color:', color)
                idx = (marker_vector == marker) & (color_vector == color)
                ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.5,
                            marker=markers[i],
                            edgecolors='black',
                            color=next(colors),
                            label='{}, {}'.format(marker, color))

    elif (color_vector is not None):
        for color in np.unique(color_vector):
            idx = (color_vector == color)
            ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.5,
                        marker='o',
                        edgecolors='black',
                        color=next(colors),
                        label='{}'.format(color))

    elif (marker_vector is not None):
        for i, marker in enumerate(np.unique(marker_vector)):
            idx = (marker_vector == marker)
            ax.scatter(pca[idx, pc0], pca[idx, pc1], alpha=0.5,
                        marker=markers[i],
                        edgecolors='black',
                        color='blue',
                        label='{}'.format(marker))

    else:
        ax.scatter(pca[:, pc0], pca[:, pc1], alpha=0.5,
                   marker='s', edgecolors='black', color='blue')

    if title: ax.set_title(title)
    ax.set_xlabel('PC' + str(components[0]))
    ax.set_ylabel('PC' + str(components[1]))
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0.0), ncol=1, borderaxespad=0, frameon=True)
    if grid:
        plt.grid(True)
    else:
        plt.grid(False)

    if verbose:
        print('Explained variance by PCA components [{}, {}]: [{:.5f}, {:.5f}]'.format(
            components[0], components[1],
            pca_obj.explained_variance_ratio_[pc0],
            pca_obj.explained_variance_ratio_[pc1]))

    return pca_obj, pca, fig