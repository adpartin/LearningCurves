import os
import sys
import warnings
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from math import sqrt
from scipy import optimize

# Utils
from cv_splitter import cv_splitter, plot_ytr_yvl_dist

        
def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))      
    
    
def scale_ticks_params(tick_scale='linear'):
    """ Helper function for learning cureve plots.
    Args:
        tick_scale : available values are [linear, log2, log10]
    """
    if tick_scale == 'linear':
        base = None
        label_scale = 'Linear Scale'
    else:
        if tick_scale == 'log2':
            base = 2
            label_scale = 'Log2 Scale'
        elif tick_scale == 'log10':
            base = 10
            label_scale = 'Log10 Scale'
        else:
            raise ValueError('The specified tick scale is not supported.')
    return base, label_scale    
    
    
    
def power_law_func_3prm(x, alpha, beta, gamma):
    """ 3 parameters. docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.power.html """
    return alpha * np.power(x, beta) + gamma
    
    
def fit_power_law_3prm(x, y, p0:list=[30, -0.5, 0.06]):
    """ Fit learning curve data to power-law (3 params). """
    try:
        prms, prms_cov = optimize.curve_fit(power_law_func_3prm, x, y, p0=p0)
        prms_dct = {}
        prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = prms[0], prms[1], prms[2]
        return prms_dct
    except:
        # print('Could not fit power-law.')
        warnings.warn('Could not fit power-law.')
        return None
        
    # prms, prms_cov = optimize.curve_fit(power_law_func_3prm, x, y, p0=p0)
    # prms_dct = {}
    # prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = prms[0], prms[1], prms[2]
    # return prms_dct



def plot_lrn_crv_new(x, y, yerr=None, metric_name:str='score',
                     xtick_scale:str='log2', ytick_scale:str='log2',
                     xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5),
                     ls='-', marker='.', color=None, alpha=0.7, label:str=None, ax=None):
    
    """ This function takes train set size in x and score in y, and generates a learning curve plot.
    Returns:
        ax : ax handle from existing plot (this allows to plot results from different runs for comparison)
    """
    x = x.ravel()
    y = y.ravel()
    
    # Init figure
    fontsize = 13
    legend_fontsize = 10
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    # p = ax.plot(x, y, marker=marker, ls='',  markerfacecolor=color, markeredgecolor='k', alpha=alpha, label=label);
    if yerr is None:
        p = ax.plot(x, y, marker=marker, ls=ls, color=color, alpha=alpha, label=label);
    else:
        p = ax.errorbar(x, y, yerr, marker=marker, ls=ls, color=color, alpha=alpha, label=label);
    c = p[0].get_color()

    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)
    
    ax.set_xlabel(f'Training Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylabel = capitalize_metric(metric_name)
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)        
        
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is not None:
        ax.set_title(title)
    
    # Location of legend --> https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    if label is not None:
        ax.legend(frameon=True, fontsize=legend_fontsize, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    return ax



def plot_lrn_crv_power_law(x, y, plot_fit:bool=True, plot_raw:bool=True, metric_name:str='Score',
                           xtick_scale:str='log2', ytick_scale:str='log2',
                           xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5),
                           marker='.', color=None, alpha=0.7, label:str='Data', ax=None):
    
    """ This function takes train set size in x and score in y, and generates a learning curve plot.
    The power-law model is fitted to the learning curve data.
    Returns:
        ax : ax handle from existing plot (this allows to plot results from different runs for comparison)
        fit_prms : power-law model parameters after fitting
        rmse : root mean squared error of the fit
    """
    x = x.ravel()
    y = y.ravel()
    assert len(x)==len(y), 'Length of vectors x and y must be the same. Got x={len(x)}, y={len(y)}.'
    
    # Init figure
    fontsize = 13
    legend_fontsize = 10
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    if plot_raw:
        p = ax.plot(x, y, marker=marker, ls='',  markerfacecolor=color, markeredgecolor='k', alpha=alpha, label=label);
        c = p[0].get_color()
    else:
        c = color

    # Fit power-law (3 params)
    fit_prms = fit_power_law_3prm(x, y)
    if fit_prms is None:
        return None
    
    # Compute the fit
    yfit = power_law_func_3prm(x, **fit_prms)
    
    # Compute goodness-of-fit
    # R2 is not valid for non-linear models
    # https://statisticsbyjim.com/regression/standard-error-regression-vs-r-squared/
    # http://tuvalu.santafe.edu/~aaronc/powerlaws/
    # https://stats.stackexchange.com/questions/3242/how-to-measure-argue-the-goodness-of-fit-of-a-trendline-to-a-power-law
    # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0170920&type=printable
    # https://www.mathworks.com/help/curvefit/evaluating-goodness-of-fit.html --> based on this we should use SSE or RMSE
    rmse = sqrt( metrics.mean_squared_error(y, yfit) )
    mae = metrics.mean_absolute_error(y, yfit)
    gof = {'rmse': rmse, 'mae': mae}

    # Plot fit
    # eq = r"e(m)={:.2f}$m^{:.2f}$ + {:.2f}".format(power_law_params['alpha'], power_law_params['beta'], power_law_params['gamma'])
    label_fit = '{} Fit; RMSE={:.4f}; a={:.2f}; b={:.2f}'.format(label, rmse, fit_prms['alpha'], fit_prms['beta'])
    # if plot_fit: ax.plot(x, yfit, '--', color=c, label=f'{label} fit; RMSE={rmse:.4f}; ' + eq);
    if plot_fit: ax.plot(x, yfit, '--', color=c, label=label_fit);
        
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)
    
    ax.set_xlabel(f'Training Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylabel = capitalize_metric(metric_name)
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)        
        
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # Add equation (text) on the plot
    # matplotlib.org/3.1.1/gallery/text_labels_and_annotations/usetex_demo.html#sphx-glr-gallery-text-labels-and-annotations-usetex-demo-py
    
    # eq = r"$\varepsilon(m) = \alpha m^{\beta}$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}"
    eq = None
    # xloc = 2.0 * x.min()
    xloc = x.min() + 0.01*(x.max() - x.min())
    yloc = y.min() + 0.9*(y.max() - y.min())
    ax.text(xloc, yloc, eq,
            {'color': 'black', 'fontsize': fontsize, 'ha': 'left', 'va': 'center',
             'bbox': {'boxstyle':'round', 'fc':'white', 'ec':'black', 'pad':0.2}})    

    # matplotlib.org/users/mathtext.html
    # ax.set_title(r"$\varepsilon_{mae}(m) = \alpha m^{\beta} + \gamma$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}, $\gamma$={power_law_params['gamma']:.2f}");
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is not None:
        ax.set_title(title)
    
    # Location of legend --> https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    ax.legend(frameon=True, fontsize=legend_fontsize, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True)
    # return ax, fit_prms, rmse
    return ax, fit_prms, gof



def lrn_crv_power_law_extrapolate(x, y, # m0:int,
        p_et:int=1,
        plot_fit:bool=True, plot_raw_it:bool=True, metric_name:str='Score',
        xtick_scale:str='log2', ytick_scale:str='log2',
        xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5),
        label:str='Data', label_et:str=None, ax=None):
    
    """ This function takes train set size in x and score in y, and generates a learning curve plot.
    The power-law model is fitted to the learning curve data.
    Args:
        plot_raw_it : wether to plot the raw that was used for curve fitting
        m0 : the number of shards to use for curve fitting (iterpolation)
        p_et : number of points to extrapolate
        ax : ax handle from existing plot (this allows to plot results from different runs for comparison)
        pwr_law_params : power-law model parameters after fitting
    """
    x = x.ravel()
    y = y.ravel()
    assert len(x)==len(y), 'Length of vectors x and y must be the same. Got x={len(x)}, y={len(y)}.'
    
    # Data for curve fitting (interpolation)
    m0 = len(x) - p_et
    x_it = x[:m0]
    y_it = y[:m0]

    # Data for extapolation (check how well the fitted curve fits the unseen future data)
    x_et = x[m0:]
    y_et = y[m0:]

    # Fit power-law (3 params)
    fit_prms = fit_power_law_3prm(x_it, y_it)
    if fit_prms is None:
        return None    

    # Plot fit for the entire available range
    y_it_fit = power_law_func_3prm(x_it, **fit_prms)
    y_et_fit = power_law_func_3prm(x_et, **fit_prms) # extrapolate using model from interpolation
    y_fit = power_law_func_3prm(x, **fit_prms)
    
    # Compute goodness-of-fit
    # rmse_it = sqrt( metrics.mean_squared_error(y_it, y_it_fit) )
    # rmse_et = sqrt( metrics.mean_squared_error(y_et, y_et_fit) )
    mae_it = metrics.mean_absolute_error( y_it, y_it_fit )
    mae_et = metrics.mean_absolute_error( y_et, y_et_fit )
    
    # Init figure
    fontsize = 13
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    # Plot raw points that are used for interpolation
    if plot_raw_it:
        # p = ax.plot(x, y, marker=marker, ls='',  markerfacecolor=color, markeredgecolor='k', alpha=alpha, label=label);
        p = ax.plot(x_it, y_it, marker='.', ls='',  markerfacecolor=None, markeredgecolor='k', label=f'{label} for curve fitting');
        c_it = p[0].get_color()
        
        # p = ax.plot(x_et, y_et, marker='o', ls='',  markerfacecolor=None, markeredgecolor='k', label=label=f'{label} for extrapolation');
        # c_et = p[0].get_color()
    else:
        pass
        # c = color        
    
    # Plot raw data
    # Plot raw points that are used for extrapolation
    # ax.plot(x_it, y_it, '.', color=None, markeredgecolor='k', label=f'{label} for interpolation');
    if label_et is None:
        ax.plot(x_et, y_et, 'o', color=None, markerfacecolor='r', markeredgecolor='k', label=f'{label} for extrapolation');
    else:
        ax.plot(x_et, y_et, 'o', color=None, markerfacecolor='r', markeredgecolor='k', label=label_et);

    # Plot fit
    if plot_fit: ax.plot(x, y_fit, '--', color=None, markeredgecolor='k', label=f'{label} Fit'); 
    # if plot_fit: ax.plot(x_it, y_it_fit, '--', color=None, label=f'{label} interpolation (MAE {mae_it:.7f})');
    # if plot_fit: ax.plot(x_et, y_et_fit, '--', color=None, label=f'{label} extrapolation (MAE {mae_et:.7f})');
        
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)
    
    ax.set_xlabel(f'Training Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylabel = capitalize_metric(metric_name)
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)        
    
    # eq = r"$\varepsilon(m) = \alpha m^{\beta}$" + rf"; $\alpha$={fit_prms['alpha']:.2f}, $\beta$={fit_prms['beta']:.2f}"
    eq = None
    xloc = x.min() + 0.01*(x.max() - x.min())
    yloc = y.min() + 0.9*(y.max() - y.min())
    ax.text(xloc, yloc, eq,
            {'color': 'black', 'fontsize': fontsize, 'ha': 'left', 'va': 'center',
             'bbox': {'boxstyle':'round', 'fc':'white', 'ec':'black', 'pad':0.2}})    

    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is not None:
        ax.set_title(title)        
    ax.set_title(title)
    
    ax.legend(frameon=True, fontsize=fontsize, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(True)
    return ax, fit_prms, mae_et



# ----------------------------------------------------------------------
# The 2 funcs below are the 1st version of plot funcs that assume output
# from sklearn learning_curve func
def plot_lrn_crv_all_metrics(df, outdir:Path, figsize=(7,5), xtick_scale='linear', ytick_scale='linear'):
    """ Takes the entire table of scores across folds and train set sizes, and generates plots of 
    learning curves for the different metrics.
    This function generates a list of results (rslt) and passes it to plot_lrn_crv(). This representation
    of results is used in sklearn's learning_curve() function, and thus, we used the same format here.
    Args:
        df : contains train and val scores for cv folds (the scores are the last cv_folds cols)
            metric |  set   | tr_size |  fold0  |  fold1  |  fold2  |  fold3  |  fold4
          ------------------------------------------------------------------------------
              r2   |  True  |   200   |   0.95  |   0.98  |   0.97  |   0.91  |   0.92
              r2   |  False |   200   |   0.21  |   0.27  |   0.22  |   0.25  |   0.24
              mae  |  True  |   200   |   0.11  |   0.12  |   0.15  |   0.10  |   0.18
              mae  |  False |   200   |   0.34  |   0.37  |   0.35  |   0.33  |   0.30
              r2   |  True  |   600   |   0.75  |   0.78  |   0.77  |   0.71  |   0.72
              r2   |  False |   600   |   0.41  |   0.47  |   0.42  |   0.45  |   0.44
              mae  |  True  |   600   |   0.21  |   0.22  |   0.25  |   0.20  |   0.28
              mae  |  False |   600   |   0.34  |   0.37  |   0.35  |   0.33  |   0.30
              ...  |  ..... |   ...   |   ....  |   ....  |   ....  |   ....  |   ....
        cv_folds : number of cv folds
        outdir : dir to save plots
    """
    tr_shards = sorted(df['tr_size'].unique())

    # figs = []
    for metric_name in df['metric'].unique():
        aa = df[df['metric']==metric_name].reset_index(drop=True)
        aa.sort_values('tr_size', inplace=True)

        tr = aa[aa['set']=='tr']
        # vl = aa[aa['set']=='vl']
        te = aa[aa['set']=='te']

        tr = tr[[c for c in tr. columns if 'fold' in c]]
        # vl = vl[[c for c in vl.columns if 'fold' in c]]
        te = te[[c for c in te.columns if 'fold' in c]]

        rslt = []
        rslt.append(tr_shards)
        rslt.append(tr.values if tr.values.shape[0]>0 else None)
        # rslt.append(vl.values if vl.values.shape[0]>0 else None)
        rslt.append(te.values if te.values.shape[0]>0 else None)

        if xtick_scale != 'linear' or ytick_scale != 'linear':
            fname = 'lrn_crv_' + metric_name + '_log.png'
        else:
            fname = 'lrn_crv_' + metric_name + '.png'
        title = 'Learning Curve'

        path = outdir / fname
        fig = plot_lrn_crv(rslt=rslt, metric_name=metric_name, figsize=figsize,
                xtick_scale=xtick_scale, ytick_scale=ytick_scale, title=title, path=path)
        # figs.append(fig)


def plot_lrn_crv(rslt:list, metric_name:str='score',
                 xtick_scale:str='log2', ytick_scale:str='log2',
                 xlim:list=None, ylim:list=None, title:str=None, path:Path=None,
                 figsize=(7,5), ax=None):
    """ 
    Plot learning curves for training and test sets.
    Args:
        rslt : output from sklearn.model_selection.learning_curve()
            rslt[0] : 1-D array (n_ticks, ) -> vector of train set sizes
            rslt[1] : 2-D array (n_ticks, n_cv_folds) -> tr scores
            rslt[2] : 2-D array (n_ticks, n_cv_folds) -> te scores
    """
    tr_shards = rslt[0]
    tr_scores = rslt[1]
    te_scores = rslt[2]
    
    def plot_single_crv(tr_shards, scores, ax, phase, color=None):
        scores_mean = np.mean(scores, axis=1)
        scores_std  = np.std( scores, axis=1)
        ax.plot(tr_shards, scores_mean, '.-', color=color, alpha=0.5, label=f'{phase} Score')
        ax.fill_between(tr_shards, scores_mean - scores_std, scores_mean + scores_std, alpha=0.1, color=color)

    # Plot learning curves
    fontsize = 13
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
        
    if tr_scores is not None:
        plot_single_crv(tr_shards, scores=tr_scores, ax=ax, color='b', phase='Train')
    if te_scores is not None:
        plot_single_crv(tr_shards, scores=te_scores, ax=ax, color='g', phase='Test')

    # Set axes scale and labels
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)

    ax.set_xlabel(f'Train Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylabel = capitalize_metric(metric_name)
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)

    # Other settings
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is None: title='Learning Curve'
    ax.set_title(title)
        
    ax.legend(loc='best', frameon=True)
    ax.grid(True)
    plt.tight_layout()

    # Save fig
    if path is not None: plt.savefig(path, bbox_inches='tight')
    return ax
# ----------------------------------------------------------------------
