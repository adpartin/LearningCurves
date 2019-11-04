""" 
This code generates CV splits and/or train/test splits of a dataset.
TODO: Add plots of the splits (e.g. drug, cell line, reponse distributions).
"""
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

import sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# Utils
from cv_splitter import cv_splitter, plot_ytr_yvl_dist

            
def plot_hist(x, var_name, fit=None, bins=100, path='hist.png'):
    """ Plot hist of a 1-D array x. """
    if fit is not None:
        (mu, sigma) = stats.norm.fit(x)
        fit = stats.norm
        label = f'norm fit: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}'
    else:
        label = None
    
    alpha = 0.6
    fig, ax = plt.subplots()
#     sns.distplot(x, bins=bins, kde=True, fit=fit, 
#                  hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'},
#                  kde_kws={'linewidth': 2, 'alpha': alpha, 'color': 'k'},
#                  fit_kws={'linewidth': 2, 'alpha': alpha, 'color': 'r',
#                           'label': label})
    sns.distplot(x, bins=bins, kde=False, fit=fit, 
                 hist_kws={'linewidth': 2, 'alpha': alpha, 'color': 'b'})
    plt.grid(True)
    if label is not None: plt.legend()
    plt.title(var_name + ' hist')
    plt.savefig(path, bbox_inches='tight')


    
# --------------------------------------------------------------------------------------------------
# Power-law utils
# --------------------------------------------------------------------------------------------------
def power_law_func_3prm(x, alpha, beta, gamma):
    """ 3 parameters. docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.power.html """
    return alpha * np.power(x, beta) + gamma
    
    
# def fit_power_law_3prm(x, y, p0:list=[30, -0.3, 0.06]):
def fit_power_law_3prm(x, y, p0:list=[30, -0.5, 0.06]):
    """ Fit learning curve data to power-law (3 params).
    TODO: How should we fit the data across multiple folds?
    This can be addressed using Bayesian methods (look at Bayesian linear regression).
    The uncertainty of parameters indicates the consistency across folds.
    MUKHERJEE et al (2003) performs significance test!
    """
    prms, prms_cov = optimize.curve_fit(power_law_func_3prm, x, y, p0=p0)
    prms_dct = {}
    prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = prms[0], prms[1], prms[2]
    return prms_dct


def plot_lrn_crv_power_law(x, y, plot_fit:bool=True, metric_name:str='score',
                           xtick_scale:str='log2', ytick_scale:str='log2',
                           xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5),
                           marker='.', color=None, alpha=0.7,  label:str='Data', ax=None):
    
    """ This function takes train set size in x and score in y, and generates a learning curve plot.
    The power-law model is fitted to the learning curve data.
    Args:
        ax : ax handle from existing plot (this allows to plot results from different runs for comparison)
        pwr_law_params : power-law model parameters after fitting
    """
    x = x.ravel()
    y = y.ravel()
    
    # Fit power-law (3 params)
    power_law_params = fit_power_law_3prm(x, y)
    yfit = power_law_func_3prm(x, **power_law_params)
    
    # Compute goodness-of-fit
    # R2 is not valid for non-linear models
    # https://statisticsbyjim.com/regression/standard-error-regression-vs-r-squared/
    # http://tuvalu.santafe.edu/~aaronc/powerlaws/
    # https://stats.stackexchange.com/questions/3242/how-to-measure-argue-the-goodness-of-fit-of-a-trendline-to-a-power-law
    # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0170920&type=printable
    # https://www.mathworks.com/help/curvefit/evaluating-goodness-of-fit.html --> based on this we should use SSE or RMSE
    rmse = sqrt( metrics.mean_squared_error(y, yfit) )

    # Init figure
    fontsize = 13
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    p = ax.plot(x, y, marker=marker, ls='',  markerfacecolor=color, alpha=alpha, label=label);
    c = p[0].get_color()

    # Plot fit
    # if plot_fit: ax.plot(x, yfit, '--', color=fit_color, label=f'{label} fit (RMSE: {rmse:.7f})');    
    if plot_fit: ax.plot(x, yfit, '--', color=c, label=f'{label} fit (RMSE: {rmse:.7f})');    
        
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
    # eq = r"$\varepsilon_{mae}(m) = \alpha m^{\beta} + \gamma$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}, $\gamma$={power_law_params['gamma']:.2f}"
    # eq = rf"$\varepsilon(m) = {power_law_params['alpha']:.2f} m^{power_law_params['beta']:.2f} + {power_law_params['gamma']:.2f}$" # TODO: make this work    
    
    eq = r"$\varepsilon(m) = \alpha m^{\beta}$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}"
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
    if title is None: title='Learning Curve (power-law)'
    ax.set_title(title)
    
    # Location of legend --> https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    ax.legend(frameon=True, fontsize=fontsize, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(True)
    # return fig, ax, power_law_params
    return ax, power_law_params



def lrn_crv_power_law_extrapolate(x, y, m0:int, 
        plot_fit:bool=True, metric_name:str='score',
        xtick_scale:str='log2', ytick_scale:str='log2',
        xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5),
        label:str='Data', ax=None):
    
    """ This function takes train set size in x and score in y, and generates a learning curve plot.
    The power-law model is fitted to the learning curve data.
    Args:
        m0 : the number of shards to use for curve fitting (iterpolation)
        ax : ax handle from existing plot (this allows to plot results from different runs for comparison)
        pwr_law_params : power-law model parameters after fitting
    """
    x = x.ravel()
    y = y.ravel()
    
    # Data for curve fitting (interpolation)
    x_it = x[:m0]
    y_it = y[:m0]

    # Data for extapolation (check how well the fitted curve fits the unseen future data)
    x_et = x[m0:]
    y_et = y[m0:]

    # Fit power-law (3 params)
    power_law_params = fit_power_law_3prm(x_it, y_it)

    # Plot fit for the entire available range
    y_it_fit = power_law_func_3prm(x_it, **power_law_params)
    y_et_fit = power_law_func_3prm(x_et, **power_law_params)
    y_fit = power_law_func_3prm(x, **power_law_params)

    # Compute goodness-of-fit
    # R2 is not valid for non-linear models
    # https://statisticsbyjim.com/regression/standard-error-regression-vs-r-squared/
    # http://tuvalu.santafe.edu/~aaronc/powerlaws/
    # https://stats.stackexchange.com/questions/3242/how-to-measure-argue-the-goodness-of-fit-of-a-trendline-to-a-power-law
    # https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0170920&type=printable
    # https://www.mathworks.com/help/curvefit/evaluating-goodness-of-fit.html --> based on this we should use SSE or RMSE
    #rmse_it = sqrt( metrics.mean_squared_error(y_it, y_it_fit) )
    #rmse_et = sqrt( metrics.mean_squared_error(y_et, y_et_fit) )
    mae_it = metrics.mean_absolute_error(y_it, y_it_fit )
    mae_et = metrics.mean_absolute_error(y_et, y_et_fit )
    
    # Init figure
    fontsize = 13
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raw data
    # ax.plot(x, y, '.', color=None, label=label);
    ax.plot(x_it, y_it, '.', color=None, label=f'{label} for interpolation');
    ax.plot(x_et, y_et, 'o', color=None, label=f'{label} for extrapolation');

    # Plot fit
    if plot_fit: ax.plot(x, y_fit, '--', color=None, label=f'{label} Fit'); 
    if plot_fit: ax.plot(x_it, y_it_fit, '--', color=None, label=f'{label} interpolation (MAE {mae_it:.7f})');
    if plot_fit: ax.plot(x_et, y_et_fit, '--', color=None, label=f'{label} extrapolation (MAE {mae_et:.7f})');
        
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
    # eq = r"$\varepsilon_{mae}(m) = \alpha m^{\beta} + \gamma$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}, $\gamma$={power_law_params['gamma']:.2f}"
    # eq = rf"$\varepsilon(m) = {power_law_params['alpha']:.2f} m^{power_law_params['beta']:.2f} + {power_law_params['gamma']:.2f}$" # TODO: make this work    
    
    eq = r"$\varepsilon(m) = \alpha m^{\beta}$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}"
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
    if title is None: title='Learning Curve (power-law)'
    ax.set_title(title)
    
    # Location of legend --> https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132
    ax.legend(frameon=True, fontsize=fontsize, bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(True)
    # return fig, ax, power_law_params
    return ax, power_law_params
# --------------------------------------------------------------------------------------------------


def plot_runtime(rt:pd.DataFrame, outdir:Path=None, figsize=(7,5),
        xtick_scale:str='linear', ytick_scale:str='linear'):
    """ Plot training time vs shard size. """
    fontsize = 13
    fig, ax = plt.subplots(figsize=figsize)
    for f in rt['fold'].unique():
        d = rt[rt['fold']==f]
        ax.plot(d['tr_sz'], d['time'], '.--', label='fold'+str(f))
       
    # Set axes scale and labels
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)

    ax.set_xlabel(f'Train Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ax.set_ylabel(f'Training Time (minutes) ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)

    ax.set_title('Runtime')
    #ax.set_xlabel(f'Training Size', fontsize=fontsize)
    #ax.set_ylabel(f'Training Time (minutes)', fontsize=fontsize)
    ax.legend(loc='best', frameon=True, fontsize=fontsize)
    ax.grid(True)

    # Save fig
    if outdir is not None: plt.savefig(outdir/'runtime.png', bbox_inches='tight')

        
def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))    
