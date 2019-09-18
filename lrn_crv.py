"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.

TODO: move utils.calc_scores to a more local function.
"""
import os
import sys
from pathlib import Path
from collections import OrderedDict

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

import keras
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import plot_model

from scipy import optimize

# Utils
import ml_models


class LearningCurve():
    """
    Train estimator using multiple train set sizes and generate learning curves for multiple metrics.
    The CV splitter splits the input dataset into cv_folds data subsets.
    Examples:
        cv = sklearn.model_selection.KFold(n_splits=5, shuffle=False, random_state=0)
        lrn_curve.my_learning_curve(X=xdata, Y=ydata, mltype='reg', cv=cv, n_shards=5)
    """
    def __init__(self,
            X, Y,
            cv=5,
            cv_lists=None,
            n_shards: int=5,
            # shard_step_scale: str='log2',
            # shard_frac=[],
            args=None,
            logger=None,
            outdir='./'):
        """
        Args:
            X : array-like (pd.DataFrame or np.ndarray)
            Y : array-like (pd.DataFrame or np.ndarray)
            cv : (optional) number of cv folds (int) or sklearn cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
            cv_lists : tuple of 2 dicts, cv_lists[0] and cv_lists[1], that contain the tr and vl folds, respectively 
            n_shards : number of dataset splits in the learning curve (used if shard_frac is None)
            
            shard_step_scale : if n_shards is provided, this will generate a list of training set sizes with steps
                specified by this arg. Available values: 'linear', 'log2', 'log10', 'log'.
                e.g., if n_shards=5 and shard_step_scale='linear', then it generates ...
            shard_frac : list of relative numbers of training samples that are used to generate learning curves
                e.g., shard_frac=[0.1, 0.2, 0.4, 0.7, 1.0].
                If this arg is not provided, then the training shards are generated from n_shards and shard_step_scale.
                
            args : command line args
        """
        self.X = pd.DataFrame(X).values
        self.Y = pd.DataFrame(Y).values
        self.cv = cv
        self.cv_lists = cv_lists
        self.n_shards = n_shards
        # self.shard_step_scale = shard_step_scale 
        # self.shard_frac = shard_frac
        self.args = args
        self.logger = logger
        self.outdir = Path(outdir)

        self.create_fold_dcts()
        self.create_tr_shards_list()

        
    def create_fold_dcts(self):
        """ Returns a tuple of two dicts (tr_dct, vl_dct) that contain the splits of all the folds. """
        tr_dct = {}
        vl_dct = {}

        # Use lists passed as input arg
        if self.cv_lists is not None:
            tr_id = self.cv_lists[0]
            vl_id = self.cv_lists[1]
            assert tr_id.shape[1] == vl_id.shape[1], 'tr and vl must have the same number of folds.'
            self.cv_folds = tr_id.shape[1]

            # Calc the split ratio if cv=1
            if self.cv_folds == 1:
                self.vl_size = vl_id.shape[0]/(vl_id.shape[0] + tr_id.shape[0])

            for fold in range(tr_id.shape[1]):
                tr_dct[fold] = tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
                vl_dct[fold] = vl_id.iloc[:, fold].dropna().values.astype(int).tolist()

        # Generate folds on the fly if no pre-defined folds were passed
        else:
            if isinstance(self.cv, int):
                self.cv_folds = self.cv
                self.cv = KFold(n_splits=self.cv_folds, shuffle=False, random_state=self.random_state)
            else:
                # cv is sklearn splitter
                self.cv_folds = cv.get_n_splits() 

            if cv_folds == 1:
                self.vl_size = cv.test_size

            # Create sklearn splitter 
            if self.mltype == 'cls':
                if self.Y.ndim > 1 and self.Y.shape[1] > 1:
                    splitter = self.cv.split(self.X, np.argmax(self.Y, axis=1))
            else:
                splitter = self.cv.split(self.X, self.Y)
            
            # Generate the splits
            for fold, (tr_vec, vl_vec) in enumerate(splitter):
                tr_dct[fold] = tr_vec
                vl_dct[fold] = vl_vec

        self.tr_dct = tr_dct
        self.vl_dct = vl_dct


    def create_tr_shards_list(self):
        """ Generate the list of training shard sizes. """
#         if len(self.shard_frac)==0:
#             # if any( [self.shard_step_scale.lower()==s for s in ['lin', 'linear']] ):
#             scale = self.shard_step_scale.lower()
#             if scale == 'linear':
#                 self.shard_frac = np.linspace(0.1, 1.0, self.n_shards)
#             else:
#                 if scale == 'log2':
#                     base = 2
#                 elif scale == 'log10':
#                     base = 10
#                 # In np.logspace the sequence starts at base ** start 
#                 # self.shard_frac = np.logspace(start=0.0, stop=1.0, num=self.n_shards, endpoint=True, base=base)/base
#                 # shard_frac_small = list(np.logspace(start=0.0, stop=1.0, num=2*self.n_shards, endpoint=True, base=base)/(self.X.shape[0]/10))
#                 # shard_frac_low_range = list(np.linspace(start=10, stop=int(0.1*self.X.shape[0]), num=2*self.n_shards, endpoint=False)/self.X.shape[0])
#                 shard_frac = list(np.logspace(start=0.0, stop=1.0, num=self.n_shards, endpoint=True, base=base)/base)
#                 # shard_frac.extend(shard_frac_low_range)
#                 self.shard_frac = np.array( sorted(list(set(shard_frac))) )

#             if self.logger: self.logger.info(f'Shard step spacing: {self.shard_step_scale}.')

#         if self.cv_folds == 1:
#             self.tr_shards = [int(n) for n in (1-self.vl_size) * self.X.shape[0] * self.shard_frac if n>0]
#         else: 
#             self.tr_shards = [int(n) for n in (self.cv_folds-1)/self.cv_folds * self.X.shape[0] * self.shard_frac if n>0]

        # --------------------------------------------
        # Fixed spacing
        if self.cv_folds == 1:
            self.max_samples = int((1-self.vl_size) * self.X.shape[0])
        else: 
            self.max_samples = int((self.cv_folds-1)/self.cv_folds * self.X.shape[0])
            
        # TODO need to add self.max_samples to the training vector
        v = 2 ** np.array(np.arange(30))[1:]
        idx = np.argmin( np.abs( v - self.max_samples ) )
        
        if v[idx] > max_samples:
            v = list(v[:idx])
            v.append(max_samples)
        else:
            # v = list(v[:idx])
            v = list(v[:idx+1])
            v.append(max_samples)
            # If the diff btw max_samples and the latest shards (v[-1] - v[-2]) is "too small", then remove max_samples from the possible shards.
            if 0.5*v[-3] > (v[-1] - v[-2]):
                print('here')
                v = v[:-1]
        
        self.tr_shards = v[:-self.n_shards]
        # --------------------------------------------
        
        if self.logger is not None: self.logger.info('Train shards: {}\n'.format(self.tr_shards))


    def trn_learning_curve(self,
            framework: str='lightgbm',
            mltype: str='reg',
            model_name: str='lgb_reg', # TODO! this is redundent
            init_kwargs: dict={},
            fit_kwargs: dict={},
            clr_keras_kwargs: dict={},
            metrics: list=['r2', 'neg_mean_absolute_error', 'neg_median_absolute_error', 'neg_mean_squared_error'],
            n_jobs: int=4,
            random_state: int=None,
            plot=True):
        """ 
        Args:
            framework : ml framework (keras, lightgbm, or sklearn)
            mltype : type to ml problem (reg or cls)
            init_kwargs : dict of parameters that initialize the estimator
            fit_kwargs : dict of parameters to the estimator's fit() method
            clr_keras_kwargs : 
            metrics : allow to pass a string of metrics  TODO!
        """
        self.framework = framework
        self.mltype = mltype
        self.model_name = model_name
        self.init_kwargs = init_kwargs
        self.fit_kwargs = fit_kwargs 
        self.clr_keras_kwargs = clr_keras_kwargs
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Start nested loop of train size and cv folds
        tr_scores_all = [] # list of dicts
        vl_scores_all = [] # list of dicts

        # CV loop
        for fold, (tr_k, vl_k) in enumerate(zip( self.tr_dct.keys(), self.vl_dct.keys() )):
            if self.logger is not None: self.logger.info(f'Fold {fold+1}/{self.cv_folds}')

            tr_id = self.tr_dct[tr_k]
            vl_id = self.vl_dct[vl_k]

            # Samples from this dataset are randomly sampled for training
            xtr = self.X[tr_id, :]
            ytr = self.Y[tr_id, :]

            # A fixed set of validation samples for the current CV split
            xvl = self.X[vl_id, :]
            yvl = np.squeeze(self.Y[vl_id, :])        

            # Shards loop (iterate across the dataset sizes and train)
            # np.random.seed(random_state)
            # idx = np.random.permutation(len(xtr))
            idx = np.arange(len(xtr))
            for i, tr_sz in enumerate(self.tr_shards):
                # For each shard: train model, save best model, calc tr_scores, calc_vl_scores
                if self.logger: self.logger.info(f'\tTrain size: {tr_sz} ({i+1}/{len(self.tr_shards)})')   

                # Sequentially get a subset of samples (the input dataset X must be shuffled)
                xtr_sub = xtr[idx[:tr_sz], :]
                ytr_sub = np.squeeze(ytr[idx[:tr_sz], :])            

                # Get the estimator
                estimator = ml_models.get_model(self.model_name, init_kwargs=self.init_kwargs)
                model = estimator.model
                
                # Train
                # self.val_split = 0 # 0.1 # used for early stopping
                self.eval_frac = 0.1 # 0.1 # used for early stopping
                eval_samples = int(self.eval_frac*xvl.shape[0])
                eval_set = (xvl[:eval_samples, :], yvl[:eval_samples])
                if self.framework=='lightgbm':
                    model, trn_outdir = self.trn_lgbm_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub, fold=fold, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='keras':
                    model, trn_outdir = self.trn_keras_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub, fold=fold, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='pytorch':
                    pass
                else:
                    raise ValueError(f'framework {self.framework} is not supported.')

                # Calc preds and scores TODO: dump preds
                # ... training set
                y_pred, y_true = calc_preds(model, x=xtr_sub, y=ytr_sub, mltype=self.mltype)
                tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                # ... val set
                y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=self.mltype)
                vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)

                del estimator, model
                # nm = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
                # dn = ((y_true - np.average(y_true, axis=0)) ** 2).sum(axis=0, dtype=np.float64)

                # Add metadata
                tr_scores['tr_set'] = True
                tr_scores['fold'] = 'fold'+str(fold)
                tr_scores['tr_size'] = tr_sz
                
                vl_scores['tr_set'] = False
                vl_scores['fold'] = 'fold'+str(fold)
                vl_scores['tr_size'] = tr_sz

                # Append scores (dicts)
                tr_scores_all.append(tr_scores)
                vl_scores_all.append(vl_scores)

                # Dump intermediate scores
                # TODO: test this!
                scores_tmp = pd.concat([scores_to_df(tr_scores_all), scores_to_df(vl_scores_all)], axis=0)
                scores_tmp.to_csv( trn_outdir / ('tmp_scores.csv'), index=False )
                del trn_outdir, tmp_scores
                
            # Dump intermediate results (this is useful if the run terminates before run ends)
            # tr_df_tmp = scores_to_df(tr_scores_all)
            # vl_df_tmp = scores_to_df(vl_scores_all)
            scores_all_df_tmp = pd.concat([scores_to_df(tr_scores_all), scores_to_df(vl_scores_all)], axis=0)
            scores_all_df_tmp.to_csv( self.outdir / ('_lrn_crv_scores_cv' + str(fold+1) + '.csv'), index=False )

        # Scores to df
        tr_scores_df = scores_to_df( tr_scores_all )
        vl_scores_df = scores_to_df( vl_scores_all )
        scores_df = pd.concat([tr_scores_df, vl_scores_df], axis=0)
        
        # Dump final results
        tr_scores_df.to_csv( self.outdir/'tr_lrn_crv_scores.csv', index=False) 
        vl_scores_df.to_csv( self.outdir/'vl_lrn_crv_scores.csv', index=False) 
        scores_df.to_csv( self.outdir/'lrn_crv_scores.csv', index=False) 
        
        # Plot learning curves
        if plot:
            plot_lrn_crv_all_metrics( scores_df, outdir=self.outdir )

        return scores_df


    def trn_keras_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ ... """
        keras.utils.plot_model(model, to_file=self.outdir/'nn_model.png')

        # Create output dir
        trn_outdir = self.outdir / ('cv'+str(fold+1) + '_sz'+str(tr_sz))
        os.makedirs(trn_outdir, exist_ok=False)
        
        # Keras callbacks
        keras_callbacks = define_keras_callbacks(trn_outdir)
        
        # if bool(self.clr_keras_kwargs):
        if self.clr_keras_kwargs['mode'] is not None:
            keras_callbacks.append( ml_models.clr_keras_callback(**self.clr_keras_kwargs) )

        # Fit params
        fit_kwargs = self.fit_kwargs
        fit_kwargs['validation_data'] = eval_set
        # fit_kwargs['validation_split'] = self.val_split
        fit_kwargs['callbacks'] = keras_callbacks

        # Train model
        history = model.fit(xtr_sub, ytr_sub, **fit_kwargs)
        ml_models.save_krs_history(history, outdir=trn_outdir)
        ml_models.plot_prfrm_metrics(history, title=f'Train size: {tr_sz}', skp_ep=20, add_lr=True, outdir=trn_outdir)

        # Load the best model (https://github.com/keras-team/keras/issues/5916)
        # model = keras.models.load_model(str(trn_outdir/'model_best.h5'), custom_objects={'r2_krs': ml_models.r2_krs})
        model = keras.models.load_model( str(trn_outdir/'model_best.h5') )
        return model, trn_outdir


    def trn_lgbm_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save LigthGBM model. """
        # Create output dir
        trn_outdir = self.outdir / ('cv'+str(fold+1) + '_sz'+str(tr_sz))
        # os.makedirs(trn_outdir, exist_ok=False)
        os.makedirs(trn_outdir, exist_ok=True)

        # Get a subset of samples for validation for early stopping
        fit_kwargs = self.fit_kwargs
        # xtr_sub, xvl_sub, ytr_sub, yvl_sub = train_test_split(xtr_sub, ytr_sub, test_size=self.val_split)
        # if xvl_sub_.shape[0] > 0:
        #     fit_kwargs['eval_set'] = (xvl_sub, yvl_sub)
        #     fit_kwargs['early_stopping_rounds'] = 10
        fit_kwargs['eval_set'] = eval_set
        fit_kwargs['early_stopping_rounds'] = 10

        # Train and save model
        model.fit(xtr_sub, ytr_sub, **fit_kwargs)
        joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir

    


def define_keras_callbacks(outdir):
    checkpointer = ModelCheckpoint(str(outdir/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    # early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=60, verbose=1)
    return [checkpointer, csv_logger, early_stop, reduce_lr]


def plot_lrn_crv_all_metrics(df, outdir:Path, figsize=(7,5), xtick_scale='linear', ytick_scale='linear'):
    """ Takes the entire table of scores across folds and train set sizes, and generates plots of 
    learning curves for the different metrics.
    Args:
        df : contains train and val scores for cv folds (the scores are the last cv_folds cols)
            metric | tr_set | tr_size |  fold0  |  fold1  |  fold2  |  fold3  |  fold4
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

        tr = aa[aa['tr_set']==True]
        vl = aa[aa['tr_set']==False]
        # tr = aa[aa['phase']=='train']
        # vl = aa[aa['phase']=='val']

        tr = tr[[c for c in tr.columns if 'fold' in c]]
        vl = vl[[c for c in vl.columns if 'fold' in c]]
        # tr = tr.iloc[:, -cv_folds:]
        # vl = vl.iloc[:, -cv_folds:]

        rslt = []
        rslt.append(tr_shards)
        rslt.append(tr.values if tr.values.shape[0]>0 else None)
        rslt.append(vl.values if vl.values.shape[0]>0 else None)

        fname = 'lrn_crv_' + metric_name + '.png'
        title = 'Learning curve'

        path = outdir / fname
        fig = plot_lrn_crv(rslt=rslt, metric_name=metric_name, figsize=figsize,
                xtick_scale=xtick_scale, ytick_scale=ytick_scale, title=title, path=path)
        # figs.append(fig)
        

def scale_ticks_params(tick_scale='linear'):
    """ Helper function for learning cureve plots.
    Args:
        tick_scale : available values are [linear, log2, log10]
    """
    if tick_scale == 'linear':
        base = None
        label_scale = 'Linear scale'
    else:
        if tick_scale == 'log2':
            base = 2
            label_scale = 'Log2 scale'
        elif tick_scale == 'log10':
            base = 10
            label_scale = 'Log10 scale'
        else:
            raise ValueError('The specified tick scale is not supported.')
    return base, label_scale


def plot_lrn_crv(rslt:list, metric_name:str='score',
                 xtick_scale:str='log2', ytick_scale:str='log2',
                 xlim:list=None, ylim:list=None, title:str=None, path:Path=None,
                 figsize=(7,5), ax=None):
    """ 
    Args:
        rslt : output from sklearn.model_selection.learning_curve()
            rslt[0] : 1-D array (n_ticks, ) -> vector of train set sizes
            rslt[1] : 2-D array (n_ticks, n_cv_folds) -> tr scores
            rslt[2] : 2-D array (n_ticks, n_cv_folds) -> vl scores
    """
    tr_shards = rslt[0]
    tr_scores = rslt[1]
    vl_scores = rslt[2]
    
    def plot_single_crv(tr_shards, scores, ax, phase, color=None):
        scores_mean = np.mean(scores, axis=1)
        scores_std  = np.std( scores, axis=1)
        ax.plot(tr_shards, scores_mean, '.-', color=color, label=f'{phase} score')
        ax.fill_between(tr_shards, scores_mean - scores_std, scores_mean + scores_std, alpha=0.1, color=color)

    # Plot learning curves
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    if tr_scores is not None:
        plot_single_crv(tr_shards, scores=tr_scores, ax=ax, color='b', phase='Train')
    if vl_scores is not None:
        plot_single_crv(tr_shards, scores=vl_scores, ax=ax, color='g', phase='Val')

    # Set axes scale and labels
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)

    ax.set_xlabel(f'Train Dataset Size ({xlabel_scale})')
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylbl = ' '.join(s.capitalize() for s in metric_name.split('_'))
    ax.set_ylabel(f'{ylbl} ({ylabel_scale})')
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)

    # Other settings
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is None: title='Learning curve'
    ax.set_title(title)
        
    ax.legend(loc='best', frameon=True)
    ax.grid(True)
    plt.tight_layout()

    # Save fig
    if path is not None: plt.savefig(path, bbox_inches='tight')
    return ax


def power_law_func(x, alpha, beta, gamma):
    """ docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.power.html """
    return alpha * np.power(x, beta) + gamma
    
    
def power_law_func_(x, alpha, beta, gamma1, gamma2):
    """ docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.power.html """
    return alpha * np.power(x, beta) + gamma1 + gamma2
    
    
def fit_power_law(x, y, p0:list=[30, -0.3, 0.06]):
    """ Fit learning curve data (train set size vs metric) to power-law.
    TODO: How should we fit the data across multiple folds? This can
    be addressed using Bayesian methods (look at Bayesian linear regression).
    The uncertainty of parameters indicates the consistency of across folds.
    """
    prms, prms_cov = optimize.curve_fit(power_law_func, x, y, p0=p0)
    prms_dct = {}
    prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma'] = prms[0], prms[1], prms[2]
    return prms_dct


def fit_power_law_(x, y, p0:list=[30, -0.3, 0.06, 0.12]):
    """ Fit learning curve data (train set size vs metric) to power-law. """
    prms, prms_cov = optimize.curve_fit(power_law_func_, x, y, p0=p0)
    prms_dct = {}
    prms_dct['alpha'], prms_dct['beta'], prms_dct['gamma1'], prms_dct['gamma2'] = prms[0], prms[1], prms[2], prms[3]
    return prms_dct


def plot_lrn_crv_power_law(x, y, plot_fit:bool=True, metric_name:str='score',
                           xtick_scale:str='log2', ytick_scale:str='log2',
                           xlim:list=None, ylim:list=None, title:str=None, figsize=(7,5)):
    """ ... """
    x = x.ravel()
    y = y.ravel()
    
    fontsize = 13
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, '.-', color=None, label='data');

    # Fit power-law
    power_law_params = fit_power_law(x, y)
    yfit = power_law_func(x, **power_law_params)
    # power_law_params_ = fit_power_law(x, y)
    # yfit = power_law_func_(x, **power_law_params_)
    if plot_fit: ax.plot(x, yfit, '--', color=None, label='fit');    
        
    basex, xlabel_scale = scale_ticks_params(tick_scale=xtick_scale)
    basey, ylabel_scale = scale_ticks_params(tick_scale=ytick_scale)
    
    ax.set_xlabel(f'Training Dataset Size ({xlabel_scale})', fontsize=fontsize)
    if 'log' in xlabel_scale.lower(): ax.set_xscale('log', basex=basex)

    ylabel = ' '.join(s.capitalize() for s in metric_name.split('_'))
    ax.set_ylabel(f'{ylabel} ({ylabel_scale})', fontsize=fontsize)
    if 'log' in ylabel_scale.lower(): ax.set_yscale('log', basey=basey)
        
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    
    # Add equation (text) on the plot
    # matplotlib.org/3.1.1/gallery/text_labels_and_annotations/usetex_demo.html#sphx-glr-gallery-text-labels-and-annotations-usetex-demo-py
    # eq = r"$\varepsilon_{mae}(m) = \alpha m^{\beta} + \gamma$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}, $\gamma$={power_law_params['gamma']:.2f}"
    # eq = rf"$\varepsilon(m) = {power_law_params['alpha']:.2f} m^{power_law_params['beta']:.2f} + {power_law_params['gamma']:.2f}$" # TODO: make this work
    
    eq = r"$\varepsilon(m) = \alpha m^{\beta}$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}"
    # xloc = 2.0 * x.ravel().min()
    xloc = x.min() + 0.1*(x.max() - x.min())
    yloc = y.min() + 0.9*(y.max() - y.min())
    ax.text(xloc, yloc, eq,
            {'color': 'black', 'fontsize': fontsize, 'ha': 'left', 'va': 'center',
             'bbox': {'boxstyle':'round', 'fc':'white', 'ec':'black', 'pad':0.2}})
    
    # matplotlib.org/users/mathtext.html
    # ax.set_title(r"$\varepsilon_{mae}(m) = \alpha m^{\beta} + \gamma$" + rf"; $\alpha$={power_law_params['alpha']:.2f}, $\beta$={power_law_params['beta']:.2f}, $\gamma$={power_law_params['gamma']:.2f}");
    if ylim is not None: ax.set_ylim(ylim)
    if xlim is not None: ax.set_ylim(xlim)
    if title is None: title='Learning curve (power-law)'
    ax.set_title(title)
    
    ax.legend(loc='best', frameon=True, fontsize=fontsize)
    ax.grid(True)
    return fig, ax, power_law_params




# Define custom metric to calc auroc from regression
# scikit-learn.org/stable/modules/model_evaluation.html#scoring
def reg_auroc(y_true, y_pred, th=0.5):
    """ Compute area under the ROC for regression. """
    y_true = np.where(y_true < th, 1, 0)
    y_score = np.where(y_pred < th, 1, 0)
    reg_auroc_score = sklearn.metrics.roc_auc_score(y_true, y_score)
    return reg_auroc_score


def reg_auroc_score():
    return sklearn.metrics.make_scorer(score_func=reg_auroc, greater_is_better=True)    


def calc_preds(model, x, y, mltype):
    """ Calc predictions. """
    if mltype == 'cls':    
        if y.ndim > 1 and y.shape[1] > 1:
            y_pred = model.predict_proba(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(ydata, axis=1)
        else:
            y_pred = model.predict_proba(x)
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y
            
    elif mltype == 'reg':
        y_pred = model.predict(x)
        y_true = y

    return y_pred, y_true


def calc_scores(y_true, y_pred, mltype, metrics=None):
    """ Create dict of scores.
    Args:
        metrics : TODO allow to pass a string of metrics
    """
    scores = {}

    if mltype == 'cls':    
        scores['auroc'] = sklearn.metrics.roc_auc_score(y_true, y_pred)
        scores['f1_score'] = sklearn.metrics.f1_score(y_true, y_pred, average='micro')
        scores['acc_blnc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    elif mltype == 'reg':
        scores['r2'] = sklearn.metrics.r2_score(y_true=y_true, y_pred=y_pred)
        scores['mean_absolute_error'] = sklearn.metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['median_absolute_error'] = sklearn.metrics.median_absolute_error(y_true=y_true, y_pred=y_pred)
        scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        # scores['auroc_reg'] = reg_auroc(y_true=y_true, y_pred=y_pred)

    # # https://scikit-learn.org/stable/modules/model_evaluation.html
    # for metric_name, metric in metrics.items():
    #     if isinstance(metric, str):
    #         scorer = sklearn.metrics.get_scorer(metric_name) # get a scorer from string
    #         scores[metric_name] = scorer(ydata, pred)
    #     else:
    #         scores[metric_name] = scorer(ydata, pred)
    return scores


def scores_to_df(scores_all):
    """ (tricky commands) """
    df = pd.DataFrame(scores_all)
    df = df.melt(id_vars=['fold', 'tr_size', 'tr_set'])
    df = df.rename(columns={'variable': 'metric'})
    df = df.pivot_table(index=['metric', 'tr_size', 'tr_set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


