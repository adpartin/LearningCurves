"""
Functions to generate learning curves.
Records performance (error or score) vs training set size.
"""
import os
import sys
from pathlib import Path
from time import time
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

from sklearn import metrics
from math import sqrt

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
from plotting import plot_hist


# --------------------------------------------------------------------------------
class LearningCurve():
    """
    Train estimator using multiple shards (train set sizes) and generate learning curves for multiple performance metrics.
    Example:
        lc = LearningCurve(xdata, ydata, cv_lists=(tr_ids, vl_ids))
        lrn_crv_scores = lc.trn_learning_curve( framework=framework, mltype=mltype, model_name=model_name,
                                                init_kwargs=init_kwargs, fit_kwargs=fit_kwargs, clr_keras_kwargs=clr_keras_kwargs)
    """
    def __init__(self,
            X, Y,
            cv=5,
            cv_lists=None,  # (tr_id, vl_id, te_id)
            cv_folds_arr=None,
            shard_step_scale: str='log2',
            min_shard = 0,
            max_shard = None,
            n_shards: int=None,
            shards_arr: list=[],
            args=None,
            logger=None,
            outdir='./'):
        """
        Args:
            X : array-like (pd.DataFrame or np.ndarray)
            Y : array-like (pd.DataFrame or np.ndarray)
            cv : (optional) number of cv folds (int) or sklearn cv splitter --> scikit-learn.org/stable/glossary.html#term-cv-splitter
            # cv_lists : tuple of 2 dicts, cv_lists[0] and cv_lists[1], that contain the tr and vl folds, respectively
            cv_lists : tuple of 3 dicts, cv_lists[0] and cv_lists[1], cv_lists[2], that contain the tr, vl, and te folds, respectively
            cv_folds_arr : list that contains the specific folds in the cross-val run

            shard_step_scale : specifies how to generate the shard values. 
                Available values: 'linear', 'log2', 'log10'.

            min_shard : min shard value in the case when shard_step_scale is 'log2' or 'log10'
            max_shard : max shard value in the case when shard_step_scale is 'log2' or 'log10'

            n_shards : number of shards in the learning curve (used only in the shard_step_scale is 'linear')
            shards_arr : list ints specifying the shards to process (e.g., [128, 256, 512])
            
            shard_frac : list of relative numbers of training samples that are used to generate learning curves
                e.g., shard_frac=[0.1, 0.2, 0.4, 0.7, 1.0].
                If this arg is not provided, then the training shards are generated from n_shards and shard_step_scale.
                
            args : command line args
        """
        self.X = pd.DataFrame(X).values
        self.Y = pd.DataFrame(Y).values
        self.cv = cv
        self.cv_lists = cv_lists
        self.cv_folds_arr = cv_folds_arr

        self.shard_step_scale = shard_step_scale 
        self.min_shard = min_shard
        self.max_shard = max_shard
        self.n_shards = n_shards
        self.shards_arr = shards_arr

        self.args = args
        self.logger = logger
        self.outdir = Path(outdir)

        self.create_fold_dcts()
        self.create_tr_shards_list()

        
    def create_fold_dcts(self):
        """ Converts a tuple of arrays self.cv_lists into two dicts, tr_dct, vl_dct, and te_dict.
        Both sets of data structures contain the splits of all the k-folds. """
        tr_dct = {}
        vl_dct = {}
        te_dct = {}

        # Use lists passed as input arg
        if self.cv_lists is not None:
            tr_id = self.cv_lists[0]
            vl_id = self.cv_lists[1]
            te_id = self.cv_lists[2]
            assert (tr_id.shape[1]==vl_id.shape[1]) and (tr_id.shape[1]==te_id.shape[1]), 'tr, vl, and te must have the same number of folds.'
            self.cv_folds = tr_id.shape[1]

            # Calc the split ratio if cv=1
            if self.cv_folds == 1:
                self.vl_size = vl_id.shape[0]/(tr_id.shape[0] + vl_id.shape[0] + te_id.shape[0])
                self.te_size = te_id.shape[0]/(tr_id.shape[0] + vl_id.shape[0] + te_id.shape[0])

            if self.cv_folds_arr is None: self.cv_folds_arr = [f+1 for f in range(self.cv_folds)]
                
            for fold in range(tr_id.shape[1]):
                if fold+1 in self.cv_folds_arr:
                    tr_dct[fold] = tr_id.iloc[:, fold].dropna().values.astype(int).tolist()
                    vl_dct[fold] = vl_id.iloc[:, fold].dropna().values.astype(int).tolist()
                    te_dct[fold] = te_id.iloc[:, fold].dropna().values.astype(int).tolist()
                

        # Generate folds on the fly if no pre-defined folds were passed
        # TODO: this option won't work after we added test set in addition to train and val sets.
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

        # Keep dicts
        self.tr_dct = tr_dct
        self.vl_dct = vl_dct
        self.te_dct = te_dct


    def create_tr_shards_list(self):
        """ Generate a list of training shards (training sizes). """
        if self.shards_arr is not None:
            # No need to generate an array of training shards if shards_arr is specified
            self.tr_shards = self.shards_arr
            
        else:
            # Fixed spacing
            if self.max_shard is None:
                self.max_shard = len(self.tr_dct[0])
                # if self.cv_folds == 1:
                #     self.max_shard = int( (1 - self.vl_size - self.te_size) * self.X.shape[0] )
                # else: 
                #     self.max_shard = int( (self.cv_folds-1)/self.cv_folds * self.X.shape[0] )

            # Full vector of shards
            # (we create a vector with very large values so that we later truncate it with max_shard)
            scale = self.shard_step_scale.lower()
            if scale == 'linear':
                m = np.linspace(0, self.max_shard, self.n_shards+1)[1:]
            else:
                # we create very large vector m, so that we later truncate it with max_shard
                if scale == 'log2':
                    m = 2 ** np.array(np.arange(30))[1:]
                elif scale == 'log':
                    m = np.exp( np.array(np.arange(8))[1:] )
                elif scale == 'log10':
                    m = 10 ** np.array(np.arange(8))[1:]

            m = np.array( [int(i) for i in m] ) # cast to int

            # Set min shard
            idx_min = np.argmin( np.abs( m - self.min_shard ) )
            if m[idx_min] > self.min_shard:
                m = m[idx_min:]  # all values larger than min_shard
                m = np.concatenate( (np.array([self.min_shard]), m) )  # preceed arr with specified min_shard
            else:
                m = m[idx_min:]

            # Set max shard
            idx_max = np.argmin( np.abs( m - self.max_shard ) )
            if m[idx_max] > self.max_shard:
                m = list(m[:idx_max])    # all values EXcluding the last one
                m.append(self.max_shard)
            else:
                m = list(m[:idx_max+1])  # all values INcluding the last one
                m.append(self.max_shard) # TODO: should this be also added??
                # If the diff btw max_samples and the latest shards (m[-1] - m[-2]) is "too small", then remove max_samples from the possible shards.
                if 0.5*m[-3] > (m[-1] - m[-2]): m = m[:-1] # heuristic to drop the last shard

            self.tr_shards = m
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
        te_scores_all = [] # list of dicts

        # Record runtime per shard
        runtime_records = []

        # CV loop
        for fold, (tr_k, vl_k, te_k) in enumerate(zip( self.tr_dct.keys(), self.vl_dct.keys(), self.te_dct.keys() )):
            fold = fold + 1
            if self.logger is not None: self.logger.info(f'Fold {fold}/{self.cv_folds}')

            # Get the indices for this fold
            tr_id = self.tr_dct[tr_k]
            vl_id = self.vl_dct[vl_k]
            te_id = self.te_dct[te_k]

            # Samples from this dataset are randomly sampled for training
            xtr = self.X[tr_id, :]
            # ytr = self.Y[tr_id, :]
            ytr = np.squeeze(self.Y[tr_id, :])        

            # A fixed set of val samples for the current CV split
            xvl = self.X[vl_id, :]
            yvl = np.squeeze(self.Y[vl_id, :])        

            # A fixed set of test samples for the current CV split
            xte = self.X[te_id, :]
            yte = np.squeeze(self.Y[te_id, :])        

            # Shards loop (iterate across the dataset sizes and train)
            """
            np.random.seed(random_state)
            idx = np.random.permutation(len(xtr))
            Note that we don't shuffle the dataset another time using the commands above.
            """
            idx = np.arange(len(xtr))
            for i, tr_sz in enumerate(self.tr_shards):
                # For each shard: train model, save best model, calc tr_scores, calc_vl_scores
                if self.logger: self.logger.info(f'\tTrain size: {tr_sz} ({i+1}/{len(self.tr_shards)})')   

                # Sequentially get a subset of samples (the input dataset X must be shuffled)
                xtr_sub = xtr[idx[:tr_sz], :]
                # ytr_sub = np.squeeze(ytr[idx[:tr_sz], :])
                ytr_sub = ytr[idx[:tr_sz]]
                
                # Get the estimator
                estimator = ml_models.get_model(self.model_name, init_kwargs=self.init_kwargs)
                model = estimator.model
                
                # Train
                # self.val_split = 0 # 0.1 # used for early stopping
                #self.eval_frac = 0.1 # 0.1 # used for early stopping
                #eval_samples = int(self.eval_frac * xvl.shape[0])
                #eval_set = (xvl[:eval_samples, :], yvl[:eval_samples]) # we don't random sample; the same eval_set is used for early stopping
                eval_set = (xvl, yvl)
                if self.framework=='lightgbm':
                    model, trn_outdir, runtime = self.trn_lgbm_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub, fold=fold, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='sklearn':
                    model, trn_outdir, runtime = self.trn_sklearn_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub, fold=fold, tr_sz=tr_sz, eval_set=None)
                elif self.framework=='keras':
                    model, trn_outdir, runtime = self.trn_keras_model(model=model, xtr_sub=xtr_sub, ytr_sub=ytr_sub, fold=fold, tr_sz=tr_sz, eval_set=eval_set)
                elif self.framework=='pytorch':
                    pass
                else:
                    raise ValueError(f'Framework {self.framework} is not supported.')

                # Save plot of target distribution
                plot_hist(ytr_sub, var_name=f'Target (Train size={tr_sz})', fit=None, bins=100, path=trn_outdir/'target_hist_tr.png')
                plot_hist(yvl, var_name=f'Target (Val size={len(yvl)})', fit=None, bins=100, path=trn_outdir/'target_hist_vl.png')
                plot_hist(yte, var_name=f'Target (Test size={len(yte)})', fit=None, bins=100, path=trn_outdir/'target_hist_te.png')
                    
                # Calc preds and scores TODO: dump preds
                # ... training set
                y_pred, y_true = calc_preds(model, x=xtr_sub, y=ytr_sub, mltype=self.mltype)
                tr_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                tr_scores['y_avg'] = np.mean(y_pred)
                # ... val set
                y_pred, y_true = calc_preds(model, x=xvl, y=yvl, mltype=self.mltype)
                vl_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                vl_scores['y_avg'] = np.mean(y_pred)
                # ... test set
                y_pred, y_true = calc_preds(model, x=xte, y=yte, mltype=self.mltype)
                te_scores = calc_scores(y_true=y_true, y_pred=y_pred, mltype=self.mltype, metrics=None)
                te_scores['y_avg'] = np.mean(y_pred)

                del estimator, model
                
                # Save predictions (need to include metadata)
                # TODO
                pass

                # Store runtime
                runtime_records.append((fold, tr_sz, runtime))

                # Add metadata
                # tr_scores['tr_set'] = True
                tr_scores['set'] = 'tr'
                tr_scores['fold'] = 'fold'+str(fold)
                tr_scores['tr_size'] = tr_sz
                
                # vl_scores['tr_set'] = False
                vl_scores['set'] = 'vl'
                vl_scores['fold'] = 'fold'+str(fold)
                vl_scores['tr_size'] = tr_sz

                # te_scores['tr_set'] = False
                te_scores['set'] = 'te'
                te_scores['fold'] = 'fold'+str(fold)
                te_scores['tr_size'] = tr_sz

                # Append scores (dicts)
                tr_scores_all.append(tr_scores)
                vl_scores_all.append(vl_scores)
                te_scores_all.append(te_scores)

                # Dump intermediate scores
                # TODO: test this!
                scores_tmp = pd.concat([scores_to_df([tr_scores]), scores_to_df([vl_scores]), scores_to_df([te_scores])], axis=0)
                scores_tmp.to_csv( trn_outdir / ('scores_tmp.csv'), index=False )
                del trn_outdir, scores_tmp
                
            # Dump intermediate results (this is useful if the run terminates before run ends)
            scores_all_df_tmp = pd.concat([scores_to_df(tr_scores_all), scores_to_df(vl_scores_all), scores_to_df(te_scores_all)], axis=0)
            scores_all_df_tmp.to_csv( self.outdir / ('_lrn_crv_scores_cv' + str(fold) + '.csv'), index=False )

        # Scores to df
        tr_scores_df = scores_to_df( tr_scores_all )
        vl_scores_df = scores_to_df( vl_scores_all )
        te_scores_df = scores_to_df( te_scores_all )
        scores_df = pd.concat([tr_scores_df, vl_scores_df, te_scores_df], axis=0)
        
        # Dump final results
        tr_scores_df.to_csv( self.outdir/'tr_lrn_crv_scores.csv', index=False) 
        vl_scores_df.to_csv( self.outdir/'vl_lrn_crv_scores.csv', index=False) 
        te_scores_df.to_csv( self.outdir/'te_lrn_crv_scores.csv', index=False) 
        scores_df.to_csv( self.outdir/'lrn_crv_scores.csv', index=False) 

        # Runtime df
        runtime_df = pd.DataFrame.from_records(runtime_records, columns=['fold', 'tr_sz', 'time'])
        runtime_df.to_csv( self.outdir/'runtime.csv', index=False) 
        
        # Plot learning curves
        if plot:
            plot_lrn_crv_all_metrics( scores_df, outdir=self.outdir )
            plot_lrn_crv_all_metrics( scores_df, outdir=self.outdir, xtick_scale='log2', ytick_scale='log2' )
            plot_runtime( runtime_df, outdir=self.outdir, xtick_scale='log2', ytick_scale='log2' )

        return scores_df


    def trn_keras_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save Keras model. """
        keras.utils.plot_model(model, to_file=self.outdir/'nn_model.png')

        # Create output dir
        trn_outdir = self.outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
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
        t0 = time()
        history = model.fit(xtr_sub, ytr_sub, **fit_kwargs)
        runtime = (time() - t0)/60
        ml_models.save_krs_history(history, outdir=trn_outdir)
        ml_models.plot_prfrm_metrics(history, title=f'Train size: {tr_sz}', skp_ep=20, add_lr=True, outdir=trn_outdir)

        # Load the best model (https://github.com/keras-team/keras/issues/5916)
        # model = keras.models.load_model(str(trn_outdir/'model_best.h5'), custom_objects={'r2_krs': ml_models.r2_krs})
        model = keras.models.load_model( str(trn_outdir/'model_best.h5') )
        return model, trn_outdir, runtime


    def trn_lgbm_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save LigthGBM model. """
        # Create output dir
        trn_outdir = self.outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
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
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **fit_kwargs)
        runtime = (time() - t0)/60
        joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime
    
    
    def trn_sklearn_model(self, model, xtr_sub, ytr_sub, fold, tr_sz, eval_set=None):
        """ Train and save sklearn model. """
        # Create output dir
        trn_outdir = self.outdir / ('cv'+str(fold) + '_sz'+str(tr_sz))
        # os.makedirs(trn_outdir, exist_ok=False)
        os.makedirs(trn_outdir, exist_ok=True)

        # Get a subset of samples for validation for early stopping
        fit_kwargs = self.fit_kwargs
        # xtr_sub, xvl_sub, ytr_sub, yvl_sub = train_test_split(xtr_sub, ytr_sub, test_size=self.val_split)
        # if xvl_sub_.shape[0] > 0:
        #     fit_kwargs['eval_set'] = (xvl_sub, yvl_sub)
        #     fit_kwargs['early_stopping_rounds'] = 10

        # Train and save model
        t0 = time()
        model.fit(xtr_sub, ytr_sub, **fit_kwargs)
        runtime = (time() - t0)/60
        joblib.dump(model, filename = trn_outdir / ('model.'+self.model_name+'.pkl') )
        return model, trn_outdir, runtime
# --------------------------------------------------------------------------------

    


def define_keras_callbacks(outdir):
    checkpointer = ModelCheckpoint(str(outdir/'model_best.h5'), verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20, verbose=1, mode='auto',
                                  min_delta=0.0001, cooldown=3, min_lr=0.000000001)
    # early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    return [checkpointer, csv_logger, early_stop, reduce_lr]


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

        # tr = aa[aa['tr_set']==True]
        # vl = aa[aa['tr_set']==False]
        tr = aa[aa['set']=='tr']
        # vl = aa[aa['set']=='vl']
        te = aa[aa['set']=='te']

        tr = tr[[c for c in tr.columns if 'fold' in c]]
        # vl = vl[[c for c in vl.columns if 'fold' in c]]
        te = te[[c for c in te.columns if 'fold' in c]]
        # tr = tr.iloc[:, -cv_folds:]
        # vl = vl.iloc[:, -cv_folds:]

        rslt = []
        rslt.append(tr_shards)
        rslt.append(tr.values if tr.values.shape[0]>0 else None)
        # rslt.append(vl.values if vl.values.shape[0]>0 else None)
        rslt.append(te.values if te.values.shape[0]>0 else None)

        if xtick_scale != 'linear' or ytick_scale != 'linear':
            fname = 'lrn_crv_' + metric_name + '_log.png'
        else:
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
        ax.plot(tr_shards, scores_mean, '.-', color=color, label=f'{phase} Score')
        ax.fill_between(tr_shards, scores_mean - scores_std, scores_mean + scores_std, alpha=0.1, color=color)

    # Plot learning curves
    fontsize = 13
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
        
    if tr_scores is not None:
        plot_single_crv(tr_shards, scores=tr_scores, ax=ax, color='b', phase='Train')
    if te_scores is not None:
        # plot_single_crv(tr_shards, scores=te_scores, ax=ax, color='g', phase='Val')
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



# --------------------------------------------------------------------------------------------------
# Power-law utils
# --------------------------------------------------------------------------------------------------
def power_law_func_3prm(x, alpha, beta, gamma):
    """ 3 parameters. docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.power.html """
    return alpha * np.power(x, beta) + gamma
    
    
def fit_power_law_3prm(x, y, p0:list=[30, -0.3, 0.06]):
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
                           label:str='Data', ax=None):
    
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
    # ax.plot(x, y, '.-', color=None, label=label);
    ax.plot(x, y, '.', color=None, label=label);

    # Plot fit
    if plot_fit: ax.plot(x, yfit, '--', color=None, label=f'{label} fit (RMSE: {rmse:.7f})');    
        
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
    if title is None: title='Learning curve (power-law)'
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
    if title is None: title='Learning curve (power-law)'
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
        # scores['mean_squared_error'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['mse'] = sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred)
        scores['rmse'] = sqrt( sklearn.metrics.mean_squared_error(y_true=y_true, y_pred=y_pred) )
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
    # df = df.melt(id_vars=['fold', 'tr_size', 'tr_set'])
    df = df.melt(id_vars=['fold', 'tr_size', 'set'])
    df = df.rename(columns={'variable': 'metric'})
    # df = df.pivot_table(index=['metric', 'tr_size', 'tr_set'], columns=['fold'], values='value')
    df = df.pivot_table(index=['metric', 'tr_size', 'set'], columns=['fold'], values='value')
    df = df.reset_index(drop=False)
    df.columns.name = None
    return df


