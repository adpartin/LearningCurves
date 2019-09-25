""" This script contains various ML models and some utility functions. """
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
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Embedding, Flatten, Lambda, merge
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

import lightgbm as lgb


def clr_keras_callback(mode=None, base_lr=1e-4, max_lr=1e-3, gamma=0.999994):
    """ Creates keras callback for cyclical learning rate. """
    # keras_contrib = './keras_contrib/callbacks'
    # sys.path.append(keras_contrib)
    from cyclical_learning_rate import CyclicLR

    if mode == 'trng1':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular')
    elif mode == 'trng2':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='triangular2')
    elif mode == 'exp':
        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, mode='exp_range', gamma=gamma) # 0.99994; 0.99999994; 0.999994
    return clr


def r2_krs(y_true, y_pred):
    # from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def get_model(model_name, init_kwargs=None):
    """ Return a model.
    Args:
        init_kwargs : init parameters to the model
        model_name : model name
    """
    if model_name == 'lgb_reg':
        model = LGBM_REGRESSOR(**init_kwargs)
    elif model_name == 'rf_reg':
        model = RF_REGRESSOR(**init_kwargs)
        
    elif model_name == 'nn_reg0':
        model = NN_REG0(**init_kwargs)
    elif model_name == 'nn_reg1':
        model = NN_REG1(**init_kwargs)        
        
    elif model_name == 'nn_reg_layer_less':
        model = NN_REG_L_LESS(**init_kwargs)
    elif model_name == 'nn_reg_layer_more':
        model = NN_REG_L_MORE(**init_kwargs)
        
    elif model_name == 'nn_reg_neuron_less':
        model = NN_REG_N_LESS(**init_kwargs)
    elif model_name == 'nn_reg_neuron_more':
        model = NN_REG_N_MORE(**init_kwargs)
                
    else:
        raise ValueError('model_name is invalid.')
    return model


def save_krs_history(history, outdir='.'):
    fname = 'krs_history.csv'
    hh = pd.DataFrame(history.history)
    hh['epoch'] = np.asarray(history.epoch) + 1    
    hh.to_csv( Path(outdir)/fname, index=False )
    return hh


def capitalize_metric(met):
    return ' '.join(s.capitalize() for s in met.split('_'))


# def plot_prfrm_metrics(history, title=None, skp_ep=0, outdir='.', add_lr=False):
def plot_prfrm_metrics(history=None, logfile_path=None, title=None, name=None, skp_ep=0, outdir='.', add_lr=False):    
    """ Plots training curves from keras history or keras traininig.log file.
    Args:
        history : history variable of keras
        path_to_logs : full path to the training log file of keras
        skp_ep : number of epochs to skip when plotting metrics 
        add_lr : add curve of learning rate progression over epochs
        
    Retruns:
        history : keras history
    """
    if history is not None:
        # Plot from keras history
        hh = history.history
        epochs = np.asarray(history.epoch) + 1
        all_metrics = list(history.history.keys())
        
    elif logfile_path is not None:
        # Plot from keras training.log file
        hh = pd.read_csv(logfile_path, sep=',', header=0)
        epochs = hh['epoch'] + 1
        all_metrics = list(hh.columns)
        
    # Get training performance metrics
    pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

    if len(epochs) <= skp_ep: skp_ep = 0
    eps = epochs[skp_ep:]
        
    # Interate over performance metrics and plot
    for p, m in enumerate(pr_metrics):
        metric_name = m
        metric_name_val = 'val_' + m

        y_tr = hh[metric_name][skp_ep:]
        y_vl = hh[metric_name_val][skp_ep:]
        
        ymin = min(set(y_tr).union(y_vl))
        ymax = max(set(y_tr).union(y_vl))
        lim = (ymax - ymin) * 0.1
        ymin, ymax = ymin - lim, ymax + lim

        # Start figure
        fig, ax1 = plt.subplots()
        
        # Plot metrics
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(capitalize_metric(metric_name))
        ax1.set_xlim([min(eps)-1, max(eps)+1])
        ax1.set_ylim([ymin, ymax])
        ax1.tick_params('y', colors='k')
        
        # ax1.tick_params(axis='both', which='major', labelsize=12)
        # ax1.tick_params(axis='both', which='minor', labelsize=12)        
        
        # Add learning rate
        if logfile_path is None: # learning rate is not logged into log file (so it's only available from history)
            if (add_lr is True) and ('lr' in hh):
                ax2 = ax1.twinx()
                ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=1,
                         alpha=0.6, markersize=5, label='LR')
                ax2.set_ylabel('Learning rate', color='g', fontsize=12)

                ax2.set_yscale('log') # 'linear'
                ax2.tick_params('y', colors='g')
        
        ax1.grid(True)
        # plt.legend([metric_name, metric_name_val], loc='best')
        # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
        legend = ax1.legend(loc='best', prop={'size': 10})
        frame = legend.get_frame()
        frame.set_facecolor('0.95')
        if title is not None: plt.title(title)
        
        # fig.tight_layout()
        fname = (metric_name + '.png') if name is None else (name + '_' + metric_name + '.png')
        figpath = Path(outdir) / fname
        plt.savefig(figpath, bbox_inches='tight')
        plt.close()
        
    return history


# def plot_metrics_from_logs(path_to_logs, title=None, name=None, skp_ep=0, outdir='.'):
#     """ Plots keras training from logs.
#     TODO: is this really necessary?? can this be replaced by plot_prfrm_metrics()?
#     Args:
#         path_to_logs : full path to log file
#         skp_ep: number of epochs to skip when plotting metrics 
#     """
# #     history = pd.read_csv(path_to_logs, sep=',', header=0)
    
# #     all_metrics = list(history.columns)
# #     pr_metrics = ['_'.join(m.split('_')[1:]) for m in all_metrics if 'val' in m]

#     epochs = history['epoch'] + 1
#     if len(epochs) <= skp_ep: skp_ep = 0
#     eps = epochs[skp_ep:]
#     hh = history
    
#     for p, m in enumerate(pr_metrics):
#         metric_name = m
#         metric_name_val = 'val_' + m

#         y_tr = hh[metric_name][skp_ep:]
#         y_vl = hh[metric_name_val][skp_ep:]
        
#         ymin = min(set(y_tr).union(y_vl))
#         ymax = max(set(y_tr).union(y_vl))
#         lim = (ymax - ymin) * 0.1
#         ymin, ymax = ymin - lim, ymax + lim

#         # Start figure
#         fig, ax1 = plt.subplots()
        
#         # Plot metrics
#         ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name))
#         ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=1, alpha=0.6, label=capitalize_metric(metric_name_val))        
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel(capitalize_metric(metric_name))
#         ax1.set_xlim([min(eps)-1, max(eps)+1])
#         ax1.set_ylim([ymin, ymax])
#         ax1.tick_params('y', colors='k')
        
#         ax1.grid(True)
#         # plt.legend([metric_name, metric_name_val], loc='best')
#         # medium.com/@samchaaa/how-to-plot-two-different-scales-on-one-plot-in-matplotlib-with-legend-46554ba5915a
#         legend = ax1.legend(loc='best', prop={'size': 10})
#         frame = legend.get_frame()
#         frame.set_facecolor('0.95')
#         if title is not None: plt.title(title)
        
#         # fig.tight_layout()
#         if name is not None:
#             fname = name + '_' + metric_name + '.png'
#         else:
#             fname = metric_name + '.png'
#         figpath = Path(outdir) / fname
#         plt.savefig(figpath, bbox_inches='tight')
#         plt.close()
        
#     return history



class BaseMLModel():
    """ A parent class with some general methods for children ML classes.
    The children classes are specific ML models such random forest regressor, lightgbm regressor, etc.
    """
    def __adj_r2_score(self, ydata, preds):
        """ Calc adjusted r^2.
        https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
        https://dziganto.github.io/data%20science/linear%20regression/machine%20learning/python/Linear-Regression-101-Metrics/
        https://stats.stackexchange.com/questions/334004/can-r2-be-greater-than-1
        """
        r2_score = sklearn.metrics.r2_score(ydata, preds)
        adj_r2 = 1 - (1 - r2_score) * (self.x_size[0] - 1)/(self.x_size[0] - self.x_size[1] - 1)
        return adj_r2


    def build_dense_block(self, layers, inputs, name=''):
        """ This function only applicable to keras NNs. """
        for i, l_size in enumerate(layers):
            if i == 0:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(inputs)
            else:
                x = Dense(l_size, kernel_initializer=self.initializer, name=f'{name}.fc{i+1}.{l_size}')(x)
            x = BatchNormalization(name=f'{name}.bn{i+1}')(x)
            x = Activation('relu', name=f'{name}.a{i+1}')(x)
            x = Dropout(self.dr_rate, name=f'{name}.drp{i+1}.{self.dr_rate}')(x)        
        return x

    
    
# ---------------------------------------------------------
class NN_REG0(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model
        

        
class NN_REG1(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg1'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        # layers = [1000, 1000,  500, 250, 125, 60, 30]
        layers = [2000, 2000, 1000, 500, 250, 125, 60]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model        
# ---------------------------------------------------------

        
        
class NN_REG_ATTN(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN with attention.
    TODO: implement attention layer!
    """
    model_name = 'nn_reg_attn'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model        
        
        
        
# ---------------------------------------------------------
class NN_REG_NEURON_LESS(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg_neuron_less'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [500, 250, 125, 60]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model 
        
        

class NN_REG_NEURON_MORE(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg_neuron_more'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 500, 250, 125]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model            
# ---------------------------------------------------------

        

# ---------------------------------------------------------    
class NN_REG_LAYER_LESS(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN with less layers.
    """
    model_name = 'nn_reg_layer_less'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 500]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model 
        
        
class NN_REG_LAYER_MORE(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN more layers.
    """
    model_name = 'nn_reg_layer_more'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', initializer='he_uniform', logger=None):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer

        layers = [1000, 500, 250, 125]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        if self.opt_name == 'sgd':
            opt = SGD(lr=1e-4, momentum=0.9)
        elif self.opt_name == 'adam':
            opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        else:
            opt = SGD(lr=1e-4, momentum=0.9) # for clr

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model         
# ---------------------------------------------------------

        
        
class LGBM_REGRESSOR(BaseMLModel):
    """ LightGBM regressor. """
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    def __init__(self, n_estimators=100, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None, logger=None):
        # TODO: use config file to set default parameters (like in candle)
        
        self.model = lgb.LGBMModel(
            objective = LGBM_REGRESSOR.ml_objective,
            n_estimators = n_estimators,
            n_jobs = n_jobs,
            random_state = random_state)


    # def fit(self, X, y, eval_set=None, **fit_params):
    #     #self.eval_set = eval_set
    #     #self.X = X
    #     #self.y = y
    #     #self.x_size = X.shape  # this is used to calc adjusteed r^2
        
    #     t0 = time.time()
    #     self.model.fit(X, y,
    #                    eval_metric=self.eval_metric,
    #                    eval_set=eval_set,
    #                    **fit_params)
    #     self.train_runtime = time.time() - t0

    #     if self.logger is not None:
    #         self.logger.info('Train time: {:.2f} mins'.format(self.train_runtime/60))


    def dump_model(self, outdir='.'):
        # lgb_reg.save_model(os.path.join(run_outdir, 'lgb_'+ml_type+'_model.txt'))
        joblib.dump(self.model, filename=Path(outdir)/('model.' + LGBM_REGRESSOR.model_name + '.pkl'))
        # lgb_reg_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))

        
    def plot_fi(self, max_num_features=20, title='LGBMRegressor', outdir=None):
        lgb.plot_importance(booster=self.model, max_num_features=max_num_features, grid=True, title=title)
        plt.tight_layout()

        filename = LGBM_REGRESSOR.model_name + '_fi.png'
        if outdir is None:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.savefig(Path(outdir)/filename, bbox_inches='tight')


    # # Plot training curves
    # # TODO: note, plot_metric didn't accept 'mae' although it's alias for 'l1' 
    # # TODO: plot_metric requires dict from train(), but train returns 'lightgbm.basic.Booster'??
    # for m in eval_metric:
    #     ax = lgb.plot_metric(booster=lgb_reg, metric=m, grid=True)
    #     plt.savefig(os.path.join(run_outdir, model_name+'_learning_curve_'+m+'.png'))
    

    
class RF_REGRESSOR(BaseMLModel):
    """ Random forest regressor. """
    # Define class attributes (www.toptal.com/python/python-class-attributes-an-overly-thorough-guide)
    model_name = 'rf_reg'

    def __init__(self, n_estimators=100, criterion='mse',
                 max_depth=None, min_samples_split=2,
                 max_features='sqrt',
                 bootstrap=True, oob_score=True, verbose=0, 
                 n_jobs=1, random_state=None,
                 logger=None):               

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features, bootstrap=bootstrap, oob_score=oob_score,
            verbose=verbose, random_state=random_state, n_jobs=n_jobs)


    def plot_fi(self):
        pass # TODO


    def dump_model(self, outdir='.'):
        joblib.dump(self.model, filename=os.path.join(outdir, 'model.' + RF_REGRESSOR.model_name + '.pkl'))
        # model_ = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))        