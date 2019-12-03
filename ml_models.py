""" This script contains various ML models and some utility functions. """
from __future__ import print_function, division

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import math

import sklearn
import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

try:
    import tensorflow as tf
except:
   print('Could not import tensorflow.')

import keras
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Embedding, Flatten, Lambda, merge
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Sequential, Model, model_from_json, model_from_yaml
from keras.utils import np_utils, multi_gpu_model
# from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

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

    elif model_name == 'nn_reg_mini':
        model = NN_REG_MINI(**init_kwargs)
    elif model_name == 'nn_reg_ap':
        model = NN_REG_AP(**init_kwargs)        
        
    elif model_name == 'nn_reg_attn':
        model = NN_REG_ATTN(**init_kwargs)

    elif model_name == 'nn_reg_res':
        model = NN_REG_RES(**init_kwargs)
        
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
        alpha = 0.6
        linewidth = 1
        fontsize = 12
        ax1.plot(eps, y_tr, color='b', marker='.', linestyle='-', linewidth=linewidth, alpha=alpha,
                 label=capitalize_metric(metric_name))
        ax1.plot(eps, y_vl, color='r', marker='.', linestyle='--', linewidth=linewidth, alpha=alpha,
                 label=capitalize_metric(metric_name_val))
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
                ax2.plot(eps, hh['lr'][skp_ep:], color='g', marker='.', linestyle=':', linewidth=linewidth,
                         alpha=alpha, markersize=5, label='LR')
                ax2.set_ylabel('Learning rate', color='g', fontsize=fontsize)

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



class Attention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Attention, self).build(input_shape)
    
    def call(self, V):
        Q = keras.backend.dot(V, self.kernel)
        Q =  Q * V
        Q = Q / math.sqrt(self.output_dim)
        Q = keras.activations.softmax(Q)
        return Q
    
    def compute_output_shape(self, input_shape):
        return input_shape


    
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


    def build_dense_block(self, layers, inputs, batchnorm=True, name=None):
        """ This function only applicable to keras NNs. """
        prfx = '' if name is None else f'{name}.'
        x = inputs
        
        for i, l_size in enumerate(layers):
            l_name = prfx + f'fc{i+1}.{l_size}'
            x = Dense(l_size, kernel_initializer=self.initializer, name=l_name)(x)
#             if i == 0:
#                 x = Dense(l_size, kernel_initializer=self.initializer, name=l_name)(inputs)
#             else:
#                 x = Dense(l_size, kernel_initializer=self.initializer, name=l_name)(x)
            if batchnorm:
                x = BatchNormalization(name=prfx+f'bn{i+1}')(x)
            x = Activation('relu', name=prfx+f'a{i+1}')(x)
            x = Dropout(self.dr_rate, name=prfx+f'drp{i+1}.{self.dr_rate}')(x)        
        return x      
    

#     def build_dense_res_block(self, layers, inputs, batchnorm=True, residual=False, name=None):
#         """ This function only applicable to keras NNs.
#         residual connection --> batchnorm --> activation
#         """
#         prfx = '' if name is None else f'{name}.'
#         x = inputs
        
#         for i, l_size in enumerate(layers):
#             l_name = prfx + f'res_fc{i+1}.{l_size}'
            
#             x_bypass = x

#             x = Dense(l_size, kernel_initializer=self.initializer, name=l_name)(x)
        
#             if residual:
#                 x = keras.layers.add([x_bypass, x], name=prfx+f'res_conn{i+1}')
            
#             if batchnorm:
#                 x = BatchNormalization(name=prfx+f'bn{i+1}')(x)
                
#             x = Activation('relu', name=prfx+f'a{i+1}')(x)
            
#             x = Dropout(self.dr_rate, name=prfx+f'drp{i+1}.{self.dr_rate}')(x)        
#         return x

    def build_dense_res_block(self, inputs, stage:int, skips=1, batchnorm=True, name=None):
        """ This function only applicable to keras NNs.
        residual connection --> batchnorm --> activation
        """
        prfx = '' if name is None else f'{name}.'
        x = inputs
        x_bypass = x
        
        for i in range(skips):
            l_size = int(x.get_shape()[-1])
            l_name = prfx + f'res{stage}_fc{i+1}.{l_size}'

            x = Dense(l_size, kernel_initializer=self.initializer)(x)
                    
            if batchnorm:
                x = BatchNormalization()(x)
                
            # x = Activation('relu', name=prfx+f'a{i+1}')(x)
            x = Activation('relu')(x)
            # x = Dropout(self.dr_rate)(x)        
            
        x = keras.layers.add([x_bypass, x], name=prfx+f'res_conn{stage}')
        
        if batchnorm:
            x = BatchNormalization()(x)
                
        x = Activation('relu')(x)    
        x = Dropout(self.dr_rate)(x)        
        
        
        return x
    
    def get_optimizer(self):
        if self.opt_name == 'sgd':
            opt = SGD(lr=self.lr, momentum=0.9) # lr=1e-4
        elif self.opt_name == 'adam':
            opt = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) # lr=1e-3
        else:
            opt = SGD(lr=self.lr, momentum=0.9) # for clr # lr=1e-4
        return opt


    
    
# ---------------------------------------------------------
class NN_REG0(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model
        
        
class NN_REG_MINI(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [1000, 1000, 500, 250, 125, 60, 30]
        layers = [l//2 for l in layers]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model 
        
        
class NN_REG_AP(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg0'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        # layers = [1000, 1000, 500, 250, 125, 60, 30]
        layers = [1000, 800, 400, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model         
        

        
class NN_REG1(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg1'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        # layers = [1000, 1000,  500, 250, 125, 60, 30]
        layers = [2000, 2000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model  
        
        
        
class NN_REG_RES(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN with residual connection.
    """
    model_name = 'nn_reg1'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        inputs = Input(shape=(self.input_dim,), name='inputs')
        
        # dense_layers = [1000, 1000, 500, 250, 125, 60]
        dense_layers = [1000, 500, 250]
                
        x = self.build_dense_block(dense_layers, inputs, batchnorm=batchnorm)
        x = self.build_dense_res_block(x, stage=1, skips=1, batchnorm=batchnorm)
        # x = self.build_dense_res_block(x, stage=2, skips=1, batchnorm=batchnorm)
        # x = self.build_dense_res_block(x, stage=3, skips=1, batchnorm=batchnorm, residual=residual)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model      
# ---------------------------------------------------------

        
        
# class NN_REG_ATTN(BaseMLModel):
#     """ Neural network regressor.
#     Fully-connected NN with attention.
#     TODO: implement attention layer!
#     """
#     model_name = 'nn_reg_attn'

#     def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
#         self.input_dim = input_dim
#         self.dr_rate = dr_rate
#         self.opt_name = opt_name
#         self.initializer = initializer
#         self.lr = lr

#         layers = [1000, 1000, 500, 250, 125, 60, 30]
#         inputs = Input(shape=(self.input_dim,), name='inputs')
#         x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

#         outputs = Dense(1, activation='relu', name='outputs')(x)
#         model = Model(inputs=inputs, outputs=outputs)
        
#         opt = self.get_optimizer()
#         model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
#         self.model = model        
        
        
class NN_REG_ATTN(BaseMLModel):
    """ Neural network regressor. 
    Fully-connected NN with attention layer.
    """
    model_name = 'nn_reg_attn'

    # def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', logger=None):
    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr
        
        # layers = [1000, 1000, 500, 250, 125, 60, 30]
        inputs = Input(shape=(self.input_dim,), name='inputs')        
        
        # inputs = Input(shape=(input_dim,))
        #x = Lambda(lambda x: x, output_shape=(1000,))(inputs)
        # attn_lin = Dense(1000, activation='relu', name='attn_lin')(inputs)
        # attn_probs = Dense(1000, activation='softmax', name='attn_probs')(inputs)
        # x = keras.layers.multiply( [attn_lin, attn_probs], name='attn')
        
        # ------------------------------------------
        # New attention layer (Rick, Austin)
        """
        a = Dense(1000)(inputs)
        a = BatchNormalization()(a)
        a = Activation('relu')(a)
        b = Attention(1000)(a)
        x = keras.layers.multiply([b, a])
        """
        # Old attention layer
        a = Dense(1000, activation='relu')(inputs)
        b = Dense(1000, activation='softmax')(inputs)
        x = keras.layers.multiply( [a, b] )        
        
        x = Dense(1000)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(500)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(250)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        
        x = Dense(125)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(60)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)

        x = Dense(30)(x)
        # x = BatchNormalization()(x)
        if batchnorm: x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dr_rate)(x)
        # ------------------------------------------

        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
#         if opt_name == 'sgd':
#             opt = SGD(lr=1e-4, momentum=0.9)
#         elif opt_name == 'adam':
#             opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         else:
#             opt = SGD(lr=1e-4, momentum=0.9) # for clr        
        opt = self.get_optimizer()
        
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model        
                
        
        
# ---------------------------------------------------------
class NN_REG_NEURON_LESS(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg_neuron_less'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [500, 250, 125, 60]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model 
        
        

class NN_REG_NEURON_MORE(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN.
    """
    model_name = 'nn_reg_neuron_more'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [1000, 500, 250, 125]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model            
# ---------------------------------------------------------

        

# ---------------------------------------------------------    
class NN_REG_LAYER_LESS(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN with less layers.
    """
    model_name = 'nn_reg_layer_less'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [1000, 500]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model 
        
        
class NN_REG_LAYER_MORE(BaseMLModel):
    """ Neural network regressor.
    Fully-connected NN more layers.
    """
    model_name = 'nn_reg_layer_more'

    def __init__(self, input_dim, dr_rate=0.2, opt_name='sgd', lr=0.001, initializer='he_uniform', batchnorm=False):
        self.input_dim = input_dim
        self.dr_rate = dr_rate
        self.opt_name = opt_name
        self.initializer = initializer
        self.lr = lr

        layers = [1000, 500, 250, 125]
        inputs = Input(shape=(self.input_dim,), name='inputs')
        x = self.build_dense_block(layers, inputs, batchnorm=batchnorm)

        outputs = Dense(1, activation='relu', name='outputs')(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        opt = self.get_optimizer()
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae']) # r2_krs 
        self.model = model         
# ---------------------------------------------------------

        
        
class LGBM_REGRESSOR(BaseMLModel):
    """ LightGBM regressor. """
    ml_objective = 'regression'
    model_name = 'lgb_reg'

    # def __init__(self, n_estimators=100, eval_metric=['l2', 'l1'], n_jobs=1, random_state=None):
    def __init__(self, **kwargs):
        # TODO: use config file to set default parameters (like in candle)
        
       #  self.model = lgb.LGBMModel(
       #      objective = LGBM_REGRESSOR.ml_objective,
       #      n_estimators = n_estimators,
       #      n_jobs = n_jobs,
       #      random_state = random_state)

        self.model = lgb.LGBMModel( objective = LGBM_REGRESSOR.ml_objective, **kwargs )

    def dump_model(self, outdir='.'):
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
                 bootstrap=True, oob_score=False, verbose=0, 
                 n_jobs=4, random_state=None):               

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
        # model_loaded = joblib.load(filename=os.path.join(run_outdir, 'lgb_reg_model.pkl'))        



