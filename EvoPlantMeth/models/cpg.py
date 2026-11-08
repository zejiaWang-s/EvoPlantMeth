from __future__ import division, print_function
import inspect
from tensorflow.keras import backend as K
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr
from tensorflow.keras import models as km
from tensorflow.keras.layers import concatenate
from .utils import Model
from ..utils import get_from_module

class CpgModel(Model):
    def __init__(self, *args, **kwargs):
        super(CpgModel, self).__init__(*args, **kwargs)
        self.scope = 'cpg'

    def inputs(self, cpg_wlen, replicate_names):
        shape = (len(replicate_names), cpg_wlen)
        return [kl.Input(shape=shape, name='cpg_state'), kl.Input(shape=shape, name='cpg_dist')]

    def _merge_inputs(self, inputs):
        if not isinstance(inputs, list): inputs = [inputs]
        reshaped = [kl.Lambda(lambda t: K.expand_dims(t, axis=-1))(inp) for inp in inputs]
        return concatenate(reshaped, axis=-1)

class FcAvg(CpgModel):
    def _replicate_model(self, input_tensor):
        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        x = kl.Dense(512, kernel_initializer=self.init, kernel_regularizer=reg)(input_tensor)
        x = kl.Activation('relu')(x)
        return km.Model(input_tensor, x)

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)
        replicate_input = kl.Input(shape=K.int_shape(x)[2:])
        replicate_model = self._replicate_model(replicate_input)
        
        x = kl.TimeDistributed(replicate_model)(x)
        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class RnnL1(CpgModel):
    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(RnnL1, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)
        input_shape = K.int_shape(x)
        num_replicates, cpg_wlen, feature_dim = input_shape[1], input_shape[2], input_shape[3]
        
        x_reshaped = kl.Lambda(lambda t: K.reshape(t, (-1, cpg_wlen, feature_dim)))(x)
        
        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        processed = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg))(x_reshaped)
        
        x = kl.Lambda(lambda t: K.reshape(t, (-1, num_replicates, 256)))(processed)
        
        x = kl.Bidirectional(kl.GRU(256, kernel_regularizer=reg))(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class RnnL1BN(CpgModel):
    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(RnnL1BN, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)
        input_shape = K.int_shape(x)
        num_replicates, cpg_wlen, feature_dim = input_shape[1], input_shape[2], input_shape[3]
        
        x_reshaped = kl.Lambda(lambda t: K.reshape(t, (-1, cpg_wlen, feature_dim)))(x)
        
        processed = kl.TimeDistributed(kl.Dense(128))(x_reshaped)
        processed = kl.BatchNormalization()(processed)
        processed = kl.Activation(self.act_replicate)(processed)

        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        processed = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg))(processed)
        processed = kl.BatchNormalization()(processed)

        x = kl.Lambda(lambda t: K.reshape(t, (-1, num_replicates, 256)))(processed)
        x = kl.Bidirectional(kl.GRU(256, kernel_regularizer=reg))(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class RnnL1BN_simple(CpgModel):
    def __init__(self, act_replicate='relu', *args, **kwargs):
        super(RnnL1BN_simple, self).__init__(*args, **kwargs)
        self.act_replicate = act_replicate

    def __call__(self, inputs):
        x = self._merge_inputs(inputs)
        input_shape = K.int_shape(x)
        num_replicates, cpg_wlen, feature_dim = input_shape[1], input_shape[2], input_shape[3]
        
        x_reshaped = kl.Lambda(lambda t: K.reshape(t, (-1, cpg_wlen, feature_dim)))(x)

        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        processed = kl.TimeDistributed(kl.Dense(64, kernel_initializer=self.init, kernel_regularizer=reg))(x_reshaped)
        processed = kl.BatchNormalization()(processed)
        processed = kl.Activation(self.act_replicate)(processed)
        
        processed = kl.Bidirectional(kl.GRU(64, kernel_regularizer=reg))(processed)
        x = kl.Lambda(lambda t: K.reshape(t, (-1, num_replicates, 128)))(processed)
        x = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg))(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class RnnL2(RnnL1):
    def __call__(self, inputs):
        x = self._merge_inputs(inputs)
        input_shape = K.int_shape(x)
        num_replicates, cpg_wlen, feature_dim = input_shape[1], input_shape[2], input_shape[3]
        
        x_reshaped = kl.Lambda(lambda t: K.reshape(t, (-1, cpg_wlen, feature_dim)))(x)
        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        
        processed = kl.TimeDistributed(kl.Dense(128, activation=self.act_replicate, kernel_regularizer=reg))(x_reshaped)
        processed = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg, return_sequences=True))(processed)
        processed = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg))(processed)

        x = kl.Lambda(lambda t: K.reshape(t, (-1, num_replicates, 256)))(processed)
        x = kl.Bidirectional(kl.GRU(128, kernel_regularizer=reg, return_sequences=True))(x)
        x = kl.Bidirectional(kl.GRU(256, kernel_regularizer=reg))(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

def list_models():
    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and issubclass(value, CpgModel) and value is not CpgModel:
            models[name] = value
    return models

def get(name):
    return get_from_module(name, globals())