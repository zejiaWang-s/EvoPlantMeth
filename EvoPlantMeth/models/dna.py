from __future__ import division, print_function
import inspect
from tensorflow.keras import layers as kl
from tensorflow.keras import regularizers as kr
from .utils import Model
from ..utils import get_from_module

class DnaModel(Model):
    def __init__(self, *args, **kwargs):
        super(DnaModel, self).__init__(*args, **kwargs)
        self.scope = 'dna'

    def inputs(self, dna_wlen):
        return [kl.Input(shape=(dna_wlen, 4), name='dna')]

class CnnL1h128(DnaModel):
    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL1h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)
        
        x = kl.Conv1D(128, 11, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)
        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class CnnL1h256(CnnL1h128):
    def __init__(self, *args, **kwargs):
        super(CnnL1h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256

class CnnL2h128(DnaModel):
    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL2h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Conv1D(256, 3, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class CnnL2h256BN(DnaModel):
    def __init__(self, nb_hidden=256, *args, **kwargs):
        super(CnnL2h256BN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, padding='same', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Conv1D(256, 3, padding='same', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class CnnL2h128BN(DnaModel):
    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL2h128BN, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, padding='same', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Conv1D(128, 3, padding='same', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class CnnL2h256(CnnL2h128):
    def __init__(self, *args, **kwargs):
        super(CnnL2h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256

class CnnL3h128(DnaModel):
    def __init__(self, nb_hidden=128, *args, **kwargs):
        super(CnnL3h128, self).__init__(*args, **kwargs)
        self.nb_hidden = nb_hidden

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Conv1D(256, 3, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)

        x = kl.Conv1D(512, 3, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(2)(x)
        x = kl.Flatten()(x)

        x = kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class CnnL3h256(CnnL3h128):
    def __init__(self, *args, **kwargs):
        super(CnnL3h256, self).__init__(*args, **kwargs)
        self.nb_hidden = 256

class CnnRnn01(DnaModel):
    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Conv1D(256, 7, kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu')(x)
        x = kl.MaxPooling1D(4)(x)

        x = kl.Bidirectional(kl.GRU(256, kernel_regularizer=reg))(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class ResNet01(DnaModel):
    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):
        name = '%02d-%02d/' % (stage, block)
        res_name = '%sres_' % name
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        x = kl.Conv1D(nb_filter[0], 1, strides=stride, name=res_name + 'conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        x = kl.Conv1D(nb_filter[1], size, padding='same', name=res_name + 'conv2', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        x = kl.Conv1D(nb_filter[2], 1, name=res_name + 'conv3', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        if nb_filter[-1] != inputs.shape[-1] or stride > 1:
            identity = kl.Conv1D(nb_filter[2], 1, strides=stride, name='%sid_conv1' % name, kernel_initializer=self.init, kernel_regularizer=reg)(inputs)
        else:
            identity = inputs

        return kl.Add(name=name + 'merge')([identity, x])

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, name='conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class ResNet02(ResNet01):
    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, name='conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], stage=1, block=3)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=3)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=3)
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class ResConv01(ResNet01):
    def _res_unit(self, inputs, nb_filter, size=3, stride=1, stage=1, block=1):
        name = '%02d-%02d/' % (stage, block)
        res_name = '%sres_' % name
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        x = kl.Conv1D(nb_filter, size, padding='same', strides=stride, name=res_name + 'conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        x = kl.Conv1D(nb_filter, size, padding='same', name=res_name + 'conv2', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        if nb_filter != inputs.shape[-1] or stride > 1:
            identity = kl.Conv1D(nb_filter, size, padding='same', strides=stride, name='%sid_conv1' % name, kernel_initializer=self.init, kernel_regularizer=reg)(inputs)
        else:
            identity = inputs

        return kl.Add(name=name + 'merge')([identity, x])

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, name='conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.BatchNormalization(name='bn1')(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        x = self._res_unit(x, 128, stage=1, block=1, stride=2)
        x = self._res_unit(x, 128, stage=1, block=2)
        x = self._res_unit(x, 256, stage=2, block=1, stride=2)
        x = self._res_unit(x, 256, stage=3, block=1, stride=2)
        x = self._res_unit(x, 512, stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

class ResAtrous01(DnaModel):
    def _res_unit(self, inputs, nb_filter, size=3, stride=1, atrous=1, stage=1, block=1):
        name = '%02d-%02d/' % (stage, block)
        res_name = '%sres_' % name
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.BatchNormalization(name=res_name + 'bn1')(inputs)
        x = kl.Activation('relu', name=res_name + 'act1')(x)
        x = kl.Conv1D(nb_filter[0], 1, strides=stride, name=res_name + 'conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn2')(x)
        x = kl.Activation('relu', name=res_name + 'act2')(x)
        x = kl.Conv1D(nb_filter[1], size, dilation_rate=atrous, padding='same', name=res_name + 'conv2', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        x = kl.BatchNormalization(name=res_name + 'bn3')(x)
        x = kl.Activation('relu', name=res_name + 'act3')(x)
        x = kl.Conv1D(nb_filter[2], 1, name=res_name + 'conv3', kernel_initializer=self.init, kernel_regularizer=reg)(x)

        if nb_filter[-1] != inputs.shape[-1] or stride > 1:
            identity = kl.Conv1D(nb_filter[2], 1, strides=stride, name='%sid_conv1' % name, kernel_initializer=self.init, kernel_regularizer=reg)(inputs)
        else:
            identity = inputs

        return kl.Add(name=name + 'merge')([identity, x])

    def __call__(self, inputs):
        x = inputs[0]
        reg = kr.L1L2(self.l1_decay, self.l2_decay)

        x = kl.Conv1D(128, 11, name='conv1', kernel_initializer=self.init, kernel_regularizer=reg)(x)
        x = kl.Activation('relu', name='act1')(x)
        x = kl.MaxPooling1D(2, name='pool1')(x)

        x = self._res_unit(x, [32, 32, 128], stage=1, block=1, stride=2)
        x = self._res_unit(x, [32, 32, 128], atrous=2, stage=1, block=2)
        x = self._res_unit(x, [32, 32, 128], atrous=4, stage=1, block=3)
        x = self._res_unit(x, [64, 64, 256], stage=2, block=1, stride=2)
        x = self._res_unit(x, [64, 64, 256], atrous=2, stage=2, block=2)
        x = self._res_unit(x, [64, 64, 256], atrous=4, stage=2, block=3)
        x = self._res_unit(x, [128, 128, 512], stage=3, block=1, stride=2)
        x = self._res_unit(x, [128, 128, 512], atrous=2, stage=3, block=2)
        x = self._res_unit(x, [128, 128, 512], atrous=4, stage=3, block=3)
        x = self._res_unit(x, [256, 256, 1024], stage=4, block=1, stride=2)

        x = kl.GlobalAveragePooling1D()(x)
        x = kl.Dropout(self.dropout)(x)
        return self._build(inputs, x)

def list_models():
    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and name.lower().find('model') == -1:
            models[name] = value
    return models

def get(name):
    return get_from_module(name, globals())