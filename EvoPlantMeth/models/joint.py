from __future__ import division, print_function
import inspect
from tensorflow.keras import layers as kl
from tensorflow.keras import models as km
from tensorflow.keras import regularizers as kr
from tensorflow.keras.layers import concatenate
from .utils import Model
from ..utils import get_from_module

class JointModel(Model):
    def __init__(self, *args, **kwargs):
        super(JointModel, self).__init__(*args, **kwargs)
        self.mode = 'concat'
        self.scope = 'joint'

    def _get_inputs_outputs(self, models):
        inputs, outputs = [], []
        for model in models:
            inputs.extend(model.inputs)
            outputs.extend(model.outputs)
        return (inputs, outputs)

    def _build(self, models, layers=[]):
        for layer in layers:
            layer._name = '%s_%s' % (self.scope, layer.name)

        inputs, outputs = self._get_inputs_outputs(models)
        x = concatenate(outputs) if len(outputs) > 1 else outputs[0]
            
        for layer in layers: x = layer(x)
        return km.Model(inputs, x, name=self.name)

class JointL0(JointModel):
    def __call__(self, models):
        return self._build(models)

class JointL1h512(JointModel):
    def __init__(self, nb_layer=1, nb_hidden=512, *args, **kwargs):
        super(JointL1h512, self).__init__(*args, **kwargs)
        self.nb_layer = nb_layer
        self.nb_hidden = nb_hidden

    def __call__(self, models):
        layers = []
        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        for _ in range(self.nb_layer):
            layers.append(kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg))
            layers.append(kl.BatchNormalization())
            layers.append(kl.Activation('relu'))
            layers.append(kl.Dropout(self.dropout))
        return self._build(models, layers)

class JointL2h512(JointL1h512):
    def __init__(self, *args, **kwargs):
        super(JointL2h512, self).__init__(*args, **kwargs)
        self.nb_layer = 2

class JointL2h256(JointL1h512):
    def __init__(self, *args, **kwargs):
        super(JointL2h256, self).__init__(nb_hidden=256, *args, **kwargs)
        self.nb_layer = 2

class JointL2h512Attention(JointModel):
    def __init__(self, *args, **kwargs):
        super(JointL2h512Attention, self).__init__(*args, **kwargs)
        self.nb_layer = 2
        self.nb_hidden = 512

    def __call__(self, models):
        inputs, outputs = self._get_inputs_outputs(models)
        x = concatenate(outputs) if len(outputs) > 1 else outputs[0]

        x_reshaped = kl.Reshape((1, -1))(x)
        attention_output = kl.MultiHeadAttention(num_heads=8, key_dim=64, name='hybrid_attention')(
            query=x_reshaped, value=x_reshaped, key=x_reshaped
        )
        
        attention_output_flat = kl.Flatten()(attention_output)
        x_with_attention = kl.Add()([x, attention_output_flat])
        x = kl.LayerNormalization()(x_with_attention)

        layers = []
        reg = kr.L1L2(l1=self.l1_decay, l2=self.l2_decay)
        for _ in range(self.nb_layer):
            layers.append(kl.Dense(self.nb_hidden, kernel_initializer=self.init, kernel_regularizer=reg))
            layers.append(kl.BatchNormalization())
            layers.append(kl.Activation('relu'))
            layers.append(kl.Dropout(self.dropout))

        final_layers_output = x
        for layer in layers:
            layer._name = '%s_%s' % (self.scope, layer.name)
            final_layers_output = layer(final_layers_output)

        return km.Model(inputs, final_layers_output, name=self.name)

class JointL3h512(JointL1h512):
    def __init__(self, *args, **kwargs):
        super(JointL3h512, self).__init__(*args, **kwargs)
        self.nb_layer = 3

def list_models():
    models = dict()
    for name, value in globals().items():
        if inspect.isclass(value) and issubclass(value, JointModel) and value is not JointModel:
            models[name] = value
    return models

def get(name):
    return get_from_module(name, globals())