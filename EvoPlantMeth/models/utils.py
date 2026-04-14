from __future__ import division, print_function

from os import path as pt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import models as km
from tensorflow.keras import layers as kl
from tensorflow.keras.utils import to_categorical

from .. import data as dat
from .. import evaluation as ev
from ..data import hdf, OUTPUT_SEP
from ..data.dna import int_to_onehot
from ..utils import to_list

class ScaledSigmoid(kl.Layer):
    def __init__(self, scaling=1.0, **kwargs):
        self.supports_masking = True
        self.scaling = scaling
        super(ScaledSigmoid, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return tf.sigmoid(x) * self.scaling

    def get_config(self):
        config = {'scaling': self.scaling}
        base_config = super(ScaledSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

CUSTOM_OBJECTS = {'ScaledSigmoid': ScaledSigmoid}

def get_first_conv_layer(layers, get_act=False):
    conv_layer, act_layer = None, None
    for layer in layers:
        if isinstance(layer, kl.Conv1D) and layer.input_shape[-1] == 4:
            conv_layer = layer
            if not get_act: break
        elif conv_layer and isinstance(layer, kl.Activation):
            act_layer = layer
            break
    if not conv_layer:
        raise ValueError('Convolutional layer not found')
    if get_act:
        if not act_layer: raise ValueError('Activation layer not found')
        return (conv_layer, act_layer)
    return conv_layer

def get_sample_weights(y, class_weights=None):
    y = y[:]
    sample_weights = np.ones(y.shape, dtype=K.floatx())
    sample_weights[y == dat.CPG_NAN] = K.epsilon()
    if class_weights is not None:
        for cla, weight in class_weights.items():
            sample_weights[y == cla] = weight
    return sample_weights

def save_model(model, model_file, weights_file=None):
    if pt.splitext(model_file)[1] == '.h5':
        model.save(model_file)
    else:
        with open(model_file, 'w') as f:
            f.write(model.to_json())
    if weights_file is not None:
        model.save_weights(weights_file, overwrite=True)

def search_model_files(dirname):
    json_file = pt.join(dirname, 'model.json')
    if pt.isfile(json_file):
        for name in ['model_weights.h5', 'model_weights_val.h5', 'model_weights_train.h5']:
            filename = pt.join(dirname, name)
            if pt.isfile(filename): return [json_file, filename]
    elif pt.isfile(pt.join(dirname, 'model.h5')):
        return pt.join(dirname, 'model.h5')
    return None

def load_model(model_files, custom_objects=CUSTOM_OBJECTS, log=None):
    if not isinstance(model_files, list): model_files = [model_files]
    if pt.isdir(model_files[0]):
        model_files = search_model_files(model_files[0])
        if model_files is None: raise ValueError('No model found in "%s"!' % model_files[0])
        if log: log('Using model files %s' % ' '.join(model_files))
        
    if pt.splitext(model_files[0])[1] == '.h5':
        model = km.load_model(model_files[0], custom_objects=custom_objects)
    else:
        with open(model_files[0], 'r') as f: model_json = f.read()
        model = km.model_from_json(model_json, custom_objects=custom_objects)
    if len(model_files) > 1:
        model.load_weights(model_files[1])
    return model

def get_objectives(output_names, is_plant=False):
    objectives = dict()
    for output_name in output_names:
        _output_name = output_name.split(OUTPUT_SEP)
        if _output_name[0] in ['bulk'] or _output_name[-1] in ['mean', 'var']:
            objective = 'mean_squared_error'
        elif _output_name[-1] in ['cat_var']:
            objective = 'categorical_crossentropy'
        else:
            objective = 'mean_squared_error' if is_plant and _output_name[0] == 'cpg' else 'binary_crossentropy'
        objectives[output_name] = objective
    return objectives

def add_output_layers(stem, output_names, init='glorot_uniform', is_plant=False, output_confidence=False):
    outputs = []
    if not output_confidence:
        for output_name in output_names:
            _output_name = output_name.split(OUTPUT_SEP)
            if _output_name[-1] in ['entropy']:
                x = kl.Dense(1, kernel_initializer=init, activation='relu')(stem)
            elif _output_name[-1] in ['var']:
                x = kl.Dense(1, kernel_initializer=init)(stem)
                x = ScaledSigmoid(0.251, name=output_name)(x)
            elif _output_name[-1] in ['cat_var']:
                x = kl.Dense(3, kernel_initializer=init, activation='softmax', name=output_name)(stem)
            else:
                x = kl.Dense(1, kernel_initializer=init, activation='sigmoid', name=output_name)(stem)
            outputs.append(x)
    else:
        for output_name in output_names:
            if output_name.split(OUTPUT_SEP)[-1] in ['cat_var']:
                raise NotImplementedError('Confidence output is not supported for "cat_var".')
            mean_layer = kl.Dense(1, kernel_initializer=init, activation='sigmoid', name=f'{output_name}_mean')(stem)
            logvar_layer = kl.Dense(1, kernel_initializer=init, activation='linear', name=f'{output_name}_logvar')(stem)
            outputs.append(kl.Concatenate(name=output_name)([mean_layer, logvar_layer]))
    return outputs

def predict_generator(model, generator, nb_sample=None):
    data = None
    nb_seen = 0
    for data_batch in generator:
        if not isinstance(data_batch, list): data_batch = list(data_batch)
        if nb_sample:
            nb_left = nb_sample - nb_seen
            for data_item in data_batch:
                for key, value in data_item.items():
                    data_item[key] = data_item[key][:nb_left]

        preds = model.predict(data_batch[0])
        if not isinstance(preds, list): preds = [preds]
        preds = {name: pred for name, pred in zip(model.output_names, preds)}

        if not data: data = [dict() for _ in range(len(data_batch))]
        dat.add_to_dict(preds, data[0])
        for i in range(1, len(data_batch)):
            dat.add_to_dict(data_batch[i], data[i])

        nb_seen += len(list(preds.values())[0])
        if nb_sample and nb_seen >= nb_sample: break

    for i in range(len(data)): data[i] = dat.stack_dict(data[i])
    return data

def evaluate_generator(model, generator, return_data=False, *args, **kwargs):
    data = predict_generator(model, generator, *args, **kwargs)
    perf = []
    for output in model.output_names:
        tmp = ev.evaluate(data[1][output], data[0][output])
        perf.append(pd.DataFrame(tmp, index=[output]))
    perf = pd.concat(perf)
    return (perf, data) if return_data else perf

def read_from(reader, nb_sample=None):
    data = None
    nb_seen = 0
    for data_batch in reader:
        if not isinstance(data_batch, list): data_batch = list(data_batch)
        if not data: data = [dict() for _ in range(len(data_batch))]
        for i in range(len(data_batch)):
            dat.add_to_dict(data_batch[i], data[i])
            
        nb_seen += len(list(data_batch[0].values())[0])
        if nb_sample and nb_seen >= nb_sample: break

    for i in range(len(data)):
        data[i] = dat.stack_dict(data[i])
        if nb_sample:
            for key, value in data[i].items():
                data[i][key] = value[:nb_sample]
    return data

def copy_weights(src_model, dst_model, must_exist=True):
    copied = []
    for dst_layer in dst_model.layers:
        src_layer = next((l for l in src_model.layers if l.name == dst_layer.name), None)
        if not src_layer:
            if must_exist: raise ValueError('Layer "%s" not found!' % dst_layer.name)
            continue
        dst_layer.set_weights(src_layer.get_weights())
        copied.append(dst_layer.name)
    return copied

def is_input_layer(layer):
    return isinstance(layer, tf.keras.layers.InputLayer)

def is_output_layer(layer, model):
    return layer.name in model.output_names

class Model(object):
    def __init__(self, dropout=0.0, l1_decay=0.0, l2_decay=0.0, init='glorot_uniform'):
        self.dropout = dropout
        self.l1_decay = l1_decay
        self.l2_decay = l2_decay
        self.init = init
        self.name = self.__class__.__name__
        self.scope = None

    def inputs(self, *args, **kwargs):
        pass

    def _build(self, input_tensor, output_tensor):
        model = km.Model(input_tensor, output_tensor, name=self.name)
        if self.scope:
            for layer in model.layers:
                if not is_input_layer(layer):
                    layer._name = '%s_%s' % (self.scope, layer.name)
        return model

    def __call__(self, inputs=None):
        pass

def encode_replicate_names(replicate_names):
    return '--'.join(replicate_names)

def decode_replicate_names(replicate_names):
    return replicate_names.split('--')

class DataReader(object):
    def __init__(self, output_names=None, use_dna=True, dna_wlen=None,
                 replicate_names=None, cpg_wlen=None, cpg_max_dist=25000, encode_replicates=False):
        self.output_names = to_list(output_names)
        self.use_dna = use_dna
        self.dna_wlen = dna_wlen
        self.replicate_names = to_list(replicate_names)
        self.cpg_wlen = cpg_wlen
        self.cpg_max_dist = cpg_max_dist
        self.encode_replicates = encode_replicates

    def _prepro_dna(self, dna):
        if self.dna_wlen:
            center, delta = dna.shape[1] // 2, self.dna_wlen // 2
            dna = dna[:, (center - delta):(center + delta + 1)]
        return int_to_onehot(dna)

    def _prepro_cpg(self, states, dists):
        prepro_states, prepro_dists = [], []
        for state, dist in zip(states, dists):
            nan = state == dat.CPG_NAN
            if np.any(nan):
                state[nan] = 0.5
                dist[nan] = self.cpg_max_dist
            dist = np.minimum(dist, self.cpg_max_dist) / self.cpg_max_dist
            prepro_states.append(np.expand_dims(state, 1))
            prepro_dists.append(np.expand_dims(dist, 1))
            
        prepro_states = np.concatenate(prepro_states, axis=1)
        prepro_dists = np.concatenate(prepro_dists, axis=1)
        
        if self.cpg_wlen:
            center, delta = prepro_states.shape[2] // 2, self.cpg_wlen // 2
            tmp = slice(center - delta, center + delta)
            prepro_states, prepro_dists = prepro_states[:, :, tmp], prepro_dists[:, :, tmp]
        return (prepro_states, prepro_dists)

    @dat.threadsafe_generator
    def __call__(self, data_files, class_weights=None, *args, **kwargs):
        names = []
        if self.use_dna: names.append('inputs/dna')
        if self.replicate_names:
            for name in self.replicate_names:
                names.extend(['inputs/cpg/%s/state' % name, 'inputs/cpg/%s/dist' % name])
        if self.output_names:
            names.extend(['outputs/%s' % name for name in self.output_names])

        for data_raw in hdf.reader(data_files, names, *args, **kwargs):
            inputs = dict()
            if self.use_dna:
                inputs['dna'] = self._prepro_dna(data_raw['inputs/dna'])

            if self.replicate_names:
                states, dists = [], []
                for name in self.replicate_names:
                    tmp = 'inputs/cpg/%s/' % name
                    states.append(data_raw[tmp + 'state'])
                    dists.append(data_raw[tmp + 'dist'])
                states, dists = self._prepro_cpg(states, dists)
                
                tmp = '_' + encode_replicate_names(self.replicate_names) if self.encode_replicates else ''
                inputs['cpg_state%s' % tmp] = states
                inputs['cpg_dist%s' % tmp] = dists

            if not self.output_names:
                yield inputs
            else:
                outputs, weights = dict(), dict()
                for name in self.output_names:
                    outputs[name] = data_raw['outputs/%s' % name]
                    cweights = class_weights[name] if class_weights else None
                    weights[name] = get_sample_weights(outputs[name], cweights)
                    if name.endswith('cat_var'):
                        output = outputs[name]
                        outputs[name] = to_categorical(output, 3)
                        outputs[name][output == dat.CPG_NAN] = 0
                yield (inputs, outputs, weights)

def data_reader_from_model(model, outputs=True, replicate_names=None):
    use_dna, dna_wlen, cpg_wlen, encode_replicates = False, None, None, False
    for input_name, input_shape in zip(model.input_names, to_list(model.input_shape)):
        if input_name == 'dna':
            use_dna, dna_wlen = True, input_shape[1]
        elif input_name.startswith('cpg_state_'):
            replicate_names = decode_replicate_names(input_name.replace('cpg_state_', '', 1))
            cpg_wlen, encode_replicates = input_shape[2], True
        elif input_name == 'cpg_state':
            if not replicate_names: raise ValueError('Replicate names required!')
            cpg_wlen = input_shape[2]

    output_names = model.output_names if outputs else None
    return DataReader(output_names=output_names, use_dna=use_dna, dna_wlen=dna_wlen,
                      cpg_wlen=cpg_wlen, replicate_names=replicate_names, encode_replicates=encode_replicates)