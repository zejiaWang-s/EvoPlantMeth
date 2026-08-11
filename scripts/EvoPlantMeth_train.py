#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import random
import re
import sys
import multiprocessing
from functools import partial

import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six
from six.moves import range

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks as kcbk
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from EvoPlantMeth import callbacks as cbk
from EvoPlantMeth import data as dat
from EvoPlantMeth import metrics as met
from EvoPlantMeth import models as mod
from EvoPlantMeth.models.utils import is_input_layer, is_output_layer
from EvoPlantMeth.data import hdf, OUTPUT_SEP
from EvoPlantMeth.utils import format_table, make_dir, EPS

LOG_PRECISION = 4
CLA_METRICS = [met.acc]
REG_METRICS = [met.mse, met.mae, met.pcc]

def _check_file_worker(filepath, required_paths):
    try:
        with h5.File(filepath, 'r') as hf:
            for path in required_paths:
                if path not in hf:
                    return filepath, f"Missing path: {path}"
        return filepath, None
    except Exception as e:
        return filepath, f"Error opening file: {str(e)}"

def remove_outputs(model):
    while is_output_layer(model.layers[-1], model):
        model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    if hasattr(model, '_output_names'):
         model._output_names = None

def rename_layers(model, scope=None):
    if not scope:
        scope = model.scope
    for layer in model.layers:
        if is_input_layer(layer) or layer.name.startswith(scope):
            continue
        layer._name = '%s_%s' % (scope, layer.name)

def get_output_stats(output):
    stats = OrderedDict()
    output = np.ma.masked_values(output, dat.CPG_NAN)
    stats['nb_tot'] = len(output)
    stats['nb_obs'] = np.sum(output != dat.CPG_NAN)
    stats['frac_obs'] = stats['nb_obs'] / stats['nb_tot']
    stats['mean'] = float(np.mean(output))
    stats['var'] = float(np.var(output))
    return stats

def get_output_weights(output_names, weight_patterns):
    regex_weights = dict()
    for weight_pattern in weight_patterns:
        tmp = [tmp.strip() for tmp in weight_pattern.split('=')]
        if len(tmp) != 2:
            raise ValueError('Invalid weight pattern "%s"!' % (weight_pattern))
        regex_weights[tmp[0]] = float(tmp[1])

    output_weights = dict()
    for output_name in output_names:
        for regex, weight in six.iteritems(regex_weights):
            if re.match(regex, output_name):
                output_weights[output_name] = weight
        if output_name not in output_weights:
            output_weights[output_name] = 1.0
    return output_weights

def get_class_weights(labels, nb_class=None):
    freq = np.bincount(labels) / len(labels)
    if nb_class is None:
        nb_class = len(freq)
    if len(freq) < nb_class:
        tmp = np.zeros(nb_class, dtype=freq.dtype)
        tmp[:len(freq)] = freq
        freq = tmp
    weights = 1 / (freq + EPS)
    weights /= weights.sum()
    return weights

def get_output_class_weights(output_name, output):
    output = output[output != dat.CPG_NAN]
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        return None
    elif _output_name[-1] == 'cat_var':
        weights = get_class_weights(output, 3)
    elif _output_name[-1] in ['cat2_var', 'diff', 'mode']:
        weights = get_class_weights(output, 2)
    else:
        return None
    weights = OrderedDict(zip(range(len(weights)), weights))
    return weights

def perf_logs_str(logs):
    return logs.to_csv(None, sep='\t', float_format='%.4f', index=False)

def get_metrics(output_name, is_plant=False):
    _output_name = output_name.split(OUTPUT_SEP)
    if _output_name[0] == 'cpg':
        metrics = REG_METRICS if is_plant else CLA_METRICS
    elif _output_name[0] == 'bulk':
        metrics = REG_METRICS + CLA_METRICS
    elif _output_name[-1] in ['diff', 'mode', 'cat2_var']:
        metrics = CLA_METRICS
    elif _output_name[-1] == 'mean':
        metrics = REG_METRICS + CLA_METRICS
    elif _output_name[-1] == 'var':
        metrics = REG_METRICS
    elif _output_name[-1] == 'cat_var':
        metrics = [met.cat_acc]
    else:
        raise ValueError('Invalid output name "%s"!' % output_name)
    return metrics


class App(object):

    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(
            prog=name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Trains EvoPlantMeth model.')

        g = p.add_argument_group('IO parameters')
        g.add_argument('--train_file_list', required=True)
        g.add_argument('--val_file_list')
        g.add_argument('-o', '--out_dir', default='./train')

        g = p.add_argument_group('Model architecture')
        models = sorted(list(mod.dna.list_models().keys()))
        g.add_argument('--dna_model', nargs='+')
        g.add_argument('--dna_wlen', type=int)
        models = sorted(list(mod.cpg.list_models().keys()))
        g.add_argument('--cpg_model', nargs='+')
        g.add_argument('--cpg_wlen', type=int)
        models = sorted(list(mod.joint.list_models().keys()))
        g.add_argument('--joint_model', default='JointL2h512')
        g.add_argument('--model_files', nargs='+')

        g = p.add_argument_group('Trainable components')
        g.add_argument('--fine_tune', action='store_true')
        g.add_argument('--train_models', choices=['dna', 'cpg', 'joint'], nargs='+')
        g.add_argument('--trainable', nargs='+')
        g.add_argument('--not_trainable', nargs='+')
        g.add_argument('--freeze_filter', action='store_true')
        g.add_argument('--filter_weights', nargs='+')

        g = p.add_argument_group('Training parameters')
        g.add_argument('--gpus', type=int, default=1)
        g.add_argument('--learning_rate', type=float, default=0.001)
        g.add_argument('--nb_epoch', type=int, default=50)
        g.add_argument('--nb_train_sample', type=int)
        g.add_argument('--nb_val_sample', type=int)
        g.add_argument('--batch_size', type=int, default=256)
        g.add_argument('--early_stopping', type=int, default=10)
        g.add_argument('--dropout', type=float, default=0.2)
        g.add_argument('--l1_decay', type=float, default=0.0)
        g.add_argument('--l2_decay', type=float, default=0.0)
        g.add_argument('--no_tensorboard', action='store_true')

        g = p.add_argument_group('Outputs and weights')
        g.add_argument('--output_names', nargs='+', default=['cpg/.*'])
        g.add_argument('--nb_output', type=int)
        g.add_argument('--no_class_weights', action='store_true')
        g.add_argument('--output_weights', nargs='+')
        g.add_argument('--replicate_names', nargs='+')
        g.add_argument('--nb_replicate', type=int)
        g.add_argument('--output_confidence', action='store_true')

        g = p.add_argument_group('Advanced parameters')
        g.add_argument('--max_time', type=float)
        g.add_argument('--stop_file')
        g.add_argument('--seed', type=int, default=0)
        g.add_argument('--no_log_outputs', action='store_true')
        g.add_argument('--verbose', action='store_true')
        g.add_argument('--log_file')
        g.add_argument('--data_q_size', type=int, default=100)
        g.add_argument('--data_nb_worker', type=int, default=8)
        g.add_argument('--is_plant', action='store_true')
        
        return p

    def verify_h5_files(self, file_list, expected_paths, log, bad_file_log_name):
        log.info(f"Verifying {len(file_list)} files...")
        if not expected_paths: return file_list

        good_files, bad_files = [], []
        num_workers = max(1, multiprocessing.cpu_count() - 2)
        check_func = partial(_check_file_worker, required_paths=expected_paths)
        chunksize = max(1, len(file_list) // (num_workers * 4))
        
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.imap_unordered(check_func, file_list, chunksize=chunksize)
                for filepath, error_msg in results:
                    if error_msg: bad_files.append((filepath, error_msg))
                    else: good_files.append(filepath)
        except Exception as e:
            log.error(f"Multiprocessing validation failed: {e}")
            return file_list
            
        if bad_files:
            bad_log_path = os.path.join(self.opts.out_dir, bad_file_log_name)
            with open(bad_log_path, 'w') as f:
                for fp, err in bad_files: f.write(f"{fp}\t{err}\n")
                    
        if not good_files:
            raise RuntimeError("No valid HDF5 files found.")
        return good_files

    def get_callbacks(self, template_model):
        opts = self.opts
        callbacks = []
        monitor_metric = 'val_pcc' if opts.is_plant or opts.output_confidence else 'val_loss'
        monitor_mode = 'max' if opts.is_plant or opts.output_confidence else 'auto'

        if opts.val_file_list:
            callbacks.append(kcbk.EarlyStopping(monitor=monitor_metric, patience=opts.early_stopping, verbose=1, mode=monitor_mode))

        callbacks.append(kcbk.ModelCheckpoint(os.path.join(opts.out_dir, 'model_weights_train.h5'), save_best_only=False))
        callbacks.append(kcbk.ModelCheckpoint(os.path.join(opts.out_dir, 'model_weights_val.h5'), monitor=monitor_metric, save_best_only=True, verbose=1, mode=monitor_mode))

        if opts.max_time:
            callbacks.append(cbk.TrainingStopper(max_time=int(opts.max_time * 3600), stop_file=opts.stop_file, verbose=1))

        callbacks.append(kcbk.ReduceLROnPlateau(monitor=monitor_metric, factor=0.2, patience=3, verbose=1, mode=monitor_mode, min_lr=1e-7))

        def save_lc(epoch, epoch_logs, val_epoch_logs):
            logs = {'lc_train.tsv': epoch_logs, 'lc_val.tsv': val_epoch_logs}
            for name, lgs in six.iteritems(logs):
                if lgs:
                    with open(os.path.join(opts.out_dir, name), 'w') as f:
                        f.write(perf_logs_str(pd.DataFrame(lgs)))

        metrics = ['loss'] + [mf.__name__ for mfs in six.itervalues(self.metrics) for mf in mfs]
        self.perf_logger = cbk.PerformanceLogger(callbacks=[save_lc], metrics=metrics, precision=LOG_PRECISION, verbose=not opts.no_log_outputs)
        callbacks.append(self.perf_logger)

        if not opts.no_tensorboard:
            callbacks.append(kcbk.TensorBoard(log_dir=opts.out_dir, histogram_freq=0, write_graph=True, write_images=True))

        return callbacks

    def build_dna_model(self, train_files_list):
        opts = self.opts
        if os.path.exists(opts.dna_model[0]):
            dna_model = mod.load_model(opts.dna_model, log=self.log.info)
            remove_outputs(dna_model)
            rename_layers(dna_model, 'dna')
        else:
            dna_model_builder = mod.dna.get(opts.dna_model[0])(l1_decay=opts.l1_decay, l2_decay=opts.l2_decay, dropout=opts.dropout)
            dna_wlen = dat.get_dna_wlen(train_files_list[0], opts.dna_wlen)
            dna_model = dna_model_builder(dna_model_builder.inputs(dna_wlen))
        return dna_model

    def build_cpg_model(self, replicate_names, train_files_list):
        opts = self.opts
        if not replicate_names: raise ValueError('No replicates found!')
        cpg_wlen = dat.get_cpg_wlen(train_files_list[0], opts.cpg_wlen)
        
        if os.path.exists(opts.cpg_model[0]):
            src_cpg_model = mod.load_model(opts.cpg_model, log=self.log.info)
            remove_outputs(src_cpg_model)
            rename_layers(src_cpg_model, 'cpg')
            if src_cpg_model.input_shape[0][1] != len(replicate_names):
                cpg_model_builder = mod.cpg.get(src_cpg_model.name)(l1_decay=opts.l1_decay, l2_decay=opts.l2_decay, dropout=opts.dropout)
                cpg_model = cpg_model_builder(cpg_model_builder.inputs(cpg_wlen, replicate_names))
                mod.copy_weights(src_cpg_model, cpg_model)
            else:
                cpg_model = src_cpg_model
        else:
            cpg_model_builder = mod.cpg.get(opts.cpg_model[0])(l1_decay=opts.l1_decay, l2_decay=opts.l2_decay, dropout=opts.dropout)
            cpg_model = cpg_model_builder(cpg_model_builder.inputs(cpg_wlen, replicate_names))
        return cpg_model

    def build_model(self, train_files_list):
        opts = self.opts
        all_output_names = set()
        files_to_check = random.sample(train_files_list, min(500, len(train_files_list)))

        for train_file in files_to_check:
            try:
                all_output_names.update(dat.get_output_names(train_file, regex=opts.output_names, nb_key=opts.nb_output))
            except Exception: pass
        
        output_names = sorted(list(all_output_names))
        if not output_names: raise ValueError('No outputs found!')

        dna_model = self.build_dna_model(train_files_list) if opts.dna_model else None
        cpg_model = self.build_cpg_model([n.replace('cpg/', '') for n in output_names if n.startswith('cpg/')], train_files_list) if opts.cpg_model else None

        if dna_model and cpg_model:
            joint_model_builder = mod.joint.get(opts.joint_model)(l1_decay=opts.l1_decay, l2_decay=opts.l2_decay, dropout=opts.dropout)
            stem = joint_model_builder([dna_model, cpg_model])
            stem._name = '_'.join([stem.name, dna_model.name, cpg_model.name])
        elif dna_model: stem = dna_model
        elif cpg_model: stem = cpg_model
        else:
            stem = mod.load_model(opts.model_files, log=self.log.info)
            if sorted(output_names) == sorted(stem.output_names): return stem
            remove_outputs(stem)

        outputs = mod.add_output_layers(stem.outputs[0], output_names, is_plant=opts.is_plant, output_confidence=opts.output_confidence)
        return Model(inputs=stem.inputs, outputs=outputs, name=stem.name)

    def set_trainability(self, model):
        opts = self.opts
        trainable, not_trainable = [], []
        
        if opts.fine_tune: not_trainable.append('.*')
        elif opts.train_models:
            not_trainable.append('.*')
            trainable.extend([f'{name}_' for name in opts.train_models])
        if opts.freeze_filter: not_trainable.append(mod.get_first_conv_layer(model.layers).name)
        
        if not trainable and opts.trainable: trainable = opts.trainable
        if not not_trainable and opts.not_trainable: not_trainable = opts.not_trainable
        if not trainable and not not_trainable: return

        for layer in model.layers:
            if is_input_layer(layer) or is_output_layer(layer, model) or not hasattr(layer, 'trainable'): continue
            for regex in not_trainable:
                if re.match(regex, layer.name): layer.trainable = False
            for regex in trainable:
                if re.match(regex, layer.name): layer.trainable = True

    def init_filter_weights(self, filename, conv_layer):
        with h5.File(filename[0], 'r') as hf:
            group = hf[filename[1]] if len(filename) > 1 else hf
            weights = group['weights'][()]
            bias = group['bias'][()] if 'bias' in group else None

        if weights.shape[1] != 1:
            weights = np.expand_dims(np.swapaxes(weights[:, :, :, 0], 0, 2), 1)

        cur_weights, cur_bias = conv_layer.get_weights()
        weights = weights[:, :, :, :min(weights.shape[-1], cur_weights.shape[-1])]

        if len(weights) > len(cur_weights):
            idx = (len(weights) - len(cur_weights)) // 2
            weights = weights[idx:(idx + len(cur_weights))]
        elif len(weights) < len(cur_weights):
            pad_weights = np.random.uniform(0, 1, [len(cur_weights)] + list(weights.shape[1:])) * 1e-2
            idx = (len(cur_weights) - len(weights)) // 2
            pad_weights[idx:(idx + len(weights))] = weights
            weights = pad_weights

        cur_weights[:, :, :, :weights.shape[-1]] = weights
        if bias is not None: cur_bias[:len(bias[:len(cur_bias)])] = bias[:len(cur_bias)]
        conv_layer.set_weights((cur_weights, cur_bias))

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file, format='%(levelname)s (%(asctime)s): %(message)s')
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.DEBUG if opts.verbose else logging.INFO)
        self.opts = opts

        if opts.seed is not None:
            tf.random.set_seed(opts.seed)
            np.random.seed(opts.seed)
            random.seed(opts.seed)

        make_dir(opts.out_dir)

        with open(opts.train_file_list, 'r') as f: train_files_list = [l.strip() for l in f if l.strip()]
        val_files_list = []
        if opts.val_file_list:
            with open(opts.val_file_list, 'r') as f: val_files_list = [l.strip() for l in f if l.strip()]

        expected_paths = []
        if opts.dna_model: expected_paths.append('inputs/dna')
        if opts.cpg_model: expected_paths.extend([f'inputs/cpg/{opts.replicate_names[0]}/dist', f'inputs/cpg/{opts.replicate_names[0]}/state'])
        expected_paths.extend([f'outputs/{o}' for o in opts.output_names])
        expected_paths = sorted(list(set(expected_paths)))

        if train_files_list: train_files_list = self.verify_h5_files(train_files_list, expected_paths, self.log, 'train_bad_files.txt')
        if val_files_list: val_files_list = self.verify_h5_files(val_files_list, expected_paths, self.log, 'val_bad_files.txt')

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

        strategy = tf.distribute.MirroredStrategy()
        global_batch_size = opts.batch_size * max(1, strategy.num_replicas_in_sync)

        with strategy.scope():
            model = self.build_model(train_files_list)
            self.set_trainability(model)
            if opts.filter_weights: self.init_filter_weights(opts.filter_weights, mod.get_first_conv_layer(model.layers))
            
            output_weights = get_output_weights(model.output_names, opts.output_weights) if opts.output_weights else None

            if opts.output_confidence:
                objectives = {name: met.gaussian_nll_loss for name in model.output_names}
                def pcc_metric(y_true, y_pred): return met.pcc(y_true, y_pred[:, 0:1])
                def mse_metric(y_true, y_pred): return met.mse(y_true, y_pred[:, 0:1])
                def mae_metric(y_true, y_pred): return met.mae(y_true, y_pred[:, 0:1])
                for m, n in zip([pcc_metric, mse_metric, mae_metric], ['pcc', 'mse', 'mae']): m.__name__ = n
                self.metrics = {name: [mse_metric, mae_metric, pcc_metric] for name in model.output_names}
            else:
                self.metrics = {name: get_metrics(name, opts.is_plant) for name in model.output_names}
                objectives = mod.get_objectives(model.output_names, is_plant=opts.is_plant)

            model.compile(optimizer=Adam(learning_rate=opts.learning_rate), loss=objectives, loss_weights=output_weights, metrics=self.metrics)

        mod.save_model(model, os.path.join(opts.out_dir, 'model.json'))

        data_reader = mod.data_reader_from_model(model, replicate_names=[n.replace('cpg/', '') for n in model.output_names if n.startswith('cpg/')])
        
        if hasattr(data_reader, 'name_map'):
            data_reader.name_map = OrderedDict({k: f"inputs/{v}" if not v.startswith('inputs/') else v for k, v in data_reader.name_map.items()})
        if hasattr(data_reader, 'output_name_map'):
            data_reader.output_name_map = OrderedDict({k: f"outputs/{v}" if not v.startswith('outputs/') else v for k, v in data_reader.output_name_map.items()})

        if opts.nb_train_sample: nb_train_sample = opts.nb_train_sample
        else:
            sample_size = min(100, len(train_files_list))
            nb_train_sample = int((dat.get_nb_sample(random.sample(train_files_list, sample_size), None) / sample_size) * len(train_files_list)) if sample_size else 0
            
        train_data = data_reader(train_files_list, class_weights=None if opts.no_class_weights else OrderedDict(), batch_size=global_batch_size, nb_sample=nb_train_sample, shuffle=True, loop=True)

        if val_files_list:
            if opts.nb_val_sample: nb_val_sample = opts.nb_val_sample
            else:
                sample_size = min(100, len(val_files_list))
                nb_val_sample = int((dat.get_nb_sample(random.sample(val_files_list, sample_size), None) / sample_size) * len(val_files_list)) if sample_size else 0
            val_data = data_reader(val_files_list, batch_size=global_batch_size, nb_sample=nb_val_sample, shuffle=False, loop=True) if nb_val_sample else None
        else:
            val_data, nb_val_sample = None, None

        callbacks = self.get_callbacks(model)
        
        steps_per_epoch = max(1, nb_train_sample // global_batch_size) if nb_train_sample else 1
        validation_steps = max(1, nb_val_sample // global_batch_size) if nb_val_sample else None

        model.fit(train_data, steps_per_epoch=steps_per_epoch, epochs=opts.nb_epoch, callbacks=callbacks, validation_data=val_data, validation_steps=validation_steps, max_queue_size=opts.data_q_size, workers=opts.data_nb_worker, verbose=2)

        filename = os.path.join(opts.out_dir, 'model_weights_val.h5')
        if os.path.isfile(filename): model.load_weights(filename)
        model.save(os.path.join(opts.out_dir, 'model.h5'))

        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)