#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import random
import sys
import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six
import tensorflow as tf

from EvoPlantMeth import data as dat
from EvoPlantMeth import evaluation as ev
from EvoPlantMeth import models as mod
from EvoPlantMeth.data import hdf
from EvoPlantMeth.utils import ProgressBar, to_list
from EvoPlantMeth import metrics as met
from EvoPlantMeth.models import utils as mod_utils

class H5Writer(object):
    def __init__(self, filename, nb_sample):
        self.out_file = h5.File(filename, 'w')
        self.nb_sample = nb_sample
        self.idx = 0

    def __call__(self, name, data, dtype=None, compression='gzip', stay=False):
        if name not in self.out_file:
            if dtype is None:
                dtype = data.dtype
            self.out_file.create_dataset(
                name=name,
                shape=[self.nb_sample] + list(data.shape[1:]),
                dtype=dtype,
                compression=compression
            )
        self.out_file[name][self.idx:(self.idx + len(data))] = data
        if not stay:
            self.idx += len(data)

    def write_dict(self, data, name='', level=0, *args, **kwargs):
        size = None
        for key, value in six.iteritems(data):
            _name = '%s/%s' % (name, key) if name else key
            if isinstance(value, dict):
                self.write_dict(value, name=_name, level=level + 1, *args, **kwargs)
            else:
                if size is not None:
                    assert size == len(value)
                else:
                    size = len(value)
                self(_name, value, stay=True, *args, **kwargs)
        if level == 0 and size is not None:
            self.idx += size

    def close(self):
        self.out_file.close()

class App(object):
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        if opts.seed is not None:
            tf.random.set_seed(opts.seed)
            np.random.seed(opts.seed)
            random.seed(opts.seed)
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(prog=name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('data_files', nargs='+')
        p.add_argument('--model_files', nargs='+', required=True)
        p.add_argument('-o', '--out_report')
        p.add_argument('--out_data')
        p.add_argument('--replicate_names', nargs='+')
        p.add_argument('--nb_replicate', type=int)
        p.add_argument('--eval_size', type=int, default=100000)
        p.add_argument('--batch_size', type=int, default=128)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--nb_sample', type=int)
        p.add_argument('--verbose', action='store_true')
        p.add_argument('--log_file')
        p.add_argument('--is_plant', action='store_true')
        p.add_argument('--output_confidence', action='store_true')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file, format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG if opts.verbose else logging.INFO)

        log.info('Loading model ...')
        custom_objects = mod_utils.CUSTOM_OBJECTS.copy()
        
        if opts.output_confidence:
            custom_objects['gaussian_nll_loss'] = met.gaussian_nll_loss
            def pcc_metric(y_true, y_pred): return met.pcc(y_true, y_pred[:, 0:1])
            def mse_metric(y_true, y_pred): return met.mse(y_true, y_pred[:, 0:1])
            def mae_metric(y_true, y_pred): return met.mae(y_true, y_pred[:, 0:1])
            pcc_metric.__name__ = 'pcc'
            mse_metric.__name__ = 'mse'
            mae_metric.__name__ = 'mae'
            custom_objects['pcc'] = pcc_metric
            custom_objects['mse'] = mse_metric
            custom_objects['mae'] = mae_metric
            
        model = mod.load_model(opts.model_files, custom_objects=custom_objects, log=log.info)

        has_confidence = opts.output_confidence
        output_shapes = to_list(model.output_shape)
        model_outputs_confidence = len(output_shapes) > 0 and output_shapes[0] is not None and output_shapes[0][-1] == 2

        if has_confidence and not model_outputs_confidence:
            raise ValueError(f'--output_confidence specified, but model output shape is {output_shapes}. Expected last dimension 2.')

        log.info('Loading data ...')
        nb_sample = dat.get_nb_sample(opts.data_files, opts.nb_sample)
        replicate_names = dat.get_replicate_names(opts.data_files[0], regex=opts.replicate_names, nb_key=opts.nb_replicate)
        
        data_reader = mod.data_reader_from_model(model, replicate_names=replicate_names)
        data_generator = data_reader(opts.data_files, nb_sample=nb_sample, batch_size=opts.batch_size, loop=False, shuffle=False)
        meta_reader = hdf.reader(opts.data_files, ['chromo', 'pos'], nb_sample=nb_sample, batch_size=opts.batch_size, loop=False, shuffle=False)

        writer = H5Writer(opts.out_data, nb_sample) if opts.out_data else None

        log.info('Predicting methylation states ...')
        nb_tot, nb_eval = 0, 0
        data_eval = dict()
        perf_eval = []
        progbar = ProgressBar(nb_sample, log.info)
        
        for inputs, outputs, _ in data_generator:
            batch_size = len(list(inputs.values())[0])
            nb_tot += batch_size
            progbar.update(batch_size)

            preds = to_list(model.predict(inputs, batch_size=batch_size))
            data_batch = {'preds': {}, 'outputs': {}}

            for i, name in enumerate(model.output_names):
                pred_data = preds[i]
                output_data = outputs[name].squeeze()

                if has_confidence:
                    data_batch['preds'][name] = pred_data[:, 0].squeeze()
                    data_batch['preds'][name + '_variance'] = np.exp(pred_data[:, 1].squeeze())
                else:
                    data_batch['preds'][name] = pred_data.squeeze()
                data_batch['outputs'][name] = output_data

            for meta_name, meta_value in six.iteritems(next(meta_reader)):
                data_batch[meta_name] = meta_value

            if writer: writer.write_dict(data_batch)

            nb_eval += batch_size
            dat.add_to_dict(data_batch, data_eval)

            if nb_tot >= nb_sample or (opts.eval_size and nb_eval >= opts.eval_size):
                data_eval = dat.stack_dict(data_eval)
                if opts.is_plant or has_confidence:
                    perf_eval.append(ev.evaluate_outputs_regression(data_eval['outputs'], data_eval['preds']))
                else:
                    perf_eval.append(ev.evaluate_outputs(data_eval['outputs'], data_eval['preds']))
                data_eval = dict()
                nb_eval = 0

        progbar.close()
        if writer: writer.close()

        if not perf_eval:
            log.error("No evaluation results collected.")
            return 1
            
        report = pd.concat(perf_eval)
        report = report.groupby(['metric', 'output']).mean().reset_index()

        if opts.out_report:
            log.info(f'Saving report to {opts.out_report}')
            report.to_csv(opts.out_report, sep='\t', index=False, float_format='%.6f')

        log.info('Evaluation report:\n' + ev.unstack_report(report).to_string(float_format='%.4f'))
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)