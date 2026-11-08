#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import h5py as h5
import logging
import numpy as np
import pandas as pd
import six

from EvoPlantMeth import data as dat
from EvoPlantMeth.utils import make_dir

def write_to_bedGraph(data, filename, compression=None):
    df = pd.DataFrame({'chromo': data['chromo'],
                       'start': data['pos'],
                       'end': data['pos'] + 1,
                       'value': data['value']},
                      columns=['chromo', 'start', 'end', 'value'])
    df['chromo'] = df['chromo'].str.decode('utf')
    df.to_csv(filename, sep='\t', index=False, header=None,
              float_format='%.5f', compression=compression)

def write_to_hdf(data, filename):
    out_file = h5.File(filename, 'w')
    keys_to_write = ['chromo', 'pos', 'value']
    if 'confidence' in data:
        keys_to_write.append('confidence')

    for name in keys_to_write:
        dtype = np.float32 if name in ['value', 'confidence'] else data[name].dtype
        out_file.create_dataset(name, data=data[name], dtype=dtype, compression='gzip')
    out_file.close()

class App(object):
    def run(self, args):
        name = os.path.basename(args[0])
        parser = self.create_parser(name)
        opts = parser.parse_args(args[1:])
        return self.main(name, opts)

    def create_parser(self, name):
        p = argparse.ArgumentParser(prog=name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument('data_file')
        p.add_argument('-o', '--out_dir', default='.')
        p.add_argument('-f', '--out_format', choices=['bedGraph', 'hdf'], default='hdf')
        p.add_argument('--chromos', nargs='+')
        p.add_argument('--output_names', nargs='+')
        p.add_argument('--nb_sample', type=int)
        p.add_argument('--verbose', action='store_true')
        p.add_argument('--log_file')
        p.add_argument('--is_plant', action='store_true')
        p.add_argument('--output_type', choices=['combined', 'pred_only', 'both'], default='combined')
        p.add_argument('--with_confidence', action='store_true')
        return p

    def main(self, name, opts):
        logging.basicConfig(filename=opts.log_file, format='%(levelname)s (%(asctime)s): %(message)s')
        log = logging.getLogger(name)
        log.setLevel(logging.DEBUG if opts.verbose else logging.INFO)

        data_file = h5.File(opts.data_file, 'r')
        nb_sample = len(data_file['pos'])
        if opts.nb_sample:
            nb_sample = min(nb_sample, opts.nb_sample)

        data = dict()
        for key in ['chromo', 'pos']:
            data[key] = data_file[key][:nb_sample]

        idx = None
        if opts.chromos:
            idx = np.in1d(data['chromo'], [chromo.encode() for chromo in opts.chromos])
            for key, value in six.iteritems(data):
                data[key] = value[idx]

        output_names = [name for name in dat.get_output_names(opts.data_file, regex=opts.output_names)
                        if not name.endswith('_confidence')]

        make_dir(opts.out_dir)

        for output_name in output_names:
            log.info('Processing: %s', output_name)
            data['output'] = data_file['outputs'][output_name][:nb_sample]
            data['pred'] = data_file['preds'][output_name][:nb_sample]
            data.pop('confidence', None)
            
            if opts.with_confidence:
                confidence_name = output_name + '_variance'
                if confidence_name in data_file['preds']:
                    data['confidence'] = data_file['preds'][confidence_name][:nb_sample]
                else:
                    log.warning('Confidence scores not found for %s', output_name)

            if opts.is_plant:
                data['pred'] = data['pred'].astype(np.float32)

            if idx is not None:
                keys_to_filter = ['output', 'pred']
                if 'confidence' in data: keys_to_filter.append('confidence')
                for key in keys_to_filter:
                    data[key] = data[key][idx]

            output_types = ['combined', 'pred_only'] if opts.output_type == 'both' else [opts.output_type]
            
            name_parts = output_name.split(dat.OUTPUT_SEP)
            base_name = name_parts[-1] if name_parts[0] == 'cpg' else '_'.join(name_parts)

            for output_type in output_types:
                if output_type == 'combined':
                    data['value'] = data['pred'].copy()
                    tmp = data['output'] != dat.CPG_NAN
                    data['value'][tmp] = data['output'][tmp]
                    file_suffix = ''
                else:
                    data['value'] = data['pred'].copy()
                    file_suffix = '_pred_only'

                if opts.is_plant:
                    data['value'] = data['value'].astype(np.float32)

                out_file_base = os.path.join(opts.out_dir, base_name + file_suffix)

                if opts.out_format == 'bedGraph':
                    write_to_bedGraph(data, out_file_base + '.bedGraph.gz', compression='gzip')
                    if 'confidence' in data:
                        confidence_data_to_write = {
                            'chromo': data['chromo'],
                            'pos': data['pos'],
                            'value': data['confidence']
                        }
                        write_to_bedGraph(confidence_data_to_write, out_file_base + '_confidence.bedGraph.gz', compression='gzip')
                elif opts.out_format == 'hdf':
                    write_to_hdf(data, out_file_base + '.h5')
                else:
                    raise ValueError('Invalid output format "%s"!' % opts.out_format)

        log.info('Done!')
        return 0

if __name__ == '__main__':
    app = App()
    app.run(sys.argv)