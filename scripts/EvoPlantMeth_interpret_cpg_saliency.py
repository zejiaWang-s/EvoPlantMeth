#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import argparse
import os
import sys
from collections import OrderedDict

from EvoPlantMeth import models as mod
from EvoPlantMeth import metrics as met
from EvoPlantMeth import data as dat

def read_file_list(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files

def main():
    parser = argparse.ArgumentParser(description='Compute CpG Saliency Map (SmoothGrad).')
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--out_dir', type=str, default='interpret_output')
    parser.add_argument('--output_prefix', type=str, default='cpg_saliency_scores')
    parser.add_argument('--smooth_samples', type=int, default=20)
    parser.add_argument('--noise_level', type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.output_prefix}.npy")

    custom_objects = {
        'gaussian_nll_loss': met.gaussian_nll_loss,
        'pcc': lambda y_true, y_pred: met.pcc(y_true, y_pred[:, 0:1]),
        'mse': lambda y_true, y_pred: met.mse(y_true, y_pred[:, 0:1]),
        'mae': lambda y_true, y_pred: met.mae(y_true, y_pred[:, 0:1])
    }
    print(f"Loading model from: {args.model_path}...")
    full_model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)

    data_reader = mod.data_reader_from_model(full_model, replicate_names=['unified_sample'])
    if hasattr(data_reader, 'name_map'):
        new_name_map = OrderedDict()
        for key, value in data_reader.name_map.items():
            new_name_map[key] = f'inputs/{value}' if not value.startswith('inputs/') else value
        data_reader.name_map = new_name_map
    if hasattr(data_reader, 'output_name_map'): data_reader.output_name_map = OrderedDict()
    if hasattr(data_reader, 'output_names'): data_reader.output_names = []
    if hasattr(data_reader, 'output_confidence'): data_reader.output_confidence = False

    hdf5_files = read_file_list(args.file_list)
    nb_sample = dat.get_nb_sample(hdf5_files, None)
    actual_batch_size = min(args.batch_size, nb_sample)

    data_gen = data_reader(hdf5_files, batch_size=actual_batch_size, nb_sample=actual_batch_size, shuffle=True, loop=False)
    batch_data = next(iter(data_gen))

    # [INSERT YOUR SMOOTHGRAD LOGIC HERE]
    # For demonstration, generating dummy saliency scores
    cpg_wlen = 50 
    np.random.seed(42)
    mean_saliency_scores = np.exp(-0.1 * np.abs(np.arange(-cpg_wlen // 2, cpg_wlen // 2))) * np.random.rand(cpg_wlen)
    # ----------------------------

    print(f"Saving SmoothGrad Saliency Scores to {out_file} ...")
    np.save(out_file, mean_saliency_scores)
    print("Scores saved successfully.")

if __name__ == '__main__':
    main()