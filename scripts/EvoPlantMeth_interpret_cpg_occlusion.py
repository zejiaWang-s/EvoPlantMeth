#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from collections import OrderedDict

from EvoPlantMeth import models as mod
from EvoPlantMeth import metrics as met
from EvoPlantMeth import data as dat

def read_file_list(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        sys.exit(1)
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description='Occlusion Test: Identifies important CpG neighbors by masking them.')
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--out_dir', type=str, default='interpret_output')
    parser.add_argument('--output_prefix', type=str, default='occlusion_test')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path_base = os.path.join(args.out_dir, args.output_prefix)

    custom_objects = {
        'gaussian_nll_loss': met.gaussian_nll_loss,
        'pcc': lambda y_true, y_pred: met.pcc(y_true, y_pred[:, 0:1]),
        'mse': lambda y_true, y_pred: met.mse(y_true, y_pred[:, 0:1]),
        'mae': lambda y_true, y_pred: met.mae(y_true, y_pred[:, 0:1])
    }
    
    print(f"Loading model: {args.model_path}")
    full_model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)

    target_layer = 'cpg/unified_sample_mean'
    try:
        output_tensor = full_model.get_layer(target_layer).output
        predict_model = tf.keras.models.Model(inputs=full_model.inputs, outputs=output_tensor)
    except:
        print(f"Warning: Layer {target_layer} not found, using default output.")
        predict_model = full_model

    data_reader = mod.data_reader_from_model(full_model, replicate_names=['unified_sample'])
    if hasattr(data_reader, 'name_map'):
        new_map = OrderedDict()
        for k, v in data_reader.name_map.items(): new_map[k] = f'inputs/{v}' if not v.startswith('inputs/') else v
        data_reader.name_map = new_map
    if hasattr(data_reader, 'output_name_map'): data_reader.output_name_map = OrderedDict()
    if hasattr(data_reader, 'output_names'): data_reader.output_names = []
    if hasattr(data_reader, 'output_confidence'): data_reader.output_confidence = False

    hdf5_files = read_file_list(args.file_list)
    nb_sample = dat.get_nb_sample(hdf5_files, None)
    actual_batch = min(args.batch_size, nb_sample)

    data_gen = data_reader(hdf5_files, batch_size=actual_batch, nb_sample=actual_batch, shuffle=True, loop=False)
    print("Loading data...")
    batch_data = next(iter(data_gen))
    input_batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data

    print("Starting occlusion test...")
    baseline_preds = predict_model.predict(input_batch, verbose=0)
    cpg_len = input_batch['cpg_state'].shape[2]
    sensitivity_scores = []

    for i in range(cpg_len):
        masked_batch = {k: v.copy() for k, v in input_batch.items()}
        masked_batch['cpg_state'][:, :, i] = 0
        masked_batch['cpg_dist'][:, :, i] = 0
        
        masked_preds = predict_model.predict(masked_batch, verbose=0)
        diff = np.mean(np.abs(masked_preds - baseline_preds))
        sensitivity_scores.append(diff)
        
        sys.stdout.write(f"\rMasking position {i}/{cpg_len-1}: Impact {diff:.6f}")
        sys.stdout.flush()

    print("\nTest complete.")

    sensitivity_scores = np.array(sensitivity_scores)
    positions = np.arange(-cpg_len // 2, cpg_len // 2)

    peak_idx = np.argmax(sensitivity_scores)
    print(f"[Diagnostics] Most sensitive relative pos: {positions[peak_idx]} (Index: {peak_idx})")

    plt.figure(figsize=(12, 6))
    plt.bar(positions, sensitivity_scores, width=0.8, color='firebrick')
    plt.title('Occlusion Sensitivity (Prediction Change when Masking Neighbor)', fontsize=14)
    plt.xlabel('Relative Neighbor Position', fontsize=12)
    plt.ylabel('Impact on Prediction (Mean Abs Diff)', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Center')
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend()

    plt.savefig(f"{out_path_base}.pdf", dpi=300)
    plt.savefig(f"{out_path_base}.png", dpi=300)
    print(f"Results saved to {out_path_base}.png")

if __name__ == '__main__':
    main()