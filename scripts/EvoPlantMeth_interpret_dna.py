#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from collections import OrderedDict
import argparse
import os
import sys
import logomaker 
import matplotlib.pyplot as plt

from EvoPlantMeth import data as dat
from EvoPlantMeth import models as mod
from EvoPlantMeth import metrics as met

def read_file_list(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Config file not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def reverse_complement_numpy(data):
    # 1. Reverse sequence (axis=1)
    data_rev = np.flip(data, axis=1)
    # 2. Complement bases (axis=2): [A, C, G, T] -> [T, G, C, A]
    data_rc = np.flip(data_rev, axis=2)
    return data_rc

def plot_logo(saliency_data, title, out_path, zoom_w, center_i):
    if len(saliency_data) == 0:
        print(f"Warning: No samples for {title}, skipping plot.")
        return

    mean_sal = np.mean(saliency_data, axis=0)
    wlen = mean_sal.shape[0]
    relative_positions = np.arange(-center_i, wlen - center_i)

    importance_per_pos = np.sum(np.abs(mean_sal), axis=1)
    df_imp = pd.DataFrame({'pos': relative_positions, 'score': importance_per_pos})
    df_imp['rolling'] = df_imp['score'].rolling(window=10, center=True).mean()

    flank = zoom_w // 2
    zoom_start_pos, zoom_end_pos = -flank, flank
    
    up_slice = slice(center_i - flank, center_i)
    down_slice = slice(center_i + 1, center_i + 1 + flank)
    
    plot_data = np.concatenate([mean_sal[up_slice], mean_sal[down_slice]])
    plot_idx = np.concatenate([np.arange(-flank, 0), np.arange(1, flank + 1)])
    
    df_logo = pd.DataFrame(plot_data, columns=['A', 'C', 'G', 'T'], index=plot_idx)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    ax1.plot(df_imp['pos'], df_imp['score'], color='lightgray', label='Raw Importance', alpha=0.6)
    ax1.plot(df_imp['pos'], df_imp['rolling'], color='blue', label='10bp Rolling Avg', linewidth=2)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Center CpG')
    ax1.axvspan(zoom_start_pos, zoom_end_pos, color='yellow', alpha=0.3, label='Zoom Region')
    
    ax1.set_title(f'{title} - Global Importance', fontsize=14)
    ax1.set_ylabel('Importance Score', fontsize=12)
    ax1.set_xlabel('Position relative to CpG (bp)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_xlim(relative_positions[0], relative_positions[-1])

    logomaker.Logo(df_logo, color_scheme='classic', ax=ax2, center_values=True)
    ax2.set_title(f'Zoomed Sequence Logo (Top {zoom_w}bp)', fontsize=14)
    ax2.set_ylabel('Saliency', fontsize=12)
    ax2.set_xlabel('Position relative to CpG (Skipping 0)', fontsize=12)
    ax2.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    
    if zoom_w <= 60:
        ax2.set_xticks(plot_idx)
        ax2.set_xticklabels(plot_idx, rotation=90, fontsize=8)
    
    plt.savefig(f"{out_path}.pdf", dpi=300)
    plt.savefig(f"{out_path}.png", dpi=300)
    print(f"Saved: {out_path}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='DNA Saliency Analysis for Plant Methylation (CG/CHG/CHH).')
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--out_dir', type=str, default='interpret_output')
    parser.add_argument('--output_prefix', type=str, default='dna_saliency')
    parser.add_argument('--zoom_width', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_out = os.path.join(args.out_dir, args.output_prefix)

    custom_objects = {
        'gaussian_nll_loss': met.gaussian_nll_loss,
        'pcc': lambda y_true, y_pred: met.pcc(y_true, y_pred[:, 0:1]),
        'mse': lambda y_true, y_pred: met.mse(y_true, y_pred[:, 0:1]),
        'mae': lambda y_true, y_pred: met.mae(y_true, y_pred[:, 0:1])
    }
    
    print(f"Loading model: {args.model_path}...")
    full_model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)

    print("Loading data...")
    data_reader = mod.data_reader_from_model(full_model, replicate_names=['unified_sample'])
    
    if hasattr(data_reader, 'name_map'):
        new_map = OrderedDict()
        for k, v in data_reader.name_map.items(): 
            new_map[k] = f'inputs/{v}' if not v.startswith('inputs/') else v
        data_reader.name_map = new_map
    if hasattr(data_reader, 'output_name_map'): data_reader.output_name_map = OrderedDict()
    if hasattr(data_reader, 'output_names'): data_reader.output_names = []
    if hasattr(data_reader, 'output_confidence'): data_reader.output_confidence = False

    hdf5_files = read_file_list(args.file_list)
    nb_sample = dat.get_nb_sample(hdf5_files, None)
    actual_batch = min(args.batch_size, nb_sample)

    data_gen = data_reader(hdf5_files, batch_size=actual_batch, nb_sample=actual_batch, shuffle=True, loop=False)
    batch_data = next(iter(data_gen))
    input_batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
    print(f"Loaded {actual_batch} samples.")

    print("Computing gradients...")
    target_layer = 'cpg/unified_sample_mean' if 'cpg/unified_sample_mean' in [l.name for l in full_model.layers] else full_model.output_names[0]
    target_output = full_model.get_layer(target_layer).output if target_layer in [l.name for l in full_model.layers] else full_model.output
    
    saliency_model = tf.keras.models.Model(inputs=full_model.inputs, outputs=target_output)
    input_tensors = {k: tf.cast(tf.convert_to_tensor(v), tf.float32) for k, v in input_batch.items()}
    dna_tensor = input_tensors['dna']

    with tf.GradientTape() as tape:
        tape.watch(dna_tensor)
        input_tensors['dna'] = dna_tensor
        preds = saliency_model(input_tensors)

    grads = tape.gradient(preds, dna_tensor).numpy()
    inputs = dna_tensor.numpy()
    raw_saliency = grads * inputs 

    print("Classifying contexts and orienting viewpoints...")
    center_idx = inputs.shape[1] // 2
    center_bases = np.argmax(inputs[:, center_idx, :], axis=1)

    mask_C = (center_bases == 1)
    mask_G = (center_bases == 2)
    
    unified_inputs, unified_saliency = np.zeros_like(inputs), np.zeros_like(raw_saliency)
    unified_inputs[mask_C] = inputs[mask_C]
    unified_saliency[mask_C] = raw_saliency[mask_C]
    unified_inputs[mask_G] = reverse_complement_numpy(inputs[mask_G])
    unified_saliency[mask_G] = reverse_complement_numpy(raw_saliency[mask_G])

    seq_plus1_is_G = (np.argmax(unified_inputs[:, center_idx + 1, :], axis=1) == 2)
    seq_plus2_is_G = (np.argmax(unified_inputs[:, center_idx + 2, :], axis=1) == 2)

    mask_valid = (mask_C | mask_G)
    mask_CG  = mask_valid & seq_plus1_is_G
    mask_CHG = mask_valid & (~seq_plus1_is_G) & seq_plus2_is_G
    mask_CHH = mask_valid & (~seq_plus1_is_G) & (~seq_plus2_is_G)

    print(f"Contexts found: CG({np.sum(mask_CG)}), CHG({np.sum(mask_CHG)}), CHH({np.sum(mask_CHH)})")

    if np.sum(mask_CG) > 0: plot_logo(unified_saliency[mask_CG], f'CG Methylation (n={np.sum(mask_CG)})', f"{base_out}_CG", args.zoom_width, center_idx)
    if np.sum(mask_CHG) > 0: plot_logo(unified_saliency[mask_CHG], f'CHG Methylation (n={np.sum(mask_CHG)})', f"{base_out}_CHG", args.zoom_width, center_idx)
    if np.sum(mask_CHH) > 0: plot_logo(unified_saliency[mask_CHH], f'CHH Methylation (n={np.sum(mask_CHH)})', f"{base_out}_CHH", args.zoom_width, center_idx)

    print("Analysis complete.")

if __name__ == '__main__':
    main()