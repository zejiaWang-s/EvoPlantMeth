#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
import re
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from EvoPlantMeth import metrics as met
from EvoPlantMeth.models import utils as mod_utils
from EvoPlantMeth.models import dna, cpg, joint

def get_args():
    parser = argparse.ArgumentParser(description="Find functional methylation sites via In Silico Mutagenesis (Gradients).")
    parser.add_argument('--model_path', required=True, help="Path to unified model.h5")
    parser.add_argument('--data_dir', required=True, help="Directory containing processed .h5 data files")
    parser.add_argument('--out_file', required=True, help="Output TSV file path")
    parser.add_argument('--gff_file', help='Optional GFF3 file for annotation and plotting')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--top_n', type=int, default=10000, help="Number of sites to save in Top TSV")
    parser.add_argument('--plot_top_genes', type=int, default=5, help="Number of top genes to label on the plot")
    parser.add_argument('--promoter_upstream', type=int, default=3000, help="Distance upstream of TSS")
    parser.add_argument('--save_all', action='store_true', help="Save all sites TSV (not just top N)")
    parser.add_argument('--plot_downsample_rate', type=float, default=0.05, help="Background plot downsample rate")
    return parser.parse_args()

@tf.function
def compute_gradients(model, inputs):
    dna_in, cpg_state_in, cpg_dist_in = inputs
    with tf.GradientTape() as tape:
        tape.watch(cpg_state_in)
        predictions = model([dna_in, cpg_state_in, cpg_dist_in])
        pred_mean = predictions[:, 0]
    return tape.gradient(pred_mean, cpg_state_in)

def load_gff_robust(gff_file):
    print(f"Loading annotations from {gff_file}...")
    genes = []
    try:
        df = pd.read_csv(gff_file, sep='\t', comment='#', header=None, on_bad_lines='skip', low_memory=False)
        target_df = df[df[2].isin(['gene', 'mRNA', 'Gene', 'transcript'])]
        for _, row in target_df.iterrows():
            try:
                attr_str = str(row[8])
                gene_id = "Unknown"
                if "ID=" in attr_str: gene_id = attr_str.split("ID=")[1].split(";")[0]
                elif "Name=" in attr_str: gene_id = attr_str.split("Name=")[1].split(";")[0]
                genes.append((str(row[0]), int(row[3]), int(row[4]), str(row[6]), gene_id))
            except: continue
    except Exception as e:
        print(f"Warning: Failed to read GFF: {e}")
        return pd.DataFrame()
    return pd.DataFrame(genes, columns=['chrom', 'start', 'end', 'strand', 'gene_id'])

def annotate_sites(sites_df, gff_df, promoter_upstream=3000):
    print(f"Annotating sites (Gene Body + {promoter_upstream}bp Promoter)...")
    annotations = ['Intergenic'] * len(sites_df)
    sites_df_reset = sites_df.reset_index(drop=True)
    gff_df['chrom'] = gff_df['chrom'].astype(str)
    sites_df_reset['chrom'] = sites_df_reset['chrom'].astype(str)
    
    for chrom in sites_df_reset['chrom'].unique():
        site_mask = sites_df_reset['chrom'] == chrom
        site_indices = sites_df_reset[site_mask].index
        site_positions = sites_df_reset.loc[site_indices, 'pos'].values
        
        gff_sub = gff_df[gff_df['chrom'] == chrom]
        if gff_sub.empty: continue
        
        g_starts, g_ends, g_strands, g_ids = gff_sub['start'].values, gff_sub['end'].values, gff_sub['strand'].values, gff_sub['gene_id'].values
        
        ext_starts = np.where((g_strands == '+') | (g_strands == '.'), np.maximum(0, g_starts - promoter_upstream), g_starts)
        ext_ends = np.where((g_strands == '-') | (g_strands == '.'), g_ends + promoter_upstream, g_ends)
        
        for idx, pos in zip(site_indices, site_positions):
            matches = np.where((ext_starts <= pos) & (ext_ends >= pos))[0]
            if len(matches) > 0: annotations[idx] = g_ids[matches[0]]
                
    return annotations

def plot_manhattan(df, top_n_threshold, gff_df, top_genes_count, out_file, promoter_dist, downsample_rate):
    print(f"Generating Manhattan plot (Downsample Rate: {downsample_rate})...")
    plot_file = out_file.replace('.tsv', '_manhattan.pdf')
    
    df = df.copy()
    def parse_chrom(c):
        digits = re.findall(r'\d+', c)
        return int(digits[0]) if digits else 999
    
    df['chrom_idx'] = df['chrom'].apply(parse_chrom)
    df = df.sort_values(by=['chrom_idx', 'pos'])
    chrom_order = sorted(df['chrom'].unique(), key=parse_chrom)
    
    plt.figure(figsize=(14, 6))
    colors, x_ticks, x_labels, current_x_offset = ['#e6e6e6', '#cccccc'], [], [], 0
    
    sorted_sens = df.sort_values('sensitivity', ascending=False)
    cutoff_idx = min(len(df)-1, top_n_threshold)
    threshold_val = sorted_sens.iloc[cutoff_idx]['sensitivity'] if len(sorted_sens) > 0 else 0
    
    global_pos_map = {}
    for i, chrom in enumerate(chrom_order):
        chrom_data = df[df['chrom'] == chrom]
        if chrom_data.empty: continue
        
        global_pos_map[chrom] = current_x_offset
        g_pos = chrom_data['pos'] + current_x_offset
        high_mask = chrom_data['sensitivity'] >= threshold_val
        
        if not high_mask.all():
            low_data = chrom_data[~high_mask]
            if 0 < downsample_rate < 1.0: low_data = low_data.sample(frac=downsample_rate, random_state=42)
            plt.scatter(low_data['pos'] + current_x_offset, low_data['sensitivity'], c=colors[i%2], s=2, alpha=0.6, edgecolors='none', rasterized=True)
                
        if high_mask.any():
            plt.scatter(g_pos[high_mask], chrom_data.loc[high_mask, 'sensitivity'], c='#d62728', s=10, alpha=0.9, edgecolors='none', rasterized=True)
            
        x_ticks.append(current_x_offset + (chrom_data['pos'].max() + chrom_data['pos'].min())/2)
        x_labels.append(chrom)
        current_x_offset += chrom_data['pos'].max()
        
    if not gff_df.empty and top_genes_count > 0:
        top_candidates = df.sort_values('sensitivity', ascending=False).head(top_genes_count * 3)
        top_candidates['gene_label'] = annotate_sites(top_candidates, gff_df, promoter_dist)
        
        labeled = 0
        for _, row in top_candidates.iterrows():
            if labeled >= top_genes_count: break
            if row['gene_label'] == 'Intergenic': continue
            g_x, g_y = row['pos'] + global_pos_map[row['chrom']], row['sensitivity']
            plt.annotate(row['gene_label'], xy=(g_x, g_y), xytext=(g_x, g_y + (df['sensitivity'].max()*0.05)),
                         arrowprops=dict(facecolor='black', arrowstyle="->", lw=0.5),
                         fontsize=8, fontweight='bold', ha='center', color='black')
            labeled += 1

    plt.xticks(x_ticks, x_labels, fontsize=8, rotation=45)
    plt.ylabel("Sensitivity Score (Gradient)", fontsize=12)
    plt.title(f"Functional Methylation Sites (Top {top_genes_count} Labeled)\nRegion: Gene Body + {promoter_dist}bp Promoter", fontsize=12)
    plt.axhline(y=threshold_val, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    
    if downsample_rate < 1.0:
        plt.text(0, plt.ylim()[1], f"Background downsampled to {downsample_rate*100}%", fontsize=6, color='gray', va='top', ha='left')

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    print(f"Manhattan plot saved to {plot_file}")
    plt.close()

def main():
    args = get_args()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try: tf.config.experimental.set_memory_growth(gpu, True)
            except: pass

    def pcc_metric(y_true, y_pred): return met.pcc(y_true, y_pred[:, 0:1])
    def mse_metric(y_true, y_pred): return met.mse(y_true, y_pred[:, 0:1])
    def mae_metric(y_true, y_pred): return met.mae(y_true, y_pred[:, 0:1])
    pcc_metric.__name__, mse_metric.__name__, mae_metric.__name__ = 'pcc', 'mse', 'mae'

    custom_objects = mod_utils.CUSTOM_OBJECTS.copy()
    custom_objects.update({
        'CnnL2h128BN': dna.CnnL2h128BN, 'RnnL1BN_simple': cpg.RnnL1BN_simple, 'JointL2h512Attention': joint.JointL2h512Attention,
        'gaussian_nll_loss': met.gaussian_nll_loss, 'pcc': pcc_metric, 'mse': mse_metric, 'mae': mae_metric
    })
    
    print(f"Loading model: {args.model_path}")
    try: model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects, compile=False)
    except Exception as e: print(f"Error loading model: {e}"); sys.exit(1)

    h5_files = glob.glob(os.path.join(args.data_dir, "*.h5"))
    if not h5_files: sys.exit(f"No h5 files in {args.data_dir}")
        
    all_results = []
    print(f"Scanning {len(h5_files)} files...")
    
    for f_idx, h5_path in enumerate(h5_files):
        if (f_idx+1)%10 == 0: print(f"Processing {f_idx+1}/{len(h5_files)}...", end='\r')
        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'inputs/cpg' not in hf: continue
                input_group = hf['inputs/cpg']
                
                valid_sample = next((key for key in input_group.keys() if 'state' in input_group[key]), None)
                if not valid_sample: continue

                dna_raw, cpg_state_raw, cpg_dist_raw = hf['inputs/dna'][:], input_group[f'{valid_sample}/state'][:], input_group[f'{valid_sample}/dist'][:]   
                chroms, poss = hf['chromo'][:].astype(str), hf['pos'][:]
                
                num_samples = len(poss)
                if num_samples == 0: continue
                
                for i in range(0, num_samples, args.batch_size):
                    sl = slice(i, min(i + args.batch_size, num_samples))
                    
                    b_dna = tf.cast(tf.one_hot(tf.convert_to_tensor(dna_raw[sl], dtype=tf.int32), depth=4) if model.input_shape[0][-1] == 4 and len(dna_raw[sl].shape) == 2 else tf.convert_to_tensor(dna_raw[sl]), dtype=tf.float32)
                    b_state = tf.expand_dims(tf.convert_to_tensor(cpg_state_raw[sl], dtype=tf.float32), 1) if len(cpg_state_raw[sl].shape) == 2 else tf.convert_to_tensor(cpg_state_raw[sl], dtype=tf.float32)
                    b_dist = tf.expand_dims(tf.convert_to_tensor(cpg_dist_raw[sl], dtype=tf.float32), 1) if len(cpg_dist_raw[sl].shape) == 2 else tf.convert_to_tensor(cpg_dist_raw[sl], dtype=tf.float32)
                    
                    grads = compute_gradients(model, [b_dna, b_state, b_dist])
                    sensitivity = tf.reduce_mean(tf.reduce_sum(tf.abs(grads), axis=2), axis=1).numpy()
                    
                    all_results.extend([{'chrom': c, 'pos': p, 'sensitivity': s} for c, p, s in zip(chroms[sl], poss[sl], sensitivity)])
        except: continue

    if not all_results: sys.exit("No sites found.")

    print(f"\nProcessing total {len(all_results)} sites...")
    df = pd.DataFrame(all_results)
    
    if args.save_all:
        all_sites_file = args.out_file.replace('.tsv', '_ALL.tsv')
        print(f"Saving ALL sites to {all_sites_file} ...")
        df.to_csv(all_sites_file, index=False, sep='\t')

    gff_df = load_gff_robust(args.gff_file) if args.gff_file and os.path.exists(args.gff_file) else pd.DataFrame()
    plot_manhattan(df, args.top_n, gff_df, args.plot_top_genes if not gff_df.empty else 0, args.out_file, args.promoter_upstream, args.plot_downsample_rate)

    print(f"Sorting Top {args.top_n} for output...")
    df_top = df.sort_values(by='sensitivity', ascending=False).head(args.top_n)
    if not gff_df.empty: df_top['annotation'] = annotate_sites(df_top, gff_df, args.promoter_upstream)
    
    print(f"Saving TSV to {args.out_file}")
    df_top.to_csv(args.out_file, index=False, sep='\t')
    print("Done!")

if __name__ == '__main__':
    main()