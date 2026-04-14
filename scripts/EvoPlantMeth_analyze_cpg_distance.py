#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description='Analyze CpG Importance by Physical Distance.')
    parser.add_argument('--saliency_scores', type=str, required=True)
    parser.add_argument('--all_dists', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='interpret_output')
    parser.add_argument('--output_prefix', type=str, default='cpg_distance_analysis')
    parser.add_argument('--bin_size', type=int, default=10)
    parser.add_argument('--max_distance', type=int, default=500)
    parser.add_argument('--min_samples_per_bin', type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_base = os.path.join(args.out_dir, args.output_prefix)
    out_tsv = os.path.join(args.out_dir, f"{args.output_prefix}_data.tsv")

    print("Loading data...")
    saliency_scores_index = np.load(args.saliency_scores)
    all_dists = np.load(args.all_dists)

    cpg_wlen = len(saliency_scores_index)
    if all_dists.shape[1] != cpg_wlen:
        print(f"Error: Dists shape ({all_dists.shape[1]}) does not match Saliency ({cpg_wlen}).")
        sys.exit(1)

    print(f"Binning by physical distance (Bin Size: {args.bin_size}bp)...")
    bins = np.arange(0, args.max_distance + args.bin_size, args.bin_size)
    binned_saliency = [[] for _ in range(len(bins) - 1)]

    for sample_idx in range(all_dists.shape[0]):
        for i in range(cpg_wlen):
            score = saliency_scores_index[i]
            distance = np.abs(all_dists[sample_idx, i])
            
            if 0 < distance <= args.max_distance:
                bin_index = np.digitize(distance, bins) - 1
                if 0 <= bin_index < len(binned_saliency):
                    binned_saliency[bin_index].append(score)

    avg_saliency_per_bin = np.array([np.mean(scores) if scores else 0 for scores in binned_saliency])
    count_per_bin = np.array([len(scores) for scores in binned_saliency])

    bin_centers = bins[:-1] + args.bin_size / 2
    df_result = pd.DataFrame({
        'center': bin_centers,
        'saliency': avg_saliency_per_bin,
        'count': count_per_bin
    })

    df_result['saliency_filtered'] = df_result['saliency'].where(df_result['count'] >= args.min_samples_per_bin)
    df_result['saliency_smoothed'] = df_result['saliency_filtered'].rolling(window=3, center=True, min_periods=1).mean()

    data_to_save = df_result[['center', 'saliency_smoothed']].dropna()
    data_to_save.columns = ['Distance_bp', 'Avg_Saliency_Smoothed']
    data_to_save.to_csv(out_tsv, index=False, sep='\t', float_format='%.6f')
    print(f"Data saved to {out_tsv}")

    print("Generating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(df_result['center'], df_result['saliency_smoothed'], color='darkorange', alpha=0.9, linewidth=3, marker='o', markersize=4, label='Smoothed Importance')
    ax.scatter(df_result['center'], df_result['saliency_filtered'], color='red', s=10, alpha=0.4, label='Filtered Raw Points')

    ax.set_title(f'CpG Saliency Aggregated by Physical Distance (Bin Size: {args.bin_size}bp)', fontsize=14)
    ax.set_xlabel('Physical Distance from Target CpG (bp)', fontsize=12)
    ax.set_ylabel('Average Saliency (Importance)', fontsize=12)
    ax.legend(loc='upper right')
    ax.set_xlim(0, args.max_distance)

    ax_count = ax.twinx()
    ax_count.plot(df_result['center'], df_result['count'], 'o--', color='gray', alpha=0.5, label='Samples Count')
    ax_count.set_ylabel('Number of Neighbors (Count)', color='gray')
    ax_count.tick_params(axis='y', colors='gray')
    ax_count.legend(loc='center right')

    plt.tight_layout()
    plt.savefig(f"{out_base}.pdf", dpi=300)
    plt.savefig(f"{out_base}.png", dpi=300)
    print(f"Plot saved to {out_base}.png")

if __name__ == '__main__':
    main()