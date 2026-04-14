#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

try:
    sns.set_theme(style="white", font_scale=1.2)
except AttributeError:
    sns.set(style="white", font_scale=1.2)
    
plt.rcParams['pdf.fonttype'] = 42 

def parse_args():
    parser = argparse.ArgumentParser(description="Joint Screening Plot (KDE & Refined Scatter).")
    parser.add_argument("--sens_file", required=True, help="Input: Sensitivity TSV file (_ALL.tsv)")
    parser.add_argument("--prof_file", required=True, help="Input: Profile/Variance TSV file")
    parser.add_argument("--gff_file", required=True, help="Input: GFF3 annotation file")
    parser.add_argument("-o", "--out_prefix", required=True, help="Output prefix")
    parser.add_argument("--top_percent", type=float, default=5.0, help="Top percentage threshold (default: 5.0)")
    parser.add_argument("--context", type=str, default="ALL", help="Filter by methylation context (e.g., CG, CHG)")
    parser.add_argument("--downsample_rate", type=float, default=0.1, help="Background downsample rate")
    parser.add_argument("--label_top_genes", type=int, default=10, help="Number of top genes to label")
    parser.add_argument("--log_scale", action="store_true", help="Apply log1p to Plasticity axis")
    return parser.parse_args()

def load_gff_robust(gff_file):
    print(f"[INFO] Loading annotations from {gff_file}...")
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
            except Exception: continue
    except Exception as e:
        print(f"[WARNING] Failed to read GFF: {e}")
        return pd.DataFrame()
    return pd.DataFrame(genes, columns=['chrom', 'start', 'end', 'strand', 'gene_id'])

def annotate_sites(sites_df, gff_df, promoter_upstream=3000):
    print(f"[INFO] Annotating top sites (Gene Body + {promoter_upstream}bp Promoter)...")
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

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    
    print(f"[INFO] Loading Sensitivity data: {args.sens_file}")
    df_sens = pd.read_csv(args.sens_file, sep='\t')
    df_sens['chrom'] = df_sens['chrom'].astype(str)
    
    print(f"[INFO] Loading Profile/Variance data: {args.prof_file}")
    df_prof = pd.read_csv(args.prof_file, sep='\t', header=None, names=['chrom', 'pos', 'Plasticity', 'context'])
    df_prof['chrom'] = df_prof['chrom'].astype(str)
    
    if args.context != "ALL":
        print(f"[INFO] Filtering for context: {args.context}")
        df_prof = df_prof[df_prof['context'] == args.context]
    
    print("[INFO] Merging datasets...")
    df_merge = pd.merge(df_sens, df_prof, on=['chrom', 'pos'], how='inner')
    if df_merge.empty:
        print("[ERROR] Merged dataset is empty. Check coordinate formats.")
        sys.exit(1)
        
    print(f"[INFO] Total valid sites after merging: {len(df_merge)}")

    df_merge['Plasticity'] = pd.to_numeric(df_merge['Plasticity'], errors='coerce')
    df_merge = df_merge.dropna(subset=['sensitivity', 'Plasticity'])

    if args.log_scale:
        df_merge['Plasticity'] = np.log1p(df_merge['Plasticity'])
        ylabel = 'Plasticity (Variance) [log1p]'
    else:
        ylabel = 'Plasticity (Variance)'
    xlabel = 'Sensitivity (Gradient Saliency)'

    sens_thresh = np.percentile(df_merge['sensitivity'], 100 - args.top_percent)
    prof_thresh = np.percentile(df_merge['Plasticity'], 100 - args.top_percent)
    
    print(f"[INFO] Thresholds (Top {args.top_percent}%): Sensitivity > {sens_thresh:.4f}, Plasticity > {prof_thresh:.4f}")

    condition_targets = (df_merge['sensitivity'] > sens_thresh) & (df_merge['Plasticity'] > prof_thresh)
    targets = df_merge[condition_targets].copy()
    background = df_merge[~condition_targets]
    
    if args.downsample_rate < 1.0 and not background.empty:
        background = background.sample(frac=args.downsample_rate, random_state=42)
    
    print(f"[INFO] Identified {len(targets)} functional targets.")
    
    gff_df = load_gff_robust(args.gff_file)
    if not gff_df.empty and not targets.empty:
        targets['gene_id'] = annotate_sites(targets, gff_df)
    else:
        targets['gene_id'] = 'Unknown'
        
    targets.to_csv(f"{args.out_prefix}_joint_targets.tsv", sep='\t', index=False)
    
    print("[INFO] Plotting Joint Screening KDE...")
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    ax_main.scatter(background['sensitivity'], background['Plasticity'], color='#e0e0e0', s=10, alpha=0.5, label='Background')
    ax_main.scatter(targets['sensitivity'], targets['Plasticity'], color='#d62728', s=25, alpha=0.8, edgecolor='white', linewidth=0.5, label='Prime Targets')

    ax_main.axvline(sens_thresh, color='gray', linestyle='--', alpha=0.7)
    ax_main.axhline(prof_thresh, color='gray', linestyle='--', alpha=0.7)

    sns.kdeplot(x=df_merge['sensitivity'], ax=ax_top, color='black', fill=True, alpha=0.2, lw=1.5)
    ax_top.axvline(sens_thresh, color='gray', linestyle='--', alpha=0.7)
    ax_top.set_ylabel('Density')
    ax_top.tick_params(axis="x", labelbottom=False)

    sns.kdeplot(y=df_merge['Plasticity'], ax=ax_right, color='black', fill=True, alpha=0.2, lw=1.5)
    ax_right.axhline(prof_thresh, color='gray', linestyle='--', alpha=0.7)
    ax_right.set_xlabel('Density')
    ax_right.tick_params(axis="y", labelleft=False)

    if args.label_top_genes > 0 and not targets.empty:
        annotated = targets.dropna(subset=['gene_id'])
        annotated = annotated[annotated['gene_id'] != 'Intergenic']
        annotated['score'] = annotated['sensitivity'] * annotated['Plasticity']
        top_genes = annotated.sort_values('score', ascending=False).head(args.label_top_genes)
        
        texts = []
        for _, row in top_genes.iterrows():
            texts.append(ax_main.text(row['sensitivity'], row['Plasticity'], str(row['gene_id']),
                                      fontsize=9, fontweight='bold', color='black', zorder=20))
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax_main, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            print("[INFO] 'adjustText' package not found. Labels might overlap.")

    ax_main.set_xlabel(xlabel, fontsize=12)
    ax_main.set_ylabel(ylabel, fontsize=12)
    
    sns.despine(ax=ax_main)
    sns.despine(ax=ax_top, left=True, bottom=True)
    sns.despine(ax=ax_right, left=True, bottom=True)

    out_file = f"{args.out_prefix}_joint_screening_kde.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to {out_file}")

if __name__ == '__main__':
    main()