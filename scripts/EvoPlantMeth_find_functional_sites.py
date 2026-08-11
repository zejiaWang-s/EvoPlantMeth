#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import argparse
import glob
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from EvoPlantMeth import metrics as met
from EvoPlantMeth import data as dat
from EvoPlantMeth.models import utils as mod_utils
from EvoPlantMeth.models import dna, cpg, joint


def get_args():
    parser = argparse.ArgumentParser(
        description="Find functional methylation sites via directional gradient sensitivity."
    )
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
        predictions = model([dna_in, cpg_state_in, cpg_dist_in], training=False)
        pred_mean = predictions[:, 0]
    return tape.gradient(pred_mean, cpg_state_in)


def normalize_neighbor_matrix(x, name):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]
    if x.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D after normalization; observed shape={x.shape}"
        )
    return x


def valid_neighbor_mask(state, dist):
    valid = np.isfinite(state) & np.isfinite(dist)
    try:
        nan_value = float(dat.CPG_NAN)
        valid &= (state != nan_value)
        valid &= (dist != nan_value)
    except Exception:
        pass
    valid &= (dist > 0)
    return valid


def get_valid_sample_name(hf):
    if 'inputs/cpg' not in hf:
        return None
    input_group = hf['inputs/cpg']
    for key in input_group.keys():
        if 'state' in input_group[key] and 'dist' in input_group[key]:
            return key
    return None


def build_global_reference(h5_files):
    pos_parts = defaultdict(list)
    value_parts = defaultdict(list)
    n_neighbors = None

    print("Pre-scanning H5 files and building genomic coordinate reference...")

    for idx, h5_path in enumerate(h5_files, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(h5_files):
            print(f"  Reference scan: {idx}/{len(h5_files)}", flush=True)

        with h5py.File(h5_path, 'r') as hf:
            sample = get_valid_sample_name(hf)
            if sample is None:
                continue

            chroms = hf['chromo'][:].astype(str)
            poss = hf['pos'][:].astype(np.int64)
            state = normalize_neighbor_matrix(
                hf[f'inputs/cpg/{sample}/state'][:], 'state'
            )

            if n_neighbors is None:
                n_neighbors = state.shape[1]
            elif state.shape[1] != n_neighbors:
                raise ValueError(
                    f"Inconsistent neighbor number: {state.shape[1]} in "
                    f"{h5_path}, expected {n_neighbors}"
                )

            out_path = f'outputs/cpg/{sample}'
            if out_path not in hf:
                raise ValueError(
                    f"Missing {out_path} in {h5_path}; observed methylation "
                    f"values are required to validate neighbor mapping."
                )

            values = hf[out_path][:].astype(np.float32)

            if len(values) != len(poss):
                raise ValueError(
                    f"Position/output length mismatch in {h5_path}: "
                    f"{len(poss)} vs {len(values)}"
                )

            for chrom in np.unique(chroms):
                mask = chroms == chrom
                pos_parts[str(chrom)].append(poss[mask])
                value_parts[str(chrom)].append(values[mask])

    if n_neighbors is None:
        raise RuntimeError("No valid CpG input was found in the H5 files.")

    if n_neighbors % 2 != 0:
        raise ValueError(
            f"Expected an even number of neighbors; observed {n_neighbors}"
        )

    reference = {}

    for chrom in pos_parts:
        pos = np.concatenate(pos_parts[chrom]).astype(np.int64)
        val = np.concatenate(value_parts[chrom]).astype(np.float32)

        order = np.argsort(pos, kind='mergesort')
        pos = pos[order]
        val = val[order]

        unique_pos, unique_idx = np.unique(pos, return_index=True)
        pos = unique_pos
        val = val[unique_idx]

        reference[chrom] = {
            'pos': pos,
            'value': val,
            'sum_abs_grad': np.zeros(len(pos), dtype=np.float64),
            'count': np.zeros(len(pos), dtype=np.int32),
        }

    print(
        f"Detected {n_neighbors} neighboring cytosines per target "
        f"(expected from --cpg_wlen 50)."
    )

    return reference, n_neighbors


def lookup_reference(ref_pos, ref_val, query_pos):
    query_pos = np.asarray(query_pos, dtype=np.int64)
    idx = np.searchsorted(ref_pos, query_pos)
    in_bounds = idx < len(ref_pos)

    found = np.zeros(len(query_pos), dtype=bool)
    found[in_bounds] = ref_pos[idx[in_bounds]] == query_pos[in_bounds]

    values = np.full(len(query_pos), np.nan, dtype=np.float32)
    values[found] = ref_val[idx[found]]

    return found, values, idx


def infer_neighbor_directions(h5_files, reference, n_neighbors,
                              max_files=20, max_rows_per_file=1000):
    """
    Infer whether each neighbor slot corresponds to the left or right side of
    the target. Both genomic coordinate and methylation value must match.
    """
    minus_tested = np.zeros(n_neighbors, dtype=np.int64)
    plus_tested = np.zeros(n_neighbors, dtype=np.int64)
    minus_match = np.zeros(n_neighbors, dtype=np.int64)
    plus_match = np.zeros(n_neighbors, dtype=np.int64)

    rng = np.random.default_rng(42)

    print("Inferring left/right direction of each neighbor slot...")

    files_used = 0

    for h5_path in h5_files:
        if files_used >= max_files:
            break

        with h5py.File(h5_path, 'r') as hf:
            sample = get_valid_sample_name(hf)
            if sample is None:
                continue

            chroms = hf['chromo'][:].astype(str)
            poss = hf['pos'][:].astype(np.int64)

            state = normalize_neighbor_matrix(
                hf[f'inputs/cpg/{sample}/state'][:], 'state'
            )
            dist = normalize_neighbor_matrix(
                hf[f'inputs/cpg/{sample}/dist'][:], 'dist'
            )

            if len(poss) == 0:
                continue

            rows = np.arange(len(poss))
            if len(rows) > max_rows_per_file:
                rows = rng.choice(rows, size=max_rows_per_file, replace=False)

            for chrom in np.unique(chroms[rows]):
                chrom_rows = rows[chroms[rows] == chrom]
                if len(chrom_rows) == 0 or str(chrom) not in reference:
                    continue

                ref_pos = reference[str(chrom)]['pos']
                ref_val = reference[str(chrom)]['value']
                target_pos = poss[chrom_rows]

                for slot in range(n_neighbors):
                    slot_state = state[chrom_rows, slot]
                    slot_dist = dist[chrom_rows, slot]

                    valid = valid_neighbor_mask(slot_state, slot_dist)
                    if not np.any(valid):
                        continue

                    tpos = target_pos[valid]
                    sstate = slot_state[valid]
                    sdist = np.rint(slot_dist[valid]).astype(np.int64)

                    minus_pos = tpos - sdist
                    plus_pos = tpos + sdist

                    minus_found, minus_val, _ = lookup_reference(
                        ref_pos, ref_val, minus_pos
                    )
                    plus_found, plus_val, _ = lookup_reference(
                        ref_pos, ref_val, plus_pos
                    )

                    minus_ok = (
                        minus_found
                        & np.isclose(minus_val, sstate, rtol=0, atol=1e-6)
                    )
                    plus_ok = (
                        plus_found
                        & np.isclose(plus_val, sstate, rtol=0, atol=1e-6)
                    )

                    minus_tested[slot] += len(sstate)
                    plus_tested[slot] += len(sstate)
                    minus_match[slot] += int(minus_ok.sum())
                    plus_match[slot] += int(plus_ok.sum())

            files_used += 1

    directions = np.zeros(n_neighbors, dtype=np.int8)

    print("\nNeighbor-slot direction validation:")
    print("slot\tminus_match\tplus_match\tdirection")

    for slot in range(n_neighbors):
        tested = max(minus_tested[slot], plus_tested[slot])

        if tested == 0:
            raise RuntimeError(
                f"Could not validate neighbor slot {slot}: no valid observations."
            )

        minus_rate = minus_match[slot] / tested
        plus_rate = plus_match[slot] / tested

        if minus_rate > plus_rate:
            directions[slot] = -1
            best_rate = minus_rate
            direction_label = 'LEFT (-)'
        else:
            directions[slot] = 1
            best_rate = plus_rate
            direction_label = 'RIGHT (+)'

        print(f"{slot}\t{minus_rate:.4f}\t{plus_rate:.4f}\t{direction_label}")

        if best_rate < 0.90:
            raise RuntimeError(
                f"Neighbor slot {slot} could not be mapped reliably "
                f"(best match rate={best_rate:.3f}). "
                f"Please inspect KnnCpgFeatureExtractor before proceeding."
            )

    n_left = int((directions == -1).sum())
    n_right = int((directions == 1).sum())

    print(
        f"\nDirection inference PASS: {n_left} left slots, "
        f"{n_right} right slots."
    )

    if n_left != n_neighbors // 2 or n_right != n_neighbors // 2:
        print(
            "WARNING: expected an equal left/right split, but the empirical "
            "mapping differs. The inferred directions will nevertheless be used."
        )

    return directions


def prepare_model_inputs(model, dna_raw, state_raw, dist_raw):
    b_dna = tf.convert_to_tensor(dna_raw, dtype=tf.int32)

    if model.input_shape[0][-1] == 4 and len(b_dna.shape) == 2:
        b_dna = tf.one_hot(b_dna, depth=4)

    b_dna = tf.cast(b_dna, dtype=tf.float32)

    b_state = tf.convert_to_tensor(state_raw, dtype=tf.float32)
    if len(b_state.shape) == 2:
        b_state = tf.expand_dims(b_state, axis=1)

    b_dist = tf.convert_to_tensor(dist_raw, dtype=tf.float32)
    if len(b_dist.shape) == 2:
        b_dist = tf.expand_dims(b_dist, axis=1)

    return b_dna, b_state, b_dist


def accumulate_batch(reference, directions, chroms, poss, state, dist, grads):
    """
    Reassign every neighbor-specific gradient to the SOURCE cytosine.

    For target j and source/neighbor i:
        gradient = d y_hat_j / d x_i

    The final score for source site i is the mean absolute gradient across all
    surrounding target sites for which i appears in the 50-neighbor input.
    """
    state = normalize_neighbor_matrix(state, 'state')
    dist = normalize_neighbor_matrix(dist, 'dist')
    grads = normalize_neighbor_matrix(grads, 'grads')

    if state.shape != dist.shape or state.shape != grads.shape:
        raise ValueError(
            f"state/dist/grads shape mismatch: "
            f"{state.shape}, {dist.shape}, {grads.shape}"
        )

    n_neighbors = state.shape[1]
    mapped = 0
    unmapped = 0

    for chrom in np.unique(chroms):
        chrom = str(chrom)

        if chrom not in reference:
            continue

        row_mask = chroms == chrom

        target_pos = poss[row_mask].astype(np.int64)
        state_sub = state[row_mask]
        dist_sub = dist[row_mask]
        grad_sub = grads[row_mask]

        ref_pos = reference[chrom]['pos']
        sum_abs_grad = reference[chrom]['sum_abs_grad']
        count = reference[chrom]['count']

        for slot in range(n_neighbors):
            slot_state = state_sub[:, slot]
            slot_dist = dist_sub[:, slot]
            slot_grad = grad_sub[:, slot]

            valid = (
                valid_neighbor_mask(slot_state, slot_dist)
                & np.isfinite(slot_grad)
            )

            if not np.any(valid):
                continue

            d = np.rint(slot_dist[valid]).astype(np.int64)
            source_pos = target_pos[valid] + int(directions[slot]) * d
            abs_grad = np.abs(slot_grad[valid]).astype(np.float64)

            idx = np.searchsorted(ref_pos, source_pos)
            in_bounds = idx < len(ref_pos)

            found = np.zeros(len(source_pos), dtype=bool)
            found[in_bounds] = ref_pos[idx[in_bounds]] == source_pos[in_bounds]

            if np.any(found):
                np.add.at(sum_abs_grad, idx[found], abs_grad[found])
                np.add.at(count, idx[found], 1)
                mapped += int(found.sum())

            unmapped += int((~found).sum())

    return mapped, unmapped


def load_gff_robust(gff_file):
    print(f"Loading annotations from {gff_file}...")
    genes = []

    try:
        df = pd.read_csv(
            gff_file, sep='\t', comment='#', header=None,
            on_bad_lines='skip', low_memory=False
        )

        target_df = df[df[2].isin(['gene', 'mRNA', 'Gene', 'transcript'])]

        for _, row in target_df.iterrows():
            try:
                attr_str = str(row[8])
                gene_id = "Unknown"

                if "ID=" in attr_str:
                    gene_id = attr_str.split("ID=")[1].split(";")[0]
                elif "Name=" in attr_str:
                    gene_id = attr_str.split("Name=")[1].split(";")[0]

                genes.append(
                    (str(row[0]), int(row[3]), int(row[4]), str(row[6]), gene_id)
                )
            except Exception:
                continue

    except Exception as e:
        print(f"Warning: Failed to read GFF: {e}")
        return pd.DataFrame()

    return pd.DataFrame(
        genes, columns=['chrom', 'start', 'end', 'strand', 'gene_id']
    )


def annotate_sites(sites_df, gff_df, promoter_upstream=3000):
    print(f"Annotating sites (Gene Body + {promoter_upstream}bp Promoter)...")

    annotations = ['Intergenic'] * len(sites_df)
    sites_df_reset = sites_df.reset_index(drop=True).copy()
    gff_df = gff_df.copy()

    gff_df['chrom'] = gff_df['chrom'].astype(str)
    sites_df_reset['chrom'] = sites_df_reset['chrom'].astype(str)

    for chrom in sites_df_reset['chrom'].unique():
        site_mask = sites_df_reset['chrom'] == chrom
        site_indices = sites_df_reset[site_mask].index
        site_positions = sites_df_reset.loc[site_indices, 'pos'].values

        gff_sub = gff_df[gff_df['chrom'] == chrom]
        if gff_sub.empty:
            continue

        g_starts = gff_sub['start'].values
        g_ends = gff_sub['end'].values
        g_strands = gff_sub['strand'].values
        g_ids = gff_sub['gene_id'].values

        ext_starts = np.where(
            (g_strands == '+') | (g_strands == '.'),
            np.maximum(0, g_starts - promoter_upstream),
            g_starts
        )

        ext_ends = np.where(
            (g_strands == '-') | (g_strands == '.'),
            g_ends + promoter_upstream,
            g_ends
        )

        for idx, pos in zip(site_indices, site_positions):
            matches = np.where(
                (ext_starts <= pos) & (ext_ends >= pos)
            )[0]

            if len(matches) > 0:
                annotations[idx] = g_ids[matches[0]]

    return annotations


def plot_manhattan(
    df,
    top_n_threshold,
    gff_df,
    top_genes_count,
    out_file,
    promoter_dist,
    downsample_rate
):
    print(f"Generating Manhattan plot (Downsample Rate: {downsample_rate})...")

    plot_file = out_file.replace('.tsv', '_manhattan.pdf')
    df = df.copy()

    def parse_chrom(c):
        digits = re.findall(r'\d+', str(c))
        return int(digits[0]) if digits else 999

    df['chrom_idx'] = df['chrom'].apply(parse_chrom)
    df = df.sort_values(by=['chrom_idx', 'pos'])

    chrom_order = sorted(df['chrom'].unique(), key=parse_chrom)

    plt.figure(figsize=(14, 6))

    colors = ['#e6e6e6', '#cccccc']
    x_ticks = []
    x_labels = []
    current_x_offset = 0

    sorted_sens = df.sort_values('sensitivity', ascending=False)

    if len(sorted_sens) == 0:
        print("No sites available for plotting.")
        plt.close()
        return

    cutoff_idx = min(
        len(df) - 1,
        max(0, int(top_n_threshold) - 1)
    )
    threshold_val = sorted_sens.iloc[cutoff_idx]['sensitivity']

    global_pos_map = {}

    for i, chrom in enumerate(chrom_order):
        chrom_data = df[df['chrom'] == chrom]

        if chrom_data.empty:
            continue

        global_pos_map[chrom] = current_x_offset
        g_pos = chrom_data['pos'] + current_x_offset
        high_mask = chrom_data['sensitivity'] >= threshold_val

        if not high_mask.all():
            low_data = chrom_data[~high_mask]

            if 0 < downsample_rate < 1.0:
                low_data = low_data.sample(
                    frac=downsample_rate,
                    random_state=42
                )

            plt.scatter(
                low_data['pos'] + current_x_offset,
                low_data['sensitivity'],
                c=colors[i % 2],
                s=2,
                alpha=0.6,
                edgecolors='none',
                rasterized=True
            )

        if high_mask.any():
            plt.scatter(
                g_pos[high_mask],
                chrom_data.loc[high_mask, 'sensitivity'],
                c='#d62728',
                s=10,
                alpha=0.9,
                edgecolors='none',
                rasterized=True
            )

        x_ticks.append(
            current_x_offset
            + (chrom_data['pos'].max() + chrom_data['pos'].min()) / 2
        )
        x_labels.append(chrom)
        current_x_offset += chrom_data['pos'].max()

    if not gff_df.empty and top_genes_count > 0:
        top_candidates = (
            df.sort_values('sensitivity', ascending=False)
            .head(top_genes_count * 3)
            .copy()
        )

        top_candidates['gene_label'] = annotate_sites(
            top_candidates, gff_df, promoter_dist
        )

        labeled = 0

        for _, row in top_candidates.iterrows():
            if labeled >= top_genes_count:
                break
            if row['gene_label'] == 'Intergenic':
                continue

            g_x = row['pos'] + global_pos_map[row['chrom']]
            g_y = row['sensitivity']

            plt.annotate(
                row['gene_label'],
                xy=(g_x, g_y),
                xytext=(g_x, g_y + (df['sensitivity'].max() * 0.05)),
                arrowprops=dict(
                    facecolor='black',
                    arrowstyle="->",
                    lw=0.5
                ),
                fontsize=8,
                fontweight='bold',
                ha='center',
                color='black'
            )

            labeled += 1

    plt.xticks(x_ticks, x_labels, fontsize=8, rotation=45)
    plt.ylabel("Regulatory Impact Score (0-1)", fontsize=12)
    plt.title(
        f"Functional Methylation Sites (Top {top_genes_count} Labeled)\n"
        f"Region: Gene Body + {promoter_dist}bp Promoter",
        fontsize=12
    )
    plt.axhline(
        y=threshold_val,
        color='blue',
        linestyle='--',
        linewidth=0.8,
        alpha=0.5
    )

    if downsample_rate < 1.0:
        plt.text(
            0,
            plt.ylim()[1],
            f"Background downsampled to {downsample_rate * 100}%",
            fontsize=6,
            color='gray',
            va='top',
            ha='left'
        )

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    print(f"Manhattan plot saved to {plot_file}")
    plt.close()


def main():
    args = get_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    def pcc_metric(y_true, y_pred):
        return met.pcc(y_true, y_pred[:, 0:1])

    def mse_metric(y_true, y_pred):
        return met.mse(y_true, y_pred[:, 0:1])

    def mae_metric(y_true, y_pred):
        return met.mae(y_true, y_pred[:, 0:1])

    pcc_metric.__name__ = 'pcc'
    mse_metric.__name__ = 'mse'
    mae_metric.__name__ = 'mae'

    custom_objects = mod_utils.CUSTOM_OBJECTS.copy()
    custom_objects.update({
        'CnnL2h128BN': dna.CnnL2h128BN,
        'RnnL1BN_simple': cpg.RnnL1BN_simple,
        'JointL2h512Attention': joint.JointL2h512Attention,
        'gaussian_nll_loss': met.gaussian_nll_loss,
        'pcc': pcc_metric,
        'mse': mse_metric,
        'mae': mae_metric
    })

    print(f"Loading model: {args.model_path}")

    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects=custom_objects,
            compile=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    h5_files = sorted(glob.glob(os.path.join(args.data_dir, "*.h5")))

    if not h5_files:
        sys.exit(f"No h5 files in {args.data_dir}")

    reference, n_neighbors = build_global_reference(h5_files)

    directions = infer_neighbor_directions(
        h5_files=h5_files,
        reference=reference,
        n_neighbors=n_neighbors
    )

    print(
        f"\nScanning {len(h5_files)} files "
        f"for directional regulatory impact..."
    )

    total_targets = 0
    total_mapped = 0
    total_unmapped = 0

    for f_idx, h5_path in enumerate(h5_files, start=1):
        print(
            f"[{f_idx}/{len(h5_files)}] {os.path.basename(h5_path)}",
            flush=True
        )

        try:
            with h5py.File(h5_path, 'r') as hf:
                sample = get_valid_sample_name(hf)

                if sample is None:
                    print("  Skipped: no valid inputs/cpg group.")
                    continue

                dna_raw = hf['inputs/dna'][:]

                state_raw = normalize_neighbor_matrix(
                    hf[f'inputs/cpg/{sample}/state'][:],
                    'state'
                )

                dist_raw = normalize_neighbor_matrix(
                    hf[f'inputs/cpg/{sample}/dist'][:],
                    'dist'
                )

                chroms = hf['chromo'][:].astype(str)
                poss = hf['pos'][:].astype(np.int64)

                if state_raw.shape[1] != n_neighbors:
                    raise ValueError(
                        f"Unexpected neighbor number in {h5_path}: "
                        f"{state_raw.shape[1]}"
                    )

                num_samples = len(poss)

                if num_samples == 0:
                    continue

                total_targets += num_samples

                for start in range(0, num_samples, args.batch_size):
                    end = min(start + args.batch_size, num_samples)
                    sl = slice(start, end)

                    b_dna, b_state, b_dist = prepare_model_inputs(
                        model=model,
                        dna_raw=dna_raw[sl],
                        state_raw=state_raw[sl],
                        dist_raw=dist_raw[sl]
                    )

                    grads = compute_gradients(
                        model,
                        [b_dna, b_state, b_dist]
                    )

                    if grads is None:
                        raise RuntimeError(
                            "Gradient calculation returned None."
                        )

                    mapped, unmapped = accumulate_batch(
                        reference=reference,
                        directions=directions,
                        chroms=chroms[sl],
                        poss=poss[sl],
                        state=state_raw[sl],
                        dist=dist_raw[sl],
                        grads=grads.numpy()
                    )

                    total_mapped += mapped
                    total_unmapped += unmapped

                    if (
                        start == 0
                        or end == num_samples
                        or (start // args.batch_size) % 20 == 0
                    ):
                        print(
                            f"  targets {end:,}/{num_samples:,}",
                            flush=True
                        )

        except Exception as e:
            print(f"ERROR while processing {h5_path}: {e}")
            raise

    print("\nGradient scan complete.")
    print(f"Total target cytosines: {total_targets:,}")
    print(
        f"Mapped neighbor-gradient contributions: "
        f"{total_mapped:,}"
    )
    print(f"Unmapped contributions: {total_unmapped:,}")

    if (
        total_mapped > 0
        and total_unmapped / (total_mapped + total_unmapped) > 0.01
    ):
        raise RuntimeError(
            "More than 1% of neighbor gradients could not be mapped "
            "back to source cytosines. Stop to avoid producing "
            "incorrect regulatory-impact scores."
        )

    rows = []

    for chrom in sorted(reference.keys()):
        ref = reference[chrom]
        valid = ref['count'] > 0

        if not np.any(valid):
            continue

        # Raw directional regulatory impact: mean absolute outgoing gradient.
        raw_score = (
            ref['sum_abs_grad'][valid]
            / ref['count'][valid]
        )

        # Bound the final score to [0, 1) with a monotonic tanh transform.
        # tanh(x) is approximately x near zero, so the previous practical
        # interpretation around 0.1 is essentially preserved
        # (tanh(0.1) = 0.09967).
        score = np.tanh(raw_score)

        # Preserve the original output schema for downstream compatibility.
        rows.append(
            pd.DataFrame({
                'chrom': chrom,
                'pos': ref['pos'][valid],
                'sensitivity': score
            })
        )

    if not rows:
        sys.exit("No regulatory-impact sites were generated.")

    df = pd.concat(rows, ignore_index=True)

    print(
        f"\nGenerated regulatory-impact scores "
        f"for {len(df):,} source cytosines."
    )
    print("\nNormalized regulatory-impact summary (0-1):")
    print(df['sensitivity'].describe())

    if args.save_all:
        all_sites_file = args.out_file.replace('.tsv', '_ALL.tsv')
        print(f"Saving ALL sites to {all_sites_file} ...")
        df.to_csv(all_sites_file, index=False, sep='\t')

    gff_df = (
        load_gff_robust(args.gff_file)
        if args.gff_file and os.path.exists(args.gff_file)
        else pd.DataFrame()
    )

    plot_manhattan(
        df=df,
        top_n_threshold=args.top_n,
        gff_df=gff_df,
        top_genes_count=(
            args.plot_top_genes if not gff_df.empty else 0
        ),
        out_file=args.out_file,
        promoter_dist=args.promoter_upstream,
        downsample_rate=args.plot_downsample_rate
    )

    print(f"Sorting Top {args.top_n} for output...")

    df_top = (
        df.sort_values(by='sensitivity', ascending=False)
        .head(args.top_n)
        .copy()
    )

    if not gff_df.empty:
        df_top['annotation'] = annotate_sites(
            df_top,
            gff_df,
            args.promoter_upstream
        )

    print(f"Saving TSV to {args.out_file}")
    df_top.to_csv(args.out_file, index=False, sep='\t')
    print("Done!")


if __name__ == '__main__':
    main()
