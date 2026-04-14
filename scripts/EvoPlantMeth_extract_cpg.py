#!/usr/bin/env python

import numpy as np
import h5py
import argparse
import sys
import os

def read_file_list(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist."); sys.exit(1)
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description='Extract CpG State and Dist arrays from HDF5 files.')
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='cpg_data_raw')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_files = read_file_list(args.file_list)
    print(f"Processing {len(hdf5_files)} files...")

    all_states, all_dists = [], []

    for i, file_path in enumerate(hdf5_files):
        try:
            with h5py.File(file_path, 'r') as hf:
                state = hf['inputs/cpg/unified_sample/state'][:]
                dist = hf['inputs/cpg/unified_sample/dist'][:]
                
                if state.ndim == 3:
                    state = state[:, 0, :]
                    dist = dist[:, 0, :]

                all_states.append(state)
                all_dists.append(dist)
                
                sys.stdout.write(f"\rProgress: {i+1}/{len(hdf5_files)}")
                sys.stdout.flush()

        except Exception as e:
            print(f"\nWarning: Skipping {file_path}. Error: {e}", file=sys.stderr)

    if not all_states:
        print("\nError: No data extracted."); sys.exit(1)

    final_states = np.vstack(all_states)
    final_dists = np.vstack(all_dists)

    print(f"\nExtraction complete. Total samples: {final_states.shape[0]}")
    np.save(os.path.join(args.out_dir, 'all_states.npy'), final_states)
    np.save(os.path.join(args.out_dir, 'all_dists.npy'), final_dists)
    print(f"Data saved to {args.out_dir}/")

if __name__ == '__main__':
    main()