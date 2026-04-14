#!/usr/bin/env python

import h5py
import sys
import os

UNIFIED_NAME = 'unified_sample'

def rename_in_file(h5_path):
    try:
        with h5py.File(h5_path, 'a') as f:
            if 'outputs/cpg' not in f:
                return
            
            output_cpg_group = f['outputs/cpg']
            old_sample_name = None
            
            for name in list(output_cpg_group.keys()):
                old_sample_name = name
                break 

            if old_sample_name is None:
                return
            
            # Rename Output
            if old_sample_name != UNIFIED_NAME:
                output_cpg_group.move(old_sample_name, UNIFIED_NAME)

            # Rename Input
            if 'inputs/cpg' in f and old_sample_name in f['inputs/cpg']:
                input_cpg_group = f['inputs/cpg']
                if old_sample_name != UNIFIED_NAME:
                    input_cpg_group.move(old_sample_name, UNIFIED_NAME)

    except Exception as e:
        print(f"ERROR processing file {h5_path}: {e}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python rename_h5_dataset.py <file_list.txt>")
        sys.exit(1)

    file_list_path = sys.argv[1]
    if not os.path.exists(file_list_path):
        print(f"Error: File list '{file_list_path}' not found.")
        sys.exit(1)

    with open(file_list_path, 'r') as f:
        files_to_process = [line.strip() for line in f if line.strip()]

    print(f"Unifying dataset names to '{UNIFIED_NAME}' for {len(files_to_process)} files...")
    
    for h5_file in files_to_process:
        rename_in_file(h5_file)