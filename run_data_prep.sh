#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. Define paths 
DATA_SCRIPT="scripts/EvoPlantMeth_data.py"
INPUT_DIR="example_data/raw_profiles"
OUTPUT_DIR="example_data/processed_h5"

# 2. Find all CGmap files in the input directory
for cpg_file in ${INPUT_DIR}/*/*.CGmap; do
    # Extract the sample name from the file path
    sample_name=$(basename "$(dirname "$cpg_file")")
    
    echo "Processing sample: ${sample_name}"
    
    # Create an output directory for this specific sample
    sample_out_dir="${OUTPUT_DIR}/${sample_name}"
    mkdir -p "${sample_out_dir}"
    
    # Run the EvoPlantMeth data preparation script
    python ${DATA_SCRIPT} \
        --cpg_profiles ${cpg_file} \
        --dna_files ${INPUT_DIR}/${sample_name}/*.fasta \
        --cpg_wlen 50 \
        --dna_wlen 1001 \
        --out_dir "${sample_out_dir}" \
        --is_plant \
        --chunk_size 10240
done
