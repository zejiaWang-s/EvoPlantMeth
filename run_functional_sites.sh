#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
MODEL_H5="train_output/EvoPlantMeth_unified.h5"
DATA_DIR_BASE="example_data/processed_h5" 
GFF_DIR_BASE="example_data/annotations"    # gff3
OUTPUT_DIR="functional_sites_output"
SAMPLES_TO_TEST="samples_to_interpret.txt"

mkdir -p ${OUTPUT_DIR}

for sample in $(cat ${SAMPLES_TO_TEST}); do
    echo "============================================="
    echo "Finding Functional Sites for: ${sample}"
    echo "============================================="

    sample_out_dir="${OUTPUT_DIR}/${sample}"
    mkdir -p "${sample_out_dir}"
    
    python scripts/EvoPlantMeth_find_functional_sites.py \
        --model_path ${MODEL_H5} \
        --data_dir "${DATA_DIR_BASE}/${sample}/" \
        --out_file "${sample_out_dir}/${sample}_functional_sites_top10k.tsv" \
        --gff_file "${GFF_DIR_BASE}/${sample}.gff3" \
        --batch_size 512 \
        --top_n 1000 \
        --promoter_upstream 3000 \
        --plot_top_genes 5 \
        --save_all \
        --plot_downsample_rate 0.05
done
