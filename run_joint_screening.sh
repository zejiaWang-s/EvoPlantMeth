#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
SENS_DIR="functional_sites_output"      
PROF_DIR="example_data/profiles"        # methylation variance
GFF_DIR="example_data/annotations"      # GFF3 
OUTPUT_DIR="joint_screening_output"
SAMPLES_TO_TEST="samples_to_interpret.txt"

mkdir -p ${OUTPUT_DIR}

for sample in $(cat ${SAMPLES_TO_TEST}); do
    echo "============================================="
    echo "Running Joint Screening for: ${sample}"
    echo "============================================="

    sample_out_dir="${OUTPUT_DIR}/${sample}"
    mkdir -p "${sample_out_dir}"
    
    python scripts/EvoPlantMeth_joint_screening.py \
        --sens_file "${SENS_DIR}/${sample}/${sample}_functional_sites_top10k_ALL.tsv" \
        --prof_file "${PROF_DIR}/${sample}_profile.tsv" \
        --gff_file "${GFF_DIR}/${sample}.gff3" \
        --out_prefix "${sample_out_dir}/${sample}_top5" \
        --top_percent 5 \
        --downsample_rate 0.1 \
        --label_top_genes 5 \
        --log_scale
done
