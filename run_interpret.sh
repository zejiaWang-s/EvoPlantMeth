#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
MODEL_H5="train_output/EvoPlantMeth_unified.h5"
ALL_SAMPLES_LIST="all_h5_val.txt"  # Used for global DNA analysis
SAMPLES_TO_TEST="samples_to_interpret.txt" # A text file containing the names of the samples you want to analyze (e.g. sample_A)

echo "============================================="
echo "1. Running Global DNA Saliency Analysis"
echo "============================================="

python scripts/EvoPlantMeth_interpret_dna.py \
    --file_list ${ALL_SAMPLES_LIST} \
    --model_path ${MODEL_H5} \
    --batch_size 512 \
    --zoom_width 30 \
    --out_dir interpret_output/dna \
    --output_prefix dna_saliency_logo

echo "============================================="
echo "2. Running CpG Context Analysis per Sample"
echo "============================================="

# Ensure you have created a 'samples_to_interpret.txt' with sample names line by line
for sample in $(cat ${SAMPLES_TO_TEST}); do
    echo "Processing CpG interpretability for sample: ${sample}"
    
    # Isolate the list for this specific sample
    sample_list="interpret_output/cpg/${sample}/${sample}_list.txt"
    mkdir -p "interpret_output/cpg/${sample}"
    grep "/${sample}/" ${ALL_SAMPLES_LIST} > ${sample_list}

    # A. Occlusion Test
    python scripts/EvoPlantMeth_interpret_cpg_occlusion.py \
        --file_list ${sample_list} \
        --model_path ${MODEL_H5} \
        --out_dir interpret_output/cpg/${sample} \
        --output_prefix ${sample}_occlusion

    # B. Extract Raw Data for Distance Analysis
    python scripts/EvoPlantMeth_extract_cpg.py \
        --file_list ${sample_list} \
        --out_dir interpret_output/cpg/${sample}/raw_arrays

    # C. SmoothGrad Saliency
    python scripts/EvoPlantMeth_interpret_cpg_saliency.py \
        --file_list ${sample_list} \
        --model_path ${MODEL_H5} \
        --out_dir interpret_output/cpg/${sample} \
        --output_prefix ${sample}_smoothgrad_scores

    # D. Aggregate Saliency by Physical Distance
    python scripts/EvoPlantMeth_analyze_cpg_distance.py \
        --saliency_scores interpret_output/cpg/${sample}/${sample}_smoothgrad_scores.npy \
        --all_dists interpret_output/cpg/${sample}/raw_arrays/all_dists.npy \
        --bin_size 10 \
        --max_distance 500 \
        --out_dir interpret_output/cpg/${sample} \
        --output_prefix ${sample}_distance_analysis
done
