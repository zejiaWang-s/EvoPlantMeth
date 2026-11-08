#!/bin/bash
#SBATCH -J model_t4
#SBATCH -p gpu_l40
#SBATCH -o out_model_t4
#SBATCH -e err_model_t4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --qos=qkliul40
#SBATCH -A qkliu_g1

export PYTHONPATH="$(pwd):$PYTHONPATH"

EVAL_SCRIPT="scripts/EvoPlantMeth_eval.py"
MODEL_JSON="train_output/model.json"
MODEL_WEIGHTS="train_output/model_weights_val.h5"
VAL_LIST="all_h5_val.txt"
OUTPUT_DIR="eval_output"

mkdir -p ${OUTPUT_DIR}


samples=$(awk -F'/' '{print $(NF-1)}' ${VAL_LIST} | sort | uniq)

for sample in $samples; do
    echo "Evaluating sample: ${sample}"
    
    sample_out_dir="${OUTPUT_DIR}/${sample}"
    mkdir -p "${sample_out_dir}"
    
    grep "/${sample}/" ${VAL_LIST} > "${sample_out_dir}/${sample}_val_h5.txt"
    
    python ${EVAL_SCRIPT} \
        $(cat "${sample_out_dir}/${sample}_val_h5.txt") \
        --model_files ${MODEL_JSON} ${MODEL_WEIGHTS} \
        --out_data "${sample_out_dir}/data.h5" \
        --out_report "${sample_out_dir}/report.tsv" \
        --is_plant \
        --output_confidence
done
