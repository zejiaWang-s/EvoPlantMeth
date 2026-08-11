#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"

EXPORT_SCRIPT="scripts/EvoPlantMeth_eval_export.py"
EVAL_DIR="eval_output"
OUTPUT_DIR="final_bedGraphs"

mkdir -p ${OUTPUT_DIR}

for sample_path in ${EVAL_DIR}/*; do
    if [ -d "${sample_path}" ]; then
        sample=$(basename "${sample_path}")
        echo "Exporting predictions for sample: ${sample}"
        
        sample_out_dir="${OUTPUT_DIR}/${sample}"
        mkdir -p "${sample_out_dir}"
        
        python ${EXPORT_SCRIPT} \
            "${sample_path}/data.h5" \
            --out_dir "${sample_out_dir}" \
            --out_format bedGraph \
            --is_plant \
            --output_type both \
            --with_confidence
    fi
done
