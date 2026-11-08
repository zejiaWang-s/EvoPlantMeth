#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
MERGE_SCRIPT="scripts/EvoPlantMeth_merge_model.py"
MODEL_JSON="train_output/model.json"
MODEL_WEIGHTS="train_output/model_weights_val.h5"
OUTPUT_MODEL="train_output/EvoPlantMeth_unified.h5"

echo "Starting model consolidation..."

python ${MERGE_SCRIPT} \
    --json ${MODEL_JSON} \
    --weights ${MODEL_WEIGHTS} \
    --out ${OUTPUT_MODEL} \
    --output_confidence
