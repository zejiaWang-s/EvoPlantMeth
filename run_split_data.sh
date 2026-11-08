#!/bin/bash
INPUT_DIR="example_data/processed_h5"
TRAIN_OUT="all_h5_train.txt"
VAL_OUT="all_h5_val.txt"

> "${TRAIN_OUT}"
> "${VAL_OUT}"

echo "Starting 80/20 train-validation split..."


for sample_dir in ${INPUT_DIR}/*; do
    if [ -d "${sample_dir}" ]; then
        sample=$(basename "${sample_dir}")
        echo "Processing sample: ${sample}"
        mapfile -t files < <(find "$PWD/${sample_dir}" -name "*.h5" | shuf)
        
        total_files=${#files[@]}
        
        if [ "$total_files" -eq 0 ]; then
            echo "  Warning: No .h5 files found in ${sample}, skipping."
            continue
        fi

        train_count=$(( total_files * 80 / 100 ))
        val_count=$(( total_files - train_count ))
        
        echo "  Total files: ${total_files} | Train: ${train_count} | Val: ${val_count}"
        for (( i=0; i<total_files; i++ )); do
            if [ "$i" -lt "$train_count" ]; then
                echo "${files[$i]}" >> "${TRAIN_OUT}"
            else
                echo "${files[$i]}" >> "${VAL_OUT}"
            fi
        done
    fi
done

echo "========================================="
echo "Data splitting completed successfully!"
echo "Train list: $(pwd)/${TRAIN_OUT} ($(wc -l < ${TRAIN_OUT}) files)"
echo "Val list:   $(pwd)/${VAL_OUT} ($(wc -l < ${VAL_OUT}) files)"
