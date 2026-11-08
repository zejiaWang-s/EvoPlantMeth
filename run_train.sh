#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"
# 1. Unify dataset names inside HDF5 files
echo "Unifying HDF5 internal dataset names..."
python scripts/rename_h5_dataset.py all_h5_train.txt
python scripts/rename_h5_dataset.py all_h5_val.txt

# 2. Run the training script
echo "Starting model training..."
python scripts/EvoPlantMeth_train.py \
    --train_file_list ./all_h5_train.txt \
    --val_file_list ./all_h5_val.txt \
    --dna_model CnnL2h128BN \
    --cpg_model RnnL1BN_simple \
    --joint_model JointL2h512Attention \
    --replicate_names 'unified_sample' \
    --output_names 'cpg/unified_sample' \
    --nb_epoch 100 \
    --early_stopping 10 \
    --learning_rate 0.0005 \
    --batch_size 512 \
    --dropout 0.6 \
    --l2_decay 1e-3 \
    -o ./train_output \
    --is_plant \
    --output_confidence \
    --gpus 8
