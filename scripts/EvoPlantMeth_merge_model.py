#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import argparse
import os
import sys
import tensorflow as tf

from EvoPlantMeth import metrics as met
from EvoPlantMeth.models import dna, cpg, joint
from EvoPlantMeth.models import utils as mod_utils

def pcc_metric(y_true, y_pred): 
    return met.pcc(y_true, y_pred[:, 0:1])

def mse_metric(y_true, y_pred): 
    return met.mse(y_true, y_pred[:, 0:1])

def mae_metric(y_true, y_pred): 
    return met.mae(y_true, y_pred[:, 0:1])

# Explicitly assign names to avoid lambda serialization errors in Keras
pcc_metric.__name__ = 'pcc'
mse_metric.__name__ = 'mse'
mae_metric.__name__ = 'mae'

def main():
    parser = argparse.ArgumentParser(description="Merge model.json and weights.h5 into a single unified model.h5")
    parser.add_argument('--json', required=True, help="Path to model.json")
    parser.add_argument('--weights', required=True, help="Path to model_weights_val.h5")
    parser.add_argument('--out', required=True, help="Path to output merged model.h5")
    parser.add_argument('--output_confidence', action='store_true', help="Set if the model outputs confidence variance")
    args = parser.parse_args()

    print(f"Loading architecture and weights...")
    
    custom_objects = mod_utils.CUSTOM_OBJECTS.copy()
    custom_objects.update({
        'CnnL2h128BN': dna.CnnL2h128BN,
        'RnnL1BN_simple': cpg.RnnL1BN_simple,
        'JointL2h512Attention': joint.JointL2h512Attention,
        'gaussian_nll_loss': met.gaussian_nll_loss,
        'pcc': pcc_metric,
        'mse': mse_metric,
        'mae': mae_metric
    })

    model_files = [args.json, args.weights]
    model = mod_utils.load_model(model_files, custom_objects=custom_objects)

    print("Compiling the unified model...")
    output_names = model.output_names

    if args.output_confidence:
        objectives = {name: met.gaussian_nll_loss for name in output_names}
        metrics_dict = {name: [mse_metric, mae_metric, pcc_metric] for name in output_names}
    else:
        objectives = {name: 'mean_squared_error' for name in output_names}
        metrics_dict = {name: [mse_metric, mae_metric, pcc_metric] for name in output_names}

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=objectives,
        metrics=metrics_dict
    )

    print(f"Saving merged model to {args.out} ...")
    model.save(args.out)
    print("Model successfully merged and saved!")

if __name__ == '__main__':
    main()