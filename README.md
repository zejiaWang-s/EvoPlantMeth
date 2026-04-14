# EvoPlantMeth

EvoPlantMeth is a deep learning framework adapted from DeepCpG, specifically optimized for modeling plant DNA methylation across different contexts (CG, CHG, and CHH). 

## Repository Structure

The code is organized into a modular structure to separate core model definitions from executable scripts.



# Installation
To ensure full reproducibility and avoid package conflicts, please create the specific Conda environment provided in this repository.

## Create the environment from the provided yaml file
```bash
conda env create -f environment.yml
```
## Activate the environment
```bash
conda activate EvoPlantMeth_env
```

# 1. Data Preparation
Before training or testing the model, raw methylation profiles (.CGmap or bedGraph format) and reference genome sequences (.fasta) must be converted into the HDF5 format required by EvoPlantMeth.

## Step 1.1: Organize your raw data
Ensure your raw data is organized in the following structure for the batch processing script:
```text
example_data/raw_profiles/
├── sample_A/
│   ├── sample_A_2.CGmap
│   └── GWHDRIN00000001.fasta
└── sample_B/
    ├── sample_B_2.CGmap
    └── Chr1.fasta
```

## Step 1.2: Run the data preparation script
We provide a bash script to process multiple samples automatically. By default, it uses a DNA sequence window of 1001 bp and extracts 50 neighboring cytosines.

Note on Plant Genomes: The --is_plant flag is enabled by default in our scripts. Unlike mammalian models, this flag ensures the extraction of sequence windows and neighboring contexts without strictly forcing a CG dinucleotide center, which is essential for capturing CHG and CHH contexts in plant epigenomes.

Save the following script as run_data_prep.sh in the root directory and execute it (bash run_data_prep.sh):
```bash
bash run_data_prep.sh
```





# 2. Model Training

Once the raw data is converted into HDF5 format, you can train the EvoPlantMeth model. 

## Step 2.1: Prepare Data Splits
```bash
run_split_data.sh
```

Before training, please prepare two text files containing the absolute or relative paths to your chunked `.h5` files (one file path per line):
* `all_h5_train.txt`: File paths for the training dataset.
* `all_h5_val.txt`: File paths for the validation dataset.

## Step 2.2: Run the training script
We provide a wrapper script (`run_train.sh`) to automate the training pipeline. 

**Note on Dataset Unification:** Because raw samples have different identifiers, the wrapper script first utilizes `rename_h5_dataset.py` to unify all internal HDF5 dataset names to `unified_sample`. This step is crucial for the deep learning model to correctly concatenate heterogeneous samples into unified training batches.

The script utilizes the following core sub-modules:
* **DNA Model:** `CnnL2h128BN` (Extracts sequence motifs)
* **CpG Model:** `RnnL1BN_simple` (Captures neighboring methylation context)
* **Joint Model:** `JointL2h512Attention` (Fuses DNA and CpG representations with attention mechanisms)

Execute the training process:

```bash
bash run_train.sh
```
Outputs
During training, the model states and validation metrics will be saved in the ./train_output directory.
* model.json: The compiled architecture.
* model_weights_val.h5: The best weights preserved via early stopping.
* lc_train.tsv & lc_val.tsv: Learning curves and metrics.


# 3. Model Evaluation

After training, you can evaluate the model's performance on the unseen validation dataset. 

## Step 3.1: Run the evaluation script
We provide a wrapper script (`run_eval.sh`) that automatically parses `all_h5_val.txt`, groups the HDF5 files by sample identity, and evaluates the model's predictions sample by sample.

Execute the evaluation process:

```bash
bash run_eval.sh
```

# 4. Exporting Predictions

The final step is to export the evaluated HDF5 predictions into standard bioinformatics formats (like `bedGraph`), allowing for visualization in tools like IGV or downstream differential methylation analysis.

## Step 4.1: Run the export script
We provide a wrapper script (`run_export.sh`) that iterates through the evaluation outputs and converts the `.h5` files into `.bedGraph.gz` files.

Execute the export process:

```bash
bash run_export.sh
```

Outputs

For each sample, the script will generate files in the final_bedGraphs/ directory:

* _pred_only.bedGraph.gz: Contains the pure model predictions for all sites.

* .bedGraph.gz (Combined): Contains experimentally observed values where available, and model predictions where missing.

* _confidence.bedGraph.gz (Optional): If --with_confidence was used during training and evaluation, this file will contain the predicted variance (uncertainty) for each site.


# 5. Model Consolidation

For downstream tasks, independent feature extraction, or deploying the model, you might prefer a single, self-contained Keras `.h5` file rather than managing the `model.json` architecture and `model_weights_val.h5` weights separately.

We provide a utility script to compile and merge these components into a single standalone model file.

## Step 5.1: Run the merge script

Execute the provided wrapper script:
```bash
bash run_merge.sh
```
Outputs

This will generate EvoPlantMeth_unified.h5 in your target directory. This file contains the complete model architecture, the best validation weights, and the compiled custom loss functions/metrics, making it fully ready for direct tensorflow.keras.models.load_model() calls in your custom downstream pipelines.

# 6. Interpretability Analysis

Understanding *why* a deep learning model makes specific predictions is vital in genomics. We provide a suite of interpretability scripts to investigate the learned motifs in DNA sequences and the spatial influence of neighboring CpG sites.

Before proceeding, ensure you have consolidated your model using the step above, yielding `EvoPlantMeth_unified.h5`. Create a text file named `samples_to_interpret.txt` containing the names of the samples you wish to analyze individually (one per line).

## Step 6.1: Run the Interpretability Pipeline
Execute the wrapper script to run all tests automatically:
```bash
bash run_interpret.sh
```

Analysis Modules included in the pipeline:

* DNA Sequence Saliency (EvoPlantMeth_interpret_dna.py)
Calculates gradients of the predictions with respect to the input DNA sequence (Saliency Maps). It automatically splits the results into distinct biological contexts (CG, CHG, CHH contexts) and plots high-resolution Sequence Logos illustrating the learned sequence motifs using logomaker. Output: interpret_output/dna/ (.png and .pdf logos)

* CpG Context Occlusion Test (EvoPlantMeth_interpret_cpg_occlusion.py)
Iteratively masks out neighboring cytosines and measures the drop in prediction accuracy. This identifies the relative positional importance of structural neighbors.Output: interpret_output/cpg/{sample}/ (Occlusion bar charts)

* SmoothGrad Saliency (EvoPlantMeth_interpret_cpg_saliency.py)
Applies the SmoothGrad technique (adding noise to inputs and averaging gradients) to generate robust importance scores for the surrounding methylation environment. Output: interpret_output/cpg/{sample}/ (.npy score arrays)

* Physical Distance Mapping (EvoPlantMeth_analyze_cpg_distance.py)
Correlates the computed SmoothGrad saliency scores with the actual, physical genomic distance (in base pairs) of the neighboring sites. This generates distribution plots that reveal how spatial decay influences epigenetic correlations. Output: interpret_output/cpg/{sample}/ (Distance decay plots and raw .tsv data files)


# 7. Double-Filtering Framework: Dimension 1 (Local Influence)

The first dimension of the EvoPlantMeth Double-Filtering Framework isolates functional methylation sites through **In Silico Saturation Mutagenesis**. 

By calculating the sensitivity (gradients) of the model's predictions with respect to specific local methylation states, we can identify regulatory loci that exert a profound local epigenetic influence.

## Step 7.1: Run the Functional Sites Screening
Ensure your reference `.gff3` annotation files are placed in an accessible directory (e.g., `example_data/annotations/`) so the script can map high-impact sites to specific gene promoters and bodies.

Execute the wrapper script:
```bash
bash run_functional_sites.sh
```

Outputs
For each sample, the script generates outputs in the functional_sites_output/{sample}/ directory:

* _functional_sites_top10k.tsv: A sorted list of the top N most influential methylation sites, annotated with their associated genes.
* _ALL.tsv (Optional): The uncompressed gradient sensitivity data for all evaluated loci.
* _manhattan.pdf: A genome-wide Manhattan plot displaying the saliency scores, visually labeling the most influential functional genes.

## Step 7.2: Dimension 2 (Joint Screening)

While Step 7.1 isolates sites based on their local regulatory impact (Sensitivity), true functional plasticity must also be reflected in the model's overall predicted variance for that site. 

The Joint Screening script merges the gradient sensitivity scores with the model-predicted methylation variance (Plasticity Profile). By isolating the upper right quadrant (e.g., Top 5% in both metrics), we identify the most highly robust functional methylation sites.

Ensure you have your variance profile data (e.g., `sample_profile.tsv` outputted directly from the model) ready in the designated profile directory.

Execute the screening process:
```bash
bash run_joint_screening.sh
```
Outputs

For each sample, the script generates outputs in the joint_screening_output/{sample}/ directory:

* _joint_targets.tsv: A comprehensive list of the prime functional targets that passed both the sensitivity and plasticity thresholds, automatically annotated with their nearest gene IDs.
* _joint_screening_kde.pdf / .png: A high-quality, publication-ready scatter plot featuring marginal Kernel Density Estimations (KDE). The background is grayed out, while the top-tier functional candidates are highlighted in red and labeled by gene name.


## Acknowledgments 
EvoPlantMeth is adapted and heavily modified from the foundational work of DeepCpG. We sincerely thank Christof Angermueller and the original authors for open-sourcing their code.

Reference: Angermueller, Christof et al. “DeepCpG: accurate prediction of single-cell DNA methylation states using deep learning.” Genome biology vol. 18,1 67. 11 Apr. 2017, doi:10.1186/s13059-017-1189-z