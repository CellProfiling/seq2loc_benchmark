# seq2loc_benchmark

## Overview

This repository integrates protein subcellular localization annotations from HPA, OpenCell, and UniProt to create a unified training set and a highly validated test set, with the latter containing only annotations supported by at least two databases. Using these curated datasets, we train and evaluate established protein sequence-to-localization predictors (DeepLoc2, MULocDeep, LAProtT5) and systematically assess combinations of protein language models (ESM2, ESM3, ProtT5, ProtBert) and aggregation strategies (Max-Pooling, Mean-Pooling, Light-Attention, Multihead-Attention). The repository also includes code for exploratory analyses: assessing whether models attend to known functional motifs or localization signals, incorporating PPI-network information into predictions, and evaluating model generalization to pathogenic missense variant that mislocalize.

## Setup

1. **Clone the repository and its submodules:**
    ```bash
    git clone --recurse-submodules https://github.com/CellProfiling/seq2loc_benchmark.git
    cd seq2loc_benchmark
    ```

2. **Install dependencies on a virtual environment:**
    ```bash
    python -m venv <environment_name>
    source <environment_name>/bin/activate
    pip install -r requirements.txt
    ```
    - If also training DeepLoc2 or MULocDeep models navigate to those submodules and set up separate virtual environments for each

3. **Set up the environment:**
    - Edit the paths in the provided `.env` file to match your system and data locations.

## Building the datasets

This directory contains Jupyter notebooks used to generate and process the datasets for benchmarking subcellular localization prediction

### 1-generating_datasets.ipynb
Integrates localization annotations from HPA, UniProt, and OpenCell. Defines canonical protein sequences, filters and maps locations from each source, and outputs consolidated data as intermediate files for further pre-processing and splitting. Also identifies a subset of proteins (referred to as HOU) with at least one localization label supported by two or more sources.

### 2-homology_partition.ipynb
Performs homology partitioning of the benchmark datasets. Uses sequence alignment results to identify similar proteins (>40% sequence-identity) between train and test sets and applies stratified group k-fold partitioning to divide data into non-homologous train/test sets and non-homologous folds.

### 3-process_lacoste_data.ipynb
Parses and maps localization annotations of wildtype and missense variants from Lacoste et al. (2024). This data is used to evaluate whether sequence predictors generalize to mislocalized pathogenic variants.How

## Workflow Scripts

The `scripts/` directory contains scripts for data processing, feature extraction, model training, and evaluation. For each workflow, both a bash (for local use) and a SLURM (for cluster use) script are provided with equivalent functionality. Below, each unique workflow script is listed and explained.

### Generating datasets

- **get_canonical_seqs.sh**  
  Extracts canonical protein sequences from various metadata CSVs using a Python utility. Iterates over several metadata sources and pulls canonical sequence information (by Uniprot or Ensembl ID) for each. This must be run before generating datasets done by notebooks/building/

- **align.sh**  
  Runs MMseqs2 easy-search alignments between specified query and target FASTA files. Used to compute sequence similarities between datasets for filtering.

- **check.sh**  
  Similar to `align.sh` but designed to align a fixed query FASTA (e.g., "hou.fasta") against multiple target FASTA datasets. Used to double check that there is not similar sequences between train sets and test set (HOU testset).

- **cluster.sh**  
  Performs MMseqs2 all-vs-all alignments to define clusters used to form k-folds of trainingset.

- **get_plm_embeddings.sh**  
  Generates protein language model (PLM) embeddings for all datasets using a specified set of models (e.g., ESM1/2/3, ProtT5, ProtBert).

### Training and evaluating baseline models

- **deeploc2_train.sh**  
  Trains DeepLoc2 models for ProtT5 and ESM1 embedding and all levels of localization hierarchy.

- **deeploc2_get_metrics.sh**  
  Evaluates DeepLoc2 models on provided test and train sets, producing performance metrics for each model variant (e.g., ProtT5, ESM1).

- **mulocdeep_make_db.sh**  
  Builds the SwissProt sequence database using psi-blast which is later used for PSSM generation.

- **mulocdeep_make_pssms.sh**  
  Generates Position-Specific Scoring Matrices (PSSMs) for input to MULocDeep using a SwissProt blast database

- **mulocdeep_train.sh**  
  Trains MULocDeep models. Note that MULocDeep predict localiation at two levels of granularity, so we train model to predict level1-level2  labels and level1-level3 labels.

- **mulocdeep_inference.sh**  
  Performs inference on saved MULocDeep models for hou_testset.csv and save metrics.

- **random.sh**
    Computer performance of random bernoulli baseline for localization prediction where bernoulli parameters are callibrated to the training set

### Training and evaluating models from PLM-Aggregation sweep

- **sweep_train.sh**  
  Runs wandb sweep (`main.py`) for training sequence localization models that combine a PLM embedding model (ESM2, ESM3, ProtT5, ProtBert) with an aggreagtions strategy (Max-Pooling, Mean-Pooling, Light-Attention, Multihead-Attention). PLM-Aggregation parameters and other hyperparameters are define by config files in ./configs/.
  
- **sweep_inference.sh**  
  Runs inference across a sweep of trained models using the main inference scripts (`main_inference.py`) and then gathers and summarizes the results (`sweep_analysis.py`).

- **ppi_sweep_train.sh**  
  Runs wandb sweep (`main_ppi.py`) for training sequence localization models that also incorporate PPI network data witha Graph-Sage model. PLM-Aggregation parameters, graph model parameters and other hyperparameters are define by config files in ./configs/
  
- **ppi_sweep_inference.sh**  
  Runs inference across a sweep of trained PPI models using the main inference scripts (`main_inference_ppi.py`) and then gathers and summarizes the results (`sweep_analysis_ppi.py`).
  
> **Note:**  
> Scripts and noteboooks load environment variables from `.env` and output files are saved to the directories defined in your environment variables.

## Analysis Notebooks

Notebooks found notebooks/analysis combine results, analyze results and make figures.

### Analyzing benchmark datasets

- **dataset_analysis.ipynb**  
  Explores and visualizes the datasets used in the benchmarks

### Analyzing model performance

- **1-benchmark_get_laprott5.ipynb**  
  Collects results from model sweep (run in `scripts/slurm/sweep_train.sh`) that correspond to LAProtT5 architeture which uses ProtT5 PLM and Light-Attention aggregation. Saves metrics for this model configuration in Benchmark-Models/LAProtT5/output

- **2-benchmark_combine_output.ipynb**  
  Aggregates the output of baseline models, MULocDeep, DeepLoc2, LaProtT5 and random, for single-localizing proteins, multi-localizing proteins and all proteins.

- **3-benchmark_models_analysis.ipynb**  
  Analyzes and performance of baseline models and produces plots for visualization

- **sweep_models_analysis.ipynb**  
  Analyzes and performance of best performing models from the PLM-Agg sweeep and produces plots for visualization


### Downstream analysis benchmark datasets

- **stratification_analysis.ipynb**  
  Evaluates performance of the best model from the PLM-Agg sweep when proteins are stratified by protein-properts (e.g., physicochemical properties, membrane-association, multilocalizing etc.).

- **motif_analysis_1.ipynb**  
   Detects attention peaks of best performing models from the sweep. Also searches from PROSITE motifs in the HOU testset. Produces intermediate files used by `motif_analysis_2.ipynb`.

- **motif_analysis_2.ipynb**  
  Analyzes if best performing model from the sweep attends to funcitonal motifs and sorting-signals directing localization. Produces plots for visualization.

- **ppi_analysis.ipynb**  
  Analyzes the performance of models trained with PPI-network information. Compares to best performing model from the non-PPI sweep and produces plots for visualization.

- **variant_analysis.ipynb**  
  Evaluates performance of best model from PLM-Agg sweep on pathogenic missense variants (data from Lacoste et al. (2024)). Produces plots for visualization.