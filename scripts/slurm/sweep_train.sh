#!/bin/bash

#SBATCH --job-name=sweep_train
#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 3-00:00:00
#SBATCH --error=slurm_out/sweep_%A_%a_err.log
#SBATCH --output=slurm_out/sweep_%A_%a_out.log
#SBATCH -p emmalu
#SBATCH --array=0-3

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

# Array of YAML config files
CONFIGS=(
    ../../configs/config_esm2.yaml
    ../../configs/config_esm3.yaml
    ../../configs/config_prott5.yaml
    ../../configs/config_protbert.yaml)

# Select config file for this task
CONFIG_FILE=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

srun python ../../main.py --sweep_config $CONFIG_FILE