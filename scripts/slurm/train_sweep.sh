#!/bin/bash

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
    ../../wandb_configs/sweep_esm2.yaml
    ../../wandb_configs/sweep_esm2.yaml
    ../../wandb_configs/sweep_prott5.yaml
    ../../wandb_configs/sweep_protbert.yaml)

# Select config file for this task
CONFIG_FILE=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

python ../../main.py --sweep_config $CONFIG_FILE