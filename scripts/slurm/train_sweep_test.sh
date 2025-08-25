#!/bin/bash

#SBATCH --gpus=1
#SBATCH -C GPU_MEM:80GB
#SBATCH -c 8
#SBATCH --mem=64GB
#SBATCH -t 3-00:00:00
#SBATCH --error=slurm_out/sweep_%A_err.log
#SBATCH --output=slurm_out/sweep_%A_out.log
#SBATCH -p emmalu

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

python ../../main.py --sweep_config ../../wandb_configs/sweep_prott5.yaml
