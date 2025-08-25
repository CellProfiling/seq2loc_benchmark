#!/bin/bash

#SBATCH -c 64 # Not sure how many CPUs needed
#SBATCH --mem=64GB
#SBATCH -t 3-00:00:00
#SBATCH --error=slurm_out/%A_%a_err.log
#SBATCH --output=slurm_out/%A_%a_out.log
#SBATCH -p emmalu

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"


python ../../main_inference.py \
  --data_folder datasets/final \
  --exp_folder $SWEEP_EXP_DIR \
  --embedding_folder $PLM_EMBEDDING_DIR\
  --save_folder $SWEEP_ANALYSIS_DIR


python ../../gather_sweep_metrics.py \
  --exp_folder $SWEEP_EXP_DIR \
  --sweep_save_folder $SWEEP_ANALYSIS_DIR\