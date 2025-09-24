#!/bin/bash


source ../../.env
source "$SEQ2LOC_ENV/bin/activate"


srun python ../../main_inference.py \
  --data_folder datasets/final \
  --exp_folder $SWEEP_EXP_DIR \
  --embedding_folder $PLM_EMBEDDING_DIR\
  --save_folder $SWEEP_ANALYSIS_DIR



srun python ../../sweep_analysis.py \
  --sweep_analysis_dir $SWEEP_ANALYSIS_DIR \