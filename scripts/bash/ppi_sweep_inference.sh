#!/bin/bash


source ../../.env
source "$SEQ2LOC_ENV/bin/activate"


srun python ../../main_inference_ppi.py \
  --exp_folder $PPI_EXP_DIR \
  --embedding_folder $PLM_EMBEDDING_DIR\
  --save_folder $PPI_ANALYSIS_DIR

srun python ../../sweep_analysis_ppi.py \
  --sweep_anaysis_dir $SWEEP_ANALYSIS_DIR \
  --ppi_analysis_dir $PPI_ANALYSIS_DIR \
  --exp_folder $PPI_EXP_DIR \
  --embedding_dir $PLM_EMBEDDING_DIR