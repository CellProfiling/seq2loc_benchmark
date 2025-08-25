#!/bin/bash

# Load environment variables
source ../../.env
source "$DEEPLOC2_ENV/bin/activate"

cd ../../Benchmark-Models/DeepLoc2/

python get_metrics.py \
  --categories_yaml ../../datasets/final/hierarchical_label_set.yaml \
  --testset ../../datasets/final/hou_testset.csv \
  --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
  --outdir "$DEEPLOC2_OUTPUT_DIR/prott5"

python get_metrics.py \
  --categories_yaml ../../datasets/final/hierarchical_label_set.yaml \
  --testset ../../datasets/final/hou_testset.csv \
  --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
  --outdir "$DEEPLOC2_OUTPUT_DIR/esm1"