#!/bin/bash

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

python ../../Benchmark-Models/Random/random_baseline.py \
    --train_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv  \
    --test_csv ../../datasets/final/hou_testset.csv \
    --yaml_classes ../../datasets/final/hierarchical_label_set.yaml \
    --save_dir $RANDOM_OUTPUT \
    --multi

python ../../Benchmark-Models/Random/random_baseline.py \
    --train_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv  \
    --test_csv ../../datasets/final/hou_testset.csv \
    --yaml_classes ../../datasets/final/hierarchical_label_set.yaml \
    --save_dir $RANDOM_OUTPUT \
    --single