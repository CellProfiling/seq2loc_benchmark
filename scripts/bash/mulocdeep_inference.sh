#!/bin/bash

source ../../.env

conda activate mulocdeep_env2

cd ../../Benchmark-Models/MULocDeep/

python inference.py \
    --crossval_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --test_csv ../../datasets/final/hou_testset.csv \
    --model_dir "$MULOCDEEP_MODELS_DIR/seq2loc_level1_3" \
    --existPSSM "$MULOCDEEP_PSSMS_DIR" \
    --savedir "$MULOCDEEP_MODELS_DIR/seq2loc_level1_3/outputs" \
    --gpu \
    --numfolds 5 \
    --numclasses 22 \
    --id_col uniprot_id \
    --level level1_3 \
    --multi \


python inference.py \
    --crossval_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --test_csv ../../datasets/final/hou_testset.csv \
    --model_dir "$MULOCDEEP_MODELS_DIR/seq2loc_level1_2" \
    --existPSSM "$MULOCDEEP_PSSMS_DIR"\
    --savedir "$MULOCDEEP_MODELS_DIR/seq2loc_level1_2/outputs" \
    --gpu \
    --numfolds 5 \
    --numclasses 22 \
    --id_col uniprot_id \
    --level level1_2 \
    --multi \