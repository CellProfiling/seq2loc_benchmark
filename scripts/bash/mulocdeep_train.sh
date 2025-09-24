#!/bin/bash

source ../../.env

conda activate mulocdeep_env2

cd ../../Benchmark-Models/MULocDeep/

python ../../Benchmark-Models/MULocDeep/train.py  \
    --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv\
    --mapping ../../datasets/intermediate/mulocdeep/mulocdeep_mapping.yaml \
    --lv1_input_dir "$MULOCDEEP_DATA_DIR/level1_2" \
    --lv2_input_dir "$MULOCDEEP_DATA_DIR/level1_2" \
    --model_output "$MULOCDEEP_MODELS_DIR/seq2loc_level1_3" \
    --MULocDeep_model \
    --existPSSM "$MULOCDEEP_PSSMS_DIR" \
    --numfolds 5 \
    --coarse 10 \
    --fine 5 \
    --db ../../datasets/intermediate/mulocdeep/swissprot_db/swissprot \


python ../../Benchmark-Models/MULocDeep/train.py  \
    --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --mapping ../../datasets/intermediate/mulocdeep/mulocdeep_mapping.yaml \
    --lv1_input_dir "$MULOCDEEP_DATA_DIR/level1_3" \
    --lv2_input_dir "$MULOCDEEP_DATA_DIR/level1_3" \
    --model_output "$MULOCDEEP_MODELS_DIR/seq2loc_level1_3" \
    --MULocDeep_model \
    --existPSSM "$MULOCDEEP_PSSMS_DIR"\
    --numfolds 5 \
    --coarse 8 \
    --fine 7 \
    --db ../../datasets/intermediate/mulocdeep/swissprot_db/swissprot \