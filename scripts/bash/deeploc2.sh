#!/bin/bash

# Load environment variables
source ../../.env
source "$DEEPLOC2_ENV/bin/activate"

cd ../../Benchmark-Models/DeepLoc2/

DL2_MODEL_TYPES=("seq2loc-prott5" "seq2loc-prott5" "seq2loc-prott5" "seq2loc-ems1" "seq2loc-ems1" "seq2loc-ems1")
LEVELS=(1 2 3 1 2 3)
CLIP_LENS=(4000 4000 4000 1022 1022 1022)
EMBEDDINGS=("ProtT5-4k.h5" "ProtT5-4k.h5" "ProtT5-4k.h5" "ESM1-4k.h5" "ESM1-4k.h5" "ESM1-4k.h5")
MODELS=("prott5" "prott5" "prott5" "esm1" "esm1" "esm1")

for i in "${!DL2_MODEL_TYPES[@]}"; do
    DL2_MODEL_TYPE=${DL2_MODEL_TYPES[$i]}
    LEVEL=${LEVELS[$i]}
    CLIP_LEN=${CLIP_LENS[$i]}
    EMBEDDING=${EMBEDDINGS[$i]}
    MODEL=${MODELS[$i]}

    echo "Running: python train_sl.py --model $DL2_MODEL_TYPE --level $LEVEL --dataset ../../datasets/final/hpa_uniprot_combined_trainset.csv --test_dataset ../../datasets/final/hou_testset.csv --clip_len $CLIP_LEN --classes_yaml ../../datasets/final/hierarchical_label_set.yaml --embeddings_path $PLM_EMBEDDING_DIR/$EMBEDDING --model_save_path $DEEPLOC2_MODELS_DIR/$MODEL --outputs_save_path $DEEPLOC2_OUTPUT_DIR/$MODEL"
    
    python train_sl.py \
        --model "$DL2_MODEL_TYPE" \
        --level "$LEVEL" \
        --dataset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
        --test_dataset ../../datasets/final/hou_testset.csv \
        --clip_len "$CLIP_LEN" \
        --classes_yaml ../../datasets/final/hierarchical_label_set.yaml \
        --embeddings_path "$PLM_EMBEDDING_DIR/$EMBEDDING" \
        --model_save_path "$DEEPLOC2_MODELS_DIR/$MODEL" \
        --outputs_save_path "$DEEPLOC2_OUTPUT_DIR/$MODEL"
done