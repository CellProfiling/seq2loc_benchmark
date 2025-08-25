#!/bin/bash

# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"


python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/all_datasets.fasta \
    --model ESM1 \
    --output_path $PLM_EMBEDDING_DIR/ESM1-4k.h5 \
    --clip_len 4000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN

python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/all_datasets.fasta \
    --model ESM3 \
    --output_path $$PLM_EMBEDDING_DIR/ESM3-3k.h5  \
    --clip_len 3000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN

python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/all_datasets.fasta \
    --model ESM2 \
    --output_path $$PLM_EMBEDDING_DIR/ESM2-4k.h5  \
    --clip_len 4000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN

python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/all_datasets.fasta \
    --model ProtT5 \
    --output_path $$PLM_EMBEDDING_DIR/ProtT5-4k.h5  \
    --clip_len 4000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN

python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/all_datasets.fasta \
    --model ProttBert \
    --output_path $$PLM_EMBEDDING_DIR/ProtBert-4k.h5  \
    --clip_len 4000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN

python ../../utils/get_embeddings.py \
    --fasta ../datasets/fastas/lacoste.fasta \
    --model ProtT5 \
    --output_path $$PLM_EMBEDDING_DIR/Lacoste_ProtT5-4k.h5  \
    --clip_len 4000 \
    --model_path $PLM_EMBEDDING_DIR \
    --token $HUGGING_FACE_TOKEN