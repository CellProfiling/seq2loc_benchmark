#!/bin/bash

# Load environment variable from .env file and activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

# Parameter arrays
DATASET=(
    "../../datasets/fastas/hpa_trainset.fasta"
    "../../datasets/fastas/uniprot_trainset.fasta"
    "../../datasets/fastas/hpa_uniprot_combined_trainset.fasta"
    "../../datasets/fastas/hpa_uniprot_combined_human_trainset.fasta"
)
OUTPUT=(
    "../../datasets/intermediate/mmseqs_out/hpa_hpa_alignment.m8"
    "../../datasets/intermediate/mmseqs_out/uniprot_uniprot_alignment.m8"
    "../../datasets/intermediate/mmseqs_out/combined_combined_alignment.m8"
    "../../datasets/intermediate/mmseqs_out/combinedhuman_combinedhuman_alignment.m8"
)

TMP="../../datasets/intermediate/mmseqs_out/tmp"
mkdir -p "$TMP"

for i in "${!DATASET[@]}"; do
    mmseqs easy-search "${DATASET[$i]}" \
                       "${DATASET[$i]}" \
                       --min-seq-id 0.4 -s 7.5 -c 0.8 \
                       --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,pident,nident,qlen,tlen,qcov,tcov, \
                       "${OUTPUT[$i]}" \
                       "$TMP"
done