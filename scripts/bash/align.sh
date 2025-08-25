#!/bin/bash

# Load environment variable from .env file and activate
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

# Parameter arrays
QUERY=(
    "../../datasets/fastas/hou.fasta"
    "../../datasets/fastas/hou.fasta"
)
TARGET=(
    "../../datasets/fastas/uniprot.fasta"
    "../datasets/fastas/hpa.fasta"
)
OUTPUT=(
    "../../datasets/intermediate/mmseqs_out/hou_uniprot_alignment.m8"
    "../datasets/intermediate/mmseqs_out/hou_hpa_alignment.m8"
)

TMP="../../datasets/intermediate/mmseqs_out/tmp"
mkdir -p "$TMP"

for i in "${!QUERY[@]}"; do
    mmseqs easy-search "${QUERY[$i]}" \
                       "${TARGET[$i]}" \
                       --min-seq-id 0.4 -s 7.5 -c 0.8 \
                       --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,pident,nident,qlen,tlen,qcov,tcov, \
                       "${OUTPUT[$i]}" \
                       "$TMP" \
done