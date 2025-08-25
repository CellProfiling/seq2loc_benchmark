#!/bin/sh
#SBATCH --job-name=align
#SBATCH --time=00:30:00
#SBATCH --partition=emmalu
#SBATCH --array=0-1
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err

mkdir -p slurm_out

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
mkdir -p $TMP

mmseqs easy-search "${QUERY[$SLURM_ARRAY_TASK_ID]}" \
                   "${TARGET[$SLURM_ARRAY_TASK_ID]}" \
                   --min-seq-id 0.4 -s 7.5 -c 0.8 \
                   --format-output query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,pident,nident,qlen,tlen,qcov,tcov, \
                   "${OUTPUT[$SLURM_ARRAY_TASK_ID]}" \
                   "$TMP"