#!/bin/bash

#SBATCH --job-name=makedb
#SBATCH --time=1:00:00
#SBATCH --output=slurm_out/makedb_%A_%a.out
#SBATCH --error=slurm_out/makedb_%A_%a.err
#SBATCH -p emmalu

#Load modules to access BLAST
ml biology
ml ncbi-blast+/2.16.0

#Make database
srun makeblastdb \
    -in ../../datasets/raw/swissprot_2025_01.fasta \
    -out ../../datasets/intermediate/swissprot_psiblast_db/swissprot \
    -blastdb_version 4 \
    -dbtype prot \
    -parse_seqids