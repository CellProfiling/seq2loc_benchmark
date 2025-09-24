#!/bin/bash

#Make database
makeblastdb \
    -in ../../datasets/raw/swissprot_2025_01.fasta \
    -out ../../datasets/intermediate/swissprot_psiblast_db/swissprot\
    -blastdb_version 4 \
    -dbtype prot \
    -parse_seqids