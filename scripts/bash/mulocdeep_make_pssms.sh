#!/bin/bash

source ../../.env

source "$MULOCDEEP_ENV/bin/activate"
#conda activate mulocdeep_env2


python make_pssms.py \
    --csv ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --dir $MULOCDEEP_PSSMS_DIR \
    --db ../../datasets/intermediate/mulocdeep/swissprot_db/swissprot \
    --n_cores 128