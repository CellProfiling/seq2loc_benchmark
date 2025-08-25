#!/bin/bash

source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

CSVS=(
    "../../datasets/raw/Lacoste_metadata_raw.csv"
    "../../datasets/raw/HPA_2024_IF-image.csv"
    "../../datasets/raw/opencell-localization-annotations.csv"
)
ID_COLS=(
    "Uniprot"
    "ensembl_ids"
    "ensg_id"
)

for i in "${!CSVS[@]}"; do
    CSV="${CSVS[$i]}"
    ID_COL="${ID_COLS[$i]}"

    # Will run VERY slowly with just 1 core!
    python ../../utils/get_canonical.py --csv "$CSV" --id_col "$ID_COL" --n_cores 1

done