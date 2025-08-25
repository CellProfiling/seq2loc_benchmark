#!/bin/sh
#SBATCH --job-name=canonical
#SBATCH --time=5:00:00
#SBATCH -c 32
#SBATCH --partition=emmalu
#SBATCH --array=0-2
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err


mkdir -p slurm_out

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
N_CORES=(
    32
    5
    5
)
TIMES=(
    "1:00:00"
    "1:00:00"
    "3-00:00:00"
)


CSV=${CSVS[$SLURM_ARRAY_TASK_ID]}
ID_COL=${ID_COLS[$SLURM_ARRAY_TASK_ID]}
N_CORE=${N_CORES[$SLURM_ARRAY_TASK_ID]}

srun python ../../utils/get_canonical.py \
    --csv $CSV \
    --id_col $ID_COL \
    --n_cores $N_CORE