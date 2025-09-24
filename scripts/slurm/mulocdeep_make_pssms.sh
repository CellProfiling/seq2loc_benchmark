#!/bin/bash

#SBATCH --job-name=pssm
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --error=slurm_out/pssms_%A_err.log
#SBATCH --output=slurm_out/pssms_%A_out.log
#SBATCH -p emmalu

source ../../.env


module load python/3.9
#source "$MULOCDEEP_ENV/bin/activate"
conda activate mulocdeep_env2


srun python ../../Benchmark-Models/MULocDeep/make_pssms.py \
    --csv ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --dir $MULOCDEEP_PSSMS_DIR \
    --db ../../datasets/intermediate/mulocdeep/swissprot_db/swissprot \
    --n_cores 8