#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=4:00:00
#SBATCH --mem=100G
#SBATCH --partition=emmalu
#SBATCH --output=slurm_out/random_%A_%a.out
#SBATCH --error=slurm_out/random_%A_%a.err
#SBATCH --array=0-1

source ../../.env
source "$SEQ2LOC_ENV/bin/activate"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    srun python ../../Benchmark-Models/Random/random_baseline.py \
        --train_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv  \
        --test_csv ../../datasets/final/hou_testset.csv \
        --yaml_classes ../../datasets/final/hierarchical_label_set.yaml \
        --save_dir $RANDOM_OUTPUT \
        --multi
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    srun python ../../Benchmark-Models/Random/random_baseline.py \
        --train_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv  \
        --test_csv ../../datasets/final/hou_testset.csv \
        --yaml_classes ../../datasets/final/hierarchical_label_set.yaml \
        --save_dir $RANDOM_OUTPUT \
        --single
fi