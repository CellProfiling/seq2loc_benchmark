#!/bin/bash
#SBATCH --job-name=dl2_metrics
#SBATCH --time=1:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_out/dl2_metrics_%A_%a.out
#SBATCH --error=slurm_out/dl2_metrics_%A_%a.err
#SBATCH -p emmalu

# Load environment variables
source ../../.env

module load python/3.9
source "$DEEPLOC2_ENV/bin/activate"

cd ../../Benchmark-Models/DeepLoc2/

MODELS=("prott5" "esm1")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun python get_metrics.py \
  --categories_yaml ../../datasets/final/hierarchical_label_set.yaml \
  --testset ../../datasets/final/hou_testset.csv \
  --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
  --outdir "$DEEPLOC2_OUTPUT_DIR/$MODEL"