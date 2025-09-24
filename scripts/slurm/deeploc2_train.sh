#!/bin/bash
#SBATCH --job-name=dl2_train
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --array=0-5
#SBATCH --output=slurm_out/dl2_%A_%a.out
#SBATCH --error=slurm_out/dl2_%A_%a.err
#SBATCH -p emmalu

source ../../.env

module load python/3.9
source "$DEEPLOC2_ENV/bin/activate"

cd ../../Benchmark-Models/DeepLoc2/

DL2_MODEL_TYPES=("seq2loc-prott5" "seq2loc-prott5" "seq2loc-prott5" "seq2loc-ems1" "seq2loc-ems1" "seq2loc-ems1")
LEVELS=(1 2 3 1 2 3)
CLIP_LENS=(4000 4000 4000 1022 1022 1022)
EMBEDDINGS=("ProtT5-4k.h5" "ProtT5-4k.h5" "ProtT5-4k.h5" "ESM1-4k.h5" "ESM1-4k.h5" "ESM1-4k.h5")
MODELS=("prott5" "prott5" "prott5" "esm1" "esm1" "esm1")

DL2_MODEL_TYPE=${DL2_MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
LEVEL=${LEVELS[$SLURM_ARRAY_TASK_ID]}
CLIP_LEN=${CLIP_LENS[$SLURM_ARRAY_TASK_ID]}
EMBEDDING=${EMBEDDINGS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

srun python train_sl.py \
    --model "$DL2_MODEL_TYPE" \
    --level "$LEVEL" \
    --dataset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --test_dataset ../../datasets/final/hou_testset.csv \
    --clip_len "$CLIP_LEN" \
    --classes_yaml ../../datasets/final/hierarchical_label_set.yaml \
    --embeddings_path "/scratch/groups/emmalu/seq2loc/embeddings3/$EMBEDDING" \
    --model_save_path "$DEEPLOC2_MODELS_DIR/$MODEL" \
    --outputs_save_path "$DEEPLOC2_OUTPUT_DIR/$MODEL" \