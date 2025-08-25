#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --time=6:00:00
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --partition=emmalu
#SBATCH --output=slurm_out/embed_%A_%a.out
#SBATCH --error=slurm_out/embed_%A_%a.err
#SBATCH --array=0-5


# Activate environment
source ../../.env
source "$SEQ2LOC_ENV/bin/activate"


# Define job parameters (one per line, quoted, space separated)
#JOBS=(
#    "--fasta ../../datasets/fastas/all_datasets.fasta --model ESM1     --output_path $PLM_EMBEDDING_DIR/ESM1-4k.h5           --clip_len 4000"
#    "--fasta ../../datasets/fastas/all_datasets.fasta --model ESM3     --output_path $PLM_EMBEDDING_DIR/ESM3-3k.h5           --clip_len 3000"
#    "--fasta ../../datasets/fastas/all_datasets.fasta --model ESM2     --output_path $PLM_EMBEDDING_DIR/ESM2-4k.h5           --clip_len 4000"
#    "--fasta ../../datasets/fastas/all_datasets.fasta --model ProtT5   --output_path $PLM_EMBEDDING_DIR/ProtT5-4k.h5         --clip_len 4000"
#    "--fasta ../../datasets/fastas/all_datasets.fasta --model ProtBert --output_path $PLM_EMBEDDING_DIR/ProtBert-4k.h5       --clip_len 4000"
#    "--fasta ../../datasets/fastas/lacoste.fasta      --model ProtT5   --output_path $PLM_EMBEDDING_DIR/Lacoste_ProtT5-4k.h5 --clip_len 4000"
#)

JOBS=(
    "--fasta ../../datasets/fastas/all_datasets.fasta --model ESM1     --output_path $PLM_EMBEDDING_DIR/ESM1-4k.h5           --clip_len 4000"
    "--fasta ../../datasets/fastas/all_datasets.fasta --model ESM2     --output_path $PLM_EMBEDDING_DIR/ESM2-4k.h5           --clip_len 4000"
    "--fasta ../../datasets/fastas/all_datasets.fasta --model ProtBert --output_path $PLM_EMBEDDING_DIR/ProtBert-4k.h5       --clip_len 4000"
    "--fasta ../../datasets/fastas/lacoste.fasta      --model ProtT5   --output_path $PLM_EMBEDDING_DIR/Lacoste_ProtT5-4k.h5 --clip_len 4000"
)

# Select job based on SLURM_ARRAY_TASK_ID
PARAMS="${JOBS[$SLURM_ARRAY_TASK_ID]}"

# Run the command
srun python ../../utils/get_embeddings.py \
    $PARAMS \
    --model_path $PLM_CACHE_DIR \
    --token $HUGGING_FACE_TOKEN