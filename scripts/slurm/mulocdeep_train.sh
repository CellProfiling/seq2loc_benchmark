#!/bin/bash
#SBATCH --job-name=mulocdeep_train
#SBATCH --time=18:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --array=0-1
#SBATCH --output=slurm_out/mld_train_%A_%a.out
#SBATCH --error=slurm_out/mld_train_%A_%a.err
#SBATCH -p emmalu

source ../../.env

module load python/3.9
#source "$MULOCDEEP_ENV/bin/activate"
conda activate mulocdeep_env2

LV_INPUT_DIRS=("$MULOCDEEP_DATA_DIR/level1_2" "$MULOCDEEP_DATA_DIR/level1_3")
MODEL_OUTPUTS=("$MULOCDEEP_MODELS_DIR/seq2loc_level1_2" "$MULOCDEEP_MODELS_DIR/seq2loc_level1_3")
COARSE_VALUES=(10 8)
FINE_VALUES=(5 7)

LV_INPUT_DIR=${LV_INPUT_DIRS[$SLURM_ARRAY_TASK_ID]}
MODEL_OUTPUT=${MODEL_OUTPUTS[$SLURM_ARRAY_TASK_ID]}
COARSE=${COARSE_VALUES[$SLURM_ARRAY_TASK_ID]}
FINE=${FINE_VALUES[$SLURM_ARRAY_TASK_ID]}

srun python ../../Benchmark-Models/MULocDeep/train.py \
    --trainset ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --mapping ../../datasets/intermediate/mulocdeep/mulocdeep_mapping.yaml \
    --lv1_input_dir "$LV_INPUT_DIR" \
    --lv2_input_dir "$LV_INPUT_DIR" \
    --model_output "$MODEL_OUTPUT" \
    --MULocDeep_model \
    --existPSSM "$MULOCDEEP_PSSMS_DIR" \
    --numfolds 5 \
    --coarse "$COARSE" \
    --fine "$FINE" \
    --db ../../datasets/intermediate/mulocdeep/swissprot_db/swissprot