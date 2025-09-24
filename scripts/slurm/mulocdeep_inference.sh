#!/bin/bash
#SBATCH --job-name=mulocdeep_infer
#SBATCH --time=18:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --array=0-1
#SBATCH --output=slurm_out/mld_infer_%A_%a.out
#SBATCH --error=slurm_out/mld_infer_%A_%a.err
#SBATCH -p emmalu

source ../../.env

module load python/3.9

#source "$MULOCDEEP_ENV/bin/activate"
conda activate mulocdeep_env2

cd ../../Benchmark-Models/MULocDeep/

MODEL_DIRS=("$MULOCDEEP_MODELS_DIR/seq2loc_level1_3" "$MULOCDEEP_MODELS_DIR/seq2loc_level1_2")
SAVEDIRS=("$MULOCDEEP_MODELS_DIR/seq2loc_level1_3/outputs" "$MULOCDEEP_MODELS_DIR/seq2loc_level1_2/outputs")
LEVELS=("level1_3" "level1_2")

MODEL_DIR=${MODEL_DIRS[$SLURM_ARRAY_TASK_ID]}
SAVEDIR=${SAVEDIRS[$SLURM_ARRAY_TASK_ID]}
LEVEL=${LEVELS[$SLURM_ARRAY_TASK_ID]}

srun python inference.py \
    --crossval_csv ../../datasets/final/hpa_uniprot_combined_trainset.csv \
    --test_csv ../../datasets/final/hou_testset.csv \
    --model_dir "$MODEL_DIR" \
    --existPSSM "$MULOCDEEP_PSSMS_DIR" \
    --savedir "$SAVEDIR" \
    --gpu \
    --numfolds 5 \
    --numclasses 22 \
    --id_col uniprot_id \
    --level "$LEVEL" \
    --multi