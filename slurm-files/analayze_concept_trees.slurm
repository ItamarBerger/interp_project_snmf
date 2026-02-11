#!/bin/bash

#SBATCH --job-name=train_hier_snmf
#SBATCH --output=/home/ADV_2526a/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/ADV_2526a/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=tslilshani@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

# Set working directory according to the user currently running the job
cd /home/ADV_2526a/$USER/interp_project_snmf


# Make sure that files that are created by this script are accessible by everyone
umask 0002

# activate conda environment
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier


if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="experiments/artifacts/concept_trees/$ACT_MODEL_NAME"
fi

# Convert '|' to ',' if needed, since slurm can't handle the "," character well in variable passing
LAYERS=$(echo "$LAYERS" | tr '|' ',')
STEPS=$(echo "$STEPS" | tr '|' ',')

SCRIPT_CMD="experiments/run_analayze_concept_trees.sh --output-path $OUTPUT_PATH"

if [[ -n "$BASE_DIR" ]]; then
  echo "Using base dir: $BASE_DIR"
  SCRIPT_CMD="$SCRIPT_CMD --base-dir $BASE_DIR"
fi

if [[ -n "$ACT_MODEL_NAME" ]]; then
  echo "Using activation model name: $ACT_MODEL_NAME"
  SCRIPT_CMD="$SCRIPT_CMD --act-model-name $ACT_MODEL_NAME"
fi

if [[ -n "$LAYERS" ]]; then
  echo "Using layers: $LAYERS"
  SCRIPT_CMD="$SCRIPT_CMD --layers $LAYERS"
fi

if [[ -n "$STEPS" ]]; then
  echo "Using steps: $STEPS"
  SCRIPT_CMD="$SCRIPT_CMD --steps $STEPS"
fi

if [[ -n "$RANKS" ]]; then
  echo "Using ranks: $RANKS"
  SCRIPT_CMD="$SCRIPT_CMD --ranks $RANKS"
fi

echo "Running command : $SCRIPT_CMD"

bash $SCRIPT_CMD
