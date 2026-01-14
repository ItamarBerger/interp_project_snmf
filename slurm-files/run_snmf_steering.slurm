#!/bin/bash

#SBATCH --job-name=hier_snmf_steering
#SBATCH --output=/home/ADV_2526a/%u/slurm_logs/%x_%j.out
#SBATCH --error=/home/ADV_2526a/%u/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=tslilshani@mail.tau.ac.il,itamarberger@mail.tau.ac.il,ohadsharet@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

# Set working directory according to the user currently running the job
cd /home/ADV_2526a/$USER/interp_project_snmf

# Make sure that files that are created by this script are accessible by everyone
umask 0002

# activate conda environment
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier

if [ -z "$STEPS" ]; then
  STEPS="all"
fi

if [ -z "$LAYERS" ]; then
  LAYERS="0,3,6,9,11"
fi

# Convert '|' to ',' if needed, since slurm can't handle the "," character well in variable passing
LAYERS=$(echo "$LAYERS" | tr '|' ',')
STEPS=$(echo "$STEPS" | tr '|' ',')

echo "Running scripts with STEPS=$STEPS and LAYERS=$LAYERS"

SCRIPT_CMD="experiments/run_hier_snmf_steering.sh --steps $STEPS --layers $LAYERS"

if [[ -n "$CAUS_SAVE_PATH" ]]; then
  echo "Using causal save path: $CAUS_SAVE_PATH"
  SCRIPT_CMD="$SCRIPT_CMD --causal-save-path $CAUS_SAVE_PATH"
fi

if [[ -n "$OUTPUT_SCORE_RESULTS" ]]; then
  echo "Using output score results path: $OUTPUT_SCORE_RESULTS"
  SCRIPT_CMD="$SCRIPT_CMD --output-score-results $OUTPUT_SCORE_RESULTS"
fi

if [[ -n "$INPUT_SCORE_RESULTS" ]]; then
  echo "Using input score results path: $INPUT_SCORE_RESULTS"
  SCRIPT_CMD="$SCRIPT_CMD --input-score-results $INPUT_SCORE_RESULTS"
fi

echo "Running command : $SCRIPT_CMD"

bash $SCRIPT_CMD

