#!/bin/bash
#SBATCH --job-name=snmf_dow_minimal
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --output=slurm_logs/dow_minimal_%j.out
#SBATCH --error=slurm_logs/dow_minimal_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=02:00:00
# Pass only explicit env vars; we'll source secrets inside the job
#SBATCH --export=NONE

# Load OPENAI_API_KEY from a private user env file if present
if [ -f "$HOME/.openai.env" ]; then
	source "$HOME/.openai.env"
fi

# Ensure OPENAI_API_KEY is available
if [ -z "${OPENAI_API_KEY:-}" ]; then
	echo "ERROR: OPENAI_API_KEY is not set. Create $HOME/.openai.env with: export OPENAI_API_KEY=..." >&2
	exit 1
fi

# 1. Initialize Conda (using Tslil's path)
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier

# 2. Run minimal steering test
bash experiments/run_minimal_dow_steering.sh
