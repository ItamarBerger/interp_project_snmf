#!/bin/bash
#SBATCH --job-name=snmf_tutorial
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --output=slurm_logs/job_%j.out
#SBATCH --error=slurm_logs/job_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --time=02:00:00

# 1. Create log directory if it's missing
mkdir -p slurm_logs

# 2. Setup Environment
cd $SLURM_SUBMIT_DIR
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier

# 3. Fix deprecation warnings
export HF_HOME=$TRANSFORMERS_CACHE

# 4. Convert and Run
echo "Converting notebook..."
python -m jupyter nbconvert --to script snmf_tutorial.ipynb --output temp_script_${SLURM_JOB_ID}

echo "Running on GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u temp_script_${SLURM_JOB_ID}.py

# 5. Cleanup
rm temp_script_${SLURM_JOB_ID}.py