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
#SBATCH --mail-user=itamarberger@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END

# Set working directory according to the user currently running the job
cd /home/ADV_2526a/$USER/interp_project_snmf


# Make sure that files that are created by this script are accesible by everyone
umask 0002

# activate conda environment
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier

bash experiments/run_analayze_concept_trees.sh 
