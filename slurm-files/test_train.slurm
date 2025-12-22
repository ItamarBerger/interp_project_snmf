#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --chdir=/home/ADV_2526a/tslilshani/interp_project_snmf
#SBATCH --output=/home/ADV_2526a/tslilshani/slurm_logs/%x_%j.out
#SBATCH --error=/home/ADV_2526a/tslilshani/slurm_logs/%x_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=tslilshani@mail.tau.ac.il,itamarberger@mail.tau.ac.il,ohadsharet@mail.tau.ac.il
#SBATCH --mail-type=FAIL,END


# Make sure that files that are created by this script are accesible by everyone
umask 0002

# activate conda environment
source /home/ADV_2526a/tslilshani/anaconda3/etc/profile.d/conda.sh
conda activate snmf_hier

bash experiments/test_snmf_steering.sh  
