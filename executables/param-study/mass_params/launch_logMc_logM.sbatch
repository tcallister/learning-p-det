#!/bin/bash

#SBATCH --job-name=array
#SBATCH --account=kicp
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=01:00:00
#SBATCH --partition=kicp
#SBATCH --ntasks=1
#SBATCH --mem=2G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computation on each job
cd /home/tcallister/repositories/learning-p-det/executables/
conda activate learning-p-det-midway
python run_logMc_logM.py $SLURM_ARRAY_TASK_ID
