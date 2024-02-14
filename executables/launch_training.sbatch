#!/bin/bash

#SBATCH --job-name=array
#SBATCH --account=kicp
#SBATCH --output=logs/log_%A_%a.out
#SBATCH --error=logs/log_%A_%a.err
#SBATCH --array=0-50
#SBATCH --time=01:00:00
#SBATCH --partition=kicp
#SBATCH --ntasks=1
#SBATCH --mem=2G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computation on each job
cd /home/tcallister/repositories/learning-p-det/executables/
conda activate learning-p-det-midway

output_dir=/project2/kicp/tcallister/learning-p-det-data/trained_models_baseline/
mkdir -p $output_dir

output_file_prefix=$output_dir/job_$(printf "%02d" $SLURM_ARRAY_TASK_ID)
python run_training.py $output_file_prefix
