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
cd /home/tcallister/repositories/learning-p-det/executables/lr_decay/
conda activate learning-p-det-midway

output_dir=/project2/kicp/tcallister/learning-p-det-data/param-study/lr_decay/lr_1e-5_step_10_delta_10
mkdir -p $output_dir

output_file=$output_dir/$(printf "%02d" $SLURM_ARRAY_TASK_ID).json
echo "Writing to: " $output_file
python run_lr_1e-5_step_10_delta_10.py $output_file