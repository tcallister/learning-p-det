#!/bin/bash

#SBATCH --job-name=array
#SBATCH --account=kicp
#SBATCH --output=logs/log.out
#SBATCH --error=logs/log.err
#SBATCH --time=02:00:00
#SBATCH --partition=kicp
#SBATCH --ntasks=1
#SBATCH --mem=4G

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computation on each job
cd /home/tcallister/repositories/learning-p-det/executables/
conda activate learning-p-det-midway

output_dir=/project2/kicp/tcallister/learning-p-det-data/trained_model_deeper/
mkdir -p $output_dir/model_checkpoints/

#output_file=$output_dir/$(printf "%02d" $SLURM_ARRAY_TASK_ID).json
#echo "Writing to: " $output_file
python run_training_deeper.py $output_dir
