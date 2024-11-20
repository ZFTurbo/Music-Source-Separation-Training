#!/bin/bash
#SBATCH --account=def-ichiro       # Replace with your account
#SBATCH --gpus-per-node=v100:1          # Request 1 V100 GPU per node
#SBATCH --cpus-per-task=6               # Request 6 CPU cores per GPU
#SBATCH --mem=32G                       # Memory per node
#SBATCH --time=0-00:05                  # Time limit (12 hours)
#SBATCH --output=job_output_%j.log      # Standard output log
#SBATCH --error=job_error_%j.log        # Standard error log

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
 

source separation_env/bin/activate

echo 'running training script'
python run_training.py