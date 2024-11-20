#!/bin/bash
#SBATCH --job-name=my_gpu_job       # Job name
#SBATCH --account=def-ichiro      # Account name (replace with your account)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --gpus-per-node=1           # Number of GPUs per node
#SBATCH --cpus-per-task=4           # Number of CPU cores per task
#SBATCH --mem=128000               # Memory per node in MiB (32GB = 32768M)
#SBATCH --time=0-03:00              # Time (DD-HH:MM)
#SBATCH --output=job_output.log     # Standard output log
#SBATCH --error=job_error.log       # Standard error log

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
 

source separation_env/bin/activate

echo 'running training script'
python run_training.py