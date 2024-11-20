#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G         # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:05           # time (DD-HH:MM)
#SBATCH --account=def-ichiro

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
 

source separation_env/bin/activate

echo 'running training script'
python run_training.py