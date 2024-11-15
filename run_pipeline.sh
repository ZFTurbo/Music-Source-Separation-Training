#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G         # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=2-00:00     # DD-HH:MM:SS
#SBATCH --account=def-ichiro

source separation_env/bin/activate

echo 'running pipeline script'
python run_pipeline.py