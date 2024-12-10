#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G         # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:10     # DD-HH:MM:SS
#SBATCH --account=def-ichiro

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")

# Create a dynamic results folder
OUTPUT_PATH="results/htdemucs_$CURRENT_DATE"
CHECKPOINTS_PATH="checkpoints/htdemucs_$CURRENT_DATE"

mkdir -p "$OUTPUT_PATH"  # Fixed path creation
mkdir -p "$CHECKPOINTS_PATH"

# Ensure the results directory exists
#mkdir -p "$RESULTS_PATH"

echo "Results will be saved to $OUTPUT_PATH"

# Redirect SLURM output to the results folder
#SBATCH --output=slurm_logs/htdemucs_$CURRENT_DATE.out  # Use a static or explicitly created folder for SLURM logs
 

source separation_env/bin/activate

echo 'running training script'
python run_training.py --model_type "htdemucs" --config_path="configs/config_musdb18_htdemucs.yaml" --results_path="$CHECKPOINTS_PATH" --start_check_point=""
#python run_training.py