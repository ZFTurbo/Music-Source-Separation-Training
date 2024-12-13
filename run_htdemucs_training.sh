#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6        # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G               # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:10           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")

# Create a dynamic results folder
CHECKPOINTS_PATH="checkpoints/htdemucs_$CURRENT_DATE"
SLURM_LOGS_PATH="slurm_logs/htdemucs_$CURRENT_DATE"

mkdir -p "$CHECKPOINTS_PATH"   # Fixed path creation
mkdir -p "$SLURM_LOGS_PATH"    # Create a folder for SLURM logs if desired

# Redirect SLURM output dynamically
exec > >(tee -a "$SLURM_LOGS_PATH/slurm-${SLURM_JOB_ID}.out") 2>&1


source separation_env/bin/activate

echo 'Running training script'
python run_training.py --model_type "htdemucs" --config_path="configs/config_musdb18_htdemucs.yaml" --results_path="$CHECKPOINTS_PATH" --start_check_point=""