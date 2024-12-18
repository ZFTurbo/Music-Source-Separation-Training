#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=0-00:60:00           # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=slurm_logs/slurm-%j.out  # Use Job ID for unique output files

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

CURRENT_DATE=$(date +"%Y-%m-%d_%H-%M-%S")

# Model-specific parameters
MODEL_TYPE="bs_roformer"
CONFIG_PATH="configs/config_musdb18_bs_roformer.yaml"

CHECKPOINTS_PATH="checkpoints/${MODEL_TYPE}_${CURRENT_DATE}"
SLURM_LOGS_PATH="slurm_logs/${MODEL_TYPE}_${CURRENT_DATE}"

mkdir -p "$CHECKPOINTS_PATH"
mkdir -p "$SLURM_LOGS_PATH"

# Redirect SLURM output dynamically
exec > >(tee -a "$SLURM_LOGS_PATH/slurm-${SLURM_JOB_ID}.out") 2>&1

source separation_env/bin/activate

echo "Running training script for model: $MODEL_TYPE"

python train_tensorboard.py \
  --model_type "$MODEL_TYPE" \
  --config_path "$CONFIG_PATH" \
  --results_path "$CHECKPOINTS_PATH" \
  --data_path "../data/MUSDB18HQ/train" \
  --valid_path "../data/MUSDB18HQ/validation" \
  --num_workers 4 \
  --start_check_point "" \
  --device_ids 0 1 2 3
