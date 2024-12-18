#!/bin/bash
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#SBATCH --gpus-per-node=v100:3  # Request 3 V100 GPUs
#SBATCH --cpus-per-task=16      # Adjust CPU allocation accordingly
#SBATCH --mem=128G              # Increase memory if necessary
=======
#SBATCH --gres=gpu:v100l:3       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16        # Adjust based on your cluster's CPU/GPU ratio
#SBATCH --mem=0               # Adjust memory as needed
>>>>>>> d0b7812 (ok)
=======
#SBATCH --nodes=1       # Request GPU "generic resources"
#SBATCH --gpus-per-node=p100l:2 # Adjust  based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
>>>>>>> c5874c0 (yes)
=======
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=v100l:2 # Adjust  based on your cluster's CPU/GPU ratio
#SBATCH --mem=125G               # Adjust memory as needed
>>>>>>> 7238389 (yes)
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
  --device_ids 0
