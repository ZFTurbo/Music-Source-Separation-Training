#!/bin/bash
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=125G         # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-00:05     # DD-HH:MM:SS
#SBATCH --account=def-ichiro
#SBATCH --output=my_job_output.out  # Save the output to this file
#SBATCH --error=my_job_error.err    # Save the error to this file

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME


 

source separation_env/bin/activate

echo 'running training script'
# python run_training.py --model_type "mdx23c" --config_path="configs/config_musdb18_mdx23c.yaml" --results_path="results/mdx23c" --start_check_point="results/mdx23c.ckpt"
python run_training.py