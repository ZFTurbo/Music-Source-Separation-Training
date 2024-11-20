#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=0-00:03
#SBATCH --account=def-ichiro
#SBATCH --output=my_job_output.out  # Save the output to this file
#SBATCH --error=my_job_error.err    # Save the error to this file

module load python/3.10 cuda/12.2 cudnn/8.9.5.29
 

source separation_env/bin/activate

echo 'running training script'
python run_training.py