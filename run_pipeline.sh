#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-ichiro

# echo 'loading venv'
# module load python/3.10
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index --upgrade pip

pip install -r requirements.txt

echo 'running pipeline script'
python run_pipeline.py