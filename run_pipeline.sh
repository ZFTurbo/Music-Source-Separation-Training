#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=kaim
echo 'running pipeline script'
python run_pipeline.py