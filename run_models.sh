#!/bin/bash

# Define arrays for each configuration parameter
model_types=("mdx23c" "bs_mamba2" "scnet")
config_paths=("configs/config_musdb18_mdx23c.yaml" "configs/config_vocals_bs_mamba2.yaml" "configs/config_musdb18_scnet.yaml")
checkpoints=("results/mdx23.ckpt" "results/bs_mamba2.ckpt" "results/scnet.ckpt")
store_dirs=("separated/mdx23c/" "separated/bs_mamba2/" "separated/scnet/")

# model_types=("bs_mamba2")
# config_paths=("configs/config_musdb18_bs_mamba2.yaml")
# checkpoints=("results/model_bs_mamba2.ckpt")
# store_dirs=("separated/bs_mamba2/")

# Base command
base_command="python inference.py --input_folder /Users/kaimikkelsen/canada_compute/data/MUSDB18/test --force_cpu"

# Loop through the configurations
for i in "${!model_types[@]}"; do
    model_type=${model_types[$i]}
    config_path=${config_paths[$i]}
    start_check_point=${checkpoints[$i]}
    store_dir=${store_dirs[$i]}
    
    # Construct and run the command
    full_command="$base_command --model_type $model_type --config_path $config_path --start_check_point $start_check_point --store_dir $store_dir"
    
    echo "Running: $full_command"

    #sdf
    eval $full_command
done