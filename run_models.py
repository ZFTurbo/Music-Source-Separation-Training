import subprocess

# Define arrays for each configuration parameter
model_types = ["mdx23c", "bs_mamba2", "scnet", "htdemucs"]
config_paths = [
    "configs/config_musdb18_mdx23c.yaml",
    "configs/config_vocals_bs_mamba2.yaml",
    "configs/config_musdb18_scnet.yaml",
    "configs/config_musdb18_htdemucs.yaml"
]
checkpoints = [
    "results/mdx23c.ckpt",
    "results/bs_mamba2.ckpt",
    "results/scnet.ckpt",
    "demucs_ckpt.th"
]
store_dirs = [
    "separated/mdx23c/",
    "separated/bs_mamba2/",
    "separated/scnet/",
    "separated/htdemucs/"
]

# Base command parts
base_command = "python inference.py"
input_folder = "/Users/kaimikkelsen/canada_compute/data/MUSDB18/test"
force_cpu = "--force_cpu"

# Loop through the configurations and execute the commands
for model_type, config_path, checkpoint, store_dir in zip(model_types, config_paths, checkpoints, store_dirs):
    command = [
        base_command,
        "--input_folder", input_folder,
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", checkpoint,
        "--store_dir", store_dir,
        force_cpu
    ]
    
    # Print the command for reference
    print("Running:", " ".join(command))

    # Run the command
    subprocess.run(" ".join(command), shell=True)