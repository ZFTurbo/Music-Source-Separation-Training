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
    "separated/next/mdx23c/",
    "separated/next/bs_mamba2/",
    "separated/next/scnet/",
    "separated/next/htdemucs/"
]

# Base command parts
base_command = "python valid.py"
valid_path = "/Users/kaimikkelsen/canada_compute/data/MUSDB18/test_stems"
#force_cpu = "--force_cpu"
metrics = ["sdr", "si_sdr", "aura_stft"]  # You can customize this list

# Loop through the configurations and execute the commands
for model_type, config_path, checkpoint, store_dir in zip(model_types, config_paths, checkpoints, store_dirs):
    command = [
        base_command,
        "--model_type", model_type,
        "--config_path", config_path,
        "--start_check_point", checkpoint,
        "--valid_path", valid_path,
        "--store_dir", store_dir,
        "--extension", "wav",
        "--metrics", " ".join(metrics),  # Join metrics into a space-separated string
        #force_cpu
    ]
    
    # Print the command for reference
    print("Running:", " ".join(command))

    # Run the command
    subprocess.run(" ".join(command), shell=True)