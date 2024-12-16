import numpy as np
np.float_ = np.float64

import musdb
import museval
import os
import platform
import datetime
import subprocess
import time
import os
import shutil
import os
import shutil

# Set the dataset folder and model name here
dataset_folder = "../data/MUSDB18-7second"  # Change this to your dataset folder path
model_choice = "htdemucs"  # Change this to "mdx23c", "bs_mamba2", "scnet", or "htdemucs"



# Model configuration mappings
models = {
    "mdx23c": {
        "config_path": "configs/config_musdb18_mdx23c.yaml",
        "checkpoint": "results/mdx23c.ckpt",
        "store_dir": "separated/mdx23c/"
    },
    "bs_mamba2": {
        "config_path": "configs/config_vocals_bs_mamba2.yaml",
        "checkpoint": "results/bs_mamba2.ckpt",
        "store_dir": "separated/bs_mamba2/"
    },
    "scnet": {
        "config_path": "configs/config_musdb18_scnet.yaml",
        "checkpoint": "results/scnet.ckpt",
        "store_dir": "separated/scnet/"
    },
    "htdemucs": {
        "config_path": "configs/config_musdb18_htdemucs.yaml",
        "checkpoint": "results/htdemucs_train/model_htdemucs_ep_0_sdr_-0.1195.ckpt",
        "store_dir": "separated/test_trained_htdemucs/"
    }
}

# Check if the model_choice is valid
if model_choice not in models:
    print("Invalid model choice. Please select a valid model name: mdx23c, bs_mamba2, scnet, or htdemucs.")
    exit(1)

# Set up selected model based on user input
selected_model = models[model_choice]
config_path = selected_model["config_path"]
checkpoint = selected_model["checkpoint"]
store_dir = selected_model["store_dir"]

print("in pipeline script")

current_directory = os.getcwd()
#print(current_directory)
# # Paths
estimates_directory = os.path.join(current_directory, store_dir)
output_directory = os.path.join(current_directory, "results", model_choice)

#Print paths and files in each directory
def print_files_in_directory(path, name):
    abs_path = os.path.abspath(path)
    print(f"{name} Path: {abs_path}")
    if os.path.exists(abs_path) and os.path.isdir(abs_path):
        files = os.listdir(abs_path)
        if files:
            print(f"Files in {name}:")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"No files found in {name}.")
    else:
        print(f"{name} path does not exist or is not a directory.")

#print_files_in_directory(dataset_folder, "Dataset")
#print_files_in_directory(estimates_directory, "Estimates Directory")
#print_files_in_directory(output_directory, "Output Directory")



# Run the separation command (assuming an inference script is available)
print(f"Starting separation for model: {model_choice}")

# Start the timer
start_time = time.time()

separation_command = [
    "python", "inference.py",
    "--input_folder", os.path.join(dataset_folder, "test"),
    "--model_type", model_choice,
    "--config_path", config_path,
    "--start_check_point", checkpoint,
    "--store_dir", estimates_directory,
    "--force_cpu"
]
subprocess.run(separation_command)

# Stop the timer
end_time = time.time()
separation_duration = end_time - start_time
print(f"Separation completed. in {separation_duration} seconds")


def move_all_to_test(folder_path):
    # Define the path for the 'test' subdirectory
    test_dir = os.path.join(folder_path, 'test')
    
    # Create 'test' directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")
    
    # Move all files and directories from the folder to the 'test' directory
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Skip the 'test' directory itself
        if item_path != test_dir:
            shutil.move(item_path, test_dir)
            print(f"Moved {item} to {test_dir}")


move_all_to_test(store_dir)


# Set the path to the folder containing the stem files
#input_folder = '/path/to/your/stem/files'
test_dir = os.path.join(store_dir, "test")

# Iterate through each file in the folder
for filename in os.listdir(test_dir):
    if '.' in filename:
        # Separate the base name (song title) and stem type
        parts = filename.split('.stem_')
        if len(parts) == 2:
            song_name, stem_type_with_ext = parts
            # Extract stem type and file extension
            stem_type, ext = os.path.splitext(stem_type_with_ext)
            
            # Create the song folder if it doesn't exist
            song_folder = os.path.join(test_dir, song_name)
            os.makedirs(song_folder, exist_ok=True)
            
            # Rename the stem file to stem type with its original extension
            new_filename = f"{stem_type}{ext}"
            source_path = os.path.join(test_dir, filename)
            dest_path = os.path.join(song_folder, new_filename)
            
            # Move the file to the new folder
            shutil.move(source_path, dest_path)
            
            print(f"Moved {filename} to {song_folder} as {new_filename}")

print("Files organized by song title.")



#estimates_directory = os.path.join("/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training", store_dir)
#output_directory = os.path.join('/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/results', model_choice)

print("Calculating metrics...")

#print(f"dataset folder is {dataset_folder}, estimates_dir {estimates_directory}, output directory {output_directory}")


# Set up musdb for testing data
mus = musdb.DB(root=dataset_folder, subsets="test")

#Now evaluate the results
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
scores = museval.eval_mus_dir(
    dataset=mus,
    estimates_dir=estimates_directory,
    output_dir=output_directory,
    ext='wav'
)

# Store evaluation results
method = museval.EvalStore(frames_agg="median", tracks_agg="median")
method.add_eval_dir(output_directory)
print(f"Method: {method}")

# Get machine name
machine_name = platform.node()

# Path for the output file
output_file = os.path.join(output_directory, 'results.txt')

# Save results with additional information
with open(output_file, 'a') as f:
    f.write(f"===== Evaluation Results =====\n")
    f.write(f"Model Name: {model_choice}")
    f.write(f"Date and Time: {current_datetime}\n")
    f.write(f"Machine Name: {machine_name}\n")
    f.write(f"Model Folder: {estimates_directory}\n")
    f.write(f"Dataset Folder: {dataset_folder}\n")
    f.write(f"Separation Duration: {separation_duration:.2f} seconds\n\n")
    f.write("Method Evaluation Results:\n")
    f.write(f"{str(method)}\n")
    f.write("\n=============================\n\n")

print(f"Method result saved to {output_file}")