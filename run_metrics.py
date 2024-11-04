import numpy as np
np.float_ = np.float64

import musdb
import museval
import os
import platform
import datetime


#dataset_folder = "../data/MUSDB18"
dataset_folder ="/Users/kaimikkelsen/canada_compute/data/MUSDB18"

estimates_directory = "/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/separated/mdx23c/"
output_directory = '/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/results/mdx23c'

#estimates_directory = "/home/kaim/projects/def-ichiro/kaim/demucs/separated/htdemucs"
#output_directory = '/home/kaim/projects/def-ichiro/kaim/demucs/results/'

# import musdb
# mus = musdb.DB(download=True)
# mus[0].audio

print(f"dataset folder is {dataset_folder}, estimates_dir {estimates_directory}, output directory {output_directory}")


mus = musdb.DB(root=dataset_folder, subsets="test")

current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# evaluate an existing estimate folder with wav files
scores = museval.eval_mus_dir(
    dataset=mus,  # instance of musdb
    estimates_dir=estimates_directory,  # path to estimate folder
    output_dir=output_directory,  # set a folder to write eval json files
    ext='wav'
)

#args = parser.parse_args()
method = museval.EvalStore(frames_agg="median", tracks_agg="median")
method.add_eval_dir(output_directory)
print(f"method {method}")

# Get current date and time
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Get machine name
machine_name = platform.node()

# Path for the output file
output_file = os.path.join(output_directory, 'results.txt')

# Open the file to write the result with additional information
with open(output_file, 'a') as f:
    f.write(f"Date and Time: {current_datetime}\n")
    f.write(f"Machine Name: {machine_name}\n\n")
    f.write(f"Model Folder: {estimates_directory}\n\n")
    f.write("Method Evaluation Results:\n")
    f.write(str(method))
    f.write("\n\n\n")

print(f"Method result saved to {output_file}")

