import numpy as np
np.float_ = np.float64
import stempeg
import os
#from IPython.display import Audio

# Define input and output directories
input_dir = '/Users/kaimikkelsen/canada_compute/data/MUSDB18/test'  # Path to folder containing .stem.mp4 files
output_dir = '/Users/kaimikkelsen/canada_compute/data/MUSDB18/test_stems'  # Set your desired output folder here

# Define stem names based on MUSDB mapping
stem_names = ["mixture", "drums", "bass", "accompaniment", "vocals"]

# Process each .stem.mp4 file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.stem.mp4'):
        # Load the stem file
        file_path = os.path.join(input_dir, filename)
        S, rate = stempeg.read_stems(file_path)
        
        # Create an output directory for each track, preserving hyphens and removing .stem
        track_name = filename.replace(".stem.mp4", "")  # Remove .stem.mp4
        track_output_dir = os.path.join(output_dir, track_name)
        os.makedirs(track_output_dir, exist_ok=True)

        # Save each stem with the corresponding name
        for i, stem_name in enumerate(stem_names):
            output_path = os.path.join(track_output_dir, f"{stem_name}.wav")
            stempeg.write_audio(output_path, S[i], sample_rate=rate, codec="pcm_s16le")
            print(f"Saved {stem_name} for {track_name} to {output_path}")
