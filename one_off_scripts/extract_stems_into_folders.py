import os
import numpy as np
np.float_ = np.float64  # Ensure compatibility
import stempeg
import soundfile as sf

# Define paths
organized_path = os.path.expanduser("/Users/kaimikkelsen/canada_compute/data/MUSDB18/validation")
output_path = os.path.join(organized_path, "separated_stems")
os.makedirs(output_path, exist_ok=True)

def extract_stems(stem_file, output_folder):
    try:
        # Load the stem file using stempeg
        stems, rate = stempeg.read_stems(stem_file, stem_id=None)  # Load all stems, including the mixture

        # Get the base name of the file (song name)
        song_name = os.path.splitext(os.path.basename(stem_file))[0]
        song_folder = os.path.join(output_folder, song_name)
        os.makedirs(song_folder, exist_ok=True)

        # Save the mixture file
        mixture_path = os.path.join(song_folder, "mixture.wav")
        sf.write(mixture_path, stems[0], rate)  # The mixture is usually the first "stem" returned
        print(f"Extracted mixture: {mixture_path}")

        # Save each individual stem
        default_stem_names = ["drums", "bass", "other", "vocals"]  # Standard MUSDB stem names
        for i, stem in enumerate(stems[1:], start=1):  # Stems start at index 1 (excluding mixture)
            stem_name = (
                f"{default_stem_names[i-1]}.wav" if i-1 < len(default_stem_names) else f"stem_{i-1}.wav"
            )
            stem_path = os.path.join(song_folder, stem_name)
            sf.write(stem_path, stem, rate)
            print(f"Extracted stem: {stem_path}")

        print(f"Finished extracting stems for {song_name}\n")
    except Exception as e:
        print(f"Error extracting stems from {stem_file}: {e}")

# Process subsets
for subset in ["."]:
    subset_path = os.path.join(organized_path, subset)
    print(subset_path)
    if not os.path.exists(subset_path):
        print(f"Subset folder not found: {subset_path}")
        continue

    # Iterate through all `.stem.mp4` files in the subset
    for file in os.listdir(subset_path):
        if file.endswith(".stem.mp4"):
            stem_file = os.path.join(subset_path, file)
            extract_stems(stem_file, output_path)

print("All stems have been processed!")