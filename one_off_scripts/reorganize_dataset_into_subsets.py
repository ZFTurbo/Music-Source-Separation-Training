import numpy as np
np.float_ = np.float64
import musdb
import os
import shutil

# Define the dataset root path (update as needed)
dataset_path = os.path.expanduser("~/musdb-sample")

# Define output paths for train, validation, and test folders
output_path = os.path.join(dataset_path, "organized")
os.makedirs(output_path, exist_ok=True)
train_path = os.path.join(output_path, "train")
val_path = os.path.join(output_path, "validation")
test_path = os.path.join(output_path, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Initialize MUSDB database
mus = musdb.DB(root=dataset_path)

# Function to copy files to their respective subsets
def organize_tracks(mus_subset, subset_path):
    for track in mus_subset.tracks:
        source_path = track.path  # Path to the original track file
        track_name = track.name.replace(" ", "_")  # Sanitize track name
        target_path = os.path.join(subset_path, f"{track_name}.stem.mp4")
        shutil.copy2(source_path, target_path)
        print(f"Copied {track.name} to {subset_path}")

# Organize train, validation, and test tracks
print("Organizing training tracks...")
train_subset = musdb.DB(root=dataset_path, subsets="train", split="train")
organize_tracks(train_subset, train_path)

print("\nOrganizing validation tracks...")
val_subset = musdb.DB(root=dataset_path, subsets="train", split="valid")
organize_tracks(val_subset, val_path)

print("\nOrganizing test tracks...")
test_subset = musdb.DB(root=dataset_path, subsets="test")
organize_tracks(test_subset, test_path)

print("\nReorganization complete!")
print(f"Organized dataset is located at: {output_path}")