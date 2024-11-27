import numpy as np
np.float_ = np.float64

import musdb
import os

# Define a path for the sample dataset (adjust as needed)
dataset_path = os.path.expanduser("~/deteletable_musdb_set")

# Initialize MUSDB with the specified path
mus = musdb.DB(root=dataset_path, download=True)

# Set up train and test subsets
mus_train = musdb.DB(root=dataset_path, subsets="train")
mus_test = musdb.DB(root=dataset_path, subsets="test")

# Train/Validation split within the train subset
mus_train_split = musdb.DB(root=dataset_path, subsets="train", split="train")
mus_valid_split = musdb.DB(root=dataset_path, subsets="train", split="valid")

# Print out the names of tracks in each subset
print("Training Tracks:")
for track in mus_train_split.tracks:
    print(f" - {track.name}")

print("\nValidation Tracks:")
for track in mus_valid_split.tracks:
    print(f" - {track.name}")

print("\nTest Tracks:")
for track in mus_test.tracks:
    print(f" - {track.name}")