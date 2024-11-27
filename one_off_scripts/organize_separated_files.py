import os
import shutil

# Path to the folder containing all the stem files
source_folder = '/Users/kaimikkelsen/canada_compute/Music-Source-Separation-Training/separated/mdx23c/test'

import os
import shutil

# Path to the folder containing all the stem files
# source_folder = '/path/to/your/source/folder'

#Define stem type mappings
stem_mappings = {
    'bass': 'bass.wav',
    'vocals': 'vocals.wav',
    'other': 'other.wav',
    'drums': 'drums.wav'
}

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".wav"):
        # Split filename to get the song name and stem type
        try:
            song_name, stem_type = filename.split(".stem_")
        except ValueError:
            print(f"Skipping file with unexpected format: {filename}")
            continue

        # Create a folder for the song
        song_folder = os.path.join(source_folder, song_name)

        # Delete the folder if it already exists
        if os.path.exists(song_folder):
            shutil.rmtree(song_folder)  # Remove the existing folder and its contents

        # Create a new folder for the song
        os.makedirs(song_folder)

        # Map the stem type to the new filename if it matches the expected pattern
        new_filename = stem_mappings.get(stem_type, None)
        if new_filename is None:
            print(f"Skipping unrecognized stem type in file: {filename}")
            continue

        # Move and rename the stem file
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(song_folder, new_filename)
        shutil.move(source_path, destination_path)

print("Files organized and renamed by song name and stem type.")
# Define stem type mappings
stem_mappings = {
    'stem_bass': 'bass.wav',
    'stem_vocals': 'vocals.wav',
    'stem_other': 'other.wav',
    'stem_drums': 'drums.wav'
}

# Loop through each file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".wav"):
        # Split filename to get the song name and stem type
        song_name, stem_type = filename.split(".stem_")
        
        # Create a folder for the song if it doesn't already exist
        song_folder = os.path.join(source_folder, song_name)
        os.makedirs(song_folder, exist_ok=True)
        
        # Map the stem type to the new filename
        new_filename = stem_mappings.get(f'stem_{stem_type}', filename)
        
        # Move and rename the stem file
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(song_folder, new_filename)
        shutil.move(source_path, destination_path)
        
print("Files organized and renamed by song name and stem type.")



#Mapping of the current naming convention to the new names
file_mapping = {
    'stem_bass.wav': 'bass.wav',
    'stem_vocals.wav': 'vocals.wav',
    'stem_other.wav': 'other.wav',
    'stem_drums.wav': 'drums.wav'
}

# Iterate through each folder in the parent directory
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)
    
    if os.path.isdir(folder_path):
        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            # Check if the file is one of the ones we want to rename
            for old_name, new_name in file_mapping.items():
                if old_name in file_name:
                    # Construct full file paths
                    old_file_path = os.path.join(folder_path, file_name)
                    new_file_path = os.path.join(folder_path, new_name)
                    
                    # Rename the file
                    os.rename(old_file_path, new_file_path)
                    print(f'Renamed: {old_file_path} to {new_file_path}')