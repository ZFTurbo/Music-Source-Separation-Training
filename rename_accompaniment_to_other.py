import os

def rename_accompaniment_to_other(parent_folder):
    # Traverse through all subfolders in the parent folder
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file == "accompaniment.wav":
                # Construct full file path
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, "other.wav")
                try:
                    # Rename the file
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                except Exception as e:
                    print(f"Failed to rename {old_path}: {e}")

# Specify the parent folder containing song subfolders
parent_folder = "/Users/kaimikkelsen/canada_compute/data/extracted_stems_test"

# Call the function
rename_accompaniment_to_other(parent_folder)