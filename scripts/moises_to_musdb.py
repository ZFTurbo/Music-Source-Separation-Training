import os
import soundfile as sf
import numpy as np
from tqdm import tqdm
import shutil
import time
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional, Union
import argparse


TARGET_SR = 44100
TARGET_CHANNELS = 2


def load_audio(path: str) -> np.ndarray:
    """
    Reads an audio file as float32 stereo array with shape (num_samples, 2).
    Mono files are duplicated to both channels. Files with a sample rate other
    than TARGET_SR or with more than 2 channels raise ValueError.
    """
    data, sr = sf.read(path, dtype='float32', always_2d=True)
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate {sr} != {TARGET_SR}: {path}")
    if data.shape[1] == 1:
        data = np.repeat(data, TARGET_CHANNELS, axis=1)
    elif data.shape[1] != TARGET_CHANNELS:
        raise ValueError(f"Unsupported number of channels {data.shape[1]}: {path}")
    return data


def fit_length(data: np.ndarray, length: int) -> np.ndarray:
    """
    Pads with zeros or trims an array of shape (num_samples, channels) to the given length.
    """
    if len(data) < length:
        return np.pad(data, ((0, length - len(data)), (0, 0)), 'constant')
    return data[:length]


def combine_audio_files(files: List[str]) -> Tuple[np.ndarray, int]:
    """
    Combines multiple audio files into one by overlaying them.
    Shorter files are padded with zeros to the length of the longest one.
    Parameters:
    - files (List[str]): List of file paths to be combined.
    Returns:
    - Tuple[np.ndarray, int]: A Tuple containing the combined float32 stereo audio and sample rate.
    """
    combined_data = load_audio(files[0])

    for file in files[1:]:
        data = load_audio(file)
        length = max(len(data), len(combined_data))
        combined_data = fit_length(combined_data, length) + fit_length(data, length)

    return combined_data, TARGET_SR


# Finds all .wav files located in folders that do not contain specified categories in their folder names
def files_to_categories(src_folder: str, categories: List[str]) -> Dict[str, List[str]]:
    """
    Finds all .wav files located in folders that do not contain specified categories
    in their folder names, within the given src_folder directory.
    Parameters:
    - src_folder (str): Path to the main directory containing subdirectories with files.
    - categories (List[str]): Keywords that should not be part of the folder's name.
    Returns:
    - Dict[str, List[str]]: A Dict with keys as categories, values as lists of paths to .wav files found.
    """
    files = {category: [] for category in categories + ['other']}

    for folder in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, folder)
        if os.path.isdir(folder_path):
            if folder.lower() in categories:
                stem = folder.lower()
            else:
                stem = 'other'
            for f in os.listdir(folder_path):
                if f.endswith('.wav'):
                    files[stem].append(os.path.join(folder_path, f))
    return files


# Processes a folder containing audio tracks, copying and combining necessary files into the target structure
def process_folder(src_folder: str, dest_folder: str, stems: List[str], trim: bool = False,
                   subtype: str = 'FLOAT') -> None:
    """
    Processes a folder containing audio tracks, copying and combining necessary files into the target structure.
    All output files are 44.1 kHz stereo with the same length, and mixture.wav is exactly
    the sum of the written stems. Missing stems are written as silence.
    Parameters:
    - src_folder (str): Path to the source folder of MoisesDB.
    - dest_folder (str): Path to the target folder for MUSDB18.
    - stems (List[str]): List of stem categories to process.
    - trim (bool): If True, trim all stems to the length of the shortest one,
      otherwise pad all stems with zeros to the length of the longest one.
    - subtype (str): soundfile subtype for output WAV files. 'FLOAT' keeps values
      outside [-1, 1] intact. For PCM subtypes all files of the track are scaled
      by one common gain if needed, so that nothing clips and mixture stays equal to the sum of stems.
    """
    categories = list(stems) + ['other']
    all_files = files_to_categories(src_folder, stems)

    if not any(all_files[category] for category in categories):
        print(f"Skip {src_folder} — no .wav files found")
        return

    stem_data = {}
    for category in categories:
        files = all_files[category]
        if files:
            stem_data[category], _ = combine_audio_files(files)

    lengths = [len(data) for data in stem_data.values()]
    length = min(lengths) if trim else max(lengths)
    if min(lengths) != max(lengths):
        print(f"Warning: stem lengths differ in {src_folder} ({min(lengths)} != {max(lengths)}), "
              f"{'trimmed' if trim else 'padded'} to {length}")

    for category in categories:
        if category in stem_data:
            stem_data[category] = fit_length(stem_data[category], length)
        else:
            stem_data[category] = np.zeros((length, TARGET_CHANNELS), dtype=np.float32)

    mixture = np.sum([stem_data[category] for category in categories], axis=0, dtype=np.float32)

    if subtype != 'FLOAT':
        peak = max(float(np.abs(data).max(initial=0.0)) for data in list(stem_data.values()) + [mixture])
        if peak > 0.999:
            gain = np.float32(0.999 / peak)
            print(f"Warning: peak {peak:.3f} in {src_folder}, all files scaled by {gain:.3f} to avoid clipping")
            stem_data = {category: data * gain for category, data in stem_data.items()}
            mixture = np.sum([stem_data[category] for category in categories], axis=0, dtype=np.float32)

    os.makedirs(dest_folder, exist_ok=True)
    for category in categories:
        sf.write(os.path.join(dest_folder, f"{category}.wav"), stem_data[category], TARGET_SR, subtype=subtype)
    sf.write(os.path.join(dest_folder, "mixture.wav"), mixture, TARGET_SR, subtype=subtype)


# Wrapper function for 'process_folder' that unpacks the arguments
def process_folder_wrapper(args: Tuple[str, str, List[str], bool, str]) -> None:
    """
    A wrapper function for 'process_folder' that unpacks the arguments.
    Parameters:
    - args (Tuple[str, str, List[str], bool, str]): A Tuple containing the source folder, destination folder paths,
      stems, trim flag and output subtype.
    """
    src_folder, dest_folder, stems, trim, subtype = args
    return process_folder(src_folder, dest_folder, stems, trim, subtype)


# Converts MoisesDB dataset to MUSDB18 format for a specified number of folders
def convert_dataset(src_root: str, dest_root: str, stems: List[str], max_folders: int = 240, num_workers: int = 4,
                    trim: bool = False, subtype: str = 'FLOAT') -> None:
    """
    Converts MoisesDB dataset to MUSDB18 format for a specified number of folders.
    Parameters:
    - src_root (str): Root directory of the MoisesDB dataset.
    - dest_root (str): Root directory where the new dataset will be saved.
    - max_folders (int): Maximum number of folders to process.
    - num_workers (int): Number of parallel workers for processing.
    - trim (bool): If True, trim all stems to the length of the shortest one.
    - subtype (str): soundfile subtype for output WAV files.
    """
    folders_to_process = []
    for folder in os.listdir(src_root):
        if len(folders_to_process) >= max_folders:
            break

        src_folder = os.path.join(src_root, folder)
        dest_folder = os.path.join(dest_root, folder)

        if os.path.isdir(src_folder):
            folders_to_process.append((src_folder, dest_folder, stems, trim, subtype))
        else:
            print(f"Skip {src_folder} — not dir")

    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_folder_wrapper, folders_to_process),
                      total=len(folders_to_process), desc="Converting tracks"):
            pass


# Count number of subfolders in a folder
def count_folders_in_folder(args_to_func) -> Dict[str, int]:
    """
    Counts the number of subfolders inside a given folder.

    Parameters:
    - folder_path (str): Path to the folder where the count is needed.

    Returns:
    - Dict[str, int]: A dictionary with folder paths as keys and subfolder counts as values.
    """
    folder_count = 0
    folder_path, stems = args_to_func
    if os.path.isdir(folder_path):
        # Count subfolders in stems
        folder_count = len([f for f in os.listdir(folder_path)
                            if os.path.isdir(os.path.join(folder_path, f)) and f in stems])
        # For other.wav
        if any(os.path.isdir(os.path.join(folder_path, f)) and f not in stems for f in os.listdir(folder_path)):
            folder_count += 1

    return {folder_path: folder_count}


# Parallel count of subfolders in each folder inside src_folder
def count_folders_parallel(src_folder: str, stems, num_workers: int = 4) -> Dict[str, int]:
    """
    Parallelly counts the number of subfolders in each folder inside src_folder.

    Parameters:
    - src_folder (str): Root folder containing subfolders to count.

    Returns:
    - Dict[str, int]: Dictionary with folder paths as keys and subfolder counts as values.
    """
    # Get list of all folders inside src_folder
    folders_to_process = [os.path.join(src_folder, folder) for folder in os.listdir(src_folder) if
                          os.path.isdir(os.path.join(src_folder, folder))]

    args_to_func = [(folder, stems) for folder in folders_to_process]

    # Parallelly process each folder using pool.map
    with Pool(num_workers) as pool:
        results = pool.map(count_folders_in_folder, args_to_func)

    # Merge results from different processes
    merged_counts = {}
    for result in results:
        merged_counts.update(result)

    return merged_counts


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Copy mixture files from VALID_DIR to INFERENCE_DIR")
    parser.add_argument('--src_dir', type=str, required=True, help="Source directory with MoisesDB tracks")
    parser.add_argument('--dest_dir', type=str, required=True, help="Directory to save tracks in MUSDB18")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help="Num of processors")
    parser.add_argument('--max_folders', type=int, default=240, help="Num of folders to use")
    parser.add_argument('--create_valid',  action='store_true', help="Create valid folders or not")
    parser.add_argument('--valid_dir', type=str, default=r'\valid', help="Directory for valid")
    parser.add_argument('--valid_size', type=int, default=10, help="Num of folders to use in valitd")
    parser.add_argument("--stems", nargs='+', type=str, default=['bass', 'drums', 'vocals'],
                        choices=['drums', 'guitar', 'vocals', 'bass', 'other_keys', 'piano', 'percussion',
                                 'bowed_strings', 'wind', 'other_plucked'], help='List of stems to use.')
    parser.add_argument('--mixture_name', type=str, default='mixture.wav', help="Name of mixture tracks")
    parser.add_argument('--trim', action='store_true', help="Trim all stems to the length of the shortest one")
    parser.add_argument('--subtype', type=str, default='FLOAT', choices=['FLOAT', 'PCM_24', 'PCM_16'],
                        help="Output WAV subtype. FLOAT keeps the mixture equal to the sum of stems without clipping; "
                             "PCM subtypes scale each track by a common gain if its peak exceeds 1.0")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    start = time.time()

    args = parse_args(args)

    source_directory = args.src_dir
    destination_directory = args.dest_dir
    num_workers = args.num_workers
    stems = args.stems
    max_folders = args.max_folders
    trim = args.trim
    subtype = args.subtype

    print(f'num_workers: {num_workers}, '
          f'categories: {stems + ["other"]}, '
          f'trim: {trim}, '
          f'subtype: {subtype}')

    convert_dataset(source_directory, destination_directory, stems, max_folders=max_folders,
                    num_workers=num_workers, trim=trim, subtype=subtype)

    print(f'All {max_folders} files have been processed, time: {time.time() - start:.2f} sec')

    if args.create_valid:
        # Count folders in the MoisesDB dataset
        result = count_folders_parallel(source_directory, stems)
        result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        # valid_size = min(args.valid_size, max_folders)
        valid_size = 10
        list_folders = list(result.keys())
        print(f'Top {valid_size} folders:')
        for track in list(result.items())[:valid_size]:
            print(track)

        valid_folder = args.valid_dir
        train_tracks_folder = destination_directory

        # Create valid folder if not exists
        if not os.path.exists(valid_folder):
            os.makedirs(valid_folder)

        num_val = 0

        # Copy folders from train_tracks to valid folder
        for folder in list_folders:
            folder_name = os.path.basename(folder)  # Get folder name

            # Form the path to the folder in train_tracks
            source_folder = os.path.join(train_tracks_folder, folder_name)

            # If the folder exists in train_tracks, copy it to valid
            if os.path.exists(source_folder):
                destination = os.path.join(valid_folder, folder_name)
                shutil.copytree(source_folder, destination)
                shutil.rmtree(source_folder)
                num_val += 1
                print(f'Folder: {folder}, num_stems: {result[folder]}')
                if num_val >= valid_size:
                    break
            else:
                print(f"Folder {source_folder} not found.")

    print('The end!')


if __name__ == '__main__':
    main(None)
