import os
import json
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
MUSDB_STEMS = {'bass', 'drums', 'vocals'}
# trackType prefix of a-tonal percussion in MoisesDB data.json
# ("a-tonal_percussion_(claps,_shakers,_congas,_cowbell_etc)"); pitched percussion
# ("pitched_percussion_(mallets,_glockenspiel,_...)") is melodic and goes to 'other'
ATONAL_PERCUSSION_PREFIX = 'a-tonal_percussion'


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


def read_track_types(src_folder: str) -> Dict[str, str]:
    """
    Reads MoisesDB data.json of a track and returns a mapping {file id: trackType}.
    Returns an empty dict if data.json is missing or broken.
    """
    path = os.path.join(src_folder, 'data.json')
    try:
        with open(path, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    return {track['id']: track.get('trackType', '')
            for stem in meta.get('stems', []) for track in stem.get('tracks', [])}


def percussion_target(file_name: str, track_types: Dict[str, str], stems: List[str], percussion: str) -> str:
    """
    Returns the output category for a file from the MoisesDB 'percussion' folder
    when 'percussion' itself is not one of the requested stems.
    - 'other': all percussion goes to 'other'
    - 'drums': all percussion goes to 'drums'
    - 'split': a-tonal percussion goes to 'drums', pitched percussion goes to 'other'
    """
    if 'drums' not in stems or percussion == 'other':
        return 'other'
    if percussion == 'drums':
        return 'drums'
    track_type = track_types.get(os.path.splitext(file_name)[0], '')
    return 'drums' if track_type.startswith(ATONAL_PERCUSSION_PREFIX) else 'other'


def files_to_categories(src_folder: str, stems: List[str], percussion: str = 'split') -> Dict[str, List[str]]:
    """
    Distributes all .wav files of a MoisesDB track between the requested stems and 'other'.
    A stem folder whose name is in stems goes to that stem, everything else goes to 'other'.
    Files from the 'percussion' folder are distributed according to the percussion mode
    (see percussion_target), unless 'percussion' is one of the requested stems.
    Parameters:
    - src_folder (str): Path to the MoisesDB track folder.
    - stems (List[str]): Requested stems (without 'other').
    - percussion (str): 'split', 'drums' or 'other'.
    Returns:
    - Dict[str, List[str]]: A Dict with keys as categories, values as sorted lists of paths to .wav files.
    """
    files = {category: [] for category in list(stems) + ['other']}
    track_types = None

    for folder in sorted(os.listdir(src_folder)):
        folder_path = os.path.join(src_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        stem_name = folder.lower()
        for f in sorted(os.listdir(folder_path)):
            if f.startswith('.') or not f.lower().endswith('.wav'):
                continue
            if stem_name in stems:
                category = stem_name
            elif stem_name == 'percussion':
                if track_types is None:
                    track_types = read_track_types(src_folder)
                    if percussion == 'split' and not track_types:
                        print(f"Warning: no data.json in {src_folder}, percussion goes to 'other'")
                category = percussion_target(f, track_types, stems, percussion)
            else:
                category = 'other'
            files[category].append(os.path.join(folder_path, f))
    return files


def is_track_done(dest_folder: str, stems: List[str]) -> bool:
    """
    Checks that all output files of a track already exist in dest_folder.
    """
    names = list(stems) + ['other', 'mixture']
    return all(os.path.isfile(os.path.join(dest_folder, f"{name}.wav")) for name in names)


# Processes a folder containing audio tracks, copying and combining necessary files into the target structure
def process_folder(src_folder: str, dest_folder: str, stems: List[str], trim: bool = False,
                   subtype: str = 'FLOAT', percussion: str = 'split') -> None:
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
    - percussion (str): How to distribute the MoisesDB 'percussion' folder, see percussion_target.
    """
    categories = list(stems) + ['other']
    all_files = files_to_categories(src_folder, stems, percussion)

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


# Wrapper function for 'process_folder' that unpacks the arguments and catches errors
def process_folder_wrapper(args: Tuple[str, str, List[str], bool, str, str]) -> Optional[Tuple[str, str]]:
    """
    A wrapper function for 'process_folder' that unpacks the arguments.
    An error in one track does not stop the whole conversion.
    Parameters:
    - args (Tuple[str, str, List[str], bool, str, str]): source folder, destination folder,
      stems, trim flag, output subtype and percussion mode.
    Returns:
    - None on success, (src_folder, error message) on failure.
    """
    src_folder, dest_folder, stems, trim, subtype, percussion = args
    try:
        process_folder(src_folder, dest_folder, stems, trim, subtype, percussion)
    except Exception as e:
        return src_folder, f"{type(e).__name__}: {e}"
    return None


def list_tracks(src_root: str, max_folders: int) -> List[str]:
    """
    Returns sorted names of track folders in src_root, at most max_folders of them.
    """
    tracks = []
    for folder in sorted(os.listdir(src_root)):
        if os.path.isdir(os.path.join(src_root, folder)):
            tracks.append(folder)
        else:
            print(f"Skip {os.path.join(src_root, folder)} — not dir")
    return tracks[:max_folders]


def count_categories(src_folder: str, stems: List[str], percussion: str) -> int:
    """
    Counts how many output categories (requested stems + 'other') of a track are not silent.
    """
    files = files_to_categories(src_folder, stems, percussion)
    return sum(1 for category_files in files.values() if category_files)


def select_valid_tracks(src_root: str, tracks: List[str], valid_dir: str, valid_size: int,
                        stems: List[str], percussion: str) -> List[str]:
    """
    Selects validation tracks. Tracks already present in valid_dir stay in validation
    (so that a rerun never moves them back to train). The rest is filled with the tracks
    that have the most non-silent categories; ties are broken by folder name, so the
    selection is reproducible.
    """
    existing = []
    if os.path.isdir(valid_dir):
        existing = [t for t in tracks if os.path.isdir(os.path.join(valid_dir, t))]
    selected = existing[:]
    if len(selected) < valid_size:
        candidates = [t for t in tracks if t not in selected]
        counts = {t: count_categories(os.path.join(src_root, t), stems, percussion) for t in candidates}
        candidates.sort(key=lambda t: (-counts[t], t))
        for t in candidates[:valid_size - len(selected)]:
            print(f'Valid track: {t}, non-silent categories: {counts[t]}')
            selected.append(t)
    return selected


# Converts MoisesDB dataset to MUSDB18 format
def convert_dataset(src_root: str, jobs: List[Tuple[str, str]], stems: List[str], num_workers: int = 4,
                    trim: bool = False, subtype: str = 'FLOAT', percussion: str = 'split',
                    skip_existing: bool = False) -> Tuple[int, List[Tuple[str, str]]]:
    """
    Converts MoisesDB tracks to MUSDB18 format.
    Parameters:
    - src_root (str): Root directory of the MoisesDB dataset.
    - jobs (List[Tuple[str, str]]): Pairs (track folder name, destination root).
    - stems (List[str]): Requested stems (without 'other').
    - num_workers (int): Number of parallel workers for processing.
    - trim (bool): If True, trim all stems to the length of the shortest one.
    - subtype (str): soundfile subtype for output WAV files.
    - percussion (str): How to distribute the MoisesDB 'percussion' folder.
    - skip_existing (bool): If True, skip tracks whose output files already exist.
    Returns:
    - Tuple[int, List[Tuple[str, str]]]: Number of processed tracks and list of failed tracks with errors.
    """
    folders_to_process = []
    skipped = 0
    for track, dest_root in jobs:
        dest_folder = os.path.join(dest_root, track)
        if skip_existing and is_track_done(dest_folder, stems):
            skipped += 1
            continue
        folders_to_process.append((os.path.join(src_root, track), dest_folder, stems, trim, subtype, percussion))
    if skipped:
        print(f'Skipped {skipped} already converted tracks')

    failed = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_folder_wrapper, folders_to_process),
                           total=len(folders_to_process), desc="Converting tracks"):
            if result is not None:
                failed.append(result)
    return len(folders_to_process) - len(failed), failed


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for converting MoisesDB to MUSDB18 format.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser(description="Convert MoisesDB to MUSDB18 format")
    parser.add_argument('--src_dir', type=str, required=True, help="Source directory with MoisesDB tracks")
    parser.add_argument('--dest_dir', type=str, required=True, help="Directory to save tracks in MUSDB18")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help="Num of processors")
    parser.add_argument('--max_folders', type=int, default=240, help="Num of folders to use")
    parser.add_argument('--create_valid',  action='store_true', help="Create valid folders or not")
    parser.add_argument('--valid_dir', type=str, default=None,
                        help="Directory for valid. Default: 'valid' folder next to dest_dir. "
                             "Tracks present in this folder are never written to dest_dir")
    parser.add_argument('--valid_size', type=int, default=10, help="Num of folders to use in valid")
    parser.add_argument("--stems", nargs='+', type=str, default=['bass', 'drums', 'vocals'],
                        choices=['drums', 'guitar', 'vocals', 'bass', 'other_keys', 'piano', 'percussion',
                                 'bowed_strings', 'wind', 'other_plucked'], help='List of stems to use.')
    parser.add_argument('--percussion', type=str, default='split', choices=['split', 'drums', 'other'],
                        help="Where to put the MoisesDB 'percussion' folder if it is not in --stems: "
                             "'split' - a-tonal percussion (claps, shakers, congas) to drums and pitched "
                             "percussion (mallets, glockenspiel) to other, by trackType from data.json; "
                             "'drums' - everything to drums; 'other' - everything to other")
    parser.add_argument('--mixture_name', type=str, default='mixture.wav', help="Name of mixture tracks")
    parser.add_argument('--trim', action='store_true',
                        help="Trim all stems to the length of the shortest one (cuts audio of longer stems). "
                             "By default all stems are padded with silence to the longest one")
    parser.add_argument('--subtype', type=str, default='FLOAT', choices=['FLOAT', 'PCM_24', 'PCM_16'],
                        help="Output WAV subtype. FLOAT keeps the mixture equal to the sum of stems without clipping; "
                             "PCM subtypes scale each track by a common gain if its peak exceeds 1.0")
    parser.add_argument('--skip_existing', action='store_true',
                        help="Skip tracks whose output files already exist (to resume an interrupted conversion)")

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
    stems = [stem.lower() for stem in args.stems]
    max_folders = args.max_folders
    trim = args.trim
    subtype = args.subtype
    percussion = args.percussion
    valid_folder = args.valid_dir
    if valid_folder is None:
        valid_folder = os.path.join(os.path.dirname(os.path.abspath(destination_directory)), 'valid')
    if os.path.abspath(valid_folder) == os.path.abspath(destination_directory):
        raise ValueError(f'--valid_dir must differ from --dest_dir: {valid_folder}')

    print(f'num_workers: {num_workers}, '
          f'categories: {stems + ["other"]}, '
          f'percussion: {"own stem" if "percussion" in stems else percussion}, '
          f'trim: {trim}, '
          f'subtype: {subtype}')

    missing_musdb = MUSDB_STEMS - set(stems)
    if missing_musdb:
        print(f'Warning: stems {sorted(missing_musdb)} are not in --stems and will be mixed into other.wav, '
              f'the result is not MUSDB18 4-stem format')

    tracks = list_tracks(source_directory, max_folders)

    if args.create_valid:
        valid_tracks = select_valid_tracks(source_directory, tracks, valid_folder, args.valid_size,
                                           stems, percussion)
    elif os.path.isdir(valid_folder):
        # Never write tracks that are already in validation to train
        valid_tracks = [t for t in tracks if os.path.isdir(os.path.join(valid_folder, t))]
    else:
        valid_tracks = []
    if valid_tracks:
        print(f'{len(valid_tracks)} tracks go to valid: {valid_folder}')
        # Remove train copies of validation tracks left from previous runs
        for track in valid_tracks:
            train_copy = os.path.join(destination_directory, track)
            if os.path.isdir(train_copy):
                print(f'Remove {train_copy} from train, it is a validation track')
                shutil.rmtree(train_copy)

    valid_set = set(valid_tracks)
    jobs = [(track, valid_folder if track in valid_set else destination_directory) for track in tracks]

    processed, failed = convert_dataset(source_directory, jobs, stems, num_workers=num_workers, trim=trim,
                                        subtype=subtype, percussion=percussion, skip_existing=args.skip_existing)

    print(f'Processed {processed} of {len(tracks)} tracks ({len(valid_tracks)} valid), '
          f'time: {time.time() - start:.2f} sec')
    if failed:
        print(f'Failed {len(failed)} tracks:')
        for src_folder, error in failed:
            print(f'  {src_folder}: {error}')

    print('The end!')


if __name__ == '__main__':
    main(None)
