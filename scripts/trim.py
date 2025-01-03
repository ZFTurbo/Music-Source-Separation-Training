import os
from sys import argv

import soundfile as sf
import argparse
import time
from typing import Union, List, Dict


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory', type=str, help="Path to the input directory.")
    parser.add_argument('--output_directory', type=str, help="Path to the output directory.")
    parser.add_argument('--start_sec', type=float, default=20.0)
    parser.add_argument('--end_sec', type=float, default=30.0)
    parser.add_argument('--codec', type=str, default='wav')
    parser.add_argument('--max_folders', type=int, default=float('inf'), help="Maximum number of folders to process.")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if not args.output_directory:
        original_dir = os.path.dirname(args.input_directory)
        tests_dir = os.path.join("tests", original_dir)
        os.makedirs(tests_dir, exist_ok=True)
        args.output_directory = os.path.join(tests_dir, os.path.basename(args.input_directory))

    return args


def trim_wav(input_file: str, output_file: str, start_sec: float, end_sec: float, codec: str):
    data, samplerate = sf.read(input_file)
    start_sample = int(start_sec * samplerate)
    end_sample = int(end_sec * samplerate)
    trimmed_data = data[start_sample:end_sample]
    sf.write(output_file, trimmed_data, samplerate, format=codec)


def trim_directory(args):
    args = parse_args(args)
    input_directory = args.input_directory
    output_directory = args.output_directory
    start_sec = args.start_sec
    end_sec = args.end_sec
    codec = args.codec
    max_folder = args.max_folders

    folder_count = 0
    start_time = time.time()

    for root, dirs, files in os.walk(input_directory):
        if folder_count >= max_folder:
            break
        if os.path.relpath(root, input_directory) != '.':
            print(f'Processing folder: {os.path.relpath(root, input_directory)}')

            relative_path = os.path.relpath(root, input_directory)
            target_folder = os.path.join(output_directory, relative_path)
            os.makedirs(target_folder, exist_ok=True)

            for filename in files:
                if filename.endswith(f'.{codec}'):
                    input_file = os.path.join(root, filename)
                    output_file = os.path.join(target_folder, filename.replace(f'.{codec}', f'.{codec}'))
                    try:
                        trim_wav(input_file, output_file, start_sec, end_sec, codec)
                    except Exception as e:
                        print(f'Error processing {filename} in folder {root}: {e}')

            folder_count += 1

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Processing complete. Total time: {total_time:.2f} seconds.')
    return output_directory


if __name__ == '__main__':
    trim_directory(None)
