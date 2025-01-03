import os
import shutil
import argparse
from typing import List, Optional, Union, Dict


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """

    parser = argparse.ArgumentParser(description="Copy mixture files from VALID_PATH to INFERENCE_DIR")
    parser.add_argument('--valid_path', type=str, help="Directory with valid tracks")
    parser.add_argument('--inference_dir', type=str, help="Directory to save inference tracks")
    parser.add_argument('--mixture_name', type=str, default='mixture.wav', help="Name of mixture tracks (default: 'mixture.wav')")
    parser.add_argument('--max_mixtures', type=int, default=float('inf'), help="Maximum number of mixtures to process.")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def copying_files(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function to copy mixture files from valid directory to inference directory.

    Parameters:
    ----------
    args : Optional[argparse.Namespace]
        The parsed arguments containing valid_path, inference_dir, and mixture_name.
    """
    args = parse_args(args)

    valid_path = args.valid_path
    inference_dir = args.inference_dir
    mixture_name = args.mixture_name
    max_mixtures = args.max_mixtures
    # Create the inference directory if it doesn't exist
    os.makedirs(inference_dir, exist_ok=True)
    mixture_count = 0
    # Walk through the valid directory to find and copy mixture files
    for root, _, files in os.walk(valid_path):
        if mixture_count >= max_mixtures:
            break
        for file in files:
            if file == mixture_name:
                mixture_count += 1
                # Full path to the valid file
                source_path = os.path.join(root, file)

                # Track ID from the parent directory name
                track_id = os.path.basename(os.path.dirname(source_path))

                # Define target file path
                target_filename = f"{track_id}.wav"
                target_path = os.path.join(inference_dir, target_filename)

                # Copy the file to the inference directory
                shutil.copy2(source_path, target_path)

                print(f"Has copied: {source_path} -> {target_path}")

    print("Copying ends.")


if __name__ == '__main__':
    copying_files(None)
