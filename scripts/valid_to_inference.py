import os
import shutil
import argparse
from typing import List, Optional

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters:
    ----------
    args : Optional[List[str]]
        List of arguments passed from the command line. If None, uses sys.argv.

    Returns:
    -------
    argparse.Namespace
        The parsed arguments containing valid_dir, inference_dir, and mixture_name.
    """
    parser = argparse.ArgumentParser(description="Copy mixture files from VALID_DIR to INFERENCE_DIR")
    parser.add_argument('--valid_dir', type=str, required=True, help="Directory with valid tracks")
    parser.add_argument('--inference_dir', type=str, required=True, help="Directory to save inference tracks")
    parser.add_argument('--mixture_name', type=str, default='mixture.wav', help="Name of mixture tracks (default: 'mixture.wav')")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


def main(args: Optional[argparse.Namespace] = None) -> None:
    """
    Main function to copy mixture files from valid directory to inference directory.

    Parameters:
    ----------
    args : Optional[argparse.Namespace]
        The parsed arguments containing valid_dir, inference_dir, and mixture_name.
    """
    args = parse_args(args)

    valid_dir = args.valid_dir
    inference_dir = args.inference_dir
    mixture_name = args.mixture_name

    # Create the inference directory if it doesn't exist
    os.makedirs(inference_dir, exist_ok=True)

    # Walk through the valid directory to find and copy mixture files
    for root, _, files in os.walk(valid_dir):
        for file in files:
            if file == mixture_name:
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
    main(None)
