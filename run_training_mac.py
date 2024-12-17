import subprocess
import argparse
from datetime import datetime

# Define the main function
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the training script with configurable options.")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # Add arguments for the command-line options
    parser.add_argument("--model_type", type=str, default="scnet", help="Type of model to train")
    parser.add_argument("--config_path", type=str, default="configs/config_musdb18_scnet.yaml", help="Path to the config file")
    parser.add_argument("--results_path", type=str, default=f"checkpoints/mac_{current_time}", help="Path to save results")
    parser.add_argument("--data_path", type=str, default="../data/MUSDB18/train", help="Path to training data")
    parser.add_argument("--valid_path", type=str, default="../data/MUSDB18/small_validation", help="Path to validation data")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0], help="GPU device IDs to use")
    parser.add_argument("--start_check_point", type=str, default="", help="modle checkpoiynt")


    # Parse the arguments
    args = parser.parse_args()

    # Build the command
    command = [
        "python", "train_optuna.py",
        "--model_type", args.model_type,
        "--config_path", args.config_path,
        "--results_path", args.results_path,
        "--data_path", args.data_path,
        "--valid_path", args.valid_path,
        "--num_workers", str(args.num_workers),
        "--start_check_point", args.start_check_point,
        "--device_ids", ",".join(map(str, args.device_ids))
    ]

    # Run the command
    try:
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
