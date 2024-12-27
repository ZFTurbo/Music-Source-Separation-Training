import yaml
import os
import sys
import argparse
from omegaconf import OmegaConf
from typing import Union, List
from ml_collections import ConfigDict


def save_config(config: Union[ConfigDict, OmegaConf], save_path: str):
    """
    Save a configuration object (ConfigDict or OmegaConf) to a file.

    Parameters:
    ----------
    config : Union[ConfigDict, OmegaConf]
        The configuration object to save.
    save_path : str
        The path where the configuration file will be saved.

    Raises:
    ------
    ValueError:
        If the configuration type is not supported.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        with open(save_path, 'w') as f:
            if isinstance(config, ConfigDict):
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            elif isinstance(config, OmegaConf):
                OmegaConf.save(config, save_path)
            else:
                raise ValueError("Unsupported configuration type. Supported types: ConfigDict, OmegaConf.")
    except Exception as e:
        raise ValueError(f"Error saving configuration: {e}")


def create_test_config(original_config_path: str, new_config_path: str, model_type: str):
    """
    Create a test configuration file based on an existing configuration.

    Parameters:
    ----------
    original_config_path : str
        Path to the original configuration file.
    new_config_path : str
        Path where the new configuration file will be saved.
    model_type : str
        The type of model (e.g., 'scnet', 'htdemucs').

    Returns:
    -------
    None
    """
    from utils import load_config

    config = load_config(model_type=model_type, config_path=original_config_path)

    config['inference']['batch_size'] = 1
    config['training']['batch_size'] = 1
    config['training']['gradient_accumulation_steps'] = 1
    config['training']['num_epochs'] = 2
    config['training']['num_steps'] = 3

    save_config(config, new_config_path)
    print(f"Test config created at: {new_config_path}")


def parse_args(args: Union[List[str], None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        args: List of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_config", type=str, default="", help="Path to the original config file.")
    parser.add_argument("--model_type", type=str, default="", help="Model type")
    parser.add_argument("--new_config", type=str, default="", help="Path to save the new test configuration file.")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    # Determine the default path for the new configuration if not provided
    if not args.new_config:
        original_dir = os.path.dirname(args.orig_config)
        tests_dir = os.path.join("tests", original_dir)
        os.makedirs(tests_dir, exist_ok=True)
        args.new_config = os.path.join(tests_dir, os.path.basename(args.orig_config))

    return args


def redact_config(args):
    # Ensure proper imports for utilities
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    args = parse_args(args)

    # Create the test configuration
    create_test_config(args.orig_config, args.new_config, args.model_type)
    return args.new_config


if __name__ == '__main__':
    args = [
        '--orig_config', 'configs/config_musdb18_scnet_large_starrytong.yaml',
        '--model_type', 'scnet'
    ]
    redact_config(None)
