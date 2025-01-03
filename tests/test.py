import os
import sys
import argparse

# Добавляем корень репозитория в системный путь
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from valid import check_validation
from inference import proc_folder
from train import train_model
from scripts.redact_config import redact_config
from scripts.valid_to_inference import copying_files
from scripts.trim import trim_directory

base_args = {
    'device_ids': '0',
    'model_type': '',
    'start_check_point': '',
    'config_path': '',
    'data_path': '',
    'valid_path': '',
    'results_path': 'tests/train_results',
    'store_dir': 'tests/valid_inference_result',
    'input_folder': '',
    'metrics': ['neg_log_wmse', 'l1_freq', 'si_sdr', 'sdr', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'],
    'max_folders': 2
}


def parse_args(dict_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_train", action='store_true', help="Check train or not")
    parser.add_argument("--check_valid", action='store_true', help="Check train or not")
    parser.add_argument("--check_inference", action='store_true', help="Check train or not")
    parser.add_argument('--device_ids', type=str, help='Device IDs for training/inference')
    parser.add_argument('--model_type', type=str, help='Model type')
    parser.add_argument('--start_check_point', type=str, help='Path to the checkpoint to start from')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--data_path', type=str, help='Path to the training data')
    parser.add_argument('--valid_path', type=str, help='Path to the validation data')
    parser.add_argument('--results_path', type=str, help='Path to save training results')
    parser.add_argument('--store_dir', type=str, help='Path to store validation/inference results')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder for inference')
    parser.add_argument('--metrics', nargs='+', help='List of metrics to evaluate')
    parser.add_argument('--max_folders', type=str, help='Maximum number of folders to process')
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3 or 4.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--use_multistft_loss", action='store_true',
                        help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        " While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def test_settings(dict_args, test_type):

    # Parse from cmd
    cli_args = parse_args(dict_args)

    # If args from cmd, add or replace in base_args
    for key, value in vars(cli_args).items():
        if value is not None:
            base_args[key] = value

    if test_type == 'user':
        # Check required arguments
        missing_args = [arg for arg in ['model_type', 'config_path', 'start_check_point', 'data_path', 'valid_path'] if
                        not base_args[arg]]
        if missing_args:
            missing_args_str = ', '.join(f'--{arg}' for arg in missing_args)
            raise ValueError(
                f"The following arguments are required but missing: {missing_args_str}."
                f" Please specify them either via command-line arguments or directly in `base_args`.")

        # Replace config
        base_args['config_path'] = redact_config({'orig_config': base_args['config_path'],
                                                  'model_type': base_args['model_type'],
                                                  'new_config': ''})

        # Trim train
        trim_args_train = {'input_directory': base_args['data_path'],
                           'max_folders': base_args['max_folders']}
        base_args['data_path'] = trim_directory(trim_args_train)
        # Trim valid
        trim_args_valid = {'input_directory': base_args['valid_path'],
                           'max_folders': base_args['max_folders']}
        base_args['valid_path'] = trim_directory(trim_args_valid)
    # Valid to inference
    if not base_args['input_folder']:
        tests_dir = os.path.join(os.path.dirname(base_args['valid_path']), 'for_inference')
        base_args['input_folder'] = tests_dir
    val_to_inf_args = {'valid_path': base_args['valid_path'],
                       'inference_dir': base_args['input_folder'],
                       'max_mixtures': 1}
    copying_files(val_to_inf_args)

    if base_args['check_valid']:
        valid_args = {key: base_args[key] for key in ['model_type', 'config_path', 'start_check_point',
                                               'store_dir', 'device_ids', 'num_workers', 'pin_memory', 'extension',
                                               'use_tta', 'metrics', 'lora_checkpoint', 'draw_spectro']}
        valid_args['valid_path'] = [base_args['valid_path']]
        print('Start validation.')
        check_validation(valid_args)
        print(f'Validation ended. See results in {base_args["store_dir"]}')

    if base_args['check_inference']:
        inference_args = {key: base_args[key] for key in ['model_type', 'config_path', 'start_check_point', 'input_folder',
                                               'store_dir', 'device_ids', 'extract_instrumental',
                                               'disable_detailed_pbar', 'force_cpu', 'flac_file', 'pcm_type',
                                               'use_tta', 'lora_checkpoint', 'draw_spectro']}

        print('Start inference.')
        proc_folder(inference_args)
        print(f'Inference ended. See results in {base_args["store_dir"]}')

    if base_args['check_train']:
        train_args = {key: base_args[key] for key in ['model_type', 'config_path', 'start_check_point', 'results_path',
                                               'data_path', 'dataset_type', 'valid_path', 'num_workers', 'pin_memory',
                                               'seed', 'device_ids', 'use_multistft_loss', 'use_mse_loss',
                                               'use_l1_loss', 'wandb_key', 'pre_valid', 'metrics',
                                               'metric_for_scheduler', 'train_lora', 'lora_checkpoint']}

        print('Start train.')
        train_model(train_args)

    print('End!')


if __name__ == "__main__":
    test_settings(None, 'user')
