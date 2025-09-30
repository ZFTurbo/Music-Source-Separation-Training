import os
import random
import time
import yaml
import wandb
import numpy as np
import torch
import argparse
from typing import Dict, List, Tuple, Union
from omegaconf import OmegaConf
from ml_collections import ConfigDict
import torch.distributed as dist
from torch import nn
import soundfile as sf


def parse_args_train(dict_args: Union[argparse.Namespace, Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    This function constructs an argument parser for model, dataset, training, and logging
    options, merges overrides from a provided dictionary (if any), and returns the parsed
    arguments. If `dict_args` is None, the arguments are parsed from `sys.argv`.

    Args:
        dict_args (Dict | None): Optional dictionary of argument overrides. Keys should
            match the defined CLI options.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all configuration
        values required for training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--load_optimizer", action='store_true',
                        help="Load optimizer state from checkpoint (if available)")
    parser.add_argument("--load_scheduler", action='store_true',
                        help="Load scheduler state from checkpoint (if available)")
    parser.add_argument("--load_epoch", action='store_true', help="Load epoch number from checkpoint (if available)")
    parser.add_argument("--load_best_metric", action='store_true',
                        help="Load best metric from checkpoint (if available)")
    parser.add_argument("--load_all_metrics", action='store_true',
                        help="Load all metrics from checkpoint (if available)")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--loss", type=str, nargs='+', choices=['masked_loss', 'mse_loss', 'l1_loss',
                                                                'multistft_loss', 'spec_masked_loss', 'spec_rmse_loss',
                                                                'log_wmse_loss'],
                        default=['masked_loss'], help="List of loss functions to use")
    parser.add_argument("--masked_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--mse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--l1_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--log_wmse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--multistft_loss_coef", type=float, default=0.001, help="Coef for loss")
    parser.add_argument("--spec_masked_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--spec_rmse_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--wandb_offline", action='store_true', help='local wandb')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora_peft", action='store_true', help="Training with LoRA from peft")
    parser.add_argument("--train_lora_loralib", action='store_true', help="Training with LoRA from loralib")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--each_metrics_in_name", action='store_true',
                        help="All stems in naming checkpoints")
    parser.add_argument("--use_standard_loss", action='store_true',
                        help="Roformers will use provided loss instead of internal")
    parser.add_argument("--save_weights_every_epoch", action='store_true',
                        help="Weights will be saved every epoch with all metric values")
    parser.add_argument("--persistent_workers", action='store_true',
                        help="dataloader persistent_workers")
    parser.add_argument("--prefetch_factor", type=int, default=None,
                        help="dataloader prefetch_factor")
    parser.add_argument("--set_per_process_memory_fraction", action='store_true',
                        help="using only VRAM, no RAM")
    parser.add_argument("--load_only_compatible_weights", action='store_true',
                        help="using only VRAM, no RAM")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    get_internal_loss = (args.model_type in ('mel_band_conformer',) or 'roformer' in args.model_type
                         ) and not args.use_standard_loss
    if get_internal_loss:
        args.loss = [f'{args.model_type}_loss']
    return args


def parse_args_valid(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for validation configuration.

    Builds the CLI for model selection, configuration paths, validation data
    locations, output/spectrogram saving options, device/runtime settings, and
    evaluation metrics. If `dict_args` is provided, its key–value pairs override
    or set the parsed arguments; otherwise arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all validation
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer,"
                             " bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint"
                                                                          " to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validate path")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="If --store_dir is set then code will generate spectrograms for resulted stems as well."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='List of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                             "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'neg_log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")


    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def parse_args_inference(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference configuration.

    Builds the CLI for model selection, configuration path, input/output handling,
    device/runtime options, test-time augmentation, and optional LoRA checkpoints.
    If `dict_args` is provided, its key–value pairs override or supply CLI options
    programmatically; otherwise, arguments are read from `sys.argv`.

    Args:
        dict_args (Union[Dict, None]): Optional mapping of argument names to values
            used to override or supply CLI options programmatically.

    Returns:
        argparse.Namespace: Parsed arguments namespace containing all inference
        configuration values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results as wav file")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Code will generate spectrograms for resulted stems."
                             " Value defines for how many seconds os track spectrogram will be generated.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24', 'FLOAT'], default='FLOAT',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint_peft", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--filename_template", type=str, default='{file_name}/{instr}',
                        help="Output filename template, without extension, using '/' for subdirectories. Default: '{file_name}/{instr}'")
    parser.add_argument("--lora_checkpoint_loralib", type=str, default='', help="Initial checkpoint to LoRA weights")
    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()
    args.pcm_type = validate_sndfile_subtype(args)

    return args


def validate_sndfile_subtype(args):
    codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
    subtype = args.pcm_type
    if subtype in sf.available_subtypes(codec):
        return subtype
    default = sf.default_subtype(codec)
    print(f"WARNING: codec {codec} doesn't support subtype {subtype}, defaulting to {default}")
    return default


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load a model configuration from a file.

    Based on `model_type`, returns either an OmegaConf (e.g., for 'htdemucs')
    or a YAML-parsed ConfigDict for other models.

    Args:
        model_type (str): Model identifier that determines the loader behavior
            (e.g., 'htdemucs', 'mdx23c', etc.).
        config_path (str): Path to the configuration file (YAML/OmegaConf).

    Returns:
        Union[ConfigDict, OmegaConf]: Loaded configuration object.

    Raises:
        FileNotFoundError: If `config_path` does not point to an existing file.
        ValueError: If the configuration cannot be parsed or is otherwise invalid.
    """
    try:
        with open(config_path, 'r') as f:
            if model_type == 'htdemucs':
                config = OmegaConf.load(config_path)
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple[nn.Module, Union[ConfigDict, OmegaConf]]:
    """
    Load and instantiate a model using a configuration file.

    Given a `model_type` and a path to a configuration, this function loads the
    configuration (YAML or OmegaConf) and constructs the corresponding model.

    Args:
        model_type (str): Identifier of the model family (e.g., 'mdx23c', 'htdemucs',
            'scnet', 'mel_band_conformer', etc.).
        config_path (str): Filesystem path to the configuration file used to
            initialize the model.

    Returns:
        Tuple[nn.Module, Union[ConfigDict, OmegaConf]]: A tuple containing the
        initialized PyTorch model and the loaded configuration object.

    Raises:
        ValueError: If `model_type` is unknown or model initialization fails.
        FileNotFoundError: If `config_path` does not exist (may be raised by the
            underlying config loader).
    """

    config = load_config(model_type, config_path)

    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'mel_band_conformer':
        from models.bs_roformer import MelBandConformer
        model = MelBandConformer(**dict(config.model))
    elif model_type == 'mel_band_roformer_experimental':
        from models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_conformer':
        from models.bs_roformer import BSConformer
        model = BSConformer(**dict(config.model))
    elif model_type == 'bs_roformer_experimental':
        from models.bs_roformer.bs_roformer_experimental import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet_tran':
        from models.scnet.scnet_tran import SCNet_Tran
        model = SCNet_Tran(**config.model)
    elif model_type == 'apollo':
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'bs_mamba2':
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'scnet_masked':
        from models.scnet.scnet_masked import SCNet
        model = SCNet(**config.model)
    elif model_type == 'conformer':
        from models.conformer_model import ConformerMSS, NeuralModel
        model = ConformerMSS(
            core=NeuralModel(**config.model),
            n_fft=config.stft.n_fft,
            hop_length=config.stft.hop_length,
            win_length=getattr(config.stft, 'win_length', config.stft.n_fft),
            center=config.stft.center
        )
    elif model_type == 'mel_band_conformer':
        from models.mel_band_conformer import MelBandConformer
        model = MelBandConformer(**config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def get_scheduler(config, optimizer):
    scheduler_name = config.training.get('scheduler', 'ReduceLROnPlateau')
    if scheduler_name == 'linear_scheduler':
        from transformers import get_linear_schedule_with_warmup
        num_training_steps = config.training.num_epochs * config.training.num_steps
        num_warmup_steps = config.training.num_warmup_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_name == 'ReduceLROnPlateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                      factor=config.training.reduce_factor)
    else:
        available_schedulers = ['linear_scheduler', 'ReduceLROnPlateau']
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available options: {available_schedulers}. "
            f"Check your config.training.scheduler setting."
        )
    scheduler.name = scheduler_name
    return scheduler


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> None:
    """
    Print a log message and optionally append it to an in-memory list.

    In Distributed Data Parallel (DDP) contexts, the message is printed only on
    rank 0; when DDP is uninitialized, it prints unconditionally. If
    `verbose_logging` is True, the message is also appended to `logs`.

    Args:
        logs (List[str]): Mutable list to which the message is appended when
            `verbose_logging` is True.
        text (str): The log message to print (rank 0 only under DDP) and
            optionally store.
        verbose_logging (bool, optional): If True, append `text` to `logs`.
            Defaults to False.

    Returns:
        None: The function prints and may mutate `logs` in place.
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(text)
        if verbose_logging:
            logs.append(text)


def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    Write accumulated log messages to a results file.

    Creates (or overwrites) a `results.txt` file inside `store_dir` and writes
    each entry from `logs` as a separate line. In Distributed Data Parallel (DDP)
    scenarios, writing is intended to occur only on rank 0.

    Args:
        store_dir (str): Directory path where `results.txt` will be saved.
        logs (List[str]): Ordered collection of log lines to write.

    Returns:
        None
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(f'{store_dir}/results.txt', 'w') as out:
            for item in logs:
                out.write(item + "\n")


def manual_seed(seed: int) -> None:
    """
    Initialize random seeds for reproducibility.

    Sets the seed across Python's `random`, NumPy, and PyTorch (CPU and CUDA)
    libraries, and updates the `PYTHONHASHSEED` environment variable. This helps
    ensure deterministic behavior where possible, though some GPU operations
    may still introduce nondeterminism.

    Args:
        seed (int): The seed value to use for all random number generators.

    Returns:
        None
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """
    Initialize runtime environment settings.

    Sets random seeds for reproducibility, adjusts PyTorch cuDNN behavior,
    configures multiprocessing with the 'spawn' start method, and ensures
    the results directory exists.

    Args:
        seed (int): Random seed value for deterministic initialization.
        results_path (str): Filesystem path to create for saving results.

    Returns:
        None
    """

    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)


def initialize_environment_ddp(rank: int, world_size: int, seed: int = 0, resuls_path: str = None) -> None:
    """
    Initialize environment for Distributed Data Parallel (DDP) training/validation.

    Sets up the DDP process group, seeds random number generators, configures
    multiprocessing to use the 'spawn' method, and creates a results directory
    if provided.

    Args:
        rank (int): Rank of the current process within the DDP group.
        world_size (int): Total number of processes participating in DDP.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        resuls_path (str, optional): Directory path to create for storing results.
            If None, no directory is created. Defaults to None.

    Returns:
        None
    """

    setup_ddp(rank, world_size)
    manual_seed(seed)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)  # force=True prevent errors
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise e
    if not (resuls_path is None):
        os.makedirs(resuls_path, exist_ok=True)


def gen_wandb_name(args, config) -> str:
    """
    Generate a descriptive name for a Weights & Biases (wandb) run.

    Combines the model type, a dash-joined list of training instruments,
    and the current date into a single string identifier.

    Args:
        args: Parsed arguments namespace containing at least `model_type`.
        config: Configuration object/dict with a `training.instruments` field.

    Returns:
        str: Formatted run name in the form
            "<model_type>_[<instrument1>-<instrument2>-...]_<YYYY-MM-DD>".
    """

    instrum = '-'.join(config['training']['instruments'])
    time_str = time.strftime("%Y-%m-%d")
    name = '{}_[{}]_{}'.format(args.model_type, instrum, time_str)
    return name


def wandb_init(args: argparse.Namespace, config: ConfigDict | OmegaConf, batch_size: int) -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.

    Depending on the provided arguments, sets up wandb in one of three modes:
    - Offline mode when `args.wandb_offline` is True.
    - Disabled mode when no valid `wandb_key` is provided.
    - Online mode with authentication using `args.wandb_key`.

    Args:
        args (argparse.Namespace): Parsed arguments containing wandb options
            (`wandb_offline`, `wandb_key`, `device_ids`).
        config (Dict): Experiment configuration dictionary to log.
        batch_size (int): Training batch size to include in the run configuration.

    Returns:
        None
    """

    if not dist.is_initialized() or dist.get_rank() == 0:
        if args.wandb_offline:
            wandb.init(mode='offline',
                       project='msst',
                       name=gen_wandb_name(args, config),
                       config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
                       )
        elif args.wandb_key is None or args.wandb_key.strip() == '':
            wandb.init(mode='disabled')
        else:
            wandb.login(key=args.wandb_key)
            wandb.init(
                project='msst',
                name=gen_wandb_name(args, config),
                config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
            )


def setup_ddp(rank: int, world_size: int) -> None:
    """
    Initialize a Distributed Data Parallel (DDP) process group.

    Configures environment variables for the DDP master node, attempts to
    initialize the process group with the NCCL backend (preferred for GPUs),
    and falls back to the Gloo backend if NCCL is unavailable. Also sets the
    current CUDA device to match the process rank.

    Args:
        rank (int): Rank of the current process in the DDP group.
        world_size (int): Total number of processes participating in DDP.

    Returns:
        None
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # We can change and use another
    os.environ["USE_LIBUV"] = "0"
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        if dist.get_rank() == 0:
            print(f'NCCL are not available. Using "gloo" backend.')

    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """
    Finalize and clean up a Distributed Data Parallel (DDP) process group.

    Calls `torch.distributed.destroy_process_group()` to release resources
    associated with the current DDP environment.

    Returns:
        None
    """
    dist.destroy_process_group()
