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
import loralib as lora
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import MSSDataset


def parse_args_train(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
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
                        'multistft_loss', 'spec_masked_loss', 'spec_rmse_loss_coef', 'log_wmse_loss'],
                        default=['masked_loss'], help="List of loss functions to use")
    parser.add_argument("--masked_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--mse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--l1_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--log_wmse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--multistft_loss_coef", type=float, default=0.001, help="Coef for loss")
    parser.add_argument("--spec_masked_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--spec_rmse_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")
    parser.add_argument("--save_metrics", action='store_true', help="Save metrics in csv file or not")
    parser.add_argument("--each_metrics_in_name", action='store_true',
                        help="Naming checkpoints consist only of vocal metric")
    parser.add_argument("--more_metrics_wandb", action='store_true',
                        help="Show metric_for_scheduler for all instuments")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def parse_args_valid(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
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
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

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
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
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
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24',
                        help="PCM type for FLAC files (PCM_16 or PCM_24)")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load the configuration from the specified path based on the model type.

    Parameters:
    ----------
    model_type : str
        The type of model to load (e.g., 'htdemucs', 'mdx23c', etc.).
    config_path : str
        The path to the YAML or OmegaConf configuration file.

    Returns:
    -------
    config : Any
        The loaded configuration, which can be in different formats (e.g., OmegaConf or ConfigDict).

    Raises:
    ------
    FileNotFoundError:
        If the configuration file at `config_path` is not found.
    ValueError:
        If there is an error loading the configuration file.
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


def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    """
    Load the model specified by the model type and configuration file.

    Parameters:
    ----------
    model_type : str
        The type of model to load (e.g., 'mdx23c', 'htdemucs', 'scnet', etc.).
    config_path : str
        The path to the configuration file (YAML or OmegaConf format).

    Returns:
    -------
    model : nn.Module or None
        The initialized model based on the `model_type`, or None if the model type is not recognized.
    config : Any
        The configuration used to initialize the model. This could be in different formats
        depending on the model type (e.g., OmegaConf, ConfigDict).

    Raises:
    ------
    ValueError:
        If the `model_type` is unknown or an error occurs during model initialization.
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
    elif model_type == 'mel_band_roformer_experimental':
        from models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def manual_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to set.
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
    Initialize the environment by setting the random seed, configuring PyTorch settings,
    and creating the results directory.

    Args:
        seed: The seed value for reproducibility.
        results_path: Path to the directory where results will be stored.
    """

    manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    os.makedirs(results_path, exist_ok=True)


def gen_wandb_name(args, config):
    instrum = '-'.join(config['training']['instruments'])
    time_str = time.strftime("%Y-%m-%d")
    name = '{}_[{}]_{}'.format(args.model_type, instrum, time_str)
    return name


def wandb_init(args: argparse.Namespace, config: Dict, device_ids: List[int], batch_size: int) -> None:
    """
    Initialize the Weights & Biases (wandb) logging system.

    Args:
        args: Parsed command-line arguments containing the wandb key.
        config: Configuration dictionary for the experiment.
        device_ids: List of GPU device IDs used for training.
        batch_size: Batch size for training.
    """

    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(
            project='msst',
            name=gen_wandb_name(args, config),
            config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size }
        )

def logging(logs: List[str], text: str, verbose_logging: bool = False) -> None:
    """
    Log validation information by printing the text and appending it to a log list.

    Parameters:
    ----------
    store_dir : str
        Directory to store the logs. If empty, logs are not stored.
    logs : List[str]
        List where the logs will be appended if the store_dir is specified.
    text : str
        The text to be logged, printed, and optionally added to the logs list.

    Returns:
    -------
    None
        This function modifies the logs list in place and prints the text.
    """

    print(text)
    if verbose_logging:
        logs.append(text)


def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """
    Write the list of results into a file in the specified directory.

    Parameters:
    ----------
    store_dir : str
        The directory where the results file will be saved.
    results : List[str]
        A list of result strings to be written to the file.

    Returns:
    -------
    None
    """
    with open(f'{store_dir}/results.txt', 'w') as out:
        for item in logs:
            out.write(item + "\n")


def setup_ddp(rank: int, world_size: int):
    """Initialize process DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # We can change and use another
    os.environ["USE_LIBUV"] = "0"
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except:
        print(f'NCCL are not available. Using "gloo" backend.')
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Finishing DDP process."""
    dist.destroy_process_group()


def initialize_environment_ddp(rank: int, world_size: int, seed: int=0, resuls_path: str=None) -> None:

    setup_ddp(rank, world_size)
    manual_seed(seed)

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)  # force=True prevent errors
    except RuntimeError as e:
        if "context has already been set" not in str(e):
            raise
    if not(resuls_path is None):
        os.makedirs(resuls_path, exist_ok=True)

def wandb_init_ddp(args: argparse.Namespace, config: Dict, batch_size: int) -> None:
    """
    Initialize the Weights & Biases (wandb) logging system.

    Args:
        args: Parsed command-line arguments containing the wandb key.
        config: Configuration dictionary for the experiment.
        device_ids: List of GPU device IDs used for training.
        batch_size: Batch size for training.
    """
    if dist.get_rank() == 0:
        if not args.wandb_key or args.wandb_key.strip() == '':
            print("WandB key is not provided. Disabling WandB logging.")
            wandb.init(mode='disabled')
        else:
            try:
                wandb.login(key=args.wandb_key)
                wandb.init(
                    project='msst',
                    name=gen_wandb_name(args, config),
                    config={'config': config, 'args': args, 'device_ids': args.device_ids, 'batch_size': batch_size}
                )
            except Exception as e:
                print(f"Error initializing WandB: {e}")
                wandb.init(mode='disabled')


def prepare_data_ddp(config: Dict, args: argparse.Namespace, batch_size: int, rank: int, world_size: int) -> DataLoader:
    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=world_size * batch_size, # to use self.config.training.num_steps without reduction
        metadata_path=os.path.join(args.results_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )

    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader


def get_optimizer_ddp(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Initializes an optimizer based on the configuration.

    Args:
        config: Configuration object containing training parameters.
        model: PyTorch model whose parameters will be optimized.

    Returns:
        A PyTorch optimizer object configured based on the specified settings.
    """

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        if dist.get_rank() == 0:
            print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'sgd':
        if dist.get_rank() == 0:
            print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        if dist.get_rank() == 0:
            print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def save_weights_ddp(store_path: str, model: torch.nn.Module, train_lora: bool) -> None:
    """
    Save model's weights. Save only if rank==0.

    Args:
        store_path (str): Path to save.
        model (torch.nn.Module): Your model to save.
        train_lora (bool): If we used LoRA.
    """
    if dist.get_rank() == 0:  # Только главный процесс сохраняет
        if train_lora:
            torch.save(lora.lora_state_dict(model), store_path)
        else:
            torch.save(model.module.state_dict(), store_path)  # model.module всегда нужен в DDP


def save_last_weights_ddp(args: argparse.Namespace, model: torch.nn.Module) -> None:

    store_path = f'{args.results_path}/last_{args.model_type}.ckpt'
    save_weights_ddp(store_path, model, args.train_lora)