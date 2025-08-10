# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'

import math
import time
import os
import glob

import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union
from utils.model_utils import demix, prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.settings import get_model_from_config, parse_args_valid, initialize_environment_ddp
from utils.audio_utils import draw_spectrogram, normalize_audio, denormalize_audio, read_audio_transposed
from utils.metrics import get_metrics
import warnings
import torch.multiprocessing as mp

import torch.distributed as dist

warnings.filterwarnings("ignore")


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
    if dist.get_rank() == 0:
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


def get_mixture_paths(
        args,
        verbose: bool,
        config: ConfigDict,
        extension: str
) -> List[str]:
    """
    Retrieve paths to mixture files in the specified validation directories.

    Parameters:
    ----------
    valid_path : List[str]
        A list of directories to search for validation mixtures.
    verbose : bool
        If True, prints detailed information about the search process.
    config : ConfigDict
        Configuration object containing parameters like `inference.num_overlap` and `inference.batch_size`.
    extension : str
        File extension of the mixture files (e.g., 'wav').

    Returns:
    -------
    List[str]
        A list of file paths to the mixture files.
    """
    try:
        valid_path = args.valid_path
    except Exception as e:
        print('No valid path in args')
        raise e

    all_mixtures_path = []
    for path in valid_path:
        part = sorted(glob.glob(f"{path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                if dist.get_rank() == 0:
                    print(f'No validation data found in: {path}')
        all_mixtures_path += part
    if verbose:
        if dist.get_rank() == 0:
            print(f'Total mixtures: {len(all_mixtures_path)}')
            print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')

    return all_mixtures_path


def update_metrics_and_pbar(
        path,
        track_metrics: Dict,
        all_metrics: Dict,
        instr: str,
        pbar_dict: Dict,
        mixture_paths: Union[List[str], tqdm],
        verbose: bool = False
) -> None:
    """
    Update metrics dictionary and progress bar with new metric values.

    Parameters:
    ----------
    track_metrics : Dict
        Dictionary with metric names as keys and their computed values as values.
    all_metrics : Dict
        Dictionary to store all metrics, organized by metric name and instrument.
    instr : str
        Name of the instrument for which the metrics are being computed.
    pbar_dict : Dict
        Dictionary for progress bar updates.
    mixture_paths : tqdm, optional
        Progress bar object, if available. Default is None.
    verbose : bool, optional
        If True, prints metric values to the console. Default is False.
    """
    for metric_name, metric_value in track_metrics.items():
        if verbose:
            if dist.get_rank() == 0:
                print(f"Metric {metric_name:11s} value: {metric_value:.4f}")
        all_metrics[metric_name][instr][path] = metric_value
        pbar_dict[f'{metric_name}_{instr}'] = metric_value

    if mixture_paths is not None:
        try:
            mixture_paths.set_postfix(pbar_dict)
        except Exception:
            pass


def process_audio_files(
        mixture_paths: List[str],
        model: torch.nn.Module,
        args,
        config,
        device: torch.device,
        verbose: bool = False,
        is_tqdm: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Process a list of audio files, perform source separation, and evaluate metrics.

    Parameters:
    ----------
    mixture_paths : List[str]
        List of file paths to the audio mixtures.
    model : torch.nn.Module
        The trained model used for source separation.
    args : Any
        Argument object containing user-specified options like metrics, model type, etc.
    config : Any
        Configuration object containing model and processing parameters.
    device : torch.device
        Device (CPU or CUDA) on which the model will be executed.
    verbose : bool, optional
        If True, prints detailed logs for each processed file. Default is False.
    is_tqdm : bool, optional
        If True, displays a progress bar for file processing. Default is True.

    Returns:
    -------
    Dict[str, Dict[str, List[float]]]
        A nested dictionary where the outer keys are metric names,
        the inner keys are instrument names, and the values are lists of metric scores.
    """
    instruments = prefer_target_instrument(config)

    use_tta = getattr(args, 'use_tta', False)
    # dir to save files, if empty no saving
    store_dir = getattr(args, 'store_dir', '')
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    # Initialize metrics dictionary
    all_metrics = {
        metric: {instr: {} for instr in config.training.instruments}
        for metric in args.metrics
    }

    if is_tqdm and dist.get_rank() == 0:
        mixture_paths = tqdm(mixture_paths)

    for path in mixture_paths:
        start_time = time.time()
        mix, sr = read_audio_transposed(path)
        mix_orig = mix.copy()
        folder = os.path.dirname(path)

        if 'sample_rate' in config.audio:
            if sr != config.audio['sample_rate']:
                orig_length = mix.shape[-1]
                if verbose:
                    if dist.get_rank() == 0:
                        print(
                            f'Warning: sample rate is different. In config: {config.audio["sample_rate"]} in file {path}: {sr}')
                mix = librosa.resample(mix, orig_sr=sr, target_sr=config.audio['sample_rate'], res_type='kaiser_best')

        if verbose:
            folder_name = os.path.abspath(folder)
            if dist.get_rank() == 0:
                print(f'Song: {folder_name} Shape: {mix.shape}')

        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mix, norm_params = normalize_audio(mix)

        waveforms_orig = demix(config, model, mix.copy(), device, model_type=args.model_type)

        if use_tta:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)

        pbar_dict = {}

        for instr in instruments:
            if verbose:
                if dist.get_rank() == 0:
                    print(f"Instr: {instr}")

            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = read_audio_transposed(f"{folder}/{instr}.{extension}", instr, skip_err=True)
                if track is None:
                    continue
            else:
                # if track=vocal+other
                track, sr1 = read_audio_transposed(f"{folder}/vocals.{extension}")
                track = mix_orig - track

            estimates = waveforms_orig[instr]

            if 'sample_rate' in config.audio:
                if sr != config.audio['sample_rate']:
                    estimates = librosa.resample(estimates, orig_sr=config.audio['sample_rate'], target_sr=sr,
                                                 res_type='kaiser_best')
                    estimates = librosa.util.fix_length(estimates, size=orig_length)

            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.wav"
                sf.write(out_wav_name, estimates.T, sr, subtype='FLOAT')
                if args.draw_spectro > 0:
                    out_img_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.jpg"
                    draw_spectrogram(estimates.T, sr, args.draw_spectro, out_img_name)
                    out_img_name_orig = f"{store_dir}/{os.path.basename(folder)}_{instr}_orig.jpg"
                    draw_spectrogram(track.T, sr, args.draw_spectro, out_img_name_orig)

            track_metrics = get_metrics(
                args.metrics,
                track,
                estimates,
                mix_orig,
                device=device,
            )

            update_metrics_and_pbar(
                path,
                track_metrics,
                all_metrics,
                instr, pbar_dict,
                mixture_paths=mixture_paths,
                verbose=verbose
            )

        if verbose:
            if dist.get_rank() == 0:
                print(f"Time for song: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
        store_dir: str,
        args,
        instruments: List[str],
        config: ConfigDict,
        all_metrics: Dict[str, Dict[str, List[float]]],
        start_time: float
) -> Dict[str, float]:
    """
    Calculate and log the average metrics for each instrument, including per-instrument metrics and overall averages.

    Parameters:
    ----------
    store_dir : str
        Directory to store the logs. If empty, logs are not stored.
    args : dict
        Dictionary containing the arguments, used for logging.
    instruments : List[str]
        List of instruments to process.
    config : ConfigDict
        Configuration dictionary containing the inference settings.
    all_metrics : Dict[str, Dict[str, List[float]]]
        A dictionary containing metric values for each instrument.
        The structure is {metric_name: {instrument_name: [metric_values]}}.
    start_time : float
        The starting time for calculating elapsed time.

    Returns:
    -------
    Dict[str, float]
        A dictionary with the average value for each metric across all instruments.
    """

    logs = []
    if store_dir:
        logs.append(str(args))
        verbose_logging = True
    else:
        verbose_logging = False

    logging(logs, text=f"Num overlap: {config.inference.num_overlap}", verbose_logging=verbose_logging)

    metric_avg = {}
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])

            mean_val = metric_values.mean()
            std_val = metric_values.std()

            logging(logs, text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})",
                    verbose_logging=verbose_logging)
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            logging(logs, text=f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}',
                    verbose_logging=verbose_logging)
    logging(logs, text=f"Elapsed time: {time.time() - start_time:.2f} sec", verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid_multi_gpu(
        model: torch.nn.Module,
        args,
        config: ConfigDict,
        rank: int,
        world_size: int,
        verbose: bool = False
) -> Tuple[Dict, Dict]:
    """
    Validation in DDP
    """

    start_time = time.time()
    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model.eval()

    store_dir = getattr(args, "store_dir", "")
    extension = config.get("inference", {}).get("extension", "wav")

    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    # Split data on gpus
    per_rank_data = all_mixtures_path[rank::world_size]
    # Num of tracks on each gpu
    target_len = math.ceil(len(all_mixtures_path) / world_size)
    if len(per_rank_data) < target_len:
        per_rank_data += [all_mixtures_path[0]] * (target_len - len(per_rank_data))

    # Local metrics for each gpu
    local_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    with torch.no_grad():
        single_metrics = process_audio_files(per_rank_data, model, args, config, device)
        for instr in config.training.instruments:
            for metric_name in args.metrics:
                local_metrics[metric_name][instr] = single_metrics[metric_name][instr]

    all_metrics = {}
    for metric in args.metrics:
        all_metrics[metric] = {}
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []
            local_data = list(local_metrics[metric][instr].values())
            if isinstance(local_data, (list, torch.Tensor)):
                if isinstance(local_data[0], np.ndarray):
                    local_data = [arr.item() for arr in local_data]
                all_metrics[metric][instr].append(torch.tensor(local_data, device=device))
            elif isinstance(local_data, np.ndarray):
                all_metrics[metric][instr].append(torch.tensor(local_data.tolist(), device=device))
            else:
                all_metrics[metric][instr].append(torch.tensor([local_data], device=device))

    for metric in args.metrics:
        for instr in config.training.instruments:
            gathered_list = [torch.zeros_like(all_metrics[metric][instr][0]) for _ in range(world_size)]
            dist.all_gather(gathered_list, all_metrics[metric][instr][0])

            all_metrics[metric][instr] = torch.cat(gathered_list).tolist()[:len(all_mixtures_path)]

    if rank == 0:
        instruments = prefer_target_instrument(config)
        return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time), all_metrics

    return None, None


def check_validation_single(rank: int, world_size: int, args=None):
    args = parse_args_valid(args)

    initialize_environment_ddp(rank, world_size)
    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='valid')

    if dist.get_rank() == 0:
        print(f"Instruments: {config.training.instruments}")

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    valid_multi_gpu(model, args, config, rank, world_size, verbose=False)


def check_validation(args=None):
    world_size = torch.cuda.device_count()
    mp.spawn(check_validation_single, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    check_validation()
