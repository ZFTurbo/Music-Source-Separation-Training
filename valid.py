# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

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

from utils.settings import get_model_from_config, logging, write_results_in_file, parse_args_valid
from utils.audio_utils import draw_spectrogram, normalize_audio, denormalize_audio, read_audio_transposed
from utils.model_utils import demix, prefer_target_instrument, apply_tta, load_start_checkpoint
from utils.metrics import get_metrics

import warnings

warnings.filterwarnings("ignore")


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
                print(f'No validation data found in: {path}')
        all_mixtures_path += part
    if verbose:
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')

    return all_mixtures_path


def update_metrics_and_pbar(
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
            print(f"Metric {metric_name:11s} value: {metric_value:.4f}")
        all_metrics[metric_name][instr].append(metric_value)
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
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    if is_tqdm:
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
                    print(f'Warning: sample rate is different. In config: {config.audio["sample_rate"]} in file {path}: {sr}')
                mix = librosa.resample(mix, orig_sr=sr, target_sr=config.audio['sample_rate'], res_type='kaiser_best')

        if verbose:
            folder_name = os.path.abspath(folder)
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
                if np.abs(estimates).max() <= 1.0:
                    out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.flac"
                    sf.write(out_wav_name, estimates.T, sr, subtype='PCM_16')
                else:
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
                track_metrics,
                all_metrics,
                instr, pbar_dict,
                mixture_paths=mixture_paths,
                verbose=verbose
            )

        if verbose:
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

            logging(logs, text=f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})", verbose_logging=verbose_logging)
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            logging(logs, text=f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}', verbose_logging=verbose_logging)
    logging(logs, text=f"Elapsed time: {time.time() - start_time:.2f} sec", verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False
) -> Tuple[dict, dict]:
    """
    Validate a trained model on a set of audio mixtures and compute metrics.

    This function performs validation by separating audio sources from mixtures,
    computing evaluation metrics, and optionally saving results to a file.

    Parameters:
    ----------
    model : torch.nn.Module
        The trained model for source separation.
    args : Namespace
        Command-line arguments or equivalent object containing configurations.
    config : dict
        Configuration dictionary with model and processing parameters.
    device : torch.device
        The device (CPU or CUDA) to run the model on.
    verbose : bool, optional
        If True, enables verbose output during processing. Default is False.

    Returns:
    -------
    dict
        A dictionary of average metrics across all instruments.
    """

    start_time = time.time()
    model.eval().to(device)

    # dir to save files, if empty no saving
    store_dir = getattr(args, 'store_dir', '')
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)
    all_metrics = process_audio_files(all_mixtures_path, model, args, config, device, verbose, not verbose)
    instruments = prefer_target_instrument(config)

    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time), all_metrics


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_mixtures_path: List[str],
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: str,
    return_dict
) -> None:
    """
    Perform validation on a subprocess with multi-processing support. Each process handles inference on a subset of the mixture files
    and updates the shared metrics dictionary.

    Parameters:
    ----------
    proc_id : int
        The process ID (used to assign metrics to the correct key in `return_dict`).
    queue : torch.multiprocessing.Queue
        Queue to receive paths to the mixture files for processing.
    all_mixtures_path : List[str]
        List of paths to the mixture files to be processed.
    model : torch.nn.Module
        The model to be used for inference.
    args : dict
        Dictionary containing various argument configurations (e.g., metrics to calculate).
    config : ConfigDict
        Configuration object containing model settings and training parameters.
    device : str
        The device to use for inference (e.g., 'cpu', 'cuda:0').
    return_dict : torch.multiprocessing.Manager().dict
        Shared dictionary to store the results from each process.

    Returns:
    -------
    None
        The function modifies the `return_dict` in place, but does not return any value.
    """

    m1 = model.eval().to(device)
    if proc_id == 0:
        progress_bar = tqdm(total=len(all_mixtures_path))

    # Initialize metrics dictionary
    all_metrics = {
        metric: {instr: [] for instr in config.training.instruments}
        for metric in args.metrics
    }

    while True:
        current_step, path = queue.get()
        if path is None:  # check for sentinel value
            break
        single_metrics = process_audio_files([path], m1, args, config, device, False, False)
        pbar_dict = {}
        for instr in config.training.instruments:
            for metric_name in all_metrics:
                all_metrics[metric_name][instr] += single_metrics[metric_name][instr]
                if len(single_metrics[metric_name][instr]) > 0:
                    pbar_dict[f"{metric_name}_{instr}"] = f"{single_metrics[metric_name][instr][0]:.4f}"
        if proc_id == 0:
            progress_bar.update(current_step - progress_bar.n)
            progress_bar.set_postfix(pbar_dict)
        # print(f"Inference on process {proc_id}", all_sdr)
    return_dict[proc_id] = all_metrics
    return


def run_parallel_validation(
    verbose: bool,
    all_mixtures_path: List[str],
    config: ConfigDict,
    model: torch.nn.Module,
    device_ids: List[int],
    args,
    return_dict
) -> None:
    """
    Run parallel validation using multiple processes. Each process handles a subset of the mixture files and computes the metrics.
    The results are stored in a shared dictionary.

    Parameters:
    ----------
    verbose : bool
        Flag to print detailed information about the validation process.
    all_mixtures_path : List[str]
        List of paths to the mixture files to be processed.
    config : ConfigDict
        Configuration object containing model settings and validation parameters.
    model : torch.nn.Module
        The model to be used for inference.
    device_ids : List[int]
        List of device IDs (for multi-GPU setups) to use for validation.
    args : dict
        Dictionary containing various argument configurations (e.g., metrics to calculate).

    Returns:
    -------
        A shared dictionary containing the validation metrics from all processes.
    """

    model = model.to('cpu')
    try:
        # For multiGPU training extract single model
        model = model.module
    except:
        pass

    queue = torch.multiprocessing.Queue()
    processes = []

    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_mixtures_path, model, args, config, device, return_dict)
        )
        p.start()
        processes.append(p)
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    for _ in range(len(device_ids)):
        queue.put((None, None))  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()  # wait for all subprocesses to finish

    return


def valid_multi_gpu(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device_ids: List[int],
    verbose: bool = False
) -> Tuple[Dict[str, float], dict]:
    """
    Perform validation across multiple GPUs, processing mixtures and computing metrics using parallel processes.
    The results from each GPU are aggregated and the average metrics are computed.

    Parameters:
    ----------
    model : torch.nn.Module
        The model to be used for inference.
    args : dict
        Dictionary containing various argument configurations, such as file saving directory and codec settings.
    config : ConfigDict
        Configuration object containing model settings and validation parameters.
    device_ids : List[int]
        List of device IDs (for multi-GPU setups) to use for validation.
    verbose : bool, optional
        Flag to print detailed information about the validation process. Default is False.

    Returns:
    -------
    Dict[str, float]
        A dictionary containing the average metrics for each metric name.
    """

    start_time = time.time()

    # dir to save files, if empty no saving
    store_dir = getattr(args, 'store_dir', '')
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    else:
        extension = getattr(args, 'extension', 'wav')

    all_mixtures_path = get_mixture_paths(args, verbose, config, extension)

    return_dict = torch.multiprocessing.Manager().dict()

    run_parallel_validation(verbose, all_mixtures_path, config, model, device_ids, args, return_dict)

    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []
            for i in range(len(device_ids)):
                all_metrics[metric][instr] += return_dict[i][metric][instr]

    instruments = prefer_target_instrument(config)

    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time), all_metrics


def check_validation(dict_args):
    args = parse_args_valid(dict_args)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method('spawn')
    except Exception as e:
        pass
    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='valid')

    print(f"Instruments: {config.training.instruments}")

    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')

    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
