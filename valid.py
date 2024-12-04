# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
from tqdm.auto import tqdm
import sys
import os
import glob
import copy
import torch
import librosa
import soundfile as sf
import numpy as np
import torch.nn as nn
import multiprocessing
from utils import demix, get_metrics, get_model_from_config, prefer_target_instrument
from typing import Tuple, Dict, List, Union
import warnings
warnings.filterwarnings("ignore")


def read_audio_transposed(path: str, instr: str = None, skip_err: bool = False) -> Tuple[np.ndarray, int]:
    """
    Reads an audio file, ensuring mono audio is converted to two-dimensional format,
    and transposes the data to have channels as the first dimension.
    Parameters
    ----------
    path : str
        Path to the audio file.
    skip_err: bool
        If true, not raise errors
    instr:
        name of instument
    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing:
        - Transposed audio data as a NumPy array with shape (channels, length).
          For mono audio, the shape will be (1, length).
        - Sampling rate (int), e.g., 44100.
    """

    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            print(f"No stem {instr}: skip!")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        if len(mix.shape) == 1:  # For mono audio
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> (np.ndarray, dict):
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: dict) -> np.ndarray:
    return audio * norm_params["std"] + norm_params["mean"]


def apply_tta(
        config,
        model: torch.nn.Module,
        mix: torch.Tensor,
        waveforms_orig: Dict[str, torch.Tensor],
        device: torch.device,
        model_type: str
) -> Dict[str, torch.Tensor]:
    """
    Apply Test-Time Augmentation (TTA) for source separation.

    This function processes the input mixture with test-time augmentations, including
    channel inversion and polarity inversion, to enhance the separation results. The
    results from all augmentations are averaged to produce the final output.

    Parameters:
    ----------
    config : Any
        Configuration object containing model and processing parameters.
    model : torch.nn.Module
        The trained model used for source separation.
    mix : torch.Tensor
        The mixed audio tensor with shape (channels, time).
    waveforms_orig : Dict[str, torch.Tensor]
        Dictionary of original separated waveforms (before TTA) for each instrument.
    device : torch.device
        Device (CPU or CUDA) on which the model will be executed.
    model_type : str
        Type of the model being used (e.g., "demucs", "custom_model").

    Returns:
    -------
    Dict[str, torch.Tensor]
        Updated dictionary of separated waveforms after applying TTA.
    """
    # Create augmentations: channel inversion and polarity inversion
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    # Process each augmented mixture
    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    # Average the results across augmentations
    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def update_metrics_and_pbar(
        track_metrics: dict,
        all_metrics: dict,
        instr: str,
        pbar_dict: dict,
        mixture_paths: Union[List[str], tqdm],
        verbose: bool = False
) -> None:
    """
    Update metrics dictionary and progress bar with new metric values.

    Parameters:
    ----------
    track_metrics : dict
        Dictionary with metric names as keys and their computed values as values.
    all_metrics : dict
        Dictionary to store all metrics, organized by metric name and instrument.
    instr : str
        Name of the instrument for which the metrics are being computed.
    pbar_dict : dict
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


def proc_list_of_files(
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

    store_dir = ''
    if hasattr(args, 'store_dir'):
        store_dir = args.store_dir

    use_tta = False
    if hasattr(args, 'use_tta'):
        use_tta = args.use_tta

    #codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    elif hasattr(args, 'extension'):
        extension = args.extension
    else:
        extension = 'wav'

    # Initialize metrics dictionary
    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []

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

            if store_dir != "":
                os.makedirs(store_dir, exist_ok=True)
                out_wav_name = f"{store_dir}/{os.path.basename(folder)}_{instr}.wav"
                sf.write(out_wav_name, estimates.T, sr, subtype='FLOAT')

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


def valid(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval().to(device)

    store_dir = ''
    if hasattr(args, 'store_dir'):
        store_dir = args.store_dir
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    elif hasattr(args, 'extension'):
        extension = args.extension
    else:
        extension = 'wav'

    all_mixtures_path = []
    for valid_path in args.valid_path:
        part = sorted(glob.glob(f"{valid_path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                print(f'No validation data found in: {valid_path}')
        all_mixtures_path += part
    if verbose:
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')

    all_metrics = proc_list_of_files(all_mixtures_path, model, args, config, device, verbose, not verbose)

    instruments = prefer_target_instrument(config)

    if store_dir != "":
        out = open(store_dir + '/results.txt', 'w')
        out.write(str(args) + "\n")
    print(f"Num overlap: {config.inference.num_overlap}")

    metric_avg = {}
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])
            mean_val = metric_values.mean()
            std_val = metric_values.std()
            print(f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})")
            if store_dir != "":
                out.write(f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})\n")
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            print(f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}')
            if store_dir != "":
                out.write(f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}\n')
    print(f"Elapsed time: {time.time() - start_time:.2f} sec")
    if store_dir != "":
        out.write(f"Elapsed time: {time.time() - start_time:.2f} sec\n")
        out.close()

    return metric_avg


def valid_mp(proc_id, queue, all_mixtures_path, model, args, config, device, return_dict):
    m1 = model.eval().to(device)
    if proc_id == 0:
        progress_bar = tqdm(total=len(all_mixtures_path))

    # Initialize metrics dictionary
    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []

    while True:
        current_step, path = queue.get()
        if path is None:  # check for sentinel value
            break
        single_metrics = proc_list_of_files([path], m1, args, config, device, False, False)
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


def valid_multi_gpu(model, args, config, device_ids, verbose=False):
    start_time = time.time()

    store_dir = ''
    if hasattr(args, 'store_dir'):
        store_dir = args.store_dir
    # codec to save files
    if 'extension' in config['inference']:
        extension = config['inference']['extension']
    elif hasattr(args, 'extension'):
        extension = args.extension
    else:
        extension = 'wav'

    all_mixtures_path = []
    for valid_path in args.valid_path:
        part = sorted(glob.glob(f"{valid_path}/*/mixture.{extension}"))
        if len(part) == 0:
            if verbose:
                print(f'No validation data found in: {valid_path}')
        all_mixtures_path += part
    if verbose:
        print(f'Total mixtures: {len(all_mixtures_path)}')
        print(f'Overlap: {config.inference.num_overlap} Batch size: {config.inference.batch_size}')
    model = model.to('cpu')
    try:
        # For multiGPU training extract single model
        if len(device_ids) > 1:
            model = model.module
    except Exception as e:
        pass

    queue = torch.multiprocessing.Queue()
    processes = []
    return_dict = torch.multiprocessing.Manager().dict()
    for i, device in enumerate(device_ids):
        if torch.cuda.is_available():
            device = f'cuda:{device}'
        else:
            device = 'cpu'
        p = torch.multiprocessing.Process(target=valid_mp, args=(i, queue, all_mixtures_path, model, args, config, device, return_dict))
        p.start()
        processes.append(p)
    for i, path in enumerate(all_mixtures_path):
        queue.put((i, path))
    for _ in range(len(device_ids)):
        queue.put((None, None))  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()  # wait for all subprocesses to finish

    all_metrics = dict()
    for metric in args.metrics:
        all_metrics[metric] = dict()
        for instr in config.training.instruments:
            all_metrics[metric][instr] = []
            for i in range(len(device_ids)):
                all_metrics[metric][instr] += return_dict[i][metric][instr]

    instruments = prefer_target_instrument(config)

    if store_dir != "":
        out = open(store_dir + '/results.txt', 'w')
        out.write(str(args) + "\n")
    print(f"Num overlap: {config.inference.num_overlap}")

    metric_avg = {}
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])
            mean_val = metric_values.mean()
            std_val = metric_values.std()
            print(f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})")
            if store_dir != "":
                out.write(f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f})\n")
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            print(f"Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}")
            if store_dir != "":
                out.write(f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}\n')
    print(f"Elapsed time: {time.time() - start_time:.2f} sec")
    if store_dir != "":
        out.write(f"Elapsed time: {time.time() - start_time:.2f} sec\n")
        out.close()

    return metric_avg


def check_validation(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--valid_path", nargs="+", type=str, help="validate path")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--extension", type=str, default='wav', help="Choose extension for validation")
    parser.add_argument("--use_tta", action='store_true', help="Flag adds test time augmentation during inference (polarity and channel inverse). While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"], choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'], help='List of metrics to use.')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != '':
        print(f'Start from checkpoint: {args.start_check_point}')
        state_dict = torch.load(args.start_check_point)
        if args.model_type in ['htdemucs', 'apollo']:
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    print(f"Instruments: {config.training.instruments}")

    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')

    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=True)


if __name__ == "__main__":
    check_validation(None)
