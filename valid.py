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
import soundfile as sf
import numpy as np
import torch.nn as nn
import multiprocessing

import warnings
warnings.filterwarnings("ignore")

from utils import demix, get_metrics, get_model_from_config, prefer_target_instrument

def proc_list_of_files(
    mixture_paths,
    model,
    args,
    config,
    device,
    verbose=False,
    is_tqdm=True
):
    instruments = prefer_target_instrument(config)

    store_dir = ''
    if hasattr(args, 'store_dir'):
        store_dir = args.store_dir
    use_tta = False
    if hasattr(args, 'use_tta'):
        use_tta = args.use_tta
    extension = 'wav'
    if hasattr(args, 'extension'):
        extension = args.extension

    if store_dir != '':
        os.makedirs(store_dir, exist_ok=True)

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
        mix, sr = sf.read(path)
        mix_orig = mix.copy()

        # Fix for mono
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=-1)

        mix = mix.T # (channels, waveform)
        folder = os.path.dirname(path)
        folder_name = os.path.abspath(folder)
        if verbose:
            print('Song: {} Shape: {}'.format(folder_name, mix.shape))

        if 'normalize' in config.inference:
            if config.inference['normalize'] is True:
                mono = mix.mean(0)
                mean = mono.mean()
                std = mono.std()
                mix = (mix - mean) / std

        if use_tta:
            # orig, channel inverse, polarity inverse
            track_proc_list = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
        else:
            track_proc_list = [mix.copy()]

        full_result = []
        for mix in track_proc_list:
            waveforms = demix(config, model, mix, device, model_type=args.model_type)
            full_result.append(waveforms)

        # Average all values in single dict
        waveforms = full_result[0]
        for i in range(1, len(full_result)):
            d = full_result[i]
            for el in d:
                if i == 2:
                    waveforms[el] += -1.0 * d[el]
                elif i == 1:
                    waveforms[el] += d[el][::-1].copy()
                else:
                    waveforms[el] += d[el]
        for el in waveforms:
            waveforms[el] = waveforms[el] / len(full_result)

        pbar_dict = {}
        for instr in instruments:
            if verbose:
                print("Instr: {}".format(instr))
            if instr != 'other' or config.training.other_fix is False:
                try:
                    track, sr1 = sf.read(folder + '/{}.{}'.format(instr, extension))

                    # Fix for mono
                    if len(track.shape) == 1:
                        track = np.expand_dims(track, axis=-1)

                except Exception as e:
                    print('No data for stem: {}. Skip!'.format(instr))
                    continue
            else:
                # other is actually instrumental
                track, sr1 = sf.read(folder + '/{}.{}'.format('vocals', extension))
                track = mix_orig - track

            estimates = waveforms[instr].T
            # print(estimates.shape)
            if 'normalize' in config.inference:
                if config.inference['normalize'] is True:
                    estimates = estimates * std + mean

            if store_dir != "":
                out_wav_name = "{}/{}_{}.wav".format(store_dir, os.path.basename(folder), instr)
                sf.write(out_wav_name, estimates, sr, subtype='FLOAT')

            track_metrics = get_metrics(
                args.metrics,
                track.T,
                estimates.T,
                mix_orig.T,
                device=device,
            )

            for metric_name in track_metrics:
                metric_value = track_metrics[metric_name]
                if verbose:
                    print("Metric {:11s} value: {:.4f}".format(metric_name, metric_value))
                all_metrics[metric_name][instr].append(metric_value)
                pbar_dict['{}_{}'.format(metric_name, instr)] = metric_value

            try:
                mixture_paths.set_postfix(pbar_dict)
            except Exception as e:
                pass
        if verbose:
            print("Time for song: {:.2f} sec".format(time.time() - start_time))

    return all_metrics


def valid(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval().to(device)

    store_dir = ''
    if hasattr(args, 'store_dir'):
        store_dir = args.store_dir
    extension = 'wav'
    if hasattr(args, 'extension'):
        extension = args.extension

    all_mixtures_path = []
    for valid_path in args.valid_path:
        part = sorted(glob.glob(valid_path + '/*/mixture.{}'.format(extension)))
        if len(part) == 0:
            if verbose:
                print('No validation data found in: {}'.format(valid_path))
        all_mixtures_path += part
    if verbose:
        print('Total mixtures: {}'.format(len(all_mixtures_path)))
        print('Overlap: {} Batch size: {}'.format(config.inference.num_overlap, config.inference.batch_size))

    all_metrics = proc_list_of_files(all_mixtures_path, model, args, config, device, verbose, not verbose)

    instruments = prefer_target_instrument(config)

    if store_dir != "":
        out = open(store_dir + '/results.txt', 'w')
        out.write(str(args) + "\n")
    print("Num overlap: {}".format(config.inference.num_overlap))

    metric_avg = {}
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])
            mean_val = metric_values.mean()
            std_val = metric_values.std()
            print("Instr {} {}: {:.4f} (Std: {:.4f})".format(instr, metric_name, mean_val, std_val))
            if store_dir != "":
                out.write("Instr {} {}: {:.4f} (Std: {:.4f})".format(instr, metric_name, mean_val, std_val) + "\n")
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            print('Metric avg {:11s}: {:.4f}'.format(metric_name, metric_avg[metric_name]))
            if store_dir != "":
                out.write('Metric avg {:11s}: {:.4f}'.format(metric_name, metric_avg[metric_name]) + "\n")
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    if store_dir != "":
        out.write("Elapsed time: {:.2f} sec".format(time.time() - start_time) + "\n")
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
                    pbar_dict['{}_{}'.format(metric_name, instr)] = "{:.4f}".format(single_metrics[metric_name][instr][0])
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
    extension = 'wav'
    if hasattr(args, 'extension'):
        extension = args.extension

    all_mixtures_path = []
    for valid_path in args.valid_path:
        part = sorted(glob.glob(valid_path + '/*/mixture.{}'.format(extension)))
        if len(part) == 0:
            if verbose:
                print('No validation data found in: {}'.format(valid_path))
        all_mixtures_path += part
    if verbose:
        print('Total mixtures: {}'.format(len(all_mixtures_path)))
        print('Overlap: {} Batch size: {}'.format(config.inference.num_overlap, config.inference.batch_size))

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
            device = 'cuda:{}'.format(device)
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
    print("Num overlap: {}".format(config.inference.num_overlap))

    metric_avg = {}
    for instr in instruments:
        for metric_name in all_metrics:
            metric_values = np.array(all_metrics[metric_name][instr])
            mean_val = metric_values.mean()
            std_val = metric_values.std()
            print("Instr {} {}: {:.4f} (Std: {:.4f})".format(instr, metric_name, mean_val, std_val))
            if store_dir != "":
                out.write("Instr {} {}: {:.4f} (Std: {:.4f})".format(instr, metric_name, mean_val, std_val) + "\n")
            if metric_name not in metric_avg:
                metric_avg[metric_name] = 0.0
            metric_avg[metric_name] += mean_val
    for metric_name in all_metrics:
        metric_avg[metric_name] /= len(instruments)

    if len(instruments) > 1:
        for metric_name in metric_avg:
            print('Metric avg {:11s}: {:.4f}'.format(metric_name, metric_avg[metric_name]))
            if store_dir != "":
                out.write('Metric avg {:11s}: {:.4f}'.format(metric_name, metric_avg[metric_name]) + "\n")
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    if store_dir != "":
        out.write("Elapsed time: {:.2f} sec".format(time.time() - start_time) + "\n")
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
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"], choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft'], help='List of metrics to use.')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        state_dict = torch.load(args.start_check_point)
        if args.model_type in ['htdemucs', 'apollo']:
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    print("Instruments: {}".format(config.training.instruments))

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
