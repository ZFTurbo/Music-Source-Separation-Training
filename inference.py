# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

from tqdm import tqdm
from typing import List
import argparse
import glob
import librosa
import numpy
import os
import soundfile
import sys
import time
import torch
import torch.nn as nn
while (tryAgain := True):
    try:
        from utils import demix_track, demix_track_demucs, get_model_from_config
        break
    except ImportError as error:
        if (pathModule := os.path.dirname(os.path.abspath(__file__))) in sys.path:
            raise error  # Reraise the error
        print(f"Added '{pathModule}' to sys.path")
        sys.path.append(pathModule)

import warnings
warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    print('Total files found: {}'.format(len(all_mixtures_path)))

    instruments: List[str] = [config.training.target_instrument] if config.training.target_instrument is not None else config.training.instruments

    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in all_mixtures_path:
        print("Starting processing track: ", path)
        if not verbose:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception as e:
            print('Cannot read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = numpy.stack([mix, mix], axis=0)

        if config.inference.get('normalize', False) is True:
            mix_orig = mix.copy()
            mono = mix.mean(0)
            mean = mono.mean()
            std = mono.std()
            mix = (mix - mean) / std
        else: # assign a value to 'normalize' if not present in config
            config.inference['normalize'] = False

        mixture = torch.tensor(mix, dtype=torch.float32)
        if args.model_type == 'htdemucs':
            waveforms = demix_track_demucs(config, model, mixture, device, pbar=detailed_pbar)
        else:
            waveforms = demix_track(config, model, mixture, device, pbar=detailed_pbar)

        if config.inference['normalize'] is True:
            mix = mix_orig.copy()
            del mix_orig # undo
            waveforms = {k: v * std + mean for k, v in waveforms.items()}

        # Output "instrumental", which is an inverse of 'vocals' or the first stem in list if 'vocals' absent
        if args.extract_instrumental:
            target = 'vocals' if 'vocals' in instruments else instruments[0]
            waveforms['instrumental'] = mix - waveforms[target]

        for target in instruments:
            estimate = waveforms[target]
            output_file = os.path.join(args.store_dir, f"{os.path.splitext(os.path.basename(path))[0]}_{target}.wav")
            soundfile.write(output_file, estimate.T, sr, subtype = 'FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer, scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true', help="invert vocals to get instrumental if provided")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action = 'store_true', help = "Force the use of CPU even if CUDA is available")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = "cuda"
        device = f'cuda:{args.device_ids}' if type(args.device_ids) == int else f'cuda:{args.device_ids[0]}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        if args.model_type == 'htdemucs':
            state_dict = torch.load(args.start_check_point, map_location = device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
        else:
            state_dict = torch.load(args.start_check_point, map_location = device, weights_only=True)
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if type(args.device_ids) != int:
        model = nn.DataParallel(model, device_ids = args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
