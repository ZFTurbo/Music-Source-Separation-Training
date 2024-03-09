# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import yaml
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
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

from utils import demix_track, demix_track_demucs, sdr, get_model_from_config


def proc_list_of_files(all_mixtures_path, model, args, config, device, verbose=False, is_tqdm=True):
    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if args.store_dir != "":
        if not os.path.isdir(args.store_dir):
            os.mkdir(args.store_dir)

    all_sdr = dict()
    for instr in config.training.instruments:
        all_sdr[instr] = []

    if is_tqdm:
        all_mixtures_path = tqdm(all_mixtures_path)

    for path in all_mixtures_path:
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        folder_name = os.path.abspath(folder)
        if verbose:
            print('Song: {}'.format(folder_name))
        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if args.model_type == 'htdemucs':
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)
        if 1:
            pbar_dict = {}
            for instr in instruments:
                if instr != 'other' or config.training.other_fix is False:
                    try:
                        track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
                    except Exception as e:
                        # print('No data for stem: {}. Skip!'.format(instr))
                        continue
                else:
                    # other is actually instrumental
                    track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                    track = mix - track

                if args.store_dir != "":
                    sf.write("{}/{}_{}.wav".format(args.store_dir, os.path.basename(folder), instr), res[instr].T, sr,
                             subtype='FLOAT')
                references = np.expand_dims(track, axis=0)
                estimates = np.expand_dims(res[instr].T, axis=0)
                sdr_val = sdr(references, estimates)[0]
                if verbose:
                    print(instr, res[instr].shape, sdr_val)
                all_sdr[instr].append(sdr_val)
                pbar_dict['sdr_{}'.format(instr)] = sdr_val

            try:
                all_mixtures_path.set_postfix(pbar_dict)
            except Exception as e:
                pass

    return all_sdr


def valid(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(args.valid_path + '/*/mixture.wav')
    print('Total mixtures: {}'.format(len(all_mixtures_path)))
    print('Overlap: {} Batch size: {}'.format(config.inference.num_overlap, config.inference.batch_size))

    all_sdr = proc_list_of_files(all_mixtures_path, model, args, config, device, verbose, not verbose)

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if args.store_dir != "":
        out = open(args.store_dir + '/results.txt', 'w')
        out.write(str(args) + "\n")
    print("Num overlap: {}".format(config.inference.num_overlap))
    sdr_avg = 0.0
    for instr in instruments:
        sdr_val = np.array(all_sdr[instr]).mean()
        print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
        if args.store_dir != "":
            out.write("Instr SDR {}: {:.4f}".format(instr, sdr_val) + "\n")
        sdr_avg += sdr_val
    sdr_avg /= len(instruments)
    if len(instruments) > 1:
        print('SDR Avg: {:.4f}'.format(sdr_avg))
    if args.store_dir != "":
        out.write('SDR Avg: {:.4f}'.format(sdr_avg) + "\n")
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    if args.store_dir != "":
        out.write("Elapsed time: {:.2f} sec".format(time.time() - start_time) + "\n")
        out.close()

    return sdr_avg


def valid_mp(proc_id, all_mixtures_path, model, args, config, device, return_dict):
    m1 = model
    m1.eval()
    m1.to(device)
    all_sdr = proc_list_of_files(all_mixtures_path, m1, args, config, device, False, False)
    return_dict[proc_id] = all_sdr
    return

def valid_multi_gpu(model, args, config, device_ids, verbose=False):
    start_time = time.time()
    all_mixtures_path = glob.glob(args.valid_path + '/*/mixture.wav')
    print('Total mixtures: {}'.format(len(all_mixtures_path)))
    print('Overlap: {} Batch size: {}'.format(config.inference.num_overlap, config.inference.batch_size))

    model = model.to('cpu')
    proc_mixtures = np.array_split(all_mixtures_path, len(device_ids))
    processes = []
    return_dict = torch.multiprocessing.Manager().dict()
    for i, device in enumerate(device_ids):
        device = 'cuda:{}'.format(device)
        print('Device: {}. Proc: {}'.format(device, len(proc_mixtures[i])))
        p = torch.multiprocessing.Process(target=valid_mp, args=(i, proc_mixtures[i], model, args, config, device, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()  # wait for all subprocesses to finish

    # print(return_dict)

    all_sdr = dict()
    for instr in config.training.instruments:
        all_sdr[instr] = []
        for i in range(len(device_ids)):
            all_sdr[instr] += return_dict[i][instr]

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if args.store_dir != "":
        out = open(args.store_dir + '/results.txt', 'w')
        out.write(str(args) + "\n")
    print("Num overlap: {}".format(config.inference.num_overlap))
    sdr_avg = 0.0
    for instr in instruments:
        sdr_val = np.array(all_sdr[instr]).mean()
        print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
        if args.store_dir != "":
            out.write("Instr SDR {}: {:.4f}".format(instr, sdr_val) + "\n")
        sdr_avg += sdr_val
    sdr_avg /= len(instruments)
    if len(instruments) > 1:
        print('SDR Avg: {:.4f}'.format(sdr_avg))
    if args.store_dir != "":
        out.write('SDR Avg: {:.4f}'.format(sdr_avg) + "\n")
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    if args.store_dir != "":
        out.write("Elapsed time: {:.2f} sec".format(time.time() - start_time) + "\n")
        out.close()

    return sdr_avg


def check_validation(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--valid_path", type=str, help="validate path")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        if args.model_type == 'htdemucs':
            config = OmegaConf.load(args.config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    print("Instruments: {}".format(config.training.instruments))

    model = get_model_from_config(args.model_type, config)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        state_dict = torch.load(args.start_check_point)
        if args.model_type == 'htdemucs':
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
        model.load_state_dict(state_dict)

    device_ids = args.device_ids
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run validation on CPU. It will be very slow...')
        model = model.to(device)

    if torch.cuda.is_available() and len(device_ids) > 1:
        valid_multi_gpu(model, args, config, device_ids, verbose=False)
    else:
        valid(model, args, config, device, verbose=False)


if __name__ == "__main__":
    check_validation(None)
