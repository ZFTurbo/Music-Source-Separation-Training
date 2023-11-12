# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.1'

import random
import argparse
import yaml
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from dataset import MSSDataset
from utils import demix_track, demix_track_demucs, sdr, get_model_from_config

import warnings

warnings.filterwarnings("ignore")


def masked_loss(y_, y, q, coarse=True):
    # shape = [num_sources, batch_size, num_channels, chunk_size]
    loss = torch.nn.MSELoss(reduction='none')(y_, y).transpose(0, 1)
    if coarse:
        loss = torch.mean(loss, dim=(-1, -2))
    loss = loss.reshape(loss.shape[0], -1)
    L = loss.detach()
    quantile = torch.quantile(L, q, interpolation='linear', dim=1, keepdim=True)
    mask = L < quantile
    return (loss * mask).mean()


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def valid(model, args, config, device, verbose=False):
    model.eval()
    all_mixtures_path = glob.glob(args.valid_path + '/*/mixture.wav')
    if verbose:
        print('Total mixtures: {}'.format(len(all_mixtures_path)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    all_sdr = dict()
    for instr in config.training.instruments:
        all_sdr[instr] = []

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    pbar_dict = {}
    for path in all_mixtures_path:
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        if verbose:
            print('Song: {}'.format(os.path.basename(folder)))
        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if args.model_type == 'htdemucs':
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)
        for instr in instruments:
            if instr != 'other' or config.training.other_fix is False:
                track, sr1 = sf.read(folder + '/{}.wav'.format(instr))
            else:
                # other is actually instrumental
                track, sr1 = sf.read(folder + '/{}.wav'.format('vocals'))
                track = mix - track
            # sf.write("{}.wav".format(instr), res[instr].T, sr, subtype='FLOAT')
            references = np.expand_dims(track, axis=0)
            estimates = np.expand_dims(res[instr].T, axis=0)
            sdr_val = sdr(references, estimates)[0]
            if verbose:
                print(instr, res[instr].shape, sdr_val)
            all_sdr[instr].append(sdr_val)
            pbar_dict['sdr_{}'.format(instr)] = sdr_val
        if not verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    sdr_avg = 0.0
    for instr in instruments:
        sdr_val = np.array(all_sdr[instr]).mean()
        print("Instr SDR {}: {:.4f}".format(instr, sdr_val))
        sdr_avg += sdr_val
    sdr_avg /= len(instruments)
    if len(instruments) > 1:
        print('SDR Avg: {:.4f}'.format(sdr_avg))
    return sdr_avg


def train_model(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, demucs, segm_models, mel_band_roformer, bs_roformer")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str, help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="dataset path. Can be several parameters.")
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", type=str, help="validate path")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    manual_seed(args.seed + int(time.time()))
    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        if args.model_type == 'htdemucs':
            config = OmegaConf.load(args.config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    print("Instruments: {}".format(config.training.instruments))

    if not os.path.isdir(args.results_path):
        os.mkdir(args.results_path)

    trainset = MSSDataset(
        config,
        args.data_path,
        metadata_path=os.path.join(args.results_path, 'metadata_{}.pkl'.format(args.dataset_type)),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    model = get_model_from_config(args.model_type, config)

    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(
                torch.load(args.start_check_point)
            )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not avilable. Run training on CPU. It will be very slow...')
        model = model.to(device)

    if 0:
        valid(model, args, config, device, verbose=True)

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr)
    elif config.training.optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, momentum=0.999)
    else:
        print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    gradient_accumulation_steps = 1
    try:
        gradient_accumulation_steps = int(config.training.gradient_accumulation_steps)
    except:
        pass

    print("Patience: {} Reduce factor: {} Batch size: {} Grad accum steps: {} Effective batch size: {}".format(
        config.training.patience,
        config.training.reduce_factor,
        config.training.batch_size,
        gradient_accumulation_steps,
        config.training.batch_size * gradient_accumulation_steps,
    ))
    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience, factor=config.training.reduce_factor)

    scaler = GradScaler()
    print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = -100
    for epoch in range(config.training.num_epochs):
        model.train()
        print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        # total_loss = None
        pbar = tqdm(train_loader)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(device)
            x = mixes.to(device)  # mixture

            with torch.cuda.amp.autocast():
                if args.model_type in ['mel_band_roformer', 'bs_roformer']:
                    # loss is computed in forward pass
                    loss = model(x, y)
                    if type(device_ids) != int:
                        # If it's multiple GPUs sum partial loss
                        loss = loss.mean()
                else:
                    y_ = model(x)
                    if args.use_mse_loss:
                        loss = nn.MSELoss()(y_, y)
                    elif args.use_l1_loss:
                        loss = F.l1_loss(y_, y)
                    else:
                        loss = masked_loss(
                            y_,
                            y,
                            q=config.training.q,
                            coarse=config.training.coarse_loss_clip
                        )

            loss /= gradient_accumulation_steps
            scaler.scale(loss).backward()
            if config.training.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            li = loss.item() * gradient_accumulation_steps
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            loss.detach()

        print('Training loss: {:.6f}'.format(loss_val / total))
        sdr_avg = valid(model, args, config, device, verbose=False)
        if sdr_avg > best_sdr:
            store_path = args.results_path + '/model_{}_ep_{}_sdr_{:.4f}.ckpt'.format(args.model_type, epoch, sdr_avg)
            print('Store weights: {}'.format(store_path))
            state_dict = model.state_dict() if type(device_ids) == int else model.module.state_dict()
            torch.save(
                state_dict,
                store_path
            )
            best_sdr = sdr_avg
        scheduler.step(sdr_avg)

        # Save last
        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        state_dict = model.state_dict() if type(device_ids) == int else model.module.state_dict()
        torch.save(
            state_dict,
            store_path
        )


def load_not_compatible_weights(model, weights, verbose=False):
    new_model = model.state_dict()
    old_model = torch.load(weights)

    for el in new_model:
        if el in old_model:
            if verbose:
                print('Match found for {}!'.format(el))
            if new_model[el].shape == old_model[el].shape:
                if verbose:
                    print('Action: Just copy weights!')
                new_model[el] = old_model[el]
            else:
                if len(new_model[el].shape) != len(old_model[el].shape):
                    if verbose:
                        print('Action: Different dimension! Too lazy to write the code... Skip it')
                else:
                    if verbose:
                        print('Shape is different: {} != {}'.format(tuple(new_model[el].shape), tuple(old_model[el].shape)))
                    ln = len(new_model[el].shape)
                    max_shape = []
                    slices_old = []
                    slices_new = []
                    for i in range(ln):
                        max_shape.append(max(new_model[el].shape[i], old_model[el].shape[i]))
                        slices_old.append(slice(0, old_model[el].shape[i]))
                        slices_new.append(slice(0, new_model[el].shape[i]))
                    # print(max_shape)
                    # print(slices_old, slices_new)
                    slices_old = tuple(slices_old)
                    slices_new = tuple(slices_new)
                    max_matrix = np.zeros(max_shape, dtype=np.float32)
                    for i in range(ln):
                        max_matrix[slices_old] = old_model[el].cpu().numpy()
                    max_matrix = torch.from_numpy(max_matrix)
                    new_model[el] = max_matrix[slices_new]
        else:
            if verbose:
                print('Match not found for {}!'.format(el))
    model.load_state_dict(
        new_model
    )


if __name__ == "__main__":
    train_model(None)
