# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.3'

# Read more here:
# https://huggingface.co/docs/accelerate/index

import argparse
import soundfile as sf
import numpy as np
import time
import glob
from tqdm.auto import tqdm
import os
import torch
import wandb
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from accelerate import Accelerator

from dataset import MSSDataset
from utils import get_model_from_config, demix, sdr
from train import masked_loss, manual_seed, load_not_compatible_weights
import warnings

warnings.filterwarnings("ignore")


def valid(model, valid_loader, args, config, device, verbose=False):
    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    all_sdr = dict()
    for instr in instruments:
        all_sdr[instr] = []

    all_mixtures_path = valid_loader
    if verbose:
        all_mixtures_path = tqdm(valid_loader)

    pbar_dict = {}
    for path_list in all_mixtures_path:
        path = path_list[0]
        mix, sr = sf.read(path)
        folder = os.path.dirname(path)
        res = demix(config, model, mix.T, device, model_type=args.model_type) # mix.T
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
            single_val = torch.from_numpy(np.array([sdr_val])).to(device)
            all_sdr[instr].append(single_val)
            pbar_dict['sdr_{}'.format(instr)] = sdr_val
        if verbose:
            all_mixtures_path.set_postfix(pbar_dict)

    return all_sdr


class MSSValidationDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        all_mixtures_path = []
        for valid_path in args.valid_path:
            part = sorted(glob.glob(valid_path + '/*/mixture.wav'))
            if len(part) == 0:
                print('No validation data found in: {}'.format(valid_path))
            all_mixtures_path += part

        self.list_of_files = all_mixtures_path

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, index):
        return self.list_of_files[index]


def train_model(args):
    accelerator = Accelerator()
    device = accelerator.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str, help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1, help="Dataset type. Must be one of: 1, 2, 3 or 4. Details here: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md")
    parser.add_argument("--valid_path", nargs="+", type=str, help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", type=bool, default=False, help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--use_multistft_loss", action='store_true', help="Use MultiSTFT Loss (from auraloss package)")
    parser.add_argument("--use_mse_loss", action='store_true', help="Use default MSE loss")
    parser.add_argument("--use_l1_loss", action='store_true', help="Use L1 loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    manual_seed(args.seed + int(time.time()))
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # Fix possible slow down with dilation convolutions
    torch.multiprocessing.set_start_method('spawn')

    model, config = get_model_from_config(args.model_type, args.config_path)
    accelerator.print("Instruments: {}".format(config.training.instruments))

    os.makedirs(args.results_path, exist_ok=True)

    device_ids = args.device_ids
    batch_size = config.training.batch_size

    # wandb
    if accelerator.is_main_process and args.wandb_key is not None and args.wandb_key.strip() != '':
        wandb.login(key = args.wandb_key)
        wandb.init(project = 'msst-accelerate', config = { 'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size })
    else:
        wandb.init(mode = 'disabled')

    # Fix for num of steps
    config.training.num_steps *= accelerator.num_processes

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, 'metadata_{}.pkl'.format(args.dataset_type)),
        dataset_type=args.dataset_type,
        verbose=accelerator.is_main_process,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    validset = MSSValidationDataset(args)
    valid_dataset_length = len(validset)

    valid_loader = DataLoader(
        validset,
        batch_size=1,
        shuffle=False,
    )

    valid_loader = accelerator.prepare(valid_loader)

    if args.start_check_point != '':
        accelerator.print('Start from checkpoint: {}'.format(args.start_check_point))
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(
                torch.load(args.start_check_point)
            )

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        accelerator.print('Optimizer params from config:\n{}'.format(optim_params))

    if config.training.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif config.training.optimizer == 'sgd':
        accelerator.print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        accelerator.print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    if accelerator.is_main_process:
        print('Processes GPU: {}'.format(accelerator.num_processes))
        print("Patience: {} Reduce factor: {} Batch size: {} Optimizer: {}".format(
            config.training.patience,
            config.training.reduce_factor,
            batch_size,
            config.training.optimizer,
        ))
    # Reduce LR if no SDR improvements for several epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        'max',
        # patience=accelerator.num_processes * config.training.patience, # This is strange place...
        patience=config.training.patience,
        factor=config.training.reduce_factor
    )

    if args.use_multistft_loss:
        try:
            loss_options = dict(config.loss_multistft)
        except:
            loss_options = dict()
        accelerator.print('Loss options: {}'.format(loss_options))
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(
            **loss_options
        )

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    if args.pre_valid:
        sdr_list = valid(model, valid_loader, args, config, device, verbose=accelerator.is_main_process)
        sdr_list = accelerator.gather(sdr_list)
        accelerator.wait_for_everyone()

        # print(sdr_list)

        sdr_avg = 0.0
        instruments = config.training.instruments
        if config.training.target_instrument is not None:
            instruments = [config.training.target_instrument]

        for instr in instruments:
            # print(sdr_list[instr])
            sdr_data = torch.cat(sdr_list[instr], dim=0).cpu().numpy()
            sdr_val = sdr_data.mean()
            accelerator.print("Valid length: {}".format(valid_dataset_length))
            accelerator.print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
            sdr_val = sdr_data[:valid_dataset_length].mean()
            accelerator.print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
            sdr_avg += sdr_val
        sdr_avg /= len(instruments)
        if len(instruments) > 1:
            accelerator.print('SDR Avg: {:.4f}'.format(sdr_avg))
        sdr_list = None

    accelerator.print('Train for: {}'.format(config.training.num_epochs))
    best_sdr = -100
    for epoch in range(config.training.num_epochs):
        model.train().to(device)
        accelerator.print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        pbar = tqdm(train_loader, disable=not accelerator.is_main_process)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch
            x = mixes

            if args.model_type in ['mel_band_roformer', 'bs_roformer']:
                # loss is computed in forward pass
                loss = model(x, y)
            else:
                y_ = model(x)
                if args.use_multistft_loss:
                    y1_ = torch.reshape(y_, (y_.shape[0], y_.shape[1] * y_.shape[2], y_.shape[3]))
                    y1 = torch.reshape(y, (y.shape[0], y.shape[1] * y.shape[2], y.shape[3]))
                    loss = loss_multistft(y1_, y1)
                    # We can use many losses at the same time
                    if args.use_mse_loss:
                        loss += 1000 * nn.MSELoss()(y1_, y1)
                    if args.use_l1_loss:
                        loss += 1000 * F.l1_loss(y1_, y1)
                elif args.use_mse_loss:
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

            accelerator.backward(loss)
            if config.training.grad_clip:
                accelerator.clip_grad_norm_(model.parameters(), config.training.grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            li = loss.item()
            loss_val += li
            total += 1
            if accelerator.is_main_process:
                wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'total': total, 'loss_val': loss_val, 'i': i })
                pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})

        if accelerator.is_main_process:
            print('Training loss: {:.6f}'.format(loss_val / total))
            wandb.log({'train_loss': loss_val / total, 'epoch': epoch})

        # Save last
        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), store_path)

        sdr_list = valid(model, valid_loader, args, config, device, verbose=accelerator.is_main_process)
        sdr_list = accelerator.gather(sdr_list)
        accelerator.wait_for_everyone()

        sdr_avg = 0.0
        instruments = config.training.instruments
        if config.training.target_instrument is not None:
            instruments = [config.training.target_instrument]

        for instr in instruments:
            if accelerator.is_main_process and 0:
                print(sdr_list[instr])
            sdr_data = torch.cat(sdr_list[instr], dim=0).cpu().numpy()
            # sdr_val = sdr_data.mean()
            sdr_val = sdr_data[:valid_dataset_length].mean()
            if accelerator.is_main_process:
                print("Instr SDR {}: {:.4f} Debug: {}".format(instr, sdr_val, len(sdr_data)))
                wandb.log({ f'{instr}_sdr': sdr_val })
            sdr_avg += sdr_val
        sdr_avg /= len(instruments)
        if len(instruments) > 1:
            if accelerator.is_main_process:
                print('SDR Avg: {:.4f}'.format(sdr_avg))
                wandb.log({'sdr_avg': sdr_avg, 'best_sdr': best_sdr})

        if accelerator.is_main_process:
            if sdr_avg > best_sdr:
                store_path = args.results_path + '/model_{}_ep_{}_sdr_{:.4f}.ckpt'.format(args.model_type, epoch, sdr_avg)
                print('Store weights: {}'.format(store_path))
                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), store_path)
                best_sdr = sdr_avg

            scheduler.step(sdr_avg)

        sdr_list = None
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    train_model(None)
