# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

import random
import argparse
import time
import copy
from tqdm.auto import tqdm
import sys
import os
import glob
import torch
import wandb
import soundfile as sf
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from dataset import MSSDataset
from utils import demix, sdr, get_model_from_config
from valid import valid_multi_gpu, valid

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


def load_not_compatible_weights(model, weights, verbose=False):
    new_model = model.state_dict()
    old_model = torch.load(weights)
    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # Fix for apollo weights loading
        old_model = old_model['state_dict']

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


def train_model(args):
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
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"], choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr", choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless', 'fullness'], help='Metric which will be used for scheduler.')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    manual_seed(args.seed + int(time.time()))
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False # Fix possible slow down with dilation convolutions
    torch.multiprocessing.set_start_method('spawn')

    model, config = get_model_from_config(args.model_type, args.config_path)
    print("Instruments: {}".format(config.training.instruments))

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]
    print('Metrics for training: {}. Metric for scheduler: {}'.format(args.metrics, args.metric_for_scheduler))

    os.makedirs(args.results_path, exist_ok=True)

    use_amp = True
    try:
        use_amp = config.training.use_amp
    except:
        pass

    device_ids = args.device_ids
    batch_size = config.training.batch_size * len(device_ids)

    # wandb
    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
    else:
        wandb.login(key=args.wandb_key)
        wandb.init(project='msst', config={'config': config, 'args': args, 'device_ids': device_ids, 'batch_size': batch_size })

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, 'metadata_{}.pkl'.format(args.dataset_type)),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(
                torch.load(args.start_check_point)
            )

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            print('Use single GPU: {}'.format(device_ids))
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            print('Use multi GPU: {}'.format(device_ids))
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not avilable. Run training on CPU. It will be very slow...')
        model = model.to(device)

    if args.pre_valid:
        valid_multi_gpu(model, args, config, args.device_ids, verbose=True)

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        print('Optimizer params from config:\n{}'.format(optim_params))

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
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print('Unknown optimizer: {}'.format(config.training.optimizer))
        exit()

    gradient_accumulation_steps = 1
    try:
        gradient_accumulation_steps = int(config.training.gradient_accumulation_steps)
    except:
        pass

    print("Patience: {} Reduce factor: {} Batch size: {} Grad accum steps: {} Effective batch size: {} Optimizer: {}".format(
        config.training.patience,
        config.training.reduce_factor,
        batch_size,
        gradient_accumulation_steps,
        batch_size * gradient_accumulation_steps,
        config.training.optimizer,
    ))
    # Reduce LR if no metric improvements for several epochs
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience, factor=config.training.reduce_factor)

    if args.use_multistft_loss:
        try:
            loss_options = dict(config.loss_multistft)
        except:
            loss_options = dict()
        print('Loss options: {}'.format(loss_options))
        loss_multistft = auraloss.freq.MultiResolutionSTFTLoss(
            **loss_options
        )

    scaler = GradScaler()
    print('Train for: {}'.format(config.training.num_epochs))
    best_metric = -10000
    for epoch in range(config.training.num_epochs):
        model.train().to(device)
        print('Train epoch: {} Learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        loss_val = 0.
        total = 0

        # total_loss = None
        pbar = tqdm(train_loader)
        for i, (batch, mixes) in enumerate(pbar):
            y = batch.to(device)
            x = mixes.to(device)  # mixture

            if 'normalize' in config.training:
                if config.training.normalize:
                    mean = x.mean()
                    std = x.std()
                    if std != 0:
                        x = (x - mean) / std
                        y = (y - mean) / std

            with torch.cuda.amp.autocast(enabled=use_amp):
                if args.model_type in ['mel_band_roformer', 'bs_roformer']:
                    # loss is computed in forward pass
                    loss = model(x, y)
                    if type(device_ids) != int:
                        # If it's multiple GPUs sum partial loss
                        loss = loss.mean()
                else:
                    y_ = model(x)
                    if args.use_multistft_loss:
                        if len(y_.shape) == 3:
                            # For models like apollo no need to reshape
                            y1_ = y_
                            y1 = y
                        else:
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
            wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})
            loss.detach()

        print('Training loss: {:.6f}'.format(loss_val / total))
        wandb.log({'train_loss': loss_val / total, 'epoch': epoch})

        # Save last
        store_path = args.results_path + '/last_{}.ckpt'.format(args.model_type)
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        torch.save(
            state_dict,
            store_path
        )

        if torch.cuda.is_available() and len(device_ids) > 1:
            metrics_avg = valid_multi_gpu(model, args, config, args.device_ids, verbose=False)
        else:
            metrics_avg = valid(model, args, config, device, verbose=False)
        metric_avg = metrics_avg[args.metric_for_scheduler]
        if metric_avg > best_metric:
            store_path = args.results_path + '/model_{}_ep_{}_{}_{:.4f}.ckpt'.format(args.model_type, epoch, args.metric_for_scheduler, metric_avg)
            print('Store weights: {}'.format(store_path))
            state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
            torch.save(
                state_dict,
                store_path
            )
            best_metric = metric_avg
        scheduler.step(metric_avg)
        wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
        for metric_name in metrics_avg:
            wandb.log({'metric_{}'.format(metric_name): metrics_avg[metric_name]})


if __name__ == "__main__":
    train_model(None)
