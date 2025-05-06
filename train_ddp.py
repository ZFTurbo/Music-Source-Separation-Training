# coding: utf-8
__author__ = 'Ilya Kiselev (kiselecheck): https://github.com/kiselecheck'
__version__ = '1.0.1'

import sys
import argparse
from tqdm.auto import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
import torch.multiprocessing as mp
import loralib as lora

from ml_collections import ConfigDict
from typing import List, Dict, Callable

from utils.losses import choice_loss
from utils.model_utils import bind_lora_to_model, load_start_checkpoint, normalize_batch
from utils.settings import get_model_from_config,  parse_args_train, save_weights_ddp, wandb_init_ddp, \
    initialize_environment_ddp, prepare_data_ddp, get_optimizer_ddp, cleanup_ddp, save_last_weights_ddp
from valid_ddp import valid_multi_gpu

import warnings

warnings.filterwarnings("ignore")


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool,
                    scaler: torch.cuda.amp.GradScaler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
    model.train()
    if dist.get_rank() == 0:
        print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]}')
    loss_val = 0.
    total = 0

    normalize = getattr(config.training, 'normalize', False)

    pbar = tqdm(train_loader,
                dynamic_ncols=True) if dist.get_rank() == 0 else train_loader  # Only main process print progress bar

    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)
        y = batch.to(device)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.model_type in ['mel_band_roformer', 'bs_roformer']:
                loss = model(x, y)
                if isinstance(device_ids, (list, tuple)):
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y, x)

        loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()

        if config.training.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        if (i + 1) % gradient_accumulation_steps == 0 or (i == len(train_loader) - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            loss_copy = loss.detach().clone()
            dist.all_reduce(loss_copy, op=dist.ReduceOp.SUM)
            loss_copy /= dist.get_world_size()

        if dist.get_rank() == 0:
            li = loss_copy.item() * gradient_accumulation_steps
            loss_val += li
            total += 1
            pbar.set_postfix({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1)})
            sys.stdout.flush()
            wandb.log({'loss': 100 * li, 'avg_loss': 100 * loss_val / (i + 1), 'i': i})

    if dist.get_rank() == 0:
        print(f'Training loss: {loss_val / total}')
        wandb.log({'train_loss': loss_val / total, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def compute_epoch_metrics(model: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          best_metric: float,
                          epoch: int, scheduler: torch.optim.lr_scheduler._LRScheduler, metrics_avg, all_metrics) -> float:
    """
    Compute and log the metrics for the current epoch, and save model weights if the metric improves.

    Args:
        model: The model to evaluate.
        args: Command-line arguments containing configuration paths and other settings.
        config: Configuration dictionary containing training settings.
        best_metric: The best metric value seen so far.
        epoch: The current epoch number.
        scheduler: The learning rate scheduler to adjust the learning rate.
        rank: The rank of the current process in DDP.
        world_size: The total number of processes in DDP.

    Returns:
        The updated best_metric.
    """

    metric_avg = metrics_avg[args.metric_for_scheduler]

    if metric_avg > best_metric:

        if args.each_metrics_in_name:
            vocal_sdr_values = np.array(all_metrics[args.metric_for_scheduler]['vocals'])
            vocal_mean_val = vocal_sdr_values.mean()
            vocal_std_val = vocal_sdr_values.std()

            other_sdr_values = np.array(all_metrics[args.metric_for_scheduler]['other'])
            other_mean_val = other_sdr_values.mean()
            other_std_val = other_sdr_values.std()

            store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}_vocals_{args.metric_for_scheduler}_{vocal_mean_val:.4f}_std_{vocal_std_val:.4f}__other_{args.metric_for_scheduler}_{other_mean_val:.4f}_std_{other_std_val:.4f}.ckpt'
        else:
            store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}_{args.metric_for_scheduler}_{metric_avg:.4f}.ckpt'
        if dist.get_rank() == 0:
            print(f'Store weights: {store_path}')
        train_lora = args.train_lora
        save_weights_ddp(store_path, model, train_lora)
        best_metric = metric_avg

    scheduler.step(metric_avg)
    wandb.log({'metric_main': metric_avg, 'best_metric': best_metric})
    for metric_name in metrics_avg:
        wandb.log({f'metric_{metric_name}': metrics_avg[metric_name]})

    return best_metric


def train_model_single(rank: int, world_size: int, args=None):
    """
    Trains the model based on the provided arguments, including data preparation, optimizer setup,
    and loss calculation. The model is trained for multiple epochs with logging via wandb.

    Args:
        args: Command-line arguments containing configuration paths, hyperparameters, and other settings.

    Returns:
        None
    """

    args = parse_args_train(args)

    initialize_environment_ddp(rank, world_size, args.seed, args.results_path)
    model, config = get_model_from_config(args.model_type, args.config_path)
    use_amp = getattr(config.training, 'use_amp', True)
    batch_size = config.training.batch_size

    wandb_init_ddp(args, config, batch_size)


    if args.start_check_point:
        load_start_checkpoint(args, model, type_='train')

    if args.train_lora:
        model = bind_lora_to_model(config, model)
        lora.mark_only_lora_as_trainable(model)

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_loader = prepare_data_ddp(config, args, batch_size, rank, world_size)
    if args.pre_valid:
        valid_multi_gpu(model, args, config, rank, world_size, verbose=False)

    optimizer = get_optimizer_ddp(config, model)
    gradient_accumulation_steps = int(getattr(config.training, 'gradient_accumulation_steps', 1))
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=config.training.patience,
                                  factor=config.training.reduce_factor)

    multi_loss = choice_loss(args, config)
    best_metric = float('-inf')
    if dist.get_rank() == 0:
        print(
            f"Instruments: {config.training.instruments}\n"
            f"Metrics for training: {args.metrics}. Metric for scheduler: {args.metric_for_scheduler}\n"
            f"Patience: {config.training.patience} "
            f"Reduce factor: {config.training.reduce_factor}\n"
            f"Batch size: {batch_size} "
            f"Grad accum steps: {gradient_accumulation_steps} "
            f"Effective batch size: {batch_size * gradient_accumulation_steps}\n"
            f"Dataset type: {args.dataset_type}\n"
            f"Optimizer: {config.training.optimizer}"
        )

        print(f'Train for: {config.training.num_epochs} epochs')

    for epoch in range(config.training.num_epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, config, args, optimizer, device, args.device_ids, epoch,
                           use_amp, scaler, gradient_accumulation_steps, train_loader, multi_loss)

        if rank == 0:
            save_last_weights_ddp(args, model)
        metrics_avg, all_metrics = valid_multi_gpu(model, args, config, rank, world_size, verbose=False)
        if rank == 0:
            best_metric = compute_epoch_metrics(model, args, config, best_metric, epoch, scheduler, metrics_avg, all_metrics)

    cleanup_ddp()  # Close DDP


def train_model(args=None):
    world_size = torch.cuda.device_count()
    mp.spawn(train_model_single, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    train_model()
