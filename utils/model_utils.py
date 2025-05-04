# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union
import loralib as lora


def demix(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """
    Unified function for audio source separation with support for multiple processing modes.

    This function separates audio into its constituent sources using either a generic custom logic
    or a Demucs-specific logic. It supports batch processing and overlapping window-based chunking
    for efficient and artifact-free separation.

    Parameters:
    ----------
    config : ConfigDict
        Configuration object containing audio and inference settings.
    model : torch.nn.Module
        The trained model used for audio source separation.
    mix : torch.Tensor
        Input audio tensor with shape (channels, time).
    device : torch.device
        The computation device (CPU or CUDA).
    model_type : str, optional
        Processing mode:
            - "demucs" for logic specific to the Demucs model.
        Default is "generic".
    pbar : bool, optional
        If True, displays a progress bar during chunk processing. Default is False.

    Returns:
    -------
    Union[Dict[str, np.ndarray], np.ndarray]
        - A dictionary mapping target instruments to separated audio sources if multiple instruments are present.
        - A numpy array of the separated source if only one instrument is present.
    """

    mix = torch.tensor(mix, dtype=torch.float32)

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic'
    # Define processing parameters based on the mode
    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_instruments = len(config.training.instruments)
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = config.audio.chunk_size
        num_instruments = len(prefer_target_instrument(config))
        num_overlap = config.inference.num_overlap

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size)
        # Add padding for generic mode to handle edge artifacts
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    batch_size = config.inference.batch_size

    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            # Initialize result and counter tensors
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False
            ) if pbar else None

            while i < mix.shape[1]:
                # Extract chunk and apply padding if necessary
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                if mode == "generic" and chunk_len > chunk_size // 2:
                    pad_mode = "reflect"
                else:
                    pad_mode = "constant"
                part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                # Process batch if it's full or the end is reached
                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    if mode == "generic":
                        window = windowing_array.clone() # using clone() fixes the clicks at chunk edges when using batch_size=1
                        if i - step == 0:  # First audio chunk, no fadein
                            window[:fade_size] = 1
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window[-fade_size:] = 1

                    for j, (start, seg_len) in enumerate(batch_locations):
                        if mode == "generic":
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu() * window[..., :seg_len]
                            counter[..., start:start + seg_len] += window[..., :seg_len]
                        else:
                            result[..., start:start + seg_len] += x[j, ..., :seg_len].cpu()
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            # Compute final estimated sources
            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            # Remove padding for generic mode
            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    # Return the result as a dictionary or a single array
    if mode == "demucs":
        instruments = config.training.instruments
    else:
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}

    if mode == "demucs" and num_instruments <= 1:
        return estimated_sources
    else:
        return ret_data



def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[Union[torch.device, str], torch.nn.Module]:
    """
    Initialize the model and assign it to the appropriate device (GPU or CPU).

    Args:
        model: The PyTorch model to be initialized.
        device_ids: List of GPU device IDs to use for parallel processing.

    Returns:
        A tuple containing the device and the model moved to that device.
    """

    if torch.cuda.is_available():
        if len(device_ids) <= 1:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")

    return device, model


def get_optimizer(config: ConfigDict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Initializes an optimizer based on the configuration.

    Args:
        config: Configuration object containing training parameters.
        model: PyTorch model whose parameters will be optimized.

    Returns:
        A PyTorch optimizer object configured based on the specified settings.
    """

    optim_params = dict()
    if 'optimizer' in config:
        optim_params = dict(config['optimizer'])
        print(f'Optimizer params from config:\n{optim_params}')

    name_optimizer = getattr(config.training, 'optimizer',
                             'No optimizer in config')

    if name_optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'prodigy':
        from prodigyopt import Prodigy
        # you can choose weight decay value based on your problem, 0 by default
        # We recommend using lr=1.0 (default) for all networks.
        optimizer = Prodigy(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'adamw8bit':
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.training.lr, **optim_params)
    elif name_optimizer == 'sgd':
        print('Use SGD optimizer')
        optimizer = SGD(model.parameters(), lr=config.training.lr, **optim_params)
    else:
        print(f'Unknown optimizer: {name_optimizer}')
        exit()
    return optimizer


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize a batch of tensors (x and y) by subtracting the mean and dividing by the standard deviation.

    Args:
        x: Tensor to normalize.
        y: Tensor to normalize (same as x, typically).

    Returns:
        A tuple of normalized tensors (x, y).
    """

    mean = x.mean()
    std = x.std()
    if std != 0:
        x = (x - mean) / std
        y = (y - mean) / std
    return x, y


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


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in at the beginning and a fade-out at the end.

    This function creates a window of size `window_size` where the first `fade_size` elements
    linearly increase from 0 to 1 (fade-in) and the last `fade_size` elements linearly decrease
    from 1 to 0 (fade-out). The middle part of the window is filled with ones.

    Parameters:
    ----------
    window_size : int
        The total size of the window.
    fade_size : int
        The size of the fade-in and fade-out regions.

    Returns:
    -------
    torch.Tensor
        A tensor of shape (window_size,) containing the generated windowing array.

    Example:
    -------
    If `window_size=10` and `fade_size=3`, the output will be:
    tensor([0.0000, 0.5000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.5000, 0.0000])
    """

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
        Return the list of target instruments based on the configuration.
        If a specific target instrument is specified in the configuration,
        it returns a list with that instrument. Otherwise, it returns the list of instruments.

        Parameters:
        ----------
        config : ConfigDict
            Configuration object containing the list of instruments or the target instrument.

        Returns:
        -------
        List[str]
            A list of target instruments.
        """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        return config.training.instruments


def load_not_compatible_weights(model: torch.nn.Module, weights: str, verbose: bool = False) -> None:
    """
    Load weights into a model, handling mismatched shapes and dimensions.

    Args:
        model: PyTorch model into which the weights will be loaded.
        weights: Path to the weights file.
        verbose: If True, prints detailed information about matching and mismatched layers.
    """

    new_model = model.state_dict()
    old_model = torch.load(weights, weights_only=False)
    if 'state' in old_model:
        # Fix for htdemucs weights loading
        old_model = old_model['state']
    if 'state_dict' in old_model:
        # Fix for apollo weights loading
        old_model = old_model['state_dict']

    for el in new_model:
        if el in old_model:
            if verbose:
                print(f'Match found for {el}!')
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
                        print(f'Shape is different: {tuple(new_model[el].shape)} != {tuple(old_model[el].shape)}')
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
                print(f'Match not found for {el}!')
    model.load_state_dict(
        new_model
    )


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    Load LoRA weights into a model.
    This function updates the given model with LoRA-specific weights from the specified checkpoint file.
    It does not require the checkpoint to match the model's full state dictionary, as only LoRA layers are updated.

    Parameters:
    ----------
    model : Module
        The PyTorch model into which the LoRA weights will be loaded.
    lora_path : str
        Path to the LoRA checkpoint file.
    device : str, optional
        The device to load the weights onto, by default 'cpu'. Common values are 'cpu' or 'cuda'.

    Returns:
    -------
    None
        The model is updated in place.
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)


def load_start_checkpoint(args: argparse.Namespace, model: torch.nn.Module, type_='train') -> None:
    """
    Load the starting checkpoint for a model.

    Args:
        args: Parsed command-line arguments containing the checkpoint path.
        model: PyTorch model to load the checkpoint into.
        type_: how to load weights - for train we can load not fully compatible weights
    """

    print(f'Start from checkpoint: {args.start_check_point}')
    if type_ in ['train']:
        if 1:
            load_not_compatible_weights(model, args.start_check_point, verbose=False)
        else:
            model.load_state_dict(torch.load(args.start_check_point))
    else:
        device='cpu'
        if args.model_type in ['htdemucs', 'apollo']:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=False)
            # Fix for htdemucs pretrained models
            if 'state' in state_dict:
                state_dict = state_dict['state']
            # Fix for apollo pretrained models
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            state_dict = torch.load(args.start_check_point, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    if args.lora_checkpoint:
        print(f"Loading LoRA weights from: {args.lora_checkpoint}")
        load_lora_weights(model, args.lora_checkpoint)


def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    """
    Replaces specific layers in the model with LoRA-extended versions.

    Parameters:
    ----------
    config : Dict[str, Any]
        Configuration containing parameters for LoRA. It should include a 'lora' key with parameters for `MergedLinear`.
    model : nn.Module
        The original model in which the layers will be replaced.

    Returns:
    -------
    nn.Module
        The modified model with the replaced layers.
    """

    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    replaced_layers = 0  # Counter for replaced layers

    for name, module in model.named_modules():
        hierarchy = name.split('.')
        layer_name = hierarchy[-1]

        # Check if this is the target layer to replace (and layer_name == 'to_qkv')
        if isinstance(module, nn.Linear):
            try:
                # Get the parent module
                parent_module = model
                for submodule_name in hierarchy[:-1]:
                    parent_module = getattr(parent_module, submodule_name)

                # Replace the module with LoRA-enabled layer
                setattr(
                    parent_module,
                    layer_name,
                    lora.MergedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        **config['lora']
                    )
                )
                replaced_layers += 1  # Increment the counter

            except Exception as e:
                print(f"Error replacing layer {name}: {e}")

    if replaced_layers == 0:
        print("Warning: No layers were replaced. Check the model structure and configuration.")
    else:
        print(f"Number of layers replaced with LoRA: {replaced_layers}")

    return model


def save_weights(store_path, model, device_ids, train_lora):

    if train_lora:
        torch.save(lora.lora_state_dict(model), store_path)
    else:
        state_dict = model.state_dict() if len(device_ids) <= 1 else model.module.state_dict()
        torch.save(
            state_dict,
            store_path
        )


def save_last_weights(args: argparse.Namespace, model: torch.nn.Module, device_ids: List[int]) -> None:
    """
    Save the model's state_dict to a file for later use.

    Args:
        args: Command-line arguments containing the results path and model type.
        model: The model whose weights will be saved.
        device_ids: List of GPU device IDs if using multiple GPUs.

    Returns:
        None
    """

    store_path = f'{args.results_path}/last_{args.model_type}.ckpt'
    train_lora = args.train_lora
    save_weights(store_path, model, device_ids, train_lora)