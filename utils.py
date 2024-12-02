# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import numpy as np
import torch
import torch.nn as nn
import yaml
import librosa
import torch.nn.functional as F
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any


def load_config(model_type: str, config_path: str) -> Any:
    """
    Load the configuration from the specified path based on the model type.

    Parameters:
    ----------
    model_type : str
        The type of model to load (e.g., 'htdemucs', 'mdx23c', etc.).
    config_path : str
        The path to the YAML or OmegaConf configuration file.

    Returns:
    -------
    config : Any
        The loaded configuration, which can be in different formats (e.g., OmegaConf or ConfigDict).

    Raises:
    ------
    FileNotFoundError:
        If the configuration file at `config_path` is not found.
    ValueError:
        If there is an error loading the configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            if model_type == 'htdemucs':
                config = OmegaConf.load(config_path)
            else:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple:
    """
    Load the model specified by the model type and configuration file.

    Parameters:
    ----------
    model_type : str
        The type of model to load (e.g., 'mdx23c', 'htdemucs', 'scnet', etc.).
    config_path : str
        The path to the configuration file (YAML or OmegaConf format).

    Returns:
    -------
    model : nn.Module or None
        The initialized model based on the `model_type`, or None if the model type is not recognized.
    config : Any
        The configuration used to initialize the model. This could be in different formats
        depending on the model type (e.g., OmegaConf, ConfigDict).

    Raises:
    ------
    ValueError:
        If the `model_type` is unknown or an error occurs during model initialization.
    """

    config = load_config(model_type, config_path)

    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'apollo':
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'bs_mamba2':
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


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

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
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
                        window = windowing_array
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

    return (
        {k: v for k, v in zip(instruments, estimated_sources)} if num_instruments > 1
        else estimated_sources
    )


def sdr(references: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    """
    Compute Signal-to-Distortion Ratio (SDR) for one or more audio tracks.

    SDR is a measure of how well the predicted source (estimate) matches the reference source.
    It is calculated as the ratio of the energy of the reference signal to the energy of the error (difference between reference and estimate).
    Return SDR in decibels (dB)
    Parameters:
    ----------
    references : np.ndarray
        A 3D numpy array of shape (num_sources, num_channels, num_samples), where num_sources is the number of sources,
        num_channels is the number of channels (e.g., 1 for mono, 2 for stereo), and num_samples is the length of the audio signal.

    estimates : np.ndarray
        A 3D numpy array of shape (num_sources, num_channels, num_samples) representing the estimated sources.

    Returns:
    -------
    np.ndarray
        A 1D numpy array containing the SDR values for each source.
    """
    eps = 1e-8  # to avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += eps
    den += eps
    return 10 * np.log10(num / den)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) for one or more audio tracks.

    SI-SDR is a variant of the SDR metric that is invariant to the scaling of the estimate relative to the reference.
    It is calculated by scaling the estimate to match the reference signal and then computing the SDR.

    Parameters:
    ----------
    reference : np.ndarray
        A 3D numpy array of shape (num_sources, num_channels, num_samples), where num_sources is the number of sources,
        num_channels is the number of channels (e.g., 1 for mono, 2 for stereo), and num_samples is the length of the audio signal.

    estimate : np.ndarray
        A 3D numpy array of shape (num_sources, num_channels, num_samples) representing the estimated sources.

    Returns:
    -------
    float
        The SI-SDR value for the source. It is a scalar representing the Signal-to-Distortion Ratio in decibels (dB).
    """
    eps = 1e-8  # To avoid numerical errors
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference ** 2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))  # Reshape to [num_sources, 1]

    reference = reference * scale
    si_sdr = np.mean(10 * np.log10(
        np.sum(reference ** 2, axis=(0, 1)) / (np.sum((reference - estimate) ** 2, axis=(0, 1)) + eps) + eps))

    return si_sdr


def L1Freq_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        fft_size: int = 2048,
        hop_size: int = 1024,
        device: str = 'cpu'
) -> float:
    """
    Compute the L1 Frequency Metric between the reference and estimated audio signals.

    This metric compares the magnitude spectrograms of the reference and estimated audio signals
    using the Short-Time Fourier Transform (STFT) and calculates the L1 loss between them. The result
    is scaled to the range [0, 100] where a higher value indicates better performance.

    Parameters:
    ----------
    reference : np.ndarray
        A 2D numpy array of shape (num_channels, num_samples) representing the reference (ground truth) audio signal.

    estimate : np.ndarray
        A 2D numpy array of shape (num_channels, num_samples) representing the estimated (predicted) audio signal.

    fft_size : int, optional
        The size of the FFT (Short-Time Fourier Transform). Default is 2048.

    hop_size : int, optional
        The hop size between STFT frames. Default is 1024.

    device : str, optional
        The device to run the computation on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
    -------
    float
        The L1 Frequency Metric in the range [0, 100], where higher values indicate better performance.
    """

    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)

    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    loss = 10 * F.l1_loss(estimate_mag, reference_mag)

    ret = 100 / (1. + float(loss.cpu().numpy()))

    return ret


def LogWMSE_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        mixture: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate the Log-WMSE (Logarithmic Weighted Mean Squared Error) between the reference, estimate, and mixture signals.

    This metric evaluates the quality of the estimated signal compared to the reference signal in the
    context of audio source separation. The result is given in logarithmic scale, which helps in evaluating
    signals with large amplitude differences.

    Parameters:
    ----------
    reference : np.ndarray
        The ground truth audio signal of shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal of shape (channels, time).

    mixture : np.ndarray
        The mixed audio signal of shape (channels, time).

    device : str, optional
        The device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    float
        The Log-WMSE value, which quantifies the difference between the reference and estimated signal on a logarithmic scale.
    """
    from torch_log_wmse import LogWMSE
    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / 44100,  # audio length in seconds
        sample_rate=44100,  # sample rate of 44100 Hz
        return_as_loss=False,  # return as loss (False means return as metric)
        bypass_filter=False,  # bypass frequency filtering (False means apply filter)
    )

    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)

    res = log_wmse(mixture, reference, estimate)

    return float(res.cpu().numpy())


def AuraSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate the AuraSTFT metric, which evaluates the spectral difference between the reference and estimated
    audio signals using Short-Time Fourier Transform (STFT) loss.

    The AuraSTFT metric computes the STFT loss in both logarithmic and linear magnitudes, and it is commonly used
    to assess the quality of audio separation tasks. The result is returned as a value scaled to the range [0, 100].

    Parameters:
    ----------
    reference : np.ndarray
        The ground truth audio signal of shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal of shape (channels, time).

    device : str, optional
        The device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    float
        The AuraSTFT metric value, scaled to the range [0, 100], which quantifies the difference between
        the reference and estimated signal in the spectral domain.
    """

    from auraloss.freq import STFTLoss

    stft_loss = STFTLoss(
        w_log_mag=1.0,  # weight for log magnitude
        w_lin_mag=0.0,  # weight for linear magnitude
        w_sc=1.0,       # weight for spectral centroid
        device=device,
    )

    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)

    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())


def AuraMRSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate the AuraMRSTFT metric, which evaluates the spectral difference between the reference and estimated
    audio signals using Multi-Resolution Short-Time Fourier Transform (STFT) loss.

    The AuraMRSTFT metric uses multi-resolution STFT analysis, which allows better representation of both
    low- and high-frequency components in the audio signals. The result is returned as a value scaled to the range [0, 100].

    Parameters:
    ----------
    reference : np.ndarray
        The ground truth audio signal of shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal of shape (channels, time).

    device : str, optional
        The device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    float
        The AuraMRSTFT metric value, scaled to the range [0, 100], which quantifies the difference between
        the reference and estimated signal in the multi-resolution spectral domain.
    """

    from auraloss.freq import MultiResolutionSTFTLoss

    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],
        hop_sizes=[256, 512, 1024],
        win_lengths=[1024, 2048, 4096],
        scale="mel",  # mel scale for frequency resolution
        n_bins=128,   # number of bins for mel scale
        sample_rate=44100,
        perceptual_weighting=True,  # apply perceptual weighting
        device=device
    )

    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())


def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Calculate the 'bleed' and 'fullness' metrics between a reference and an estimated audio signal.

    The 'bleed' metric measures how much the estimated signal bleeds into the reference signal,
    while the 'fullness' metric measures how much the estimated signal retains its distinctiveness
    in relation to the reference signal, both using mel spectrograms and decibel scaling.

    Parameters:
    ----------
    reference : np.ndarray
        The reference audio signal, shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal, shape (channels, time).

    sr : int, optional
        The sample rate of the audio signals. Default is 44100 Hz.

    n_fft : int, optional
        The FFT size used to compute the STFT. Default is 4096.

    hop_length : int, optional
        The hop length for STFT computation. Default is 1024.

    n_mels : int, optional
        The number of mel frequency bins. Default is 512.

    device : str, optional
        The device for computation, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    tuple
        A tuple containing two values:
        - `bleedless` (float): A score indicating how much 'bleeding' the estimated signal has (higher is better).
        - `fullness` (float): A score indicating how 'full' the estimated signal is (higher is better).
    """

    from torchaudio.transforms import AmplitudeToDB

    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    window = torch.hann_window(n_fft).to(device)

    # Compute STFTs with the Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    diff = S2_db - S1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return bleedless.cpu().numpy(), fullness.cpu().numpy()


def get_metrics(
        metrics: List[str],
        reference: np.ndarray,
        estimate: np.ndarray,
        mix: np.ndarray,
        device: str = 'cpu',
) -> Dict[str, float]:
    """
    Calculate a list of metrics to evaluate the performance of audio source separation models.

    The function computes the specified metrics based on the reference, estimate, and mixture.

    Parameters:
    ----------
    metrics : List[str]
        A list of metric names to compute (e.g., ['sdr', 'si_sdr', 'l1_freq']).

    reference : np.ndarray
        The reference audio (true signal) with shape (channels, length).

    estimate : np.ndarray
        The estimated audio (predicted signal) with shape (channels, length).

    mix : np.ndarray
        The mixed audio signal with shape (channels, length).

    device : str, optional, default='cpu'
        The device ('cpu' or 'cuda') to perform the calculations on.

    Returns:
    -------
    Dict[str, float]
        A dictionary containing the computed metric values.
    """
    result = dict()

    # Adjust the length to be the same across all inputs
    min_length = min(reference.shape[1], estimate.shape[1])
    reference = reference[:, :min_length]
    estimate = estimate[:, :min_length]
    mix = mix[:, :min_length]

    if 'sdr' in metrics:
        references = np.expand_dims(reference, axis=0)
        estimates = np.expand_dims(estimate, axis=0)
        result['sdr'] = sdr(references, estimates)[0]

    if 'si_sdr' in metrics:
        result['si_sdr'] = si_sdr(reference, estimate)

    if 'l1_freq' in metrics:
        result['l1_freq'] = L1Freq_metric(reference, estimate, device=device)

    if 'log_wmse' in metrics:
        result['log_wmse'] = LogWMSE_metric(reference, estimate, mix, device)

    if 'aura_stft' in metrics:
        result['aura_stft'] = AuraSTFT_metric(reference, estimate, device)

    if 'aura_mrstft' in metrics:
        result['aura_mrstft'] = AuraMRSTFT_metric(reference, estimate, device)

    if 'bleedless' in metrics or 'fullness' in metrics:
        bleedless, fullness = bleed_full(reference, estimate, device=device)
        if 'bleedless' in metrics:
            result['bleedless'] = bleedless
        if 'fullness' in metrics:
            result['fullness'] = fullness

    return result


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
    if config.training.get('target_instrument'):
        return [config.training.target_instrument]
    else:
        return config.training.instruments
