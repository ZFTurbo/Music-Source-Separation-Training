# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import numpy as np
import torch
import torch.nn as nn
import yaml
import librosa
import torch.nn.functional as F
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict, List


def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        if model_type == 'htdemucs':
            config = OmegaConf.load(config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

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
        model = MelBandRoformer(
            **dict(config.model)
        )
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config.model)
        )
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(
            **config.model
        )
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(
            **config.kwargs
        )
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(
            **config.model
        )
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(
            **config.model
        )
    elif model_type == 'apollo':
        from models.look2hear.models import BaseModel
        model = BaseModel.apollo(**config.model)
    elif model_type == 'bs_mamba2':
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model, config

def _getWindowingArray(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    N = config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = config.inference.batch_size

    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')

    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            req_shape = (len(prefer_target_instrument(config)),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = windowingArray
                    if i - step == 0:  # First audio chunk, no fadein
                        window[:fade_size] = 1
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window[-fade_size:] = 1

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    return {k: v for k, v in zip(prefer_target_instrument(config), estimated_sources)}

def demix_track_demucs(config, model, mix, device, pbar=False):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    N = config.inference.num_overlap
    batch_size = config.inference.batch_size
    step = C // N
    # print(S, C, N, step, mix.shape, mix.device)

    with torch.cuda.amp.autocast(enabled=config.training.use_amp):
        with torch.inference_mode():
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step


                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return estimated_sources


def sdr(references, estimates):
    # compute SDR for one song
    delta = 1e-7  # avoid numerical errors
    num = np.sum(np.square(references), axis=(1, 2))
    den = np.sum(np.square(references - estimates), axis=(1, 2))
    num += delta
    den += delta
    return 10 * np.log10(num / den)


def si_sdr(reference, estimate):
    eps = 1e-07
    scale = np.sum(estimate * reference + eps, axis=(0, 1)) / np.sum(reference**2 + eps, axis=(0, 1))
    scale = np.expand_dims(scale, axis=(0, 1))  # shape - [50, 1]
    reference = reference * scale
    sisdr = np.mean(10 * np.log10(np.sum(reference**2, axis=(0, 1)) / (np.sum((reference - estimate)**2, axis=(0, 1)) + eps) + eps))
    return sisdr


def L1Freq_metric(
    reference,
    estimate,
    fft_size=2048,
    hop_size=1024,
    device='cpu'
):
    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)
    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)
    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)
    loss = 10 * F.l1_loss(estimate_mag, reference_mag)
    # Metric is on the range from 0 to 100 - larger the better
    ret = 100 / (1. + float(loss.cpu().numpy()))
    return ret


def LogWMSE_metric(
    reference,
    estimate,
    mixture,
    device='cpu',
):
    from torch_log_wmse import LogWMSE
    log_wmse = LogWMSE(
        audio_length=reference.shape[-1] / 44100,
        sample_rate=44100,
        return_as_loss=False,  # optional
        bypass_filter=False,  # optional
    )
    reference = torch.from_numpy(reference).unsqueeze(0).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).unsqueeze(0).to(device)
    mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)
    # print(reference.shape, estimate.shape, mixture.shape)
    res = log_wmse(mixture, reference, estimate)
    return float(res.cpu().numpy())


def AuraSTFT_metric(
    reference,
    estimate,
    device='cpu',
):
    from auraloss.freq import STFTLoss
    stft_loss = STFTLoss(
        w_log_mag=1.0,
        w_lin_mag=0.0,
        w_sc=1.0,
        device=device,
    )
    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)
    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())


def AuraMRSTFT_metric(
    reference,
    estimate,
    device='cpu',
):
    from auraloss.freq import MultiResolutionSTFTLoss
    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],
        hop_sizes=[256, 512, 1024],
        win_lengths=[1024, 2048, 4096],
        scale="mel",
        n_bins=128,
        sample_rate=44100,
        perceptual_weighting=True,
        device=device
    )
    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)
    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())


def bleed_full(
    reference,
    estimate,
    sr=44100,
    n_fft=4096,
    hop_length=1024,
    n_mels=512,
    device='cpu',
):
    from torchaudio.transforms import AmplitudeToDB

    # Move tensors to GPU if available
    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    # Create a Hann window
    window = torch.hann_window(n_fft).to(device)

    # Compute STFTs with the Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="constant"))

    # create mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)  # (melbandroformer is doing it that way) edit: sent to right device now

    # apply mel scale
    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    # Convert to decibels
    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    # Calculate difference
    diff = S2_db - S1_db

    # Separate positive and negative differences
    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    # Calculate averages
    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    # Scale with 100 as best score
    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return bleedless.cpu().numpy(), fullness.cpu().numpy()


def get_metrics(
    metrics,
    reference, # (ch, length)
    estimate,  # (ch, length)
    mix,       # (ch, length)
    device='cpu',
):
    result = dict()
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


def demix(config, model, mix: NDArray, device, pbar=False, model_type: str = None) -> Dict[str, NDArray]:
    mix = torch.tensor(mix, dtype=torch.float32)
    if model_type == 'htdemucs':
        return demix_track_demucs(config, model, mix, device, pbar=pbar)
    else:
        return demix_track(config, model, mix, device, pbar=pbar)


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    if config.training.get('target_instrument'):
        return [config.training.target_instrument]
    else:
        return config.training.instruments
