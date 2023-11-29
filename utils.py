# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import numpy as np
import torch
import torch.nn as nn


def get_model_from_config(model_type, config):
    if model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
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
    else:
        print('Unknown model: {}'.format(model_type))
        model = None

    return model


def demix_track(config, model, mix, device):
    C = config.audio.hop_length * (config.inference.dim_t - 1)
    N = config.inference.num_overlap
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}


def demix_track_demucs(config, model, mix, device):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    N = config.inference.num_overlap
    step = C // N
    # print(S, C, N, step, mix.shape, mix.device)

    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            mix = mix.to(device)
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0
            all_parts = []
            all_lengths = []
            all_steps = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                all_parts.append(part)
                all_lengths.append(length)
                all_steps.append(i)
                i += step
            all_parts = torch.stack(all_parts, dim=0)
            # print(all_parts.shape)

            start_time = time.time()
            res = model(all_parts)
            # print(res.shape)
            # print("Time:", time.time() - start_time)
            # print(part.mean(), part.max(), part.min())
            # print(x.mean(), x.max(), x.min())

            for j in range(res.shape[0]):
                x = res[j]
                length = all_lengths[j]
                i = all_steps[j]
                # Sometimes model gives nan...
                if torch.isnan(x[..., :length]).any():
                    result[..., i:i+length] += all_parts[j][..., :length].to(device)
                else:
                    result[..., i:i + length] += x[..., :length]
                counter[..., i:i+length] += 1.

            # print(result.mean(), result.max(), result.min())
            # print(counter.mean(), counter.max(), counter.min())
            estimated_sources = result / counter

    if S > 1:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources.cpu().numpy())}
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
