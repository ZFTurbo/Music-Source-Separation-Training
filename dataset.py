# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
from tqdm import tqdm
from glob import glob
import audiomentations as AU
import pedalboard as PB
import warnings
warnings.filterwarnings("ignore")


def load_chunk(path, length, chunk_size, offset=None):
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        x = sf.read(path, dtype='float32')[0]
        pad = np.zeros([chunk_size - length, 2])
        x = np.concatenate([x, pad])
    return x.T


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1, batch_size=None):
        self.config = config
        self.dataset_type = dataset_type # 1, 2, 3 or 4
        self.instruments = instruments = config.training.instruments
        if batch_size is None:
            batch_size = config.training.batch_size
        self.batch_size = batch_size
        self.file_types = ['wav', 'flac']

        # Augmentation block
        self.aug = False
        if 'augmentations' in config:
            if config['augmentations'].enable is True:
                print('Use augmentation for training')
                self.aug = True
        else:
            print('There is no augmentations block in config. Augmentations disabled for training...')

        # metadata_path = data_path + '/metadata'
        try:
            metadata = pickle.load(open(metadata_path, 'rb'))
            print('Loading songs data from cache: {}. If you updated dataset remove {} before training!'.format(metadata_path, os.path.basename(metadata_path)))
        except Exception:
            print('Collecting metadata for', str(data_path), 'Dataset type:', self.dataset_type)
            if self.dataset_type in [1, 4]:
                metadata = []
                track_paths = []
                if type(data_path) == list:
                    for tp in data_path:
                        track_paths += sorted(glob(tp + '/*'))
                else:
                    track_paths += sorted(glob(data_path + '/*'))

                track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
                for path in tqdm(track_paths):
                    # Check lengths of all instruments (it can be different in some cases)
                    lengths_arr = []
                    for instr in instruments:
                        length = -1
                        for extension in self.file_types:
                            path_to_audio_file = path + '/{}.{}'.format(instr, extension)
                            if os.path.isfile(path_to_audio_file):
                                length = len(sf.read(path_to_audio_file)[0])
                                break
                        if length == -1:
                            print('Cant find file "{}" in folder {}'.format(instr, path))
                            continue
                        lengths_arr.append(length)
                    lengths_arr = np.array(lengths_arr)
                    if lengths_arr.min() != lengths_arr.max():
                        print('Warning: lengths of stems are different for path: {}. ({} != {})'.format(
                            path,
                            lengths_arr.min(),
                            lengths_arr.max())
                        )
                    # We use minimum to allow overflow for soundfile read in non-equal length cases
                    metadata.append((path, lengths_arr.min()))
            elif self.dataset_type == 2:
                metadata = dict()
                for instr in self.instruments:
                    metadata[instr] = []
                    track_paths = []
                    if type(data_path) == list:
                        for tp in data_path:
                            track_paths += sorted(glob(tp + '/{}/*.wav'.format(instr)))
                            track_paths += sorted(glob(tp + '/{}/*.flac'.format(instr)))
                    else:
                        track_paths += sorted(glob(data_path + '/{}/*.wav'.format(instr)))
                        track_paths += sorted(glob(data_path + '/{}/*.flac'.format(instr)))

                    for path in tqdm(track_paths):
                        length = len(sf.read(path)[0])
                        metadata[instr].append((path, length))
            elif self.dataset_type == 3:
                import pandas as pd
                if type(data_path) != list:
                    data_path = [data_path]

                metadata = dict()
                for i in range(len(data_path)):
                    print('Reading tracks from: {}'.format(data_path[i]))
                    df = pd.read_csv(data_path[i])

                    skipped = 0
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        print('Tracks found for {}: {}'.format(instr, len(part)))
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        metadata[instr] = []
                        track_paths = list(part['path'].values)
                        for path in tqdm(track_paths):
                            if not os.path.isfile(path):
                                print('Cant find track: {}'.format(path))
                                skipped += 1
                                continue
                            # print(path)
                            try:
                                length = len(sf.read(path)[0])
                            except:
                                print('Problem with path: {}'.format(path))
                                skipped += 1
                                continue
                            metadata[instr].append((path, length))
                    if skipped > 0:
                        print('Missing tracks: {} from {}'.format(skipped, len(df)))
            else:
                print('Unknown dataset type: {}. Must be 1, 2 or 3'.format(self.dataset_type))
                exit()

            pickle.dump(metadata, open(metadata_path, 'wb'))

        if self.dataset_type in [1, 4]:
            if len(metadata) > 0:
                print('Found tracks in dataset: {}'.format(len(metadata)))
            else:
                print('No tracks found for training. Check paths you provided!')
                exit()
        else:
            for instr in self.instruments:
                print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        self.chunk_size = config.audio.chunk_size
        self.min_mean_abs = config.audio.min_mean_abs

    def __len__(self):
        return self.config.training.num_steps * self.batch_size

    def load_source(self, metadata, instr):
        while True:
            if self.dataset_type in [1, 4]:
                track_path, track_length = random.choice(metadata)
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(instr, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                        except Exception as e:
                            # Sometimes error during FLAC reading, catch it and use zero stem
                            print('Error: {} Path: {}'.format(e, path_to_audio_file))
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break
            else:
                track_path, track_length = random.choice(metadata[instr])
                try:
                    source = load_chunk(track_path, track_length, self.chunk_size)
                except Exception as e:
                    # Sometimes error during FLAC reading, catch it and use zero stem
                    print('Error: {} Path: {}'.format(e, track_path))
                    source = np.zeros((2, self.chunk_size), dtype=np.float32)

            if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                break
        if self.aug:
            source = self.augm_data(source, instr)
        return torch.tensor(source, dtype=torch.float32)

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            s1 = self.load_source(self.metadata, instr)
            # Mixup augmentation. Multiple mix of same type of stems
            if self.aug:
                if 'mixup' in self.config['augmentations']:
                    if self.config['augmentations'].mixup:
                        mixup = [s1]
                        for prob in self.config.augmentations.mixup_probs:
                            if random.uniform(0, 1) < prob:
                                s2 = self.load_source(self.metadata, instr)
                                mixup.append(s2)
                        mixup = torch.stack(mixup, dim=0)
                        loud_values = np.random.uniform(
                            low=self.config.augmentations.loudness_min,
                            high=self.config.augmentations.loudness_max,
                            size=(len(mixup),)
                        )
                        loud_values = torch.tensor(loud_values, dtype=torch.float32)
                        mixup *= loud_values[:, None, None]
                        s1 = mixup.mean(dim=0, dtype=torch.float32)
            res.append(s1)
        res = torch.stack(res)
        return res

    def load_aligned_data(self):
        track_path, track_length = random.choice(self.metadata)
        res = []
        for i in self.instruments:
            attempts = 10
            while attempts:
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(i, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                        except Exception as e:
                            # Sometimes error during FLAC reading, catch it and use zero stem
                            print('Error: {} Path: {}'.format(e, path_to_audio_file))
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break
                if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                    break
                attempts -= 1
                if attempts <= 0:
                    print('Attempts max!', track_path)
            res.append(source)
        res = np.stack(res, axis=0)
        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
        return torch.tensor(res, dtype=torch.float32)

    def augm_data(self, source, instr):
        # source.shape = (2, 261120) - first channels, second length
        source_shape = source.shape
        applied_augs = []
        if 'all' in self.config['augmentations']:
            augs = self.config['augmentations']['all']
        else:
            augs = dict()

        # We need to add to all augmentations specific augs for stem. And rewrite values if needed
        if instr in self.config['augmentations']:
            for el in self.config['augmentations'][instr]:
                augs[el] = self.config['augmentations'][instr][el]

        # Channel shuffle
        if 'channel_shuffle' in augs:
            if augs['channel_shuffle'] > 0:
                if random.uniform(0, 1) < augs['channel_shuffle']:
                    source = source[::-1].copy()
                    applied_augs.append('channel_shuffle')
        # Random inverse
        if 'random_inverse' in augs:
            if augs['random_inverse'] > 0:
                if random.uniform(0, 1) < augs['random_inverse']:
                    source = source[:, ::-1].copy()
                    applied_augs.append('random_inverse')
        # Random polarity (multiply -1)
        if 'random_polarity' in augs:
            if augs['random_polarity'] > 0:
                if random.uniform(0, 1) < augs['random_polarity']:
                    source = -source.copy()
                    applied_augs.append('random_polarity')
        # Random pitch shift
        if 'pitch_shift' in augs:
            if augs['pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pitch_shift']:
                    apply_aug = AU.PitchShift(
                        min_semitones=augs['pitch_shift_min_semitones'],
                        max_semitones=augs['pitch_shift_max_semitones'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('pitch_shift')
        # Random seven band parametric eq
        if 'seven_band_parametric_eq' in augs:
            if augs['seven_band_parametric_eq'] > 0:
                if random.uniform(0, 1) < augs['seven_band_parametric_eq']:
                    apply_aug = AU.SevenBandParametricEQ(
                        min_gain_db=augs['seven_band_parametric_eq_min_gain_db'],
                        max_gain_db=augs['seven_band_parametric_eq_max_gain_db'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('seven_band_parametric_eq')
        # Random tanh distortion
        if 'tanh_distortion' in augs:
            if augs['tanh_distortion'] > 0:
                if random.uniform(0, 1) < augs['tanh_distortion']:
                    apply_aug = AU.TanhDistortion(
                        min_distortion=augs['tanh_distortion_min'],
                        max_distortion=augs['tanh_distortion_max'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('tanh_distortion')
        # Random MP3 Compression
        if 'mp3_compression' in augs:
            if augs['mp3_compression'] > 0:
                if random.uniform(0, 1) < augs['mp3_compression']:
                    apply_aug = AU.Mp3Compression(
                        min_bitrate=augs['mp3_compression_min_bitrate'],
                        max_bitrate=augs['mp3_compression_max_bitrate'],
                        backend=augs['mp3_compression_backend'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('mp3_compression')
        # Random AddGaussianNoise
        if 'gaussian_noise' in augs:
            if augs['gaussian_noise'] > 0:
                if random.uniform(0, 1) < augs['gaussian_noise']:
                    apply_aug = AU.AddGaussianNoise(
                        min_amplitude=augs['gaussian_noise_min_amplitude'],
                        max_amplitude=augs['gaussian_noise_max_amplitude'],
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('gaussian_noise')
        # Random TimeStretch
        if 'time_stretch' in augs:
            if augs['time_stretch'] > 0:
                if random.uniform(0, 1) < augs['time_stretch']:
                    apply_aug = AU.TimeStretch(
                        min_rate=augs['time_stretch_min_rate'],
                        max_rate=augs['time_stretch_max_rate'],
                        leave_length_unchanged=True,
                        p=1.0
                    )
                    source = apply_aug(samples=source, sample_rate=44100)
                    applied_augs.append('time_stretch')

        # Possible fix of shape
        if source_shape != source.shape:
            source = source[..., :source_shape[-1]]

        # Random Reverb
        if 'pedalboard_reverb' in augs:
            if augs['pedalboard_reverb'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_reverb']:
                    room_size = random.uniform(
                        augs['pedalboard_reverb_room_size_min'],
                        augs['pedalboard_reverb_room_size_max'],
                    )
                    damping = random.uniform(
                        augs['pedalboard_reverb_damping_min'],
                        augs['pedalboard_reverb_damping_max'],
                    )
                    wet_level = random.uniform(
                        augs['pedalboard_reverb_wet_level_min'],
                        augs['pedalboard_reverb_wet_level_max'],
                    )
                    dry_level = random.uniform(
                        augs['pedalboard_reverb_dry_level_min'],
                        augs['pedalboard_reverb_dry_level_max'],
                    )
                    width = random.uniform(
                        augs['pedalboard_reverb_width_min'],
                        augs['pedalboard_reverb_width_max'],
                    )
                    board = PB.Pedalboard([PB.Reverb(
                        room_size=room_size,  # 0.1 - 0.9
                        damping=damping,  # 0.1 - 0.9
                        wet_level=wet_level,  # 0.1 - 0.9
                        dry_level=dry_level,  # 0.1 - 0.9
                        width=width,  # 0.9 - 1.0
                        freeze_mode=0.0,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_reverb')

        # Random Chorus
        if 'pedalboard_chorus' in augs:
            if augs['pedalboard_chorus'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_chorus']:
                    rate_hz = random.uniform(
                        augs['pedalboard_chorus_rate_hz_min'],
                        augs['pedalboard_chorus_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_chorus_depth_min'],
                        augs['pedalboard_chorus_depth_max'],
                    )
                    centre_delay_ms = random.uniform(
                        augs['pedalboard_chorus_centre_delay_ms_min'],
                        augs['pedalboard_chorus_centre_delay_ms_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_chorus_feedback_min'],
                        augs['pedalboard_chorus_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_chorus_mix_min'],
                        augs['pedalboard_chorus_mix_max'],
                    )
                    board = PB.Pedalboard([PB.Chorus(
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_delay_ms=centre_delay_ms,
                        feedback=feedback,
                        mix=mix,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_chorus')

        # Random Phazer
        if 'pedalboard_phazer' in augs:
            if augs['pedalboard_phazer'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_phazer']:
                    rate_hz = random.uniform(
                        augs['pedalboard_phazer_rate_hz_min'],
                        augs['pedalboard_phazer_rate_hz_max'],
                    )
                    depth = random.uniform(
                        augs['pedalboard_phazer_depth_min'],
                        augs['pedalboard_phazer_depth_max'],
                    )
                    centre_frequency_hz = random.uniform(
                        augs['pedalboard_phazer_centre_frequency_hz_min'],
                        augs['pedalboard_phazer_centre_frequency_hz_max'],
                    )
                    feedback = random.uniform(
                        augs['pedalboard_phazer_feedback_min'],
                        augs['pedalboard_phazer_feedback_max'],
                    )
                    mix = random.uniform(
                        augs['pedalboard_phazer_mix_min'],
                        augs['pedalboard_phazer_mix_max'],
                    )
                    board = PB.Pedalboard([PB.Phaser(
                        rate_hz=rate_hz,
                        depth=depth,
                        centre_frequency_hz=centre_frequency_hz,
                        feedback=feedback,
                        mix=mix,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_phazer')

        # Random Distortion
        if 'pedalboard_distortion' in augs:
            if augs['pedalboard_distortion'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_distortion']:
                    drive_db = random.uniform(
                        augs['pedalboard_distortion_drive_db_min'],
                        augs['pedalboard_distortion_drive_db_max'],
                    )
                    board = PB.Pedalboard([PB.Distortion(
                        drive_db=drive_db,
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_distortion')

        # Random PitchShift
        if 'pedalboard_pitch_shift' in augs:
            if augs['pedalboard_pitch_shift'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_pitch_shift']:
                    semitones = random.uniform(
                        augs['pedalboard_pitch_shift_semitones_min'],
                        augs['pedalboard_pitch_shift_semitones_max'],
                    )
                    board = PB.Pedalboard([PB.PitchShift(
                        semitones=semitones
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_pitch_shift')

        # Random Resample
        if 'pedalboard_resample' in augs:
            if augs['pedalboard_resample'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_resample']:
                    target_sample_rate = random.uniform(
                        augs['pedalboard_resample_target_sample_rate_min'],
                        augs['pedalboard_resample_target_sample_rate_max'],
                    )
                    board = PB.Pedalboard([PB.Resample(
                        target_sample_rate=target_sample_rate
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_resample')

        # Random Bitcrash
        if 'pedalboard_bitcrash' in augs:
            if augs['pedalboard_bitcrash'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_bitcrash']:
                    bit_depth = random.uniform(
                        augs['pedalboard_bitcrash_bit_depth_min'],
                        augs['pedalboard_bitcrash_bit_depth_max'],
                    )
                    board = PB.Pedalboard([PB.Bitcrush(
                        bit_depth=bit_depth
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_bitcrash')

        # Random MP3Compressor
        if 'pedalboard_mp3_compressor' in augs:
            if augs['pedalboard_mp3_compressor'] > 0:
                if random.uniform(0, 1) < augs['pedalboard_mp3_compressor']:
                    vbr_quality = random.uniform(
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_min'],
                        augs['pedalboard_mp3_compressor_pedalboard_mp3_compressor_max'],
                    )
                    board = PB.Pedalboard([PB.MP3Compressor(
                        vbr_quality=vbr_quality
                    )])
                    source = board(source, 44100)
                    applied_augs.append('pedalboard_mp3_compressor')

        # print(applied_augs)
        return source

    def __getitem__(self, index):
        if self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()
        else:
            res = self.load_aligned_data()

        # Randomly change loudness of each stem
        if self.aug:
            if 'loudness' in self.config['augmentations']:
                if self.config['augmentations']['loudness']:
                    loud_values = np.random.uniform(
                        low=self.config['augmentations']['loudness_min'],
                        high=self.config['augmentations']['loudness_max'],
                        size=(len(res),)
                    )
                    loud_values = torch.tensor(loud_values, dtype=torch.float32)
                    res *= loud_values[:, None, None]

        mix = res.sum(0)

        if self.aug:
            if 'mp3_compression_on_mixture' in self.config['augmentations']:
                apply_aug = AU.Mp3Compression(
                    min_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_min'],
                    max_bitrate=self.config['augmentations']['mp3_compression_on_mixture_bitrate_max'],
                    backend=self.config['augmentations']['mp3_compression_on_mixture_backend'],
                    p=self.config['augmentations']['mp3_compression_on_mixture']
                )
                mix_conv = mix.cpu().numpy().astype(np.float32)
                required_shape = mix_conv.shape
                mix = apply_aug(samples=mix_conv, sample_rate=44100)
                # Sometimes it gives longer audio (so we cut)
                if mix.shape != required_shape:
                    mix = mix[..., :required_shape[-1]]
                mix = torch.tensor(mix, dtype=torch.float32)

        # If we need only given stem (for roformers)
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            return res[index], mix

        return res, mix
