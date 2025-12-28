# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from tqdm import tqdm
from typing import Union
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from glob import glob
import audiomentations as AU
import pedalboard as PB
import warnings
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings("ignore")
import argparse

def prepare_data(config: Union[ConfigDict, OmegaConf], args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Build the training DataLoader. If torch.distributed.is_initialized() is True,
    construct a DDP DataLoader with DistributedSampler; otherwise, construct a regular DataLoader.

    Args:
        config: Dataset configuration passed to MSSDataset.
        args: Must provide data_path, results_path, dataset_type, and DataLoader settings.
        batch_size: Per-process mini-batch size.

    Returns:
        Configured DataLoader for the training split.
    """
    # DDP
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if args.dataset_type != 5:
            ddp_batch = batch_size * world_size # maintain "num_steps" semantics across the whole world
        else:
            ddp_batch = batch_size

        trainset = MSSDataset(
            config,
            args.data_path,
            batch_size=ddp_batch,
            metadata_path=os.path.join(args.results_path, f"metadata_{args.dataset_type}.pkl"),
            dataset_type=args.dataset_type,
        )

        sampler = DistributedSampler(
            trainset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,             # per-process batch size
            sampler=sampler,                   # sampler handles shuffling in DDP
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        trainset = MSSDataset(
            config,
            args.data_path,
            batch_size=batch_size,
            metadata_path=os.path.join(args.results_path, f"metadata_{args.dataset_type}.pkl"),
            dataset_type=args.dataset_type,
        )

        train_loader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )

    return train_loader


def load_chunk(path, length, chunk_size, offset=None):
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        x = sf.read(path, dtype='float32')[0]
        if len(x.shape) == 1:
            # Mono case
            pad = np.zeros((chunk_size - length))
        else:
            pad = np.zeros([chunk_size - length, x.shape[-1]])
        x = np.concatenate([x, pad], axis=0)
    # Mono fix
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=1)
    return x.T


def get_track_set_length(params):
    path, instruments, file_types, dataset_type = params
    should_print = (not dist.is_initialized() or dist.get_rank() == 0) and dataset_type != 7
    # Check lengths of all instruments (it can be different in some cases)
    lengths_arr = []
    for instr in instruments:
        length = -1
        for extension in file_types:
            path_to_audio_file = path + '/{}.{}'.format(instr, extension)
            if os.path.isfile(path_to_audio_file):
                length = sf.info(path_to_audio_file).frames
                break
        if length == -1 and should_print:
            print('Cant find file "{}" in folder {}'.format(instr, path))
            continue
        lengths_arr.append(length)
    lengths_arr = np.array(lengths_arr)
    if lengths_arr.min() != lengths_arr.max() and should_print:
        print(f'Warning: lengths of stems are different for path: {path}. ({lengths_arr.min()} != {lengths_arr.max()})')
    # We use minimum to allow overflow for soundfile read in non-equal length cases
    return path, lengths_arr.min()


# For multiprocessing
def get_track_length(params):
    path = params
    length = sf.info(path).frames
    return (path, length)


def process_chunk_worker(args):
    task, instruments, file_types, min_mean_abs, default_chunk_size = args
    track_path, track_length, offset, chunk_size = task

    try:
        for instrument in instruments:
            instrument_loud_enough = False
            for extension in file_types:
                path_to_audio_file = track_path + '/{}.{}'.format(instrument, extension)
                if os.path.isfile(path_to_audio_file):
                    try:
                        source = load_chunk(path_to_audio_file, length=track_length, offset=offset,
                                            chunk_size=chunk_size)
                        if np.abs(source).mean() >= min_mean_abs:
                            instrument_loud_enough = True
                            break
                    except Exception as e:
                        return (track_path, offset, False)

            if not instrument_loud_enough:
                return (track_path, offset, False)

        return (track_path, offset, True)

    except Exception:
        return (track_path, offset, False)


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1, batch_size=None, verbose=True):
        self.verbose = verbose
        self.config = config
        self.dataset_type = dataset_type  # 1, 2, 3, 4 or 5
        self.data_path = data_path
        self.instruments = instruments = config.training.instruments
        if batch_size is None:
            batch_size = config.training.batch_size
        self.batch_size = batch_size
        self.file_types = ['wav', 'flac']
        self.metadata_path = metadata_path

        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        # Augmentation block
        self.aug = False
        if 'augmentations' in config:
            if config['augmentations'].enable is True:
                if self.verbose and should_print:
                    print('Use augmentation for training')
                self.aug = True
        else:
            if self.verbose and should_print:
                print('There is no augmentations block in config. Augmentations disabled for training...')

        metadata = self.get_metadata()

        if self.dataset_type in [1, 4, 5, 6, 7]:
            if len(metadata) > 0:
                if self.verbose and should_print:
                    print('Found tracks in dataset: {}'.format(len(metadata)))
            else:
                if should_print:
                    print('No tracks found for training. Check paths you provided!')
                exit()
        else:
            for instr in self.instruments:
                if self.verbose and should_print:
                    print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        self.chunk_size = config.audio.chunk_size
        self.min_mean_abs = config.audio.min_mean_abs
        self.do_chunks = config.training.get('precompute_chunks', False) and float(self.min_mean_abs) > 0
        # For dataset_type 5 - precompute all chunks
        if self.dataset_type == 5 or (self.dataset_type == 4 or self.dataset_type == 6) and self.do_chunks:
             self._initialize_chunks_metadata()
        if self.dataset_type == 7:
            self._build_class_to_tracks()
    def __len__(self):
        if self.dataset_type == 5:
            return len(self.chunks_metadata)
        return self.config.training.num_steps * self.batch_size


    def __getitem__(self, index):
        if self.dataset_type == 7:
            res, mix, active_stem_ids = self.load_class_balanced_aligned()
        elif self.dataset_type == 5:
            track_path, offset = self.chunks_metadata[index]
            res = self._load_chunk_by_offset(track_path, offset)
        elif self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()
        else:  # type 4 or 6
            if self.do_chunks:
                track_path, offset = self.chunks_metadata[np.random.randint(len(self.chunks_metadata))]
                res = self._load_chunk_by_offset(track_path, offset)
            else:
                if self.dataset_type == 6:
                    res, mix = self.load_aligned_data()
                else:
                    res, _ = self.load_aligned_data()

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
        if self.dataset_type != 6 and self.dataset_type!=7:
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

        # If we need to optimize only given stem
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            return res[index:index+1], mix

        if self.dataset_type==7:
            return res, mix, active_stem_ids

        return res, mix

    def _build_class_to_tracks(self):
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        total_tracks = len(self.metadata)
        max_ratio = self.config.training.get('max_class_presence_ratio', 0.4)
        max_tracks = int(total_tracks * max_ratio)

        class_to_tracks = {instr: [] for instr in self.instruments}

        for track_path, _ in self.metadata:
            for instr in self.instruments:
                for ext in self.file_types:
                    path = f"{track_path}/{instr}.{ext}"
                    if os.path.isfile(path):
                        class_to_tracks[instr].append(track_path)
                        break

        filtered_class_to_tracks = {}

        for instr, tracks in class_to_tracks.items():
            count = len(tracks)
            ratio = count / total_tracks

            if count == 0:
                continue  # stem нигде не встречается

            if ratio > max_ratio:
                if should_print:
                    print(
                        f"[dataset_type=7] Skip frequent stem '{instr}': "
                        f"{count}/{total_tracks} ({ratio:.1%})"
                    )
                continue

            filtered_class_to_tracks[instr] = tracks

        if len(filtered_class_to_tracks) == 0:
            raise RuntimeError(
                "dataset_type 7: all classes were filtered out by frequency threshold"
            )

        self.class_to_tracks = filtered_class_to_tracks
        self.available_classes = list(filtered_class_to_tracks.keys())

        if should_print:
            print(
                f"[dataset_type=7] Using {len(self.available_classes)} balanced classes "
                f"out of {len(self.instruments)} instruments"
            )

    def load_class_balanced_aligned(self):
        """
        1) Randomly choose instrument (class)
        2) Randomly choose track containing this instrument
        3) Load aligned chunk from this track
        """
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        instr = random.choice(self.available_classes)
        track_path = random.choice(self.class_to_tracks[instr])

        # Find track length
        track_length = None
        for path, length in self.metadata:
            if path == track_path:
                track_length = length
                break

        if track_length is None:
            raise RuntimeError(f"Track length not found: {track_path}")

        if track_length >= self.chunk_size:
            offset = np.random.randint(track_length - self.chunk_size + 1)
        else:
            offset = None

        mix = None
        for extension in self.file_types:
            path_to_mix_file = f"{track_path}/mixture.{extension}"
            if os.path.isfile(path_to_mix_file):
                try:
                    mix = load_chunk(
                        path_to_mix_file,
                        track_length,
                        self.chunk_size,
                        offset=offset
                    )
                    break
                except Exception as e:
                    print(e)
        res = []
        active_stem_ids = []

        for idx, instr in enumerate(self.instruments):
            found = False
            for extension in self.file_types:
                path_to_audio_file = f"{track_path}/{instr}.{extension}"
                if os.path.isfile(path_to_audio_file):
                    try:
                        source = load_chunk(
                            path_to_audio_file,
                            track_length,
                            self.chunk_size,
                            offset=offset
                        )
                        active_stem_ids.append(idx)
                        found = True
                        break
                    except Exception as e:
                        print(e)

            if not found:
                source = np.zeros((2, self.chunk_size), dtype=np.float32)

            res.append(source)

        res = np.stack(res, axis=0)

        if mix is None:
            mix = np.sum(res, axis=0)

        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)

        return (
            torch.tensor(res, dtype=torch.float32),
            torch.tensor(mix, dtype=torch.float32),
            active_stem_ids
        )


    def _initialize_chunks_metadata(self):
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)
        chunks_cache_path = self.metadata_path.replace('.pkl', '_chunks.pkl')
        current_config = {
            'chunk_size': self.chunk_size,
            'min_mean_abs': self.min_mean_abs,
            'instruments': sorted(self.instruments)
        }
        if os.path.exists(chunks_cache_path):
            try:
                cached_chunks = pickle.load(open(chunks_cache_path, 'rb'))
                cached_config = cached_chunks.get('config', {})
                config_matches = (
                        cached_config.get('chunk_size') == current_config['chunk_size'] and
                        cached_config.get('min_mean_abs') == current_config['min_mean_abs'] and
                        cached_config.get('instruments') == current_config['instruments']
                )
                if config_matches:
                    self.chunks_metadata = cached_chunks['chunks_metadata']
                    if self.verbose and should_print:
                        print(f'Loaded {len(self.chunks_metadata)} cached chunks from {chunks_cache_path}')
                else:
                    if self.verbose and should_print:
                        print('Config changed, recomputing chunks...')
                        print(f'Cached config: {cached_config}')
                        print(f'Current config: {current_config}')
                    self.chunks_metadata = self._precompute_and_cache_chunks(
                        chunks_cache_path, current_config)
            except Exception as e:
                if self.verbose and should_print:
                    print(f'Chunks cache corrupted ({e}), recomputing...')
                self.chunks_metadata = self._precompute_and_cache_chunks(
                    chunks_cache_path, current_config)
        else:
            self.chunks_metadata = self._precompute_and_cache_chunks(
                chunks_cache_path, current_config)

        if self.verbose and should_print:
            print(f'Precomputed {len(self.chunks_metadata)} chunks')


    def _precompute_and_cache_chunks(self, cache_path, config):
        """Precompute all chunks and save to cache with config"""
        if self.dataset_type == 4 or self.dataset_type == 6:
            chunks_metadata = self._precompute_random_chunks()
        elif self.dataset_type == 5:
            chunks_metadata = self._precompute_chunks()
        else:
            raise 'Only dataset type 4, 5 can be precomputed'
        cache_data = {
            'chunks_metadata': chunks_metadata,
            'config': config
        }
        pickle.dump(cache_data, open(cache_path, 'wb'))

        return chunks_metadata


    def _precompute_chunks(self):
        """Precompute all chunks for dataset_type 5 with overlap 2 using multiprocessing"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        tasks = []
        for track_path, track_length in self.metadata:
            if track_length < self.chunk_size:
                tasks.append((track_path, track_length, 0, track_length))
            else:
                step = self.chunk_size // 2
                num_chunks = (track_length - self.chunk_size) // step + 1
                for i in range(num_chunks):
                    offset = i * step
                    tasks.append((track_path, track_length, offset, self.chunk_size))

        if should_print:
            print(f"Total tasks to process: {len(tasks)}")

        if multiprocessing.cpu_count() > 1:
            chunks_metadata = self._process_tasks_parallel(tasks, should_print)
        else:
            chunks_metadata = self._process_tasks_sequential(tasks, should_print)

        if self.verbose and should_print:
            print(
                f'Created {len(chunks_metadata)} good chunks from {len(self.metadata)} tracks')

        return chunks_metadata

    def _precompute_random_chunks(self):
        """Precompute exact number of good chunks"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)

        target_count = self.config.training.get('num_precompute_chunks', self.config.training.num_steps * self.batch_size * self.config.training.num_epochs)
        chunks_metadata = []

        if should_print:
            print(f"Generating exactly {target_count} good chunks...")

        with tqdm(total=target_count, desc='Progress good chunks') as pbar:
            while len(chunks_metadata) < target_count:
                batch_size = self.config.training.get('precompute_batch_for_chunks', 500)
                tasks = []
                need = target_count - len(chunks_metadata)
                for i in range(batch_size):
                    track_path, track_length = random.choice(self.metadata)
                    if track_length < self.chunk_size:
                        tasks.append((track_path, track_length, 0, track_length))
                    else:
                        offset = np.random.randint(track_length - self.chunk_size + 1)
                        tasks.append((track_path, track_length, offset, self.chunk_size))

                if multiprocessing.cpu_count() > 1:
                    good_chunks = self._process_tasks_parallel(tasks, False)
                else:
                    good_chunks = self._process_tasks_sequential(tasks, False)

                chunks_metadata.extend(good_chunks)
                pbar.update(min(len(good_chunks),need))

        chunks_metadata = chunks_metadata[:target_count]

        return chunks_metadata


    def _process_tasks_sequential(self, tasks, should_print):
        chunks_metadata = []

        pbar = tqdm(tasks, desc='Processing chunks') if should_print else tasks
        for task in pbar:
            track_path, track_length, offset, chunk_size = task
            if self._is_chunk_loud_enough(track_path, offset, chunk_size, track_length):
                chunks_metadata.append((track_path, offset))

        return chunks_metadata


    def _process_tasks_parallel(self, tasks, should_print):
        chunks_metadata = []

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

            worker_args = [(task, self.instruments, self.file_types, self.min_mean_abs, self.chunk_size) for task in
                           tasks]

            results = []
            if should_print:
                with tqdm(total=len(tasks), desc='Processing chunks') as pbar:
                    for i, result in enumerate(pool.imap_unordered(process_chunk_worker, worker_args)):
                        results.append(result)
                        pbar.update(1)
            else:
                for result in pool.imap_unordered(process_chunk_worker, worker_args):
                    results.append(result)

            for result in results:
                track_path, offset, is_loud_enough = result
                if is_loud_enough:
                    chunks_metadata.append((track_path, offset))

        return chunks_metadata


    def _is_chunk_loud_enough(self, track_path, offset, chunk_size, track_length):

        try:
            for instrument in self.instruments:
                instrument_loud_enough = False
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(instrument, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, length=track_length, offset=offset,
                                                chunk_size=chunk_size)
                            if np.abs(source).mean() >= self.min_mean_abs:
                                instrument_loud_enough = True
                                break
                        except Exception as e:
                            if not dist.is_initialized() or dist.get_rank() == 0:
                                print('Error loading: {} Path: {}'.format(e, path_to_audio_file))
                            return False

                if not instrument_loud_enough:
                    return False

            return True

        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print('Error checking chunk loudness: {} Path: {}'.format(e, track_path))
            return False


    def read_from_metadata_cache(self, track_paths, instr=None):
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)
        metadata = []
        if os.path.isfile(self.metadata_path):
            if self.verbose and should_print:
                print('Found metadata cache file: {}'.format(self.metadata_path))
            old_metadata = pickle.load(open(self.metadata_path, 'rb'))
        else:
            return track_paths, metadata

        if instr:
            old_metadata = old_metadata[instr]

        # We will not re-read tracks existed in old metadata file
        track_paths_set = set(track_paths)
        for old_path, file_size in old_metadata:
            if old_path in track_paths_set:
                metadata.append([old_path, file_size])
                track_paths_set.remove(old_path)
        track_paths = list(track_paths_set)
        if len(metadata) > 0 and should_print:
            print('Old metadata was used for {} tracks.'.format(len(metadata)))
        return track_paths, metadata


    def get_metadata(self):
        read_metadata_procs = multiprocessing.cpu_count() - 2
        should_print = not dist.is_initialized() or dist.get_rank() == 0
        if 'read_metadata_procs' in self.config['training']:
            read_metadata_procs = int(self.config['training']['read_metadata_procs'])

        if self.verbose and should_print:
            print(
                'Dataset type:', self.dataset_type,
                'Processes to use:', read_metadata_procs,
                '\nCollecting metadata for', str(self.data_path),
            )

        if self.dataset_type in [1, 4, 5, 6, 7]:  # Added type 7
            track_paths = []
            if type(self.data_path) == list:
                for tp in self.data_path:
                    tracks_for_folder = sorted(glob(tp + '/*'))
                    if len(tracks_for_folder) == 0 and should_print:
                        print('Warning: no tracks found in folder \'{}\'. Please check it!'.format(tp))
                    track_paths += tracks_for_folder
            else:
                track_paths += sorted(glob(self.data_path + '/*'))

            track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
            track_paths, metadata = self.read_from_metadata_cache(track_paths, None)

            if read_metadata_procs <= 1:
                pbar = tqdm(track_paths) if should_print else track_paths
                for path in pbar:
                    track_path, track_length = get_track_set_length((path, self.instruments, self.file_types, self.dataset_type))
                    metadata.append((track_path, track_length))
            else:
                with ThreadPoolExecutor(max_workers=read_metadata_procs) as executor:
                    futures = [
                        executor.submit(
                            get_track_set_length,
                            args
                        )
                        for args in zip(
                            track_paths,
                            itertools.repeat(self.instruments),
                            itertools.repeat(self.file_types),
                        )
                    ]

                    if should_print:
                        for f in tqdm(as_completed(futures), total=len(futures)):
                            track_path, track_length = f.result()
                            metadata.append((track_path, track_length))
                    else:
                        for f in as_completed(futures):
                            metadata.append(f.result())

        elif self.dataset_type == 2:
            metadata = dict()
            for instr in self.instruments:
                metadata[instr] = []
                track_paths = []
                if type(self.data_path) == list:
                    for tp in self.data_path:
                        track_paths += sorted(glob(tp + '/{}/*.wav'.format(instr)))
                        track_paths += sorted(glob(tp + '/{}/*.flac'.format(instr)))
                else:
                    track_paths += sorted(glob(self.data_path + '/{}/*.wav'.format(instr)))
                    track_paths += sorted(glob(self.data_path + '/{}/*.flac'.format(instr)))

                track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                if read_metadata_procs <= 1:
                    pbar = tqdm(track_paths) if should_print else track_paths
                    for path in pbar:
                        length = sf.info(path).frames
                        metadata[instr].append((path, length))
                else:
                    p = multiprocessing.Pool(processes=read_metadata_procs)
                    track_iter = p.imap(get_track_length, track_paths)
                    if should_print:
                        track_iter = tqdm(track_iter, total=len(track_paths))

                    for out in track_iter:
                        metadata[instr].append(out)
                    p.close()

        elif self.dataset_type == 3:
            import pandas as pd
            if type(self.data_path) != list:
                data_path = [self.data_path]

            metadata = dict()
            for i in range(len(self.data_path)):
                if self.verbose and should_print:
                    print('Reading tracks from: {}'.format(self.data_path[i]))
                df = pd.read_csv(self.data_path[i])

                skipped = 0
                for instr in self.instruments:
                    part = df[df['instrum'] == instr].copy()
                    if should_print:
                        print('Tracks found for {}: {}'.format(instr, len(part)))
                for instr in self.instruments:
                    part = df[df['instrum'] == instr].copy()
                    metadata[instr] = []
                    track_paths = list(part['path'].values)
                    track_paths, metadata[instr] = self.read_from_metadata_cache(track_paths, instr)

                    pbar = tqdm(track_paths) if should_print else track_paths
                    for path in pbar:
                        if not os.path.isfile(path):
                            if should_print:
                                print('Cant find track: {}'.format(path))
                            skipped += 1
                            continue
                        # print(path)
                        try:
                            length = sf.info(path).frames
                        except:
                            if should_print:
                                print('Problem with path: {}'.format(path))
                            skipped += 1
                            continue
                        metadata[instr].append((path, length))
                if skipped > 0 and should_print:
                    print('Missing tracks: {} from {}'.format(skipped, len(df)))
        else:
            if should_print:
                print('Unknown dataset type: {}. Must be 1, 2, 3, 4, 5 or 6'.format(self.dataset_type))
            exit()

        # Save metadata
        pickle.dump(metadata, open(self.metadata_path, 'wb'))
        return metadata


    def load_source(self, metadata, instr):
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)
        while True:
            if self.dataset_type in [1, 4, 5, 6, 7]:
                track_path, track_length = random.choice(metadata)
                for extension in self.file_types:
                    path_to_audio_file = track_path + '/{}.{}'.format(instr, extension)
                    if os.path.isfile(path_to_audio_file):
                        try:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                        except Exception as e:
                            # Sometimes error during FLAC reading, catch it and use zero stem
                            if should_print:
                                print('Error: {} Path: {}'.format(e, path_to_audio_file))
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break
            else:
                track_path, track_length = random.choice(metadata[instr])
                try:
                    source = load_chunk(track_path, track_length, self.chunk_size)
                except Exception as e:
                    # Sometimes error during FLAC reading, catch it and use zero stem
                    if should_print:
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


    def _load_chunk_by_offset(self, track_path, offset):
        """Load specific chunk by track path and offset"""
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)
        res = []

        for instr in self.instruments:
            for extension in self.file_types:
                path_to_audio_file = track_path + '/{}.{}'.format(instr, extension)
                if os.path.isfile(path_to_audio_file):
                    try:
                        # Get track length from metadata
                        track_length = None
                        for path, length in self.metadata:
                            if path == track_path:
                                track_length = length
                                break

                        if track_length is None:
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        else:
                            source = load_chunk(path_to_audio_file, track_length, self.chunk_size, offset=offset)
                    except Exception as e:
                        if should_print:
                            print('Error: {} Path: {}'.format(e, path_to_audio_file))
                        source = np.zeros((2, self.chunk_size), dtype=np.float32)
                    break
            else:
                source = np.zeros((2, self.chunk_size), dtype=np.float32)

            res.append(source)

        res = np.stack(res, axis=0)

        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)

        return torch.tensor(res, dtype=torch.float32)


    def load_aligned_data(self):
        track_path, track_length = random.choice(self.metadata)
        should_print = (not dist.is_initialized() or dist.get_rank() == 0)
        attempts = 10
        while attempts:
            if track_length >= self.chunk_size:
                common_offset = np.random.randint(track_length - self.chunk_size + 1)
            else:
                common_offset = None
            res = []
            silent_chunks = 0
            for i in self.instruments:
                found = False
                for extension in self.file_types:
                    path_to_audio_file = f"{track_path}/{i}.{extension}"
                    if os.path.isfile(path_to_audio_file):
                        found = True
                        try:
                            source = load_chunk(
                                path_to_audio_file,
                                track_length,
                                self.chunk_size,
                                offset=common_offset
                            )
                        except Exception as e:
                            if should_print:
                                print(f"Error: {e} Path: {path_to_audio_file}")
                            source = np.zeros((2, self.chunk_size), dtype=np.float32)
                        break

                if not found:
                    source = np.zeros((2, self.chunk_size), dtype=np.float32)

                res.append(source)
                if np.abs(source).mean() < self.min_mean_abs:  # remove quiet chunks
                    silent_chunks += 1

            mix = None
            for extension in self.file_types:
                path_to_mix_file = track_path + '/mixture.{}'.format(extension)
                if os.path.isfile(path_to_mix_file):
                    try:
                        mix = load_chunk(path_to_mix_file, track_length, self.chunk_size, offset=common_offset)
                    except Exception as e:
                        if should_print:
                            print('Error loading mix: {} Path: {}'.format(e, path_to_mix_file))
                    break

            if silent_chunks == 0:
                break

            attempts -= 1
            if attempts <= 0 and should_print:
                print('Attempts max!', track_path)
            if common_offset is None:
                break

        try:
            res = np.stack(res, axis=0)
        except Exception as e:
            print('Error during stacking stems: {} Track Length: {} Track path: {}'.format(str(e), track_length,
                                                                                           track_path))
            res = np.zeros((len(self.instruments), 2, self.chunk_size), dtype=np.float32)
        if mix is None:
            mix = res.sum(0)
        if self.aug:
            for i, instr in enumerate(self.instruments):
                res[i] = self.augm_data(res[i], instr)
        return torch.tensor(res, dtype=torch.float32), torch.tensor(mix, dtype=torch.float32)


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