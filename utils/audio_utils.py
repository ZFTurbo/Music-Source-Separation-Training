import argparse
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from utils.dataset import MSSDataset
from torch.utils.data import DataLoader


def prepare_data(config: Dict, args: argparse.Namespace, batch_size: int) -> DataLoader:
    """
    Prepare the training dataset and data loader.

    Args:
        config: Configuration dictionary for the dataset.
        args: Parsed command-line arguments containing dataset paths and settings.
        batch_size: Batch size for training.

    Returns:
        DataLoader object for the training dataset.
    """

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'metadata_{args.dataset_type}.pkl'),
        dataset_type=args.dataset_type,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    return train_loader


def read_audio_transposed(path: str, instr: str = None, skip_err: bool = False) -> Tuple[np.ndarray, int]:
    """
    Reads an audio file, ensuring mono audio is converted to two-dimensional format,
    and transposes the data to have channels as the first dimension.
    Parameters
    ----------
    path : str
        Path to the audio file.
    skip_err: bool
        If true, not raise errors
    instr:
        name of instument
    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple containing:
        - Transposed audio data as a NumPy array with shape (channels, length).
          For mono audio, the shape will be (1, length).
        - Sampling rate (int), e.g., 44100.
    """

    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            print(f"No stem {instr}: skip!")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        if len(mix.shape) == 1:  # For mono audio
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize an audio signal by subtracting the mean and dividing by the standard deviation.

    Parameters:
    ----------
    audio : np.ndarray
        Input audio array with shape (channels, time) or (time,).

    Returns:
    -------
    tuple[np.ndarray, dict[str, float]]
        - Normalized audio array with the same shape as the input.
        - Dictionary containing the mean and standard deviation of the original audio.
    """

    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Denormalize an audio signal by reversing the normalization process (multiplying by the standard deviation
    and adding the mean).

    Parameters:
    ----------
    audio : np.ndarray
        Normalized audio array to be denormalized.
    norm_params : dict[str, float]
        Dictionary containing the 'mean' and 'std' values used for normalization.

    Returns:
    -------
    np.ndarray
        Denormalized audio array with the same shape as the input.
    """

    return audio * norm_params["std"] + norm_params["mean"]


def draw_spectrogram(waveform, sample_rate, length, output_file):
    import librosa.display

    # Cut only required part of spectorgram
    x = waveform[:int(length * sample_rate), :]
    X = librosa.stft(x.mean(axis=-1))  # perform short-term fourier transform on mono signal
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)  # convert an amplitude spectrogram to dB-scaled spectrogram.
    fig, ax = plt.subplots()
    # plt.figure(figsize=(30, 10))  # initialize the fig size
    img = librosa.display.specshow(
        Xdb,
        cmap='plasma',
        sr=sample_rate,
        x_axis='time',
        y_axis='linear',
        ax=ax
    )
    ax.set(title='File: ' + os.path.basename(output_file))
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    if output_file is not None:
        plt.savefig(output_file)