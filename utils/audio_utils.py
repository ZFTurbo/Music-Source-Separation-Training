
import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

import torch.distributed as dist


def read_audio_transposed(path: str, instr: Optional[str] = None, skip_err: bool = False) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Read an audio file and return transposed waveform data with channels first.

    Loads the audio file from `path`, converts mono signals to 2D format, and
    transposes the array so that its shape is (channels, length). In case of
    errors, either raises an exception or skips gracefully depending on
    `skip_err`.

    Args:
        path (str): Path to the audio file to load.
        instr (Optional[str], optional): Instrument name, used for informative
            messages when `skip_err` is True. Defaults to None.
        skip_err (bool, optional): If True, skip files with read errors and
            return `(None, None)` instead of raising. Defaults to False.

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: A tuple containing:
            - NumPy array of shape (channels, length), or None if skipped.
            - Sampling rate as an integer, or None if skipped.
    """

    should_print = not dist.is_initialized() or dist.get_rank() == 0

    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            if should_print:
                print(f"No stem {instr}: skip!")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        if len(mix.shape) == 1:  # For mono audio
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize an audio signal using mean and standard deviation.

    Computes the mean and standard deviation from the mono mix of the input
    signal, then applies normalization to each channel.

    Args:
        audio (np.ndarray): Input audio array of shape (channels, time) or (time,).

    Returns:
        Tuple[np.ndarray, Dict[str, float]]: A tuple containing:
            - Normalized audio with the same shape as the input.
            - A dictionary with keys "mean" and "std" from the original audio.
    """

    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std()
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Reverse normalization on an audio signal.

    Applies the stored mean and standard deviation to restore the original
    scale of a previously normalized signal.

    Args:
        audio (np.ndarray): Normalized audio array to be denormalized.
        norm_params (Dict[str, float]): Dictionary containing the keys
            "mean" and "std" used during normalization.

    Returns:
        np.ndarray: Denormalized audio with the same shape as the input.
    """

    return audio * norm_params["std"] + norm_params["mean"]


def draw_spectrogram(waveform: np.ndarray, sample_rate: int, length: float, output_file: str) -> None:
    """
    Generate and save a spectrogram image from an audio waveform.

    Converts the provided waveform into a mono signal, computes its Short-Time
    Fourier Transform (STFT), converts the amplitude spectrogram to dB scale,
    and plots it using a plasma colormap.

    Args:
        waveform (np.ndarray): Input audio waveform array of shape (time, channels)
            or (time,).
        sample_rate (int): Sampling rate of the waveform in Hz.
        length (float): Duration (in seconds) of the waveform to include in the
            spectrogram.
        output_file (str): Path to save the resulting spectrogram image.

    Returns:
        None
    """

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


def draw_mel_spectrogram(waveform: np.ndarray, sample_rate: int, length: float, output_file: str) -> None:
    """
    Generate and save a spectrogram image from an audio waveform.

    Converts the provided waveform into a mono signal, computes its Short-Time
    Fourier Transform (STFT), converts the amplitude spectrogram to dB scale,
    and plots it using a plasma colormap.

    Args:
        waveform (np.ndarray): Input audio waveform array of shape (time, channels)
            or (time,).
        sample_rate (int): Sampling rate of the waveform in Hz.
        length (float): Duration (in seconds) of the waveform to include in the
            spectrogram.
        output_file (str): Path to save the resulting spectrogram image.

    Returns:
        None
    """

    import librosa.display

    # Cut only required part of spectrogram
    x = waveform

    # Compute mel-spectrogram instead of STFT
    S = librosa.feature.melspectrogram(
        y=x.mean(axis=-1),  # mono signal
        sr=sample_rate
    )

    # Convert to dB scale
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    try:
        img = librosa.display.specshow(
            S_db,
            cmap='plasma',
            sr=sample_rate,
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        ax.set(title='Mel-spectrogram: ' + os.path.basename(output_file))
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        if output_file is not None:
            plt.savefig(output_file)
    finally:
        plt.close(fig)