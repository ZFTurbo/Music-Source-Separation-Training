
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


def draw_2_mel_spectrogram(
    estimates_waveform: np.ndarray,
    track_waveform: np.ndarray,
    sample_rate: int,
    length: float,
    output_base: str
) -> None:
    """
    Generate and save separate images for spectrograms and waveforms
    for both estimated and original audio.

    Creates two separate images:
    - One with mel-spectrograms (estimated vs original)
    - One with waveforms (estimated vs original)

    Args:
        estimates_waveform (np.ndarray): Estimated audio waveform
        track_waveform (np.ndarray): Original audio waveform
        sample_rate (int): Sampling rate in Hz
        length (float): Duration in seconds to include
        output_base (str): Base path for output files (without extension)

    Returns:
        None
    """
    import librosa.display

    # Prepare both waveforms
    waveforms = [estimates_waveform, track_waveform]
    titles = ["Estimated", "Original"]

    # Store processed (mono, possibly decimated) waveforms
    processed_waveforms: list[tuple[np.ndarray, int]] = []

    for waveform in waveforms:
        # Convert to mono if multi-channel
        mono_signal = waveform.mean(axis=-1) if len(waveform.shape) > 1 else waveform

        # Apply decimation for long audio signals
        if len(mono_signal) > 60 * sample_rate:
            # Decimation: take every second sample
            mono_signal = mono_signal[::2]
            effective_sr = sample_rate // 2
        else:
            effective_sr = sample_rate

        processed_waveforms.append((mono_signal, effective_sr))

    # Create mel-spectrograms figure
    fig_spec, axes_spec = plt.subplots(2, 1, figsize=(16, 10))

    for i, ((mono_signal, effective_sr), title) in enumerate(
        zip(processed_waveforms, titles)
    ):
        # Compute mel-spectrogram with reduced number of mel bins
        S = librosa.feature.melspectrogram(
            y=mono_signal,
            sr=effective_sr,
            n_mels=128
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Plot mel-spectrogram
        img = librosa.display.specshow(
            S_db,
            cmap="plasma",
            sr=effective_sr,
            x_axis="time",
            y_axis="mel",
            ax=axes_spec[i]
        )
        axes_spec[i].set_title(
            f"Mel-spectrogram: {title}",
            fontsize=14,
            fontweight="bold"
        )
        axes_spec[i].set_xlabel("Time (seconds)", fontsize=12)
        axes_spec[i].set_ylabel("Frequency (Mel)", fontsize=12)

        # Colorbar intentionally disabled
        # fig_spec.colorbar(img, ax=axes_spec, format="%+2.f dB",
        #                   shrink=0.8, pad=0.02, location="right")

    # Set global title for spectrograms
    fig_spec.suptitle(
        f"Mel-spectrograms: {os.path.basename(output_base)}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.4, right=0.88)

    # Save spectrograms image with reduced DPI
    spec_output = f"{output_base}_spectrograms.jpg"
    plt.savefig(spec_output, dpi=150, bbox_inches="tight")
    plt.close(fig_spec)

    # Create waveforms figure
    fig_wave, axes_wave = plt.subplots(2, 1, figsize=(16, 8))

    for i, ((mono_signal, effective_sr), title) in enumerate(
        zip(processed_waveforms, titles)
    ):
        # Generate time axis
        time = np.linspace(
            0,
            len(mono_signal) / effective_sr,
            len(mono_signal)
        )

        # Plot simplified waveform for very long signals
        if len(mono_signal) > 100000:
            # Take every 10th sample for plotting
            plot_indices = np.arange(0, len(mono_signal), 10)
            axes_wave[i].plot(
                time[plot_indices],
                mono_signal[plot_indices],
                color="#00ff88",
                alpha=0.9,
                linewidth=0.5
            )
        else:
            axes_wave[i].plot(
                time,
                mono_signal,
                color="#00ff88",
                alpha=0.9,
                linewidth=0.8
            )

        axes_wave[i].fill_between(
            time,
            mono_signal,
            alpha=0.3,
            color="#00ff8833"
        )
        axes_wave[i].set_xlabel("Time (seconds)", fontsize=12)
        axes_wave[i].set_ylabel("Amplitude", fontsize=12)
        axes_wave[i].set_title(
            f"Waveform: {title}",
            fontsize=14,
            fontweight="bold"
        )
        axes_wave[i].grid(True, alpha=0.3, color="gray")
        axes_wave[i].set_xlim(0, time[-1])

    # Set global title for waveforms
    fig_wave.suptitle(
        f"Waveforms: {os.path.basename(output_base)}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.4)

    # Save waveforms image
    wave_output = f"{output_base}_waveforms.jpg"
    plt.savefig(wave_output, dpi=150, bbox_inches="tight")
    plt.close(fig_wave)

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

    plot_waveform_basic(waveform, sample_rate, output_file.replace('.jpg', '_waveform.jpg'))


def plot_waveform_basic(waveform, samplerate, output_path=None,  theme='dark'):
    data = waveform
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    try:

        themes = {
            'dark': {'bg': '#0f0f0f', 'wave': '#00ff88', 'fill': '#00ff8833'},
            'light': {'bg': 'white', 'wave': '#2563eb', 'fill': '#3b82f633'},
            'purple': {'bg': '#1a1a2e', 'wave': '#e94560', 'fill': '#e9456033'}
        }

        colors = themes.get(theme, themes['dark'])

        fig, ax = plt.subplots(figsize=(12, 3), facecolor=colors['bg'])

        time = np.linspace(0, len(data) / samplerate, len(data))

        ax.plot(time, data, color=colors['wave'], alpha=0.9, linewidth=0.8)
        ax.fill_between(time, data, alpha=0.3, color=colors['fill'])

        ax.set_facecolor(colors['bg'])
        if theme == 'dark' or theme == 'purple':
            ax.tick_params(colors='white', labelsize=8)
            ax.set_xlabel('Time (seconds)', color='white', fontsize=10)
            ax.set_ylabel('Amplitude', color='white', fontsize=10)
        else:
            ax.tick_params(colors='black', labelsize=8)

        ax.grid(True, alpha=0.2, color='gray')
        ax.set_xlim(0, time[-1])

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight',
                        facecolor=colors['bg'], edgecolor='none')

    finally:
        plt.close()
