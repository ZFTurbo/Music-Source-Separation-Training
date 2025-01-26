import argparse
import os
import sys
import time
import numpy as np
import soundfile as sf
import torch
import pyaudio
from torch import nn
from typing import Tuple, Any, Union, Dict
import queue
import keyboard
import threading
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import get_model_from_config, load_start_checkpoint, demix, apply_tta



RATE: int = 44100  # Sampling rate (44.1 kHz)
CHUNK: int = 1024  # Buffer size
FORMAT: int = pyaudio.paFloat32  # Audio format

kornevaya_dir = os.path.dirname(os.path.abspath(__file__))


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.

    Args:
        dict_args: Dict of command-line arguments. If None, arguments will be parsed from sys.argv.

    Returns:
        Namespace object containing parsed arguments and their values.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--out_dir", type=str, default="", help="Path to directory with results as wav file")
    parser.add_argument("--out_name", type=str, default="final", help="Path to directory with results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU even if CUDA is available")
    parser.add_argument("--use_tta", action='store_true',
                        help="Flag adds test time augmentation during inference (polarity and channel inverse)."
                        "While this triples the runtime, it reduces noise and slightly improves prediction quality.")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def initialize_device(args: Any) -> str:
    """
    Initializes the device for model inference.

    Args:
        args: Command-line or script arguments containing device preferences.

    Returns:
        A string specifying the device to use (e.g., 'cpu', 'cuda', 'mps').
    """
    if args.force_cpu:
        return "cpu"
    elif torch.cuda.is_available():
        return f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(args: Any, device: str) -> Tuple[nn.Module, Any]:
    """
    Loads the model and its configuration.

    Args:
        args: Command-line or script arguments.
        device: The device to which the model will be loaded.

    Returns:
        A tuple containing the model and its configuration.
    """
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='inference')

    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    return model, config

def initialize_audio_streams() -> Tuple[pyaudio.PyAudio, pyaudio.Stream]:
    """
    Initializes the audio input stream.

    Returns:
        A tuple containing the PyAudio instance and the input stream.
    """
    p = pyaudio.PyAudio()
    stream_input = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    return p, stream_input

def close_audio_streams(stream_input: pyaudio.Stream, p: pyaudio.PyAudio) -> None:
    """
    Closes the audio input stream and terminates PyAudio.

    Args:
        stream_input: The audio input stream.
        p: The PyAudio instance.
    """
    stream_input.stop_stream()
    stream_input.close()
    p.terminate()

def record_audio(stream_input: pyaudio.Stream) -> bytes:
    """
    Records audio until the 'S' key is pressed.

    Args:
        stream_input: The audio input stream.

    Returns:
        The recorded audio data as bytes.
    """
    stop_flag = threading.Event()

    def check_stop_key() -> None:
        """Background thread to monitor the 'S' key press."""
        while not stop_flag.is_set():
            if keyboard.is_pressed('s'):
                print("Recording stopped...")
                stop_flag.set()

    # Start a background thread to monitor the stop key
    keyboard_thread = threading.Thread(target=check_stop_key, daemon=True)
    keyboard_thread.start()

    audio_queue = queue.Queue()
    print("Recording started... Press 'S' to stop.")

    while not stop_flag.is_set():
        try:
            audio_data = stream_input.read(256, exception_on_overflow=False)
            audio_queue.put(audio_data)
        except IOError as e:
            print("Error reading audio data:", e)
            break

    # Combine all audio data from the queue
    audio_data = b"".join(list(audio_queue.queue))
    with audio_queue.mutex:
        audio_queue.queue.clear()
    return audio_data

def process_audio_chunks(
    audio_data: bytes, model: nn.Module, args: Any, config: Any, device: str
) -> None:
    """
    Processes the recorded audio data and saves the result to a file.

    Args:
        audio_data: The recorded audio data as bytes.
        model: The loaded model for processing.
        args: Command-line or script arguments.
        config: The model configuration.
        device: The device used for model inference.
    """
    start_time = time.time()
    print("Processing audio...")

    # Convert bytes to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.float32)

    audio_array = np.expand_dims(audio_array, axis=0)  # Add a batch dimension
    audio_array = np.concatenate([audio_array, audio_array], axis=0)  # Make stereo

    # Save raw audio data
    output_path = os.path.abspath(os.path.join(kornevaya_dir, '..', args.out_dir, 'raw_audio.wav'))
    sf.write(output_path, audio_array.T, RATE, 'FLOAT')

    # Process audio using the model
    waveforms_orig = demix(config, model, audio_array, device, model_type=args.model_type)
    if args.use_tta:
        waveforms_orig = apply_tta(config, model, audio_array, waveforms_orig, device, args.model_type)
    waveforms_orig =  waveforms_orig['vocals']
    # Save processed audio data
    output_path = os.path.abspath(os.path.join(kornevaya_dir, '..', args.out_dir, f'{args.out_name}.wav'))
    sf.write(output_path, waveforms_orig.T, RATE, 'FLOAT')
    print(f"Processing completed. Output saved to {output_path}. Time taken: {time.time() - start_time:.2f} seconds")

def main() -> None:
    """
    Main function to record and process audio.
    """
    args = parse_args(None)
    args.config_path = os.path.abspath(os.path.join(kornevaya_dir, '..', args.config_path))
    args.start_check_point = os.path.abspath(os.path.join(kornevaya_dir, '..', args.start_check_point))

    device = initialize_device(args)
    print("Using device:", device)

    os.makedirs(args.out_dir, exist_ok=True)
    model, config = load_model(args, device)

    p, stream_input = initialize_audio_streams()

    # Record audio
    audio_data = record_audio(stream_input)
    close_audio_streams(stream_input, p)

    # Process recorded audio
    process_audio_chunks(audio_data, model, args, config, device)

    print("End.")

if __name__ == "__main__":
    main()
