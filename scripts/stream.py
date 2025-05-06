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
import asyncio

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_utils import load_start_checkpoint, demix, apply_tta
from utils.settings import get_model_from_config
RATE: int = 44100  # Sampling rate (44.1 kHz)
FORMAT: int = pyaudio.paFloat32  # Audio format

base_dir = str(os.path.dirname(os.path.abspath(__file__)))

def parse_args(dict_args: Union[Dict[str, Any], None]) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        dict_args (Union[Dict[str, Any], None]): Optional dictionary of arguments
                                                  to override the command-line input.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=int, required=True, choices=[1, 2], help="Choose script type: 1 or 2")
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--out_dir", type=str, default="stream_dir", help="Path to directory with results as wav file")
    parser.add_argument("--out_name", type=str, default="final", help="Path to directory with results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='List of GPU IDs')
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

def initialize_device(args: argparse.Namespace) -> str:
    """
    Initialize device based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.

    Returns:
        str: Device type ('cpu', 'cuda', or 'mps').
    """
    if args.force_cpu:
        return "cpu"
    elif torch.cuda.is_available():
        return f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(args: argparse.Namespace, device: str) -> Tuple[nn.Module, Any]:
    """
    Load the model and configuration from the given arguments.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (str): Device to load the model on ('cpu', 'cuda', 'mps').

    Returns:
        Tuple[nn.Module, Any]: The loaded model and its configuration.
    """
    torch.backends.cudnn.benchmark = True
    model, config = get_model_from_config(args.model_type, args.config_path)

    if args.start_check_point:
        load_start_checkpoint(args, model, type_='inference')

    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)
    return model, config

def initialize_audio_streams(chunk_size: int) -> Tuple[pyaudio.PyAudio, pyaudio.Stream, pyaudio.Stream]:
    """
    Initialize input and output audio streams.

    Args:
        chunk_size (int): Size of each audio chunk.

    Returns:
        Tuple[pyaudio.PyAudio, pyaudio.Stream, pyaudio.Stream]: PyAudio instance and the input/output streams.
    """
    p = pyaudio.PyAudio()
    stream_input = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=chunk_size
    )
    stream_output = p.open(
        format=FORMAT,
        channels=2,
        rate=RATE,
        output=True,
        frames_per_buffer=chunk_size
    )
    return p, stream_input, stream_output


def close_audio_streams(stream_input: pyaudio.Stream, stream_output: pyaudio.Stream, p: pyaudio.PyAudio) -> None:
    """
    Close input and output audio streams.

    Args:
        stream_input (pyaudio.Stream): Input audio stream.
        stream_output (pyaudio.Stream): Output audio stream.
        p (pyaudio.PyAudio): PyAudio instance.
    """
    stream_input.stop_stream()
    stream_input.close()
    stream_output.stop_stream()
    stream_output.close()
    p.terminate()


# Implementation for type 1
def type_1_main(args: argparse.Namespace, device: str, model: nn.Module, config: Any, chunk_size: int) -> None:
    """
    Main function for type 1 script, which records audio, processes it, and saves the output.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (str): Device to run the model on.
        model (nn.Module): The model to use for audio separation.
        config (Any): Configuration used for demixing.
        chunk_size (int): Size of each audio chunk.
    """

    def record_audio(stream_input: pyaudio.Stream) -> bytes:
        """
        Record audio from the input stream until the user presses 'S' to stop.

        Args:
            stream_input (pyaudio.Stream): The input audio stream.

        Returns:
            bytes: Recorded audio data.
        """
        stop_flag = threading.Event()

        def check_stop_key() -> None:
            while not stop_flag.is_set():
                if keyboard.is_pressed('s'):
                    print("Recording stopped...")
                    stop_flag.set()

        keyboard_thread = threading.Thread(target=check_stop_key, daemon=True)
        keyboard_thread.start()

        audio_queue = queue.Queue()
        print("Recording started... Press 'S' to stop.")

        while not stop_flag.is_set():
            try:
                audio_data = stream_input.read(44100, exception_on_overflow=False)
                audio_queue.put(audio_data)
            except IOError as e:
                print("Error reading audio data:", e)
                break

        audio_data = b"".join(list(audio_queue.queue))
        with audio_queue.mutex:
            audio_queue.queue.clear()
        return audio_data


    def process_audio(audio_data: bytes, model: nn.Module, args: argparse.Namespace, config: Any, device: str, stream_output: pyaudio.Stream) -> None:
        """
        Process the recorded audio using the model and save the output.

        Args:
            audio_data (bytes): The recorded audio data.
            model (nn.Module): The model to use for audio separation.
            args (argparse.Namespace): Parsed arguments.
            config (Any): Configuration for the model.
            device (str): Device to run the model on.
            stream_output (pyaudio.Stream): The output audio stream.
        """
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        audio_array = np.expand_dims(audio_array, axis=0)
        audio_array = np.concatenate([audio_array, audio_array], axis=0)

        output_path = os.path.abspath(os.path.join(args.out_dir, 'raw_audio.wav'))
        sf.write(output_path, audio_array.T, RATE, 'FLOAT')

        waveforms_orig = demix(config, model, audio_array, device, model_type=args.model_type)
        if args.use_tta:
            waveforms_orig = apply_tta(config, model, audio_array, waveforms_orig, device, args.model_type)
        waveforms_orig = waveforms_orig['vocals']

        output_path = os.path.abspath(os.path.join(args.out_dir, f'{args.out_name}.wav'))
        sf.write(output_path, waveforms_orig.T, RATE, 'FLOAT')
        print(f"Processing completed. Output saved to {output_path}.")
        stream_output.write(waveforms_orig.T.tobytes())


    os.makedirs(args.out_dir, exist_ok=True)
    p, stream_input, stream_output = initialize_audio_streams(chunk_size)
    audio_data = record_audio(stream_input)
    process_audio(audio_data, model, args, config, device, stream_output)
    close_audio_streams(stream_input, stream_output, p)
    print("End.")


# Implementation for type 2
async def type_2_main(args: argparse.Namespace, device: str, model: nn.Module, config: Any, chunk_size: int) -> None:
    """
    Main function for type 2 script, which continuously records audio, processes it,
    and streams the output in real-time.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (str): Device to run the model on.
        model (nn.Module): The model to use for audio separation.
        config (Any): Configuration used for demixing.
        chunk_size (int): Size of each audio chunk.
    """
    def fake(config: Any, model: nn.Module, device: str, args: argparse.Namespace) -> None:
        """
        Fake initialization function to simulate processing and model loading.

        Args:
            config (Any): Configuration used for demixing.
            model (nn.Module): The model to use for audio separation.
            device (str): Device to run the model on.
            args (argparse.Namespace): Parsed arguments.
        """
        print('Please wait...')
        audio_array = np.random.randn(chunk_size)
        audio_array = np.expand_dims(audio_array, axis=0)
        audio_array = np.concatenate([audio_array, audio_array], axis=0)
        demix(config, model, audio_array, device, model_type=args.model_type)
        print("Model initialized. Speak...")

    async def output_to_speakers(waveforms_orig: np.ndarray, stream_output: pyaudio.Stream) -> None:
        """
        Stream the output audio to the speakers.

        Args:
            waveforms_orig (Dict[str, np.ndarray]): Dictionary containing the separated audio sources.
            stream_output (pyaudio.Stream): The output audio stream.
        """
        stream_output.write(waveforms_orig.T.tobytes())

    async def record_and_process_audio(stream_input: pyaudio.Stream, stream_output: pyaudio.Stream, model: nn.Module,
                                        args: argparse.Namespace, config: Any, device: str) -> None:
        """
        Continuously record audio from the input stream, process it using the model,
        and stream the output to the speakers in real-time.

        Args:
            stream_input (pyaudio.Stream): The input audio stream.
            stream_output (pyaudio.Stream): The output audio stream.
            model (nn.Module): The model to use for audio separation.
            args (argparse.Namespace): Parsed arguments.
            config (Any): Configuration used for demixing.
            device (str): Device to run the model on.
        """
        while True:
            try:

                audio_data = stream_input.read(chunk_size, exception_on_overflow=False)
                start_time = time.time()
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_array = np.expand_dims(audio_array, axis=0)
                audio_array = np.concatenate([audio_array, audio_array], axis=0)

                waveforms_orig = demix(config, model, audio_array, device, model_type=args.model_type)
                if args.use_tta:
                    waveforms_orig = apply_tta(config, model, audio_array, waveforms_orig, device, args.model_type)

                waveforms_orig = waveforms_orig['vocals']
                if time.time() - start_time > 0.7:
                    print(f'Elapsed time: {time.time() - start_time:.2f}')
                await output_to_speakers(waveforms_orig, stream_output)
            except IOError as e:
                print("Error reading audio data:", e)
                break

    fake(config, model, device, args)
    p, stream_input, stream_output = initialize_audio_streams(chunk_size)
    await record_and_process_audio(stream_input, stream_output, model, args, config, device)
    close_audio_streams(stream_input, stream_output, p)


if __name__ == "__main__":
    args = parse_args(None)
    args.config_path = str(os.path.abspath(os.path.join(base_dir, '..', args.config_path)))
    args.start_check_point = str(os.path.abspath(os.path.join(base_dir, '..', args.start_check_point)))
    args.out_dir = str(os.path.abspath(os.path.join(base_dir, '..', args.out_dir)))

    device = initialize_device(args)
    model, config = load_model(args, device)

    if args.model_type == 'htdemucs':
        chunk_size = config.training.samplerate * config.training.segment
    else:
        chunk_size = config.audio.chunk_size



    if args.type == 1:
        type_1_main(args, device, model, config, chunk_size)
    elif args.type == 2:
        try:
            asyncio.run(type_2_main(args, device, model,config, chunk_size))
        finally:
            print('End!')
            exit(0)
