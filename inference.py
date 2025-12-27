# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


def run_folder(
    model: "torch.nn.Module",
    args: "argparse.Namespace",
    config: dict,
    device: "torch.device",
    verbose: bool = False
) -> None:
    """
    Process a folder of audio files for source separation.

    Parameters:
    ----------
    model : torch.nn.Module
        Pre-trained model for source separation.
    args : argparse.Namespace
        Arguments containing input folder, output folder, and processing options.
    config : dict
        Configuration object with audio and inference settings.
    device : torch.device
        Device for model inference (CPU or CUDA).
    verbose : bool, optional
        If True, prints detailed information during processing. Default is False.
    """

    start_time = time.time()
    model.eval()

    # Recursively collect all files from input directory
    mixture_paths = sorted(
        glob.glob(os.path.join(args.input_folder, "**/*.*"), recursive=True)
    )
    mixture_paths = [p for p in mixture_paths if os.path.isfile(p)]

    sample_rate: int = getattr(config.audio, "sample_rate", 44100)

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    instruments: list[str] = prefer_target_instrument(config)[:]
    os.makedirs(args.store_dir, exist_ok=True)

    # Wrap paths with progress bar if not in verbose mode
    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    # Determine whether to use detailed progress bar
    if args.disable_detailed_pbar:
        detailed_pbar = False
    else:
        detailed_pbar = True

    for path in mixture_paths:
        # Get relative path from input folder
        relative_path: str = os.path.relpath(path, args.input_folder)
        # Extract directory and file name
        dir_name: str = os.path.dirname(relative_path)
        file_name: str = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print(f"Cannot read track: {format(path)}")
            print(f"Error message: {str(e)}")
            continue

        # Convert mono audio to expected channel format if needed
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)
            if "num_channels" in config.audio:
                if config.audio["num_channels"] == 2:
                    print("Convert mono track to stereo...")
                    mix = np.concatenate([mix, mix], axis=0)

        mix_orig = mix.copy()

        # Normalize input audio if enabled
        if "normalize" in config.inference:
            if config.inference["normalize"] is True:
                mix, norm_params = normalize_audio(mix)

        # Perform source separation
        waveforms_orig = demix(
            config,
            model,
            mix,
            device,
            model_type=args.model_type,
            pbar=detailed_pbar
        )

        # Apply test-time augmentation if enabled
        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                args.model_type
            )

        # Extract instrumental track if requested
        if args.extract_instrumental:
            instr = "vocals" if "vocals" in instruments else instruments[0]
            waveforms_orig["instrumental"] = mix_orig - waveforms_orig[instr]
            if "instrumental" not in instruments:
                instruments.append("instrumental")

        for instr in instruments:
            estimates = waveforms_orig[instr]

            # Denormalize output audio if normalization was applied
            if "normalize" in config.inference:
                if config.inference["normalize"] is True:
                    estimates = denormalize_audio(estimates, norm_params)

            peak: float = float(np.abs(estimates).max())
            if peak <= 1.0:
                codec = "flac"
            else:
                codec = "wav"

            subtype = args.pcm_type

            # Generate output directory structure using relative paths
            dirnames, fname = format_filename(
                args.filename_template,
                instr=instr,
                start_time=int(start_time),
                file_name=file_name,
                dir_name=dir_name,
                model_type=args.model_type,
                model=os.path.splitext(
                    os.path.basename(args.start_check_point)
                )[0],
            )

            # Create output directory
            output_dir: str = os.path.join(args.store_dir, *dirnames)
            os.makedirs(output_dir, exist_ok=True)

            output_path: str = os.path.join(output_dir, f"{fname}.{codec}")
            sf.write(output_path, estimates.T, sr, subtype=subtype)

            # Draw and save spectrogram if enabled
            if args.draw_spectro > 0:
                output_img_path = os.path.join(output_dir, f"{fname}.jpg")
                draw_spectrogram(estimates.T, sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")

def format_filename(template, **kwargs):
    '''
    Formats a filename from a template. e.g "{file_name}/{instr}"
    Using slashes ('/') in template will result in directories being created
    Returns [dirnames, fname], i.e. an array of dir names and a single file name
    '''
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname

def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(args.model_type, args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=False)


def do(input_folder, store_dir, config_path, start_check_point):

    args = {
        'model_type': 'bs_roformer',
        'input_folder': input_folder,
        'store_dir': store_dir,
        'config_path': config_path,
        'start_check_point': start_check_point,
        'filename_template': '{dir_name}/{instr}',

    }
    proc_folder(args)


def main():

    input_folder = r"E:\datasets\uvr_1"
    store_dir = r"E:\datasets\uvr_50_stems"

#     # wind-chimes
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn230_bs_roformer_wind_chimes\config.yaml",
#     r"E:\trash\weights\mvsep-2021\mss_code\weights\nn230_bs_roformer_wind_chimes\model_bs_roformer_wind_chimes_sdr_8.9242.ckpt")
#
#     # dobro
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn229_bs_roformer_dobro\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn229_bs_roformer_dobro\model_bs_roformer_dobro_sdr_8.4506.ckpt")
#
    # ukulele
    do(input_folder,
           store_dir,
           r"E:\trash\weights\mvsep-2021\mss_code\weights\nn228_bs_roformer_ukulele\config.yaml",
        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn228_bs_roformer_ukulele\model_bs_roformer_ukulele_sdr_9.1350.ckpt")

    # bells
    do(input_folder,
           store_dir,
           r"weights/mvsep-2021/mss_code/weights/nn227_bs_roformer_bells/config.yaml",
        r"weights/mvsep-2021/mss_code/weights/nn227_bs_roformer_bells/model_bs_roformer_bells_sdr_5.9385.ckpt")
#
#     # congas
#     do(input_folder,
#            store_dir,
#            r"E:\trash\weights\mvsep-2021\mss_code\weights\nn226_bs_roformer_congas\config.yaml",
#         r"E:\trash\weights\mvsep-2021\mss_code\weights\nn226_bs_roformer_congas\model_bs_roformer_congas_sdr_12.6609.ckpt")
#
#     # bassoon
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn225_bs_roformer_bassoon\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn225_bs_roformer_bassoon\model_bs_roformer_bassoon_sdr_6.5313.ckpt")
#
    # tuba
    do(input_folder,
       store_dir,
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn224_bs_roformer_tuba\config.yaml",
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn224_bs_roformer_tuba\model_bs_roformer_tuba_sdr_11.2987.ckpt")
#
    # harpsichord
    do(input_folder,
       store_dir,
       r"weights\mvsep-2021\mss_code\weights\nn223_bs_roformer_harpsichord\config.yaml",
       r"weights\mvsep-2021\mss_code\weights\nn223_bs_roformer_harpsichord\model_bs_roformer_harpsichord_sdr_4.7593.ckpt")

#     # sitar
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn222_bs_roformer_sitar\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn222_bs_roformer_sitar\model_bs_roformer_sitar_sdr_5.9296.ckpt")
#
#     # triangle
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn221_bs_roformer_triangle\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn221_bs_roformer_triangle\model_bs_roformer_triangle_sdr_11.0587.ckpt")
#
#     # harmonica
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn220_bsroformer_harmonica\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn220_bsroformer_harmonica\harmonica_bs_roformer_sdr_11.39.ckpt"
#        )
#
    # timpani
    do('timpani',
       input_folder,
       store_dir,
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn219_bsroformer_timpani\config.yaml",
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn219_bsroformer_timpani\timpani_bs_roformer_sdr_7.05.ckpt",
       )
#
#
#
#     # glockenspiel
#     do(input_folder,
#        store_dir,
#     r"E:\trash\weights\mvsep-2021\mss_code\weights\nn218_bsroformer_glockenspiel\config.yaml",
#     r"E:\trash\weights\mvsep-2021\mss_code\weights\nn218_bsroformer_glockenspiel\glockenspiel_bs_roformer_sdr_8.43.ckpt"
#        )
#
#     # marimba
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn217_bsroformer_marimba\config.yaml",
# r"E:\trash\weights\mvsep-2021\mss_code\weights\nn217_bsroformer_marimba\marimba_bs_roformer_sdr_6.88.ckpt")
#
#     # banjo
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn216_bsroformer_banjo\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn216_bsroformer_banjo\banjo_bs_roformer_sdr_6.11.ckpt"
# )
#
#     # french_horn
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn215_bsroformer_french_horn\config.yaml",
# r"E:\trash\weights\mvsep-2021\mss_code\weights\nn215_bsroformer_french_horn\french_horn_bs_roformer_sdr_5.18.ckpt")
#
#     # electric guitar
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn214_bsroformer_electric_guitar\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn214_bsroformer_electric_guitar\electric_guitar_bs_roformer_sdr_8.63.ckpt")
#
#     # synth
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn213_bsroformer_synth\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn213_bsroformer_synth\fused_model_synth_sdr_3.77.ckpt")
#
#     # digital_piano
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn210_bsroformer_digital_piano\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn210_bsroformer_digital_piano\fused_model_digital_piano_2.61.ckpt")
#
#     # clarinet
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn209_bsroformer_clarinet\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn209_bsroformer_clarinet\fused_model_clarinet_5.96.ckpt")
#
#     # oboe
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn208_bsroformer_oboe\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn208_bsroformer_oboe\fused_model_oboe_7.37.ckpt")
#
#     # tambourine
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn207_bsroformer_tambourine\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn207_bsroformer_tambourine\fused_model_tambourine_4.33.ckpt")
#
#     # trombone
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn205_bsroformer_trombone\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn205_bsroformer_trombone\fused_model_trombone_7.13.ckpt")
#
    # mandolin
    do(input_folder,
       store_dir,
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn204_bsroformer_mandolin\config.yaml",
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn204_bsroformer_mandolin\fused_model_mandolin_4.05.ckpt")

#     # # organ
#     # do(input_folder,
#     #    store_dir,
#     #    r"E:\trash\weights\mvsep-2021\mss_code\weights\nn203_bsroformer_organ\config.yaml",
#     #    r"E:\trash\weights\mvsep-2021\mss_code\weights\nn203_bsroformer_organ\fused_model_organ_5.08.ckpt")
#
#     # double_bass
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn202_bsroformer_double_bass\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn202_bsroformer_double_bass\fused_model_double_bass_15.07.ckpt")
#
#     # harp
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn200_bsroformer_harp\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn200_bsroformer_harp\fused_model_harp_6.61.ckpt")
#
#     # trumpet
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn198_bsroformer_trumpet\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn198_bsroformer_trumpet\fused_model_9.77.ckpt")
#
#     # cello
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn196_bsroformer_cello\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn196_bsroformer_cello\fused_model_11.81.ckpt")
#
#     # viola
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn195_bsroformer_viola\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn195_bsroformer_viola\fused_model_5.46.ckpt")
#
#     # strings
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn194_bsroformer_strings\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn194_bsroformer_strings\fused_model_5.41.ckpt")
#
#     # flute
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn189_bsroformer_flute\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn189_bsroformer_flute\fused_model_9.46.ckpt")
#
    # acoustic_guitar
    do('acoustic-guitar',
        input_folder,
       store_dir,
       r"weights\mvsep-2021\mss_code\weights\nn188_bsroformer_acoustic_guitar\config.yaml",
       r"weights\mvsep-2021\mss_code\weights\nn188_bsroformer_acoustic_guitar\fused_model_6.54_11.51.ckpt")

#     # violin
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn186_bsroformer_violin\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn186_bsroformer_violin\fused_model_sdr_7.2997.ckpt")
#
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn199_bsroformer_lead_back_vocals_model_v1\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn199_bsroformer_lead_back_vocals_model_v1\fused_model_12.11_v2.ckpt")
#
#     do(input_folder,
#        store_dir,
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn175_bs_roformer_sw_6_stem\config.yaml",
#        r"E:\trash\weights\mvsep-2021\mss_code\weights\nn175_bs_roformer_sw_6_stem\bs_roformer_sw_6_stems_fixed.chpt")

    do(input_folder,
       store_dir,
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn201_bsroformer_saxophone\config.yaml",
       r"E:\trash\weights\mvsep-2021\mss_code\weights\nn201_bsroformer_saxophone\fused_model_saxophone_9.77.ckpt")

    # wind
    do(input_folder,
       store_dir,
       r"weights\mvsep-2021\mss_code\weights\nn191_bsroformer_wind\config.yaml",
       r"weights\mvsep-2021\mss_code\weights\nn191_bsroformer_wind\fused_model_9.82.ckpt")



    args = {
        'model_type': 'mel_band_roformer',
        'input_folder': input_folder,
        'store_dir': store_dir,
        'config_path': r"weights\mss_code/weights/nn156_drumsep_melband_roformer_6_stems/config.yaml",
        'start_check_point': r"weights\mss_code/weights/nn156_drumsep_melband_roformer_6_stems/model_mel_band_roformer_6_stems_ep_86_sdr_12.4821.ckpt",
        'filename_template': '{dir_name}/{instr}',

    }
    proc_folder(args)


if __name__ == "__main__":
    main()
