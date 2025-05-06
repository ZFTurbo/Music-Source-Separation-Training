from test import test_settings
from scripts.redact_config import redact_config
from utils.settings import load_config
from pathlib import Path
import os
import numpy as np
import soundfile as sf
from typing import List, Dict

MODEL_CONFIGS = {
    'config_apollo.yaml': {'model_type': 'apollo'},
    'config_dnr_bandit_bsrnn_multi_mus64.yaml': {'model_type': 'bandit'},
    'config_dnr_bandit_v2_mus64.yaml': {'model_type': 'bandit_v2'},
    'config_drumsep.yaml': {'model_type': 'htdemucs'},
    'config_htdemucs_6stems.yaml': {'model_type': 'htdemucs'},
    'config_musdb18_bs_roformer.yaml': {'model_type': 'bs_roformer'},
    'config_musdb18_demucs3_mmi.yaml': {'model_type': 'htdemucs'},
    'config_musdb18_htdemucs.yaml': {'model_type': 'htdemucs'},
    'config_musdb18_mdx23c.yaml': {'model_type': 'mdx23c'},
    'config_musdb18_mel_band_roformer.yaml': {'model_type': 'mel_band_roformer'},
    'config_musdb18_mel_band_roformer_all_stems.yaml': {'model_type': 'mel_band_roformer'},
    'config_musdb18_scnet.yaml': {'model_type': 'scnet'},
    'config_musdb18_scnet_large.yaml': {'model_type': 'scnet'},
    # 'config_musdb18_scnet_large_starrytong.yaml': {'model_type': 'scnet'},
    'config_vocals_bandit_bsrnn_multi_mus64.yaml': {'model_type': 'bandit'},
    'config_vocals_bs_roformer.yaml': {'model_type': 'bs_roformer'},
    'config_vocals_htdemucs.yaml': {'model_type': 'htdemucs'},
    'config_vocals_mdx23c.yaml': {'model_type': 'mdx23c'},
    'config_vocals_mel_band_roformer.yaml': {'model_type': 'mel_band_roformer'},
    'config_vocals_scnet.yaml': {'model_type': 'scnet'},
    'config_vocals_scnet_large.yaml': {'model_type': 'scnet'},
    'config_vocals_scnet_unofficial.yaml': {'model_type': 'scnet_unofficial'},
    'config_vocals_segm_models.yaml': {'model_type': 'segm_models'},


    # 'config_vocals_swin_upernet.yaml': {'model_type': 'swin_upernet'},
    # 'config_musdb18_torchseg.yaml': {'model_type': 'torchseg'},
    # 'config_musdb18_segm_models.yaml': {'model_type': 'segm_models'},
    # 'config_musdb18_bs_mamba2.yaml': {'model_type': 'bs_mamba2'},
    # 'config_vocals_bs_mamba2.yaml': {'model_type': 'bs_mamba2'},
    # 'config_vocals_torchseg.yaml': {'model_type': 'torchseg'}
}


# Folders for tests
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = ROOT_DIR / 'configs/'
TEST_DIR = ROOT_DIR / "tests_cache/"
TRAIN_DIR = TEST_DIR / "train_tracks/"
VALID_DIR = TEST_DIR / "valid_tracks/"


def create_dummy_tracks(directory: Path, num_tracks: int, instruments: List[str],
                        duration: float = 5.0, sample_rate: int = 44100) -> None:
    """
    Generates random audio tracks for stems in two subdirectories within the specified directory.

    Parameters:
    ----------
    directory : Path
        Path to the directory where the tracks will be saved.
    num_tracks : int
        Number of tracks to generate in each folder.
    instruments : List[str]
        List of instrument names (stems) to create.
    duration : float, optional
        Duration of each track in seconds. Default is 5.0.
    sample_rate : int, optional
        Sampling rate of the generated audio. Default is 44100 Hz.

    Returns:
    -------
    None
    """

    os.makedirs(directory, exist_ok=True)

    for folder_name in [str(i) for i in range(1, num_tracks+1)]:
        folder_path = directory / folder_name
        os.makedirs(folder_path, exist_ok=True)
        for instrument in instruments:
            # Generate random noice for each track
            samples = int(duration * sample_rate)
            track = np.random.uniform(-1.0, 1.0, (2, samples)).astype(np.float32)
            file_path = folder_path / f"{instrument}.wav"
            sf.write(file_path, track.T, sample_rate)


def cleanup_test_tracks() -> None:
    """
    Removes all cached test tracks.

    This function deletes the entire directory specified by the global `TEST_DIR` variable
    if it exists.

    Returns:
    -------
    None
        This function does not return a value. It performs cleanup of test data.
    """


def modify_configs() -> Dict[str, Path]:
    """
    Updates configuration files in the `configs` directory for use with test data.

    This function processes configuration files defined in the global `MODEL_CONFIGS` dictionary,
    modifies them to be compatible with test scenarios, and saves the updated configurations
    in a test-specific directory.

    Returns:
    -------
    Dict[str, Path]
        A dictionary where the keys are the original configuration file names, and the values
        are the paths to the updated configuration files.
    """
    config_dir = CONFIGS_DIR
    updated_configs = {}
    for config, args in MODEL_CONFIGS.items():
        model_type = args['model_type']
        config_path = config_dir / config
        updated_config_path = redact_config({
            'orig_config': str(config_path),
            'model_type': model_type,
            'new_config': str(TEST_DIR / 'configs' / config)
        })
        updated_configs[config] = updated_config_path
    return updated_configs


def run_tests() -> None:
    """
    Executes validation tests for all configurations.

    This function updates configurations, generates random dummy data for testing,
    and runs a series of tests (training, validation, and inference checks) for each
    model configuration specified in the global `MODEL_CONFIGS` dictionary.

    Returns:
    -------
    None
    """

    updated_configs = modify_configs()

    # For every config
    for config, args in MODEL_CONFIGS.items():
        model_type = args['model_type']
        cfg = load_config(model_type=model_type, config_path=TEST_DIR / 'configs' / config)
        # Random tracks
        create_dummy_tracks(TRAIN_DIR, instruments=cfg.training.instruments+['mixture'], num_tracks=2)
        create_dummy_tracks(VALID_DIR, instruments=cfg.training.instruments+['mixture'], num_tracks=2)

        print(f"\nRunning tests for model: {model_type} (config: {config})")

        test_args = {
            'check_train': False,
            'check_valid': True,
            'check_inference': True,
            'config_path': updated_configs[config],
            'data_path': str(TRAIN_DIR),
            'valid_path': str(VALID_DIR),
            'results_path': str(TEST_DIR / "results" / model_type),
            'store_dir': str(TEST_DIR / "inference_results" / model_type),
            'metrics': ['sdr', 'si_sdr', 'l1_freq']
        }

        test_args.update(args)

        test_settings(test_args, 'admin')
        print(f"Tests for model {model_type} completed successfully.")

    # Remove test_cache
    cleanup_test_tracks()


if __name__ == "__main__":
    run_tests()
