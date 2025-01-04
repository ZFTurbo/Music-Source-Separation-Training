`tests` Documentation
========================

Overview
--------

The `tests.py` script is designed to verify the functionality of a specific configuration, model weights, and dataset before proceeding with training, validation, or inference. Additionally, it allows the specification of other parameters, which can be passed either through the command line or via the `base_args` variable in the script.

Usage
-----

To use `tests.py`, provide the desired arguments via the command line using the `--` prefix. It is mandatory to specify the following arguments:

*   `--model_type`
    
*   `--config_path`
    
*   `--start_check_point`
    
*   `--data_path`
    
*   `--valid_path`
    

For example:

```
python tests.py --check_train \
--config_path config.yaml \
--model_type scnet \
--data_path /path/to/data \
--valid_path /path/to/valid 
```

Alternatively, you can define default arguments in the `base_args` variable directly in the script.

Arguments
---------

The script accepts the following arguments:

*   `--check_train`: Check training functionality.
    
*   `--check_valid`: Check validation functionality.
    
*   `--check_inference`: Check inference functionality.
    
*   `--device_ids`: Specify device IDs for training or inference.
    
*   `--model_type`: Specify the type of model to use.
    
*   `--start_check_point`: Path to the checkpoint to start from.
    
*   `--config_path`: Path to the configuration file.
    
*   `--data_path`: Path to the training data.
    
*   `--valid_path`: Path to the validation data.
    
*   `--results_path`: Path to save training results.
    
*   `--store_dir`: Path to store validation or inference results.
    
*   `--input_folder`: Path to the input folder for inference.
    
*   `--metrics`: List of metrics to evaluate, provided as space-separated values.
    
*   `--max_folders`: Maximum number of folders to process.
    
*   `--dataset_type`: Dataset type. Must be one of: 1, 2, 3, or 4. Default is 1.
    
*   `--num_workers`: Number of workers for the dataloader. Default is 0.
    
*   `--pin_memory`: Use pinned memory in the dataloader.
    
*   `--seed`: Random seed for reproducibility. Default is 0.
    
*   `--use_multistft_loss`: Use MultiSTFT Loss from the auraloss package.
    
*   `--use_mse_loss`: Use Mean Squared Error (MSE) loss.
    
*   `--use_l1_loss`: Use L1 loss.
    
*   `--wandb_key`: API Key for Weights and Biases (wandb). Default is an empty string.
    
*   `--pre_valid`: Run validation before training.
    
*   `--metric_for_scheduler`: Metric to be used for the learning rate scheduler. Choices are `sdr`, `l1_freq`, `si_sdr`, `neg_log_wmse`, `aura_stft`, `aura_mrstft`, `bleedless`, or `fullness`. Default is `sdr`.
    
*   `--train_lora`: Enable training with LoRA.
    
*   `--lora_checkpoint`: Path to the initial LoRA weights checkpoint. Default is an empty string.
    
*   `--extension`: File extension for validation. Default is `wav`.
    
*   `--use_tta`: Enable test-time augmentation during inference. This triples runtime but improves prediction quality.
    
*   `--extract_instrumental`: Invert vocals to obtain instrumental output if available.
    
*   `--disable_detailed_pbar`: Disable the detailed progress bar.
    
*   `--force_cpu`: Force the use of the CPU, even if CUDA is available.
    
*   `--flac_file`: Output FLAC files instead of WAV.
    
*   `--pcm_type`: PCM type for FLAC files. Choices are `PCM_16` or `PCM_24`. Default is `PCM_24`.
    
*   `--draw_spectro`: Generate spectrograms for the resulting stems. Specify the value in seconds of the track. Requires `--store_dir` to be set. Default is 0.
    

Example
-------

To check train, validate and inference with a configuration file with a specific dataset and checkpoint we can use:

```
python tests/test.py \
--check_train \
--check_valid \
--check_inference \
--model_type scnet \
--config_path configs/config_musdb18_scnet_large_starrytong.yaml \
--start_check_point weights/model_scnet_ep_30_neg_log_wmse_-11.8688.ckpt \
--data_path datasets/moisesdb/train_tracks \
--valid_path datasets/moisesdb/valid \
--use_tta \
--use_mse_loss
```

This command validates the setup by:

*   Specifying `scnet` as the model type.
    
*   Loading the configuration from `configs/config_musdb18_scnet_large_starrytong.yaml`.
    
*   Using the dataset located at `datasets/moisesdb/train_tracks` for training.
    
*   Using `datasets/moisesdb/valid` for validation.
    
*   Starting from the checkpoint at `weights/model_scnet_ep_30_neg_log_wmse_-11.8688.ckpt`.
    
*   Enabling test-time augmentation and using MSE loss.
    

Additional Script: `admin_test.py`
----------------------------------

The `admin_test.py` script provides a way to verify the functionality of all configurations and models without specifying model weights or datasets. By default, it performs validation and inference. The configurations and corresponding parameters can be modified using the `MODEL_CONFIGS` variable in the script.

This script is useful for bulk testing and ensuring that multiple configurations are correctly set up. It can help identify potential issues with configurations or models before proceeding to detailed testing with `tests.py` or full-scale training.
