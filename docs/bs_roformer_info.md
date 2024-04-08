### Batch sizes for BSRoformer 

You can use table below to choose BS Roformer `batch_size` parameter for training based on your GPUs. Batch size values provided for single GPU. If you have several GPUs you need to multiply value on number of GPUs. 

| chunk_size | dim | depth | batch_size (A6000 48GB) | batch_size (3090/4090 24GB) | batch_size (16GB) |
|:----------:|:---:|:-----:|:-----------------------:|:---------------------------:|:-----------------:|
|   131584   | 128 |   6   |           10            |              5              |         3         |
|   131584   | 256 |   6   |            8            |              4              |         2         |
|   131584   | 384 |   6   |            7            |              3              |         2         |
|   131584   | 512 |   6   |            6            |              3              |         2         |
|   131584   | 256 |   8   |            6            |              3              |         2         |
|   131584   | 256 |  12   |            4            |              2              |         1         |
|   263168   | 128 |   6   |            4            |              2              |         1         |
|   263168   | 256 |   6   |            3            |              1              |         1         |
|   352800   | 128 |   6   |            2            |              1              |         -         |
|   352800   | 256 |   6   |            2            |              1              |         -         |
|   352800   | 384 |  12   |            1            |              -              |         -         |
|   352800   | 512 |  12   |            -            |              -              |         -         |


Parameters obtained with initial config:

```
audio:
  chunk_size: 131584
  dim_f: 1024
  dim_t: 515
  hop_length: 512
  n_fft: 2048
  num_channels: 2
  sample_rate: 44100
  min_mean_abs: 0.000

model:
  dim: 384
  depth: 12
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  linear_transformer_depth: 0
  freqs_per_bands: !!python/tuple
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 2
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 4
    - 12
    - 12
    - 12
    - 12
    - 12
    - 12
    - 12
    - 12
    - 24
    - 24
    - 24
    - 24
    - 24
    - 24
    - 24
    - 24
    - 48
    - 48
    - 48
    - 48
    - 48
    - 48
    - 48
    - 48
    - 128
    - 129
  dim_head: 64
  heads: 8
  attn_dropout: 0.1
  ff_dropout: 0.1
  flash_attn: false
  dim_freqs_in: 1025
  stft_n_fft: 2048
  stft_hop_length: 512
  stft_win_length: 2048
  stft_normalized: false
  mask_estimator_depth: 2
  multi_stft_resolution_loss_weight: 1.0
  multi_stft_resolutions_window_sizes: !!python/tuple
  - 4096
  - 2048
  - 1024
  - 512
  - 256
  multi_stft_hop_size: 147
  multi_stft_normalized: False

training:
  batch_size: 1
  gradient_accumulation_steps: 1
  grad_clip: 0
  instruments:
  - vocals
  - other
  lr: 3.0e-05
  patience: 2
  reduce_factor: 0.95
  target_instrument: vocals
  num_epochs: 1000
  num_steps: 1000
  q: 0.95
  coarse_loss_clip: true
  ema_momentum: 0.999
  optimizer: adam
  other_fix: false # it's needed for checking on multisong dataset if other is actually instrumental
  use_amp: true # enable or disable usage of mixed precision (float16) - usually it must be true
```
