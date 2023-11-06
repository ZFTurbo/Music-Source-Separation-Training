# Music Source Separation Universal Training Code

Repository for training models for music source separation. Repository is based on [kuielab code](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src) for [SDX23 challenge](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src). The main idea of this repository is to create training code, which is easy to modify for experiments. Brought to you by [MVSep.com](https://mvsep.com).

## Models

Model can be chosen with `--model_type` arg.

Available models for training:
* MDX23C based on [KUIELab TFC TDF v3 architecture](https://github.com/kuielab/sdx23/). Key: `mdx23c`.
* Demucs4HT [[Paper](https://arxiv.org/abs/2211.08553)]. Key: `htdemucs`.
* VitLarge23 based on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch). Key: `segm_models`. 
* Band Split RoFormer [[Paper](https://arxiv.org/abs/2309.02612)]. Key: `bs_roformer`. 
* Mel-Band RoFormer [[Paper](https://arxiv.org/abs/2309.02612)]. Key: `mel_band_roformer`.
 
 **Note 1**: For `segm_models` there are many different encoders is possible. [Look here](https://github.com/qubvel/segmentation_models.pytorch#encoders-).
 
 **Note 2**: Thanks to [@lucidrains](https://github.com/lucidrains) for recreating the RoFormer models based on papers.

## How to train

To train model you need to:

1) Choose model type with key `--model_type`. Possible values: `mdx23c`, `htdemucs`, `segm_models`, `mel_band_roformer`, `bs_roformer`.
2) Choose location of config for model `--config_path` `<config path>`. You can find examples of configs in [configs folder](configs/). Suffix `_musdb18` are examples for [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html).
3) If you have some check-point from the same model or from the similar model you can use it with: `--start_check_point` `<weights path>`
4) Choose path where to store results of training `--results_path` `<results folder path>`

#### Example
```bash
python train.py \ 
    --model_type mel_band_roformer \ 
    --config_path configs/config_mel_band_roformer_vocals.yaml \
    --start_check_point results/model.ckpt \
    --results_path results/ \
    --data_path 'datasets/dataset1' 'datasets/dataset2' \
    --valid_path datasets/musdb18hq/test \
    --num_workers 4 \
    --device_ids 0
```

## How to inference

#### Example

```bash
python inference.py \  
    --model_type mdx23c \
    --config_path configs/config_mdx23c_musdb18.yaml \
    --start_check_point results/last_mdx23c.ckpt \
    --input_folder input/wavs/ \
    --store_dir separation_results/
```

## Useful notes

* All batch sizes in config are adjusted to use with single NVIDIA A6000 48GB. If you have less memory please adjust correspodningly.
* It's usually always better to start with old weights even if shapes not fully match. Code supports loading weights for not fully same models (but it must have the same architecture). Training will be much faster. 

## Code description

* `configs/config_*.yaml` - configuration files for models
* `models/*` - set of available models for training and inference 
* `dataset.py` - dataset which creates new samples for training
* `inference.py` - process folder with music files and separate them
* `train.py` - main training code
* `utils.py` - common functions used by train/valid 
* `valid.py` - validation of model with metrics


## Pre-trained models

If you trained some good models, please, share them. You can post config and model weights [in this issue](https://github.com/ZFTurbo/Music-Source-Separation-Training/issues/1).

| Model Type | Instruments | Metrics | Config | Checkpoint |
|:-------------:|:-------------:|:-----:|:-----:|:-----:|
| MDX23C | vocals / other | SDR vocals: 10.17 | [Link]() | [Link]() |
| HT Demucs | vocals / other | SDR vocals: 8.78 | [Link]() | [Link]() |
| Segm Models (VitLarge23) | vocals / other | SDR vocals: 9.77 | [Link]() | [Link]() |
| Mel Band RoFormer | **vocals** / other | SDR vocals: 8.42 | [Link]() | [Link]() |

## Dataset types

Look here: [Dataset types](docs/dataset_types.md)

## Citation

* [arxiv paper](https://arxiv.org/abs/2305.07489)

```
@misc{solovyev2023benchmarks,
      title={Benchmarks and leaderboards for sound demixing tasks}, 
      author={Roman Solovyev and Alexander Stempkovskiy and Tatiana Habruseva},
      year={2023},
      eprint={2305.07489},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```