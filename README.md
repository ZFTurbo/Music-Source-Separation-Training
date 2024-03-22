# Music Source Separation Universal Training Code

Repository for training models for music source separation. Repository is based on [kuielab code](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src) for [SDX23 challenge](https://github.com/kuielab/sdx23/tree/mdx_AB/my_submission/src). The main idea of this repository is to create training code, which is easy to modify for experiments. Brought to you by [MVSep.com](https://mvsep.com).

## Models

Model can be chosen with `--model_type` arg.

Available models for training:
* MDX23C based on [KUIELab TFC TDF v3 architecture](https://github.com/kuielab/sdx23/). Key: `mdx23c`.
* Demucs4HT [[Paper](https://arxiv.org/abs/2211.08553)]. Key: `htdemucs`.
* VitLarge23 based on [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch). Key: `segm_models`. 
* Band Split RoFormer [[Paper](https://arxiv.org/abs/2309.02612), [Repository](https://github.com/lucidrains/BS-RoFormer)] . Key: `bs_roformer`. 
* Mel-Band RoFormer [[Paper](https://arxiv.org/abs/2310.01809), [Repository](https://github.com/lucidrains/BS-RoFormer)]. Key: `mel_band_roformer`.
* Swin Upernet [[Paper](https://arxiv.org/abs/2103.14030)] Key: `swin_upernet`.
* BandIt Plus [[Paper](https://arxiv.org/abs/2309.02539), [Repository](https://github.com/karnwatcharasupat/bandit)] Key: `bandit`.
* SCNet [[Paper](https://arxiv.org/abs/2401.13276), [Unofficial Repository](https://github.com/amanteur/SCNet-PyTorch)] Key: `scnet`.

 **Note 1**: For `segm_models` there are many different encoders is possible. [Look here](https://github.com/qubvel/segmentation_models.pytorch#encoders-).
 
 **Note 2**: Thanks to [@lucidrains](https://github.com/lucidrains) for recreating the RoFormer models based on papers.

## How to train

To train model you need to:

1) Choose model type with key `--model_type`. Possible values: `mdx23c`, `htdemucs`, `segm_models`, `mel_band_roformer`, `bs_roformer`.
2) Choose location of config for model `--config_path` `<config path>`. You can find examples of configs in [configs folder](configs/). Prefixes `config_musdb18_` are examples for [MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html).
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

All available training parameters you can find [here](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/train.py#L109).

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

All available inference parameters you can find [here](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/inference.py#L54).

## Useful notes

* All batch sizes in config are adjusted to use with single NVIDIA A6000 48GB. If you have less memory please adjust correspodningly in model config `training.batch_size` and `training.gradient_accumulation_steps`.
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

### Vocal models
| Model Type | Instruments | Metrics (SDR) | Config | Checkpoint |
|:-------------:|:-------------:|:-----:|:-----:|:-----:|
| MDX23C | vocals / other | SDR vocals: 10.17 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mdx23c.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mdx23c_sdr_10.17.ckpt) |
| HTDemucs4 (MVSep finetuned) | vocals / other | SDR vocals: 8.78 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_htdemucs.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_htdemucs_sdr_8.78.ckpt) |
| Segm Models (VitLarge23) | vocals / other | SDR vocals: 9.77 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt) |
| Mel Band RoFormer | vocals (*) / other | SDR vocals: 8.42 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_mel_band_roformer.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_mel_band_roformer_sdr_8.42.ckpt) |
| Swin Upernet | vocals / other | SDR vocals: 7.57 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/config_vocals_swin_upernet.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.2/model_swin_upernet_ep_56_sdr_10.6703.ckpt) |

**Note**: Metrics measured on [Multisong Dataset](https://mvsep.com/en/quality_checker).

### Single stem models
| Model Type | Instruments | Metrics (SDR) | Config | Checkpoint |
|:-------------:|:-------------:|:-----:|:-----:|:-----:|
| HTDemucs4 FT Drums | drums | SDR drums: 11.13 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_htdemucs.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th) |
| HTDemucs4 FT Bass | bass | SDR bass: 11.96 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_htdemucs.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th) |
| HTDemucs4 FT Other | other | SDR other: 5.85 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_htdemucs.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th) |
| HTDemucs4 FT Vocals (Official repository) | vocals | SDR vocals: 8.38 | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_htdemucs.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th) |

**Note**: All models output 4 stems, but quality is best only on target stem (all other stems are dummy).

### Multi-stem models

|     Model Type      | Instruments |                                                                 Metrics (SDR)                                                                  | Config | Checkpoint |
|:-------------------:|:-------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:-----:|:-----:|
|     MDX23C~~*~~     | bass / drums / vocals / other | MUSDB test avg: 7.15 (bass: 5.77, drums: 7.93 vocals: 9.23 other: 5.68) Multisong avg: 7.02 (bass: 8.40, drums: 7.73 vocals: 7.36 other: 4.57) | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt) |
|     BandIt Plus     | speech / music / effects |                                         DnR test avg: 11.50 (speech: 15.64, music: 9.18 effects: 9.69)                                         | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/config_dnr_bandit_bsrnn_multi_mus64.yaml) | [Weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.3/model_bandit_plus_dnr_sdr_11.47.chpt) |
|      HTDemucs4      | bass / drums / vocals / other |                                    Multisong avg: 9.16 (bass: 11.76, drums: 10.88 vocals: 8.24 other: 5.74)                                    | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_htdemucs.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th) |
| HTDemucs4 (6 stems) | bass / drums / vocals / other / piano / guitar |                              Multisong (bass: 11.22, drums: 10.22 vocals: 8.05 other: --- piano: --- guitar: ---)                              | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_htdemucs_6stems.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th) |
|     Demucs3 mmi     | bass / drums / vocals / other |                                    Multisong avg: 8.88 (bass: 11.17, drums: 10.70 vocals: 8.22 other: 5.42)                                    | [Config](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/configs/config_musdb18_demucs3_mmi.yaml) | [Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th) |

~~*~~ **Note**: Model was trained only on MUSDB18HQ dataset (100 songs train data)

## Dataset types

Look here: [Dataset types](docs/dataset_types.md)

## Augmentations

Look here: [Augmentations](docs/augmentations.md)

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