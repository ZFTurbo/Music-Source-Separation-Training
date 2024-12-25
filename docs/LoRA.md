## Training with LoRA

### What is LoRA?

LoRA (Low-Rank Adaptation) is a technique designed to reduce the computational and memory cost of fine-tuning large-scale neural networks. Instead of fine-tuning all the model parameters, LoRA introduces small trainable low-rank matrices that are injected into the network. This allows significant reductions in the number of trainable parameters, making it more efficient to adapt pre-trained models to new tasks. For more details, you can refer to the original paper.

### Enabling LoRA in Training

To include LoRA in your training pipeline, you need to:

Add the `--train_lora` flag to the training command.

Add the following configuration for LoRA in your config file:

Example:
```
lora:
  r: 8
  lora_alpha: 16 # alpha / rank > 1
  lora_dropout: 0.05
  merge_weights: False
  fan_in_fan_out: False
  enable_lora: [True]
```

Configuration Parameters Explained:

*  `r` (Rank): This determines the rank of the low-rank adaptation matrices. A smaller rank reduces memory usage and file size but may limit the model's adaptability to new tasks. Common values are 4, 8, or 16.

*  `lora_alpha`: Scaling factor for the LoRA weights. The ratio lora_alpha / r should generally be greater than 1 to ensure sufficient expressive power. For example, with r=8 and lora_alpha=16, the scaling factor is 2.
  
*  `lora_dropout`: Dropout rate applied to LoRA layers. It helps regularize the model and prevent overfitting, especially for smaller datasets. Typical values are in the range [0.0, 0.1].
  
*  `merge_weights`: Whether to merge the LoRA weights into the original model weights during inference. Set this to True only if you want to save the final model with merged weights for deployment.
  
*  `fan_in_fan_out`: Defines the weight initialization convention. Leave this as False for most scenarios unless your model uses a specific convention requiring it.
  
*  `enable_lora`: A list of booleans specifying whether LoRA should be applied to certain layers.
   * For example, `[True, False, True]` enables LoRA for the 1st and 3rd layers but not the 2nd.
   * The number of output neurons in the layer must be divisible by the length of enable_lora to ensure proper distribution of LoRA parameters across layers.
   * For transformer architectures such as GPT models, `enable_lora = [True, False, True]` is typically used to apply LoRA to the Query (Q) and Value (V) projection matrices while skipping the Key (K) projection matrix. This setup allows efficient fine-tuning of the attention mechanism while maintaining computational efficiency.

### Benefits of Using LoRA

* File Size Reduction: With LoRA, only the LoRA layer weights are saved, which significantly reduces the size of the saved model.

* Flexible Fine-Tuning: You can fine-tune the LoRA layers while keeping the base model frozen, preserving the original model's generalization capabilities.

* Using Pretrained Weights with LoRA

### To train a model using both pretrained weights and LoRA weights, you need to:

1. Include the `--lora_checkpoint` parameter in the training command.

2. Specify the path to the LoRA checkpoint file.

### Validating and Inferencing with LoRA

When using a model fine-tuned with LoRA for validation or inference, you must provide the LoRA checkpoint using the `--lora_checkpoint` parameter.

### Example Commands

* Training with LoRA

```
python train.py --model_type scnet \
  --config_path configs/config_musdb18_scnet_large_starrytong.yaml \
  --start_check_point weights/last_scnet.ckpt \
  --results_path results/ \
  --data_path datasets/moisesdb/train_tracks \
  --valid_path datasets/moisesdb/valid \
  --device_ids 0 \
  --metrics neg_log_wmse l1_freq sdr \
  --metric_for_scheduler neg_log_wmse \
  --train_lora
```

* Validating with LoRA
```
python valid.py --model_type scnet \
  --config_path configs/config_musdb18_scnet_large_starrytong.yaml \
  --start_check_point weights/last_scnet.ckpt \
  --store_dir results_store/ \
  --valid_path datasets/moisesdb/valid \
  --device_ids 0 \
  --metrics neg_log_wmse l1_freq si_sdr sdr aura_stft aura_mrstft bleedless fullness
```

* Inference with LoRA
```
python inference.py --lora_checkpoint weights/lora_last_scnet.ckpt \
  --model_type scnet \
  --config_path configs/config_musdb18_scnet_large_starrytong.yaml \
  --start_check_point weights/last_scnet.ckpt \
  --store_dir inference_results/ \
  --input_folder datasets/moisesdb/mixtures_for_inference \
  --device_ids 0
```

### Train example with BSRoformer and LoRA

You can use this [config](configs/config_musdb18_bs_roformer_with_lora.yaml) and this [weights](https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt) to finetune BSRoformer on your dataset.

```
python train.py --model_type bs_roformer \
  --config_path configs/config_musdb18_bs_roformer_with_lora.yaml \
  --start_check_point weights/model_bs_roformer_ep_17_sdr_9.6568.ckpt \
  --results_path results/ \
  --data_path musdb18hq/train \
  --valid_path musdb18hq/test \
  --device_ids 0 \
  --metrics sdr \
  --train_lora
```
