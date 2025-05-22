# SSLAM Model Details

## Overview

SSLAM (Self-Supervised Learning for Audio Models) refers to a set of models and techniques aimed at learning robust audio representations from unlabeled data. These representations can then be used for various downstream audio tasks, including audio classification, sound event detection, and potentially music source separation or feature extraction.

The core idea is often to leverage self-supervised pre-training tasks (like masked autoencoding, contrastive learning, etc.) on large audio datasets to build powerful encoders.

## Original Source

The specific SSLAM models being integrated into this project are based on the work found in the following GitHub repository:
*   **SSLAM Repository**: [https://github.com/ta012/SSLAM](https://github.com/ta012/SSLAM)

This repository includes implementations for pre-training (e.g., `Data2VecMultiModel`) and fine-tuning (e.g., `MaeImageClassificationModel` adapted for audio).

## Current Integration Status

The model code, primarily from the `SSLAM_Inference/models` directory of the original repository, has been copied into the `models/sslam/` directory within this project.

**Important: The SSLAM model is currently NON-FUNCTIONAL within this project.**

The integration process involves significant refactoring due to the following reasons:
*   **Framework Mismatch**: The original SSLAM models are built using the [Fairseq](https://github.com/facebookresearch/fairseq) framework. This project, however, is primarily based on plain PyTorch.
*   **Dependency Removal**: To align with the project's structure and avoid adding a large dependency like Fairseq, the SSLAM model code is being adapted to remove all Fairseq-specific components. This includes:
    *   Changing base model classes from Fairseq's `BaseFairseqModel` to `torch.nn.Module`.
    *   Replacing Fairseq's configuration dataclasses (`FairseqDataclass`) with standard Python dataclasses or project-specific configuration mechanisms.
    *   Removing Fairseq's model registration, task system, and specialized training loop utilities.

**Progress:**
*   The model files have been moved to `models/sslam/`.
*   The class definitions for `MaeImageClassificationModel` (for audio classification) and `Data2VecMultiModel` (the pre-trained backbone) have been structurally modified to inherit from `torch.nn.Module` and use standard dataclass configurations. Fairseq-specific imports and decorators have been removed.
*   A placeholder configuration file `configs/config_sslam.yaml` has been created.
*   The model type `sslam_mae_classification` is now recognized by the training and inference scripts (e.g., `train.py`, `inference.py`) via `utils/settings.py`.

**Current State:**
*   While the model files can be imported without error and the model type is selectable, the `__init__` and `forward` methods within the SSLAM model classes (`MaeImageClassificationModel`, `Data2VecMultiModel`) are currently **stubbed out** (i.e., they are mostly placeholders or commented-out code).
*   Attempting to instantiate or train/run inference with the SSLAM model will result in a `NotImplementedError`, indicating that the core logic is not yet functional.
*   Unit tests in `tests/test_sslam.py` verify the config loading and the `NotImplementedError` for instantiation.

**Next Steps for Full Integration:**
*   Complete the refactoring of the `__init__` methods to correctly build the model architecture using PyTorch components, based on the configurations.
*   Adapt the `forward` methods to perform the correct computations.
*   Implement a mechanism to load pre-trained weights from the original SSLAM checkpoints into the refactored PyTorch models. This might involve careful state dictionary key mapping.
*   Thoroughly test the functionality and performance of the ported models.

Until these steps are completed, SSLAM should be considered under active development and not ready for use.
