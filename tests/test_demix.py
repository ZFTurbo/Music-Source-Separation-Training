import unittest
from unittest.mock import patch

import numpy as np
import torch
from ml_collections import ConfigDict

from utils import model_utils


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x[:, None]


class ChunkPositionModel(torch.nn.Module):
    def forward(self, x):
        # Predictions near chunk edges can differ from those at the center.
        position = torch.linspace(0, 1, x.shape[-1], device=x.device)
        return (x + position)[:, None]


class TestDemixWindows(unittest.TestCase):
    def config(self, batch_size, overlap=2):
        return ConfigDict({
            'audio': {'chunk_size': 100},
            'inference': {'num_overlap': overlap, 'batch_size': batch_size},
            'training': {'instruments': ['vocals'], 'use_amp': False},
        })

    def demix(self, mix, model, batch_size, overlap=2):
        return model_utils.demix(
            self.config(batch_size, overlap), model, mix, 'cpu', 'mdx23c'
        )['vocals']

    def test_short_clips_preserve_identity(self):
        # Include one sample, a clip shorter than one chunk, a full chunk,
        # and clips just below and above the reflect-padding threshold.
        for length, overlap, batch_size in (
            (1, 2, 8), (80, 2, 2), (100, 2, 2),
            (100, 4, 8), (150, 4, 8), (151, 4, 8),
        ):
            with self.subTest(length=length, overlap=overlap, batch=batch_size):
                mix = np.ones((2, length), dtype=np.float32)
                result = self.demix(mix, IdentityModel(), batch_size, overlap)
                np.testing.assert_allclose(result, mix, rtol=1e-6)

    def test_predictions_do_not_depend_on_batch_size(self):
        for length, overlap, batch_size, pad_batch in (
            (500, 2, 3, False), (513, 2, 8, False), (425, 4, 8, True),
        ):
            with self.subTest(length=length, overlap=overlap, batch=batch_size,
                              pad_batch=pad_batch):
                mix = np.ones((2, length), dtype=np.float32)
                model = ChunkPositionModel()
                # Exercise the batch padding used by compiled models,
                # without compiling or downloading a model.
                model.compiled_submodules = pad_batch
                expected = self.demix(mix, model, 1, overlap)
                result = self.demix(mix, model, batch_size, overlap)
                np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_single_chunk_disables_both_endpoint_fades(self):
        mix = np.ones((2, 100), dtype=np.float32)
        for batch_size in (1, 8):
            with self.subTest(batch=batch_size):
                result = self.demix(mix, IdentityModel(), batch_size, overlap=1)
                np.testing.assert_allclose(result, mix, rtol=1e-6)

    def test_base_window_is_not_modified(self):
        window = model_utils._getWindowingArray(100, 10)
        expected = window.clone()
        with patch.object(model_utils, '_getWindowingArray', return_value=window):
            self.demix(np.ones((2, 500), dtype=np.float32), ChunkPositionModel(), 3)
        torch.testing.assert_close(window, expected)

    def test_shift_and_tta_wrappers_preserve_short_clip(self):
        mix = np.linspace(0.1, 1, 200, dtype=np.float32).reshape(2, 100)
        config = self.config(batch_size=8, overlap=4)
        model = IdentityModel()
        result = model_utils.bigshifts_wrapper(
            config, model, mix, 'cpu', 'mdx23c', bigshifts=3
        )
        result = model_utils.apply_tta(
            config, model, mix, result, 'cpu', 'mdx23c', bigshifts=3
        )
        np.testing.assert_allclose(result['vocals'], mix, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
