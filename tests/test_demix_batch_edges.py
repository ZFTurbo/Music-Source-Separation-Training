import numpy as np
import pytest
import torch
from ml_collections import ConfigDict
from utils.model_utils import demix


class IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


@pytest.mark.parametrize('length,overlap,batch', [(70, 2, 2), (30, 4, 4), (190, 2, 3), (70, 2, 1)])
def test_demix_identity_preserves_edges_across_batches(length, overlap, batch):
    config = ConfigDict({'audio': {'chunk_size': 100},
        'training': {'instruments': ['vocals'], 'target_instrument': None, 'use_amp': False},
        'inference': {'num_overlap': overlap, 'batch_size': batch}})
    mix = np.ones((2, length), dtype=np.float32)
    result = demix(config, IdentityModel(), mix, torch.device('cpu'), 'fake')['vocals']
    np.testing.assert_allclose(result, mix, atol=1e-6)


class ChunkDependentModel(torch.nn.Module):
    def forward(self, x):
        return (x + x.mean(dim=-1, keepdim=True)).unsqueeze(1)


@pytest.mark.parametrize("length,overlap", [(30, 4), (70, 2), (100, 1), (190, 2), (250, 4)])
def test_demix_batch_size_does_not_change_overlap_weights(length, overlap):
    config = ConfigDict({'audio': {'chunk_size': 100},
        'training': {'instruments': ['vocals'], 'target_instrument': None, 'use_amp': False},
        'inference': {'num_overlap': overlap, 'batch_size': 1}})
    mix = np.tile(np.linspace(-1, 1, length, dtype=np.float32), (2, 1))
    expected = demix(config, ChunkDependentModel(), mix, torch.device('cpu'), 'fake')['vocals']
    config.inference.batch_size = 3
    actual = demix(config, ChunkDependentModel(), mix, torch.device('cpu'), 'fake')['vocals']
    np.testing.assert_allclose(actual, expected, atol=1e-6)
