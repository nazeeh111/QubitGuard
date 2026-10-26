"""Independent invariants for graph recovery, data separation and ablations.

Written from the requested scientific contracts rather than matching internal
implementation outputs. These do not establish empirical neural superiority.
"""
import json
import numpy as np
import pytest
import torch

from qubitguard.simulation import Decoder, Setting, context_features
from qubitguard.models import FrameNet
from qubitguard.data import dataset
from qubitguard.statistics import wilson, paired


@pytest.mark.parametrize('distance', [3, 5, 7])
@pytest.mark.parametrize('basis', ['x', 'z'])
def test_both_matching_recoveries_and_neural_frame_toggle_are_syndrome_consistent(distance, basis):
    decoder = Decoder(Setting(distance, .007, basis=basis))
    dets, _ = decoder.sample(20, 634241)
    loop = decoder.logical_loop()
    assert loop
    assert decoder.certificate(np.zeros(decoder.circuit.num_detectors, np.uint8), loop, 1)
    for correlated in (False, True):
        logical_predictions = decoder.predict(dets, correlated)
        for syndrome, logical in zip(dets, logical_predictions):
            recovery = decoder.correction(syndrome, correlated)
            assert decoder.certificate(syndrome, recovery, int(logical))
            # A logical-frame flip must not change a single detector parity.
            assert decoder.certificate(syndrome, recovery ^ loop, int(logical) ^ 1)
            assert not decoder.certificate(syndrome, recovery, int(logical) ^ 1)


def test_seeded_circuit_replay_and_detector_grid_are_lossless():
    decoder = Decoder(Setting(5, .007))
    dets, obs = decoder.sample(100, 7529462)
    replay, replay_obs = decoder.sample(100, 7529462)
    assert np.array_equal(dets, replay)
    assert np.array_equal(obs, replay_obs)
    grid = decoder.grid(dets)
    t, y, x = decoder.indices.T
    assert len(set(map(tuple, decoder.indices))) == decoder.circuit.num_detectors
    assert np.array_equal(grid[:, 0, t, y, x], dets)
    assert np.all(grid[:, 1, t, y, x] == 1)
    assert grid[:, 1].sum(axis=(1, 2, 3)).tolist() == [decoder.circuit.num_detectors] * len(dets)


def test_dataset_split_namespaces_are_disjoint_and_cache_tampering_rejected(tmp_path):
    seeds = []
    for split in ('train', 'validation', 'test'):
        arrays, metadata = dataset(tmp_path, 3, .007, 'uniform', split, 24)
        seeds.append(metadata['seed'])
        assert len(arrays['truth']) == 24
    assert len(set(seeds)) == 3
    assert len({seed >> 62 for seed in seeds}) == 3
    target = next(tmp_path.glob('test_*.npz'))
    target.write_bytes(target.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        dataset(tmp_path, 3, .007, 'uniform', 'test', 24)


@torch.inference_mode()
def test_temporal_ablation_cannot_observe_time_order():
    torch.manual_seed(7)
    model = FrameNet(3, 8, 'no_time').eval()
    grid = torch.rand(5, 2, 4, 4, 4)
    context = torch.rand(5, 5)
    result = model(grid, context)
    permuted = model(grid[:, :, [2, 0, 3, 1]], context)
    assert torch.allclose(result, permuted, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('variant,masked_columns', [('direct', [0, 1, 2]), ('no_correlation', [1, 2])])
@torch.inference_mode()
def test_ablated_matching_context_is_inaccessible(variant, masked_columns):
    torch.manual_seed(7)
    model = FrameNet(3, 8, variant).eval()
    grid = torch.rand(5, 2, 4, 4, 4)
    context = torch.rand(5, 5)
    changed = context.clone()
    changed[:, masked_columns] = torch.rand(5, len(masked_columns)) * 100
    assert torch.equal(model(grid, context), model(grid, changed))


def test_binomial_and_paired_statistics_handle_extremes_without_superiority_claim():
    low, high = wilson(0, 100)
    assert low == 0 and 0 < high < .1
    low, high = wilson(100, 100)
    assert .9 < low < 1 and high == 1
    truth = np.array([0, 0, 1, 1])
    result = paired(truth, 1-truth, truth)
    assert result['wins'] == 4 and result['losses'] == 0
    assert result['delta_ler'] == -1
    assert result['mcnemar_p'] == pytest.approx(.125)
    identical = paired(truth, truth, truth)
    assert identical['mcnemar_p'] == 1
