"""Independent scientific contracts, including live computation behind the API.

These tests establish recovery and provenance invariants, not decoder quality.
Empirical quality belongs to independently sampled held-out benchmark results.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from qubitguard.server import app
from qubitguard.statistics import paired
from qubitguard.training import load_model
from qubitguard.evaluation import prediction
from qubitguard.simulation import Decoder, Setting
from qubitguard.provenance import validate_config

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def test_sampled_mechanism_trace_and_api_corrections_reconstruct_real_syndrome():
    """Trace parity must explain the actual shot, independently of its certificate."""
    with TestClient(app) as client:
        response = client.post('/api/simulate', json={
            'distance': 3, 'noise': .02, 'seed': 13447,
            'profile': 'uniform', 'basis': 'x',
        })
    assert response.status_code == 200, response.text
    shot = response.json()
    observed = np.array([d['event'] for d in shot['detectors']], np.uint8)
    trace_parity = np.zeros(len(observed), np.uint8)
    trace_logical = 0
    assert shot['injected_errors'], 'Chosen high-noise test should exercise sampled mechanisms'
    for mechanism in shot['injected_errors']:
        for detector in mechanism['detectors']:
            trace_parity[detector] ^= 1
        trace_logical ^= mechanism['logical']
    assert np.array_equal(trace_parity, observed)
    assert trace_logical == shot['predictions']['actual']
    assert 'representative' in shot['trace_note']
    assert 'not a unique' in shot['trace_note']
    for name in ('standard', 'correlated', 'neural'):
        if shot['predictions'][name] is None:
            continue
        parity = np.zeros(len(observed), np.uint8)
        logical = 0
        for edge in shot['corrections'][name]:
            parity[edge['a']] ^= 1
            if edge['b'] is not None:
                parity[edge['b']] ^= 1
            logical ^= edge['logical']
        assert np.array_equal(parity, observed)
        assert logical == shot['predictions'][name]
        assert shot['success'][name] == (logical == trace_logical)


def test_api_seed_replay_preserves_scientific_output_and_rejects_invalid_controls():
    payload = {'distance': 3, 'noise': .007, 'seed': 1337, 'profile': 'biased', 'basis': 'x'}
    with TestClient(app) as client:
        first = client.post('/api/simulate', json=payload)
        second = client.post('/api/simulate', json=payload)
        invalid = client.post('/api/simulate', json={**payload, 'distance': 4})
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    for field in ('detectors', 'injected_errors', 'corrections', 'predictions', 'certificate', 'success', 'provenance'):
        assert a[field] == b[field], field
    assert invalid.status_code == 422


def test_untrained_z_memory_never_reports_a_neural_prediction():
    with TestClient(app) as client:
        response = client.post('/api/simulate', json={'distance': 3, 'noise': .007, 'seed': 42, 'basis': 'z'})
    assert response.status_code == 200
    shot = response.json()
    assert shot['model']['supported'] is False
    assert shot['predictions']['neural'] is None
    assert shot['predictions']['neural_probability'] is None
    assert shot['success']['neural'] is None
    assert shot['certificate']['neural'] is None
    assert shot['certificate']['standard'] and shot['certificate']['correlated']


@torch.inference_mode()
def test_trained_prediction_cannot_read_hidden_truth():
    checkpoint = ROOT / 'artifacts/d3_residual_s17.pt'
    if not checkpoint.exists():
        pytest.skip('Fresh checkout before training; benchmark release must ship this checkpoint')
    model, bundle = load_model(checkpoint)
    decoder = Decoder(Setting(3, .007))
    dets, obs = decoder.sample(64, 384759)
    standard = decoder.predict(dets)
    correlated = decoder.predict(dets, True)
    from qubitguard.simulation import context_features
    arrays = dict(dets=dets, truth=obs[:, 0], standard=standard, correlated=correlated,
                  context=context_features(dets, standard, correlated, .007))
    a, a_score = prediction(model, bundle, arrays, decoder)
    arrays['truth'] = 1 - arrays['truth']
    b, b_score = prediction(model, bundle, arrays, decoder)
    assert np.array_equal(a, b)
    assert np.array_equal(a_score, b_score)


def test_zero_paired_discordances_do_not_claim_equivalence():
    truth = np.zeros(1000, np.uint8)
    result = paired(truth, truth, truth)
    assert result['delta_ler'] == 0
    assert result['delta_ci_low'] < 0 < result['delta_ci_high']
    assert result['mcnemar_p'] == 1


@pytest.mark.parametrize('changed', [
    {'basis': 'z'}, {'rounds_multiplier': 2}, {'train_profiles': ['biased']},
    {'split_seeds': {'train': 1, 'validation': 2, 'test': 3}},
    {'seed_policy': 'unsupported'},
])
def test_unsupported_experiment_knobs_fail_explicitly(changed):
    with pytest.raises(ValueError):
        validate_config(changed)


def test_standard_baseline_is_identical_to_independent_uncorrelated_graph():
    import pymatching
    decoder = Decoder(Setting(5, .012))
    dets, _ = decoder.sample(256, 248994)
    reference = pymatching.Matching.from_detector_error_model(decoder.dem)
    independent = reference.decode_batch(dets)[:, 0]
    decoder.predict(dets, True)  # Exercise reweighting before switching back.
    assert np.array_equal(decoder.predict(dets, False), independent)


def test_checkpoint_partial_pair_is_preserved_and_rejected(tmp_path):
    from qubitguard.training import train
    checkpoint = tmp_path / 'artifacts/d3_residual_s17.pt'
    checkpoint.parent.mkdir()
    original = b'irreplaceable-partial-artifact'
    checkpoint.write_bytes(original)
    with pytest.raises(ValueError, match='Incomplete checkpoint pair'):
        train(tmp_path, {'basis': 'x'}, 3)
    assert checkpoint.read_bytes() == original
    assert not checkpoint.with_suffix('.json').exists()


def test_cached_checkpoint_rejects_changed_training_data(tmp_path):
    """Exercise reuse integrity without running another expensive training job."""
    from qubitguard.training import train
    from qubitguard.data import digest
    from qubitguard.provenance import experiment_signature
    checkpoint = tmp_path / 'artifacts/d3_residual_s17.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'cache-guard-fixture-not-a-model')
    data = tmp_path / 'data/recorded.npz'
    data.parent.mkdir()
    data.write_bytes(b'recorded-training-data')
    config = {'basis': 'x', 'seed_policy': 'sha256-split-v1'}
    summary = {
        'experiment_signature': experiment_signature(config, 3, 'residual', 17, False),
        'checkpoint_sha256': digest(checkpoint),
        'train_provenance': [{'file': data.name, 'sha256': digest(data)}],
        'validation_provenance': [],
    }
    checkpoint.with_suffix('.json').write_text(json.dumps(summary))
    assert train(tmp_path, config, 3) == summary
    data.write_bytes(b'mutated-training-data')
    with pytest.raises(ValueError, match='dataset missing or changed'):
        train(tmp_path, config, 3)


@pytest.mark.parametrize('field', ['seed', 'shots', 'split', 'setting', 'generator_sha256'])
def test_dataset_reuse_rejects_metadata_mutation(tmp_path, field):
    from qubitguard.data import dataset
    _, original = dataset(tmp_path, 3, .007, 'uniform', 'test', 24)
    path = tmp_path / original['file']
    metadata_path = path.with_suffix('.json')
    metadata = json.loads(metadata_path.read_text())
    replacement = {'seed': 1, 'shots': 25, 'split': 'train', 'setting': {}, 'generator_sha256': 'changed-source'}
    metadata[field] = replacement[field]
    metadata_path.write_text(json.dumps(metadata))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        dataset(tmp_path, 3, .007, 'uniform', 'test', 24)
    assert path.read_bytes() == before


def test_source_fingerprint_covers_training_data_simulation_and_architecture(tmp_path, monkeypatch):
    import qubitguard.provenance as provenance
    source_dir = Path(provenance.__file__).parent
    names = ('simulation.py', 'data.py', 'models.py', 'training.py', 'provenance.py')
    for name in names:
        (tmp_path / name).write_bytes((source_dir / name).read_bytes())
    monkeypatch.setattr(provenance, '__file__', str(tmp_path / 'provenance.py'))
    baseline = provenance.source_fingerprint()
    for name in names:
        path = tmp_path / name
        original = path.read_bytes()
        path.write_bytes(original + b'\n# simulated edit in isolated copy\n')
        assert provenance.source_fingerprint() != baseline, name
        path.write_bytes(original)
        assert provenance.source_fingerprint() == baseline


def test_evaluation_rejects_missing_expected_model_roster(tmp_path):
    from qubitguard.evaluation import evaluate
    config = {'distances': [3], 'training_seeds': [17], 'models': ['residual'],
              'ablation_distances': [], 'threads': 2}
    with pytest.raises(ValueError, match='Checkpoint roster differs'):
        evaluate(tmp_path, config)


def test_parallel_api_replay_does_not_corrupt_shared_decoder_workspace():
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from qubitguard.server import app
    payload={'distance':3,'noise':.012,'seed':77651,'profile':'uniform','basis':'x'}
    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            shots=list(pool.map(lambda _:client.post('/api/simulate',json=payload).json(),range(8)))
    for shot in shots:
        assert shot['detectors']==shots[0]['detectors']
        assert shot['predictions']==shots[0]['predictions']
        assert shot['corrections']==shots[0]['corrections']
        assert all(x for x in shot['certificate'].values() if x is not None)


def test_partial_dataset_is_preserved_and_exact_source_tampering_is_rejected(tmp_path):
    from qubitguard.data import dataset
    orphan=tmp_path/'train_d3_x_uniform_p0.007000_n5.npz'
    orphan.write_bytes(b'preserve me')
    with pytest.raises(ValueError,match='Incomplete dataset'):
        dataset(tmp_path,3,.007,'uniform','train',5)
    assert orphan.read_bytes()==b'preserve me'
    safe=tmp_path/'safe'
    _,m=dataset(safe,3,.007,'uniform','train',5)
    circuit=safe/m['file'].replace('.npz','.stim')
    circuit.write_text(circuit.read_text()+'\nX_ERROR(0.1) 1\n')
    with pytest.raises(ValueError,match='Retained circuit'):
        dataset(safe,3,.007,'uniform','train',5)


def test_experimental_fullmap_checkpoint_restores_saved_predictions():
    from pathlib import Path
    import json
    import torch
    from qubitguard.fullmap import load_fullmap
    from qubitguard.evaluation import prediction
    from qubitguard.simulation import Decoder,Setting
    root=Path(__file__).resolve().parents[1]
    folder=root/'experiments'/'fullmap-v2'
    checkpoint=folder/'artifacts'/'fullmap_d3_s17.pt'
    if not checkpoint.exists(): pytest.skip('No supplied follow-up checkpoint')
    model,bundle=load_fullmap(checkpoint)
    data=folder/'data'/'test_d3_x_uniform_p0.007000.npz'
    arrays=dict(np.load(data,allow_pickle=False))
    predictions,scores=prediction(model,bundle,{k:v[:200] for k,v in arrays.items()},Decoder(Setting(3,.007)))
    saved=np.load(folder/'artifacts'/'heldout-predictions.npz',allow_pickle=False)
    assert np.array_equal(predictions,saved['d3_x_uniform_p0.007000:fullmap_v2'][:200])
    assert model.parameters_count>4209
