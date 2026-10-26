"""Fresh follow-up runs must create the provenance consumed by release audits."""
import hashlib
import json
from pathlib import Path
import shutil
import yaml
from qubitguard.fullmap import run_followup


def test_new_followup_retains_executed_source_and_prediction_hash(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'artifacts').mkdir()
    shutil.copy2(root / 'artifacts/d3_residual_s17.pt', tmp_path / 'artifacts/d3_residual_s17.pt')
    config = yaml.safe_load((root / 'configs/research-v1.yaml').read_text())
    config.update(epochs=1, train_noise=[.007], train_shots_per_setting=32,
                  validation_shots_per_setting=16, test_noise=[.007],
                  test_profiles=['uniform'], test_shots_per_setting=16, threads=1)
    result = run_followup(tmp_path, config)
    folder = tmp_path / 'experiments/fullmap-v2'
    snapshot = folder / result['source_snapshot']
    raw = folder / 'artifacts/heldout-predictions.npz'
    assert snapshot.read_bytes() == (root / 'src/qubitguard/fullmap.py').read_bytes()
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == result['executed_source_sha256']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == result['raw_predictions_sha256']
    assert json.loads((folder / 'artifacts/followup.json').read_text()) == result
