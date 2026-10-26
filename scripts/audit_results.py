"""Read-only, cross-platform numerical reconciliation of the supplied study.

No fitting, simulation, latency refresh, artifact writes or fit-cache reuse.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from qubitguard.integrity import verify_release
from qubitguard.simulation import Setting
from qubitguard.statistics import wilson, paired


def audit(root):
    integrity = verify_release(root)
    if integrity['checksum_failures']:
        raise ValueError(json.dumps(integrity))
    rows = comparisons = 0
    for folder, filename, primary in [
        (root, 'benchmark.json', True),
        (root / 'experiments/fullmap-v2', 'followup.json', False),
    ]:
        result = json.loads((folder / 'artifacts' / filename).read_text())
        with np.load(folder / 'artifacts/heldout-predictions.npz', allow_pickle=False) as raw:
            for row in result['rows']:
                setting = Setting(row['distance'], row['noise'], row['profile'])
                prefix = f'test_{setting.key}_n{row["n"]}.npz' if primary else setting.key
                suffix = ':prediction' if primary else ''
                prediction = raw[prefix + ':' + row['decoder'] + suffix]
                truth = raw[prefix + ':truth']
                errors = int(np.sum(prediction != truth))
                low, high = wilson(errors, len(truth))
                if errors != row['errors'] or len(truth) != row['n']:
                    raise ValueError(f'Failure count mismatch: {row}')
                if not np.allclose([errors / len(truth), low, high],
                                   [row['ler'], row['ci_low'], row['ci_high']], rtol=0, atol=1e-12):
                    raise ValueError(f'Interval mismatch: {row}')
                rows += 1
            for comparison in result['comparisons']:
                setting = Setting(comparison['distance'], comparison['noise'], comparison['profile'])
                prefix = f'test_{setting.key}_n20000.npz' if primary else setting.key
                suffix = ':prediction' if primary else ''
                checked = paired(raw[prefix + ':' + comparison['decoder'] + suffix],
                                 raw[prefix + ':' + comparison['baseline'] + suffix], raw[prefix + ':truth'])
                for key in ('wins', 'losses', 'delta_ler', 'delta_ci_low', 'delta_ci_high', 'mcnemar_p'):
                    if not np.isclose(checked[key], comparison[key], rtol=1e-12, atol=1e-14):
                        raise ValueError(f'Paired statistic mismatch: {key}, {comparison}')
                expected = min(1., checked['mcnemar_p'] * len(result['comparisons']))
                if not np.isclose(expected, comparison['bonferroni_p'], rtol=1e-12, atol=1e-14):
                    raise ValueError('Multiplicity adjustment mismatch')
                comparisons += 1
    if (rows, comparisons) != (322, 406):
        raise ValueError(f'Incomplete supplied study: {rows} rows, {comparisons} comparisons')
    return dict(status='passed', files_checked=integrity['files_checked'],
                result_rows=rows, paired_comparisons=comparisons,
                scope='Read-only reconciliation, not a new inference sweep or training replication')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(audit(args.root.resolve()), indent=2))
