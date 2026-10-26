"""Replay retained Stim samples and both baselines, without trusting cached results.

Usage: python scripts/replay_datasets.py --root .
Reproducibility is conditional on the recorded simulator version and platform.
"""
import argparse,json
from pathlib import Path
import numpy as np
from qubitguard.simulation import Decoder,Setting,context_features
from qubitguard.data import digest,dataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('.')); args=ap.parse_args()
    records=[]
    for path in sorted((args.root/'data').glob('*.json')):
        m=json.loads(path.read_text()); setting=Setting(**m['setting']); dec=Decoder(setting)
        a,checked=dataset(args.root/'data',setting.distance,setting.noise,setting.profile,m['split'],m['shots'],setting.basis)
        det,obs=dec.sample(m['shots'],m['seed']); std=dec.predict(det); cor=dec.predict(det,True)
        comparisons=dict(detectors=np.array_equal(a['dets'],det),truth=np.array_equal(a['truth'],obs[:,0]),
                         standard=np.array_equal(a['standard'],std),correlated=np.array_equal(a['correlated'],cor),
                         context=np.array_equal(a['context'],context_features(det,std,cor,setting.noise)))
        if not all(comparisons.values()): raise ValueError(f'Replay differs: {path}: {comparisons}')
        records.append(dict(file=checked['file'],sha256=checked['sha256'],shots=m['shots'],checks=comparisons))
        print(f'Replayed {checked["file"]}',flush=True)
    print(json.dumps(dict(datasets=len(records),shots=sum(r['shots'] for r in records),all_exact=True)))

if __name__=='__main__':main()
