"""Source/configuration compatibility: stale experiments must never be reused silently."""
import hashlib,json
from pathlib import Path
import torch,stim,pymatching,numpy
ALGORITHM_VERSION='framenet-v1'

def source_fingerprint(names=('simulation.py','data.py','models.py','training.py','provenance.py')):
    base=Path(__file__).parent
    return hashlib.sha256(b''.join(name.encode()+b'\0'+(base/name).read_bytes() for name in names)).hexdigest()

def dataset_fingerprint():
    return source_fingerprint(('simulation.py','data.py'))

def experiment_signature(config,distance,variant,seed,single_noise):
    return hashlib.sha256(json.dumps(dict(config=config,distance=distance,variant=variant,seed=seed,single_noise=single_noise,
        algorithm=ALGORITHM_VERSION,source_sha256=source_fingerprint(),torch=torch.__version__,stim=stim.__version__,
        pymatching=pymatching.__version__,numpy=numpy.__version__),sort_keys=True).encode()).hexdigest()

def validate_config(c):
    if c.get('basis','x')!='x' or c.get('rounds_multiplier',1)!=1 or c.get('train_profiles',['uniform'])!=['uniform']:
        raise ValueError('v1 training requires X memory, rounds=distance, uniform training profile. These are fixed scope, not supported experiment knobs.')
    if c.get('seed_policy','sha256-split-v1')!='sha256-split-v1': raise ValueError('Unsupported seed policy')
    if 'split_seeds' in c: raise ValueError('Use seed_policy: sha256-split-v1; numeric split offsets are not supported')
