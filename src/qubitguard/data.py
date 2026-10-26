"""Reusable datasets with disjoint deterministic split streams and provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import stim, pymatching
from .simulation import Decoder, Setting, context_features
from .provenance import dataset_fingerprint

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def dataset(root, distance, noise, profile, split, shots, basis='x'):
    s=Setting(distance,float(noise),profile,basis)
    path=Path(root)/f'{split}_{s.key}_n{shots}.npz'
    meta_path=path.with_suffix('.json')
    stream = int.from_bytes(hashlib.sha256(f'{s.key}:{split}'.encode()).digest()[:8], 'little')
    seed = (stream & ((1 << 62)-1)) | ({'train':0,'validation':1,'test':2}[split] << 62)
    if path.exists()!=meta_path.exists():
        raise ValueError('Incomplete dataset pair; preserve the existing artifact and use a fresh experiment root')
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta.get('generator_sha256')!=dataset_fingerprint(): raise ValueError('Dataset generator source differs; use a fresh experiment root')
        if meta.get('setting')!=s.__dict__ or meta.get('split')!=split or meta.get('seed')!=seed or meta.get('shots')!=shots:
            raise ValueError('Dataset metadata does not match requested setting/split/seed/shots')
        if meta['sha256'] != digest(path): raise ValueError(f'Dataset checksum mismatch: {path}')
        if meta['stim_version'] != stim.__version__ or meta['pymatching_version'] != pymatching.__version__: raise ValueError('Simulator or decoder version differs from dataset provenance')
        current=Decoder(s)
        if meta['circuit_sha256'] != current.circuit_hash: raise ValueError('Circuit source differs from cached dataset')
        if not path.with_suffix('.stim').exists() or not path.with_suffix('.dem').exists(): raise ValueError('Missing exact circuit or DEM source')
        if digest(path.with_suffix('.stim'))!=current.circuit_hash or path.with_suffix('.dem').read_text()!=str(current.dem):
            raise ValueError('Retained circuit or DEM bytes differ from the current source')
        return dict(np.load(path,allow_pickle=False)),meta
    # Independent stable 64-bit streams, with disjoint split namespaces.
    decoder=Decoder(s)
    dets,obs=decoder.sample(shots,seed)
    std=decoder.predict(dets); corr=decoder.predict(dets,True)
    arrays=dict(dets=dets.astype(np.uint8),truth=obs[:,0].astype(np.uint8),standard=std,correlated=corr,
                context=context_features(dets,std,corr,noise))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial.npz')
    np.savez_compressed(temporary,**arrays); temporary.replace(path)
    meta=dict(schema_version=1,generator_sha256=dataset_fingerprint(),setting=s.__dict__,split=split,seed=seed,shots=shots,
              stim_version=stim.__version__,pymatching_version=pymatching.__version__,
              circuit_sha256=decoder.circuit_hash,sha256=digest(path),file=path.name,
              target='single final logical-observable flip, not detector parity')
    path.with_suffix('.stim').write_text(str(decoder.circuit))
    path.with_suffix('.dem').write_text(str(decoder.dem))
    temporary_meta=meta_path.with_suffix('.partial.json')
    temporary_meta.write_text(json.dumps(meta,indent=2)+'\n'); temporary_meta.replace(meta_path)
    return arrays,meta

def merged(root,distance,noises,split,shots,profiles=('uniform',)):
    blocks=[]; metas=[]
    for profile in profiles:
        for p in noises:
            block,meta=dataset(root,distance,p,profile,split,shots)
            blocks.append(block); metas.append(meta)
    return {key:np.concatenate([b[key] for b in blocks]) for key in blocks[0]},metas
