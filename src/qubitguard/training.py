"""CPU training with validation-only threshold and checkpoint selection."""
from pathlib import Path
import copy, json, time, random
import numpy as np
import torch
from torch import nn
from .models import FrameNet
from .simulation import Decoder, Setting
from .data import merged, digest
from .provenance import experiment_signature,validate_config

@torch.inference_mode()
def scores(model,grid,context,batch_size=512):
    model.eval()
    return np.concatenate([torch.sigmoid(model(grid[i:i+batch_size],context[i:i+batch_size])).numpy() for i in range(0,len(grid),batch_size)])

def choose_threshold(prob,truth,standard,variant):
    # Includes identity residual (threshold 1.0), without examining test data.
    thresholds=np.r_[np.linspace(.05,.95,91),.975,.99,.995,.999,1.0]
    best=None
    for t in thresholds:
        pred=(prob>t).astype(np.uint8)
        if variant!='direct': pred ^= standard
        err=int(np.sum(pred!=truth))
        # Tie favors fewer interventions, then threshold closer to 0.5.
        candidate=(err,int(np.sum(prob>t)),abs(t-.5),float(t))
        if best is None or candidate<best: best=candidate
    return best[-1],best[0]

def train(root,config,distance,variant='residual',seed=17,single_noise=False):
    validate_config(config)
    signature=experiment_signature(config,distance,variant,seed,single_noise)
    root=Path(root); name=f'd{distance}_{variant}_s{seed}'+('_single' if single_noise else '')
    checkpoint=root/'artifacts'/f'{name}.pt'; summary=checkpoint.with_suffix('.json')
    if checkpoint.exists()!=summary.exists():
        raise ValueError(f'Incomplete checkpoint pair: {checkpoint}; preserve it and use a fresh experiment root')
    if checkpoint.exists() and summary.exists():
        cached=json.loads(summary.read_text())
        if cached.get('experiment_signature')!=signature or cached.get('checkpoint_sha256')!=digest(checkpoint):
            raise ValueError(f'Incompatible cached checkpoint: {checkpoint}. Use a fresh experiment root; preserve previous results.')
        for provenance in cached['train_provenance']+cached['validation_provenance']:
            ds=root/'data'/provenance['file']
            if not ds.exists() or digest(ds)!=provenance['sha256']:
                raise ValueError(f'Cached checkpoint dataset missing or changed: {ds}')
        return cached
    torch.set_num_threads(config.get('threads',4)); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.use_deterministic_algorithms(True)
    noises=config['train_noise']
    if single_noise:
        # Same total shot budget as mixed-noise model.
        noises=[.007]; shots=config['train_shots_per_setting']*len(config['train_noise'])
    else: shots=config['train_shots_per_setting']
    tr,trmeta=merged(root/'data',distance,noises,'train',shots)
    va,vameta=merged(root/'data',distance,config['train_noise'],'validation',config['validation_shots_per_setting'])
    decoder=Decoder(Setting(distance))
    g=torch.from_numpy(decoder.grid(tr['dets'])); c=torch.from_numpy(tr['context'])
    vg=torch.from_numpy(decoder.grid(va['dets'])); vc=torch.from_numpy(va['context'])
    target=tr['truth'] if variant=='direct' else tr['truth']^tr['standard']
    y=torch.from_numpy(target.astype(np.float32))
    model=FrameNet(distance,config.get('width',12),variant)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    pos=max(1.,min(8.,np.sqrt((len(target)-target.sum())/max(1,target.sum()))))
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos)))
    history=[]; best_key=(float('inf'),float('inf')); best=None; start=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train(); order=torch.randperm(len(g)); loss_sum=0.; epoch_start=time.perf_counter()
        for idx in order.split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(model(g[idx],c[idx]),y[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step(); loss_sum+=loss.item()*len(idx)
        prob=scores(model,vg,vc,config['batch_size'])
        threshold,errors=choose_threshold(prob,va['truth'],va['standard'],variant)
        validation_target=va['truth'] if variant=='direct' else va['truth']^va['standard']
        bce=float(-(validation_target*np.log(prob+1e-7)+(1-validation_target)*np.log(1-prob+1e-7)).mean())
        key=(errors,bce)
        if key<best_key:
            best_key=key; best=copy.deepcopy(model.state_dict()); best_threshold=threshold; best_epoch=epoch+1
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(g),validation_errors=errors,validation_ler=errors/len(vg),threshold=threshold,seconds=time.perf_counter()-epoch_start)
        history.append(row); print(json.dumps(dict(run=name,**row)),flush=True)
    elapsed=time.perf_counter()-start
    model.load_state_dict(best)
    bundle=dict(schema_version=1,state_dict=best,distance=distance,width=model.width,variant=variant,seed=seed,threshold=best_threshold,epoch=best_epoch,
                train_noise=noises,training_profiles=['uniform'],basis='x')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    temporary=checkpoint.with_suffix('.partial.pt')
    torch.save(bundle,temporary); temporary.replace(checkpoint)
    result=dict(experiment_signature=signature,training_config=config,name=name,distance=distance,variant=variant,seed=seed,single_noise=single_noise,parameters=model.parameters_count,
                checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=digest(checkpoint),train_shots=len(g),validation_shots=len(vg),
                train_noise=noises,training_seconds=elapsed,shots_per_second=config['epochs']*len(g)/elapsed,
                threshold=best_threshold,best_epoch=best_epoch,validation_errors=int(best_key[0]),history=history,
                train_provenance=trmeta,validation_provenance=vameta,torch_version=torch.__version__,device='cpu',threads=torch.get_num_threads(),
                probability_note='class-weighted sigmoid score, not calibrated probability; threshold chosen on validation')
    summary.write_text(json.dumps(result,indent=2)+'\n'); return result

def load_model(path):
    b=torch.load(path,map_location='cpu',weights_only=True)
    model=FrameNet(b['distance'],b['width'],b['variant']); model.load_state_dict(b['state_dict']); model.eval()
    return model,b
