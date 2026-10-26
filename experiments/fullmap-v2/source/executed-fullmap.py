"""Controlled follow-up: preserve the spatial/temporal readout map at d3.

Same encoder, data and optimization budget as v1. This is an adaptive,
exploratory capacity/readout revision, not an independent novelty claim.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import torch
from torch import nn
from .models import FrameNet
from .simulation import Decoder,Setting,context_features
from .data import merged,digest
from .training import scores,choose_threshold,load_model
from .evaluation import prediction,latencies
from .statistics import wilson,paired

class FullMapFrameNet(FrameNet):
    def __init__(self,distance=3,width=8,variant='residual'):
        if distance!=3: raise ValueError('Full-map follow-up is scoped to d3')
        super().__init__(distance,width,variant)
        self.pool=nn.Identity()
        self.head=nn.Sequential(nn.Linear(width*(distance+1)**3+5,48),nn.SiLU(),nn.Linear(48,1))

def run_followup(root,config):
    root=Path(root); out=root/'experiments'/'fullmap-v2'; art=out/'artifacts'; art.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(config['threads']); torch.manual_seed(17); torch.use_deterministic_algorithms(True)
    tr,trmeta=merged(root/'data',3,config['train_noise'],'train',config['train_shots_per_setting'])
    va,vameta=merged(root/'data',3,config['train_noise'],'validation',config['validation_shots_per_setting'])
    dec=Decoder(Setting(3)); grid=torch.from_numpy(dec.grid(tr['dets'])); context=torch.from_numpy(tr['context'])
    vg=torch.from_numpy(dec.grid(va['dets'])); vc=torch.from_numpy(va['context']); y=torch.from_numpy((tr['truth']^tr['standard']).astype(np.float32))
    model=FullMapFrameNet(); optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    target=y.numpy(); pos=max(1.,min(8.,np.sqrt((len(y)-target.sum())/max(1,target.sum()))))
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos)))
    best_key=(float('inf'),float('inf')); history=[]; start=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train(); loss_sum=0.; epoch_start=time.perf_counter()
        for idx in torch.randperm(len(grid)).split(config['batch_size']):
            optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(grid[idx],context[idx]),y[idx]); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); loss_sum+=float(loss.item())*len(idx)
        prob=scores(model,vg,vc); threshold,errors=choose_threshold(prob,va['truth'],va['standard'],'residual')
        vt=va['truth']^va['standard']; bce=float(-(vt*np.log(prob+1e-7)+(1-vt)*np.log(1-prob+1e-7)).mean())
        if (errors,bce)<best_key:
            best_key=(errors,bce); best=copy.deepcopy(model.state_dict()); best_threshold=threshold; best_epoch=epoch+1
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(grid),validation_errors=errors,threshold=threshold,seconds=time.perf_counter()-epoch_start)
        history.append(row); print(json.dumps(row),flush=True)
    elapsed=time.perf_counter()-start; model.load_state_dict(best); model.eval()
    bundle=dict(state_dict=best,distance=3,width=8,variant='residual',architecture='fullmap-v2',seed=17,threshold=best_threshold,epoch=best_epoch)
    checkpoint=art/'fullmap_d3_s17.pt'; temporary=checkpoint.with_suffix('.partial.pt'); torch.save(bundle,temporary); temporary.replace(checkpoint)
    v1,b1=load_model(root/'artifacts'/'d3_residual_s17.pt')
    rows=[]; comparisons=[]; raw={}; provenance=[]
    test_dir=out/'data'; test_dir.mkdir(exist_ok=True)
    for profile in config['test_profiles']:
        for p in config['test_noise']:
            s=Setting(3,p,profile); decoder=Decoder(s)
            # Independent confirmatory stream, disjoint from original split namespaces.
            stream=int.from_bytes(hashlib.sha256(f'fullmap-v2-confirmatory:{s.key}'.encode()).digest()[:8],'little')
            seed=(stream&((1<<62)-1))|(3<<62)
            dets,obs=decoder.sample(config['test_shots_per_setting'],seed); std=decoder.predict(dets); corr=decoder.predict(dets,True)
            arrays=dict(dets=dets.astype(np.uint8),truth=obs[:,0].astype(np.uint8),standard=std,correlated=corr,context=context_features(dets,std,corr,p))
            dp=test_dir/f'test_{s.key}.npz'; np.savez_compressed(dp,**arrays)
            meta=dict(setting=s.__dict__,seed=seed,shots=len(dets),sha256=digest(dp),circuit_sha256=decoder.circuit_hash,seed_policy='fullmap-v2-confirmatory')
            dp.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); dp.with_suffix('.stim').write_text(str(decoder.circuit)); dp.with_suffix('.dem').write_text(str(decoder.dem)); provenance.append(meta)
            predictions=dict(standard=std,correlated=corr)
            predictions['v1_residual'],raw[s.key+':v1_score']=prediction(v1,b1,arrays,decoder)
            predictions['fullmap_v2'],raw[s.key+':v2_score']=prediction(model,bundle,arrays,decoder)
            truth=arrays['truth']; n=len(truth)
            for name,pred in predictions.items():
                k=int(np.sum(pred!=truth)); low,high=wilson(k,n)
                rows.append(dict(distance=3,profile=profile,noise=p,decoder=name,n=n,errors=k,ler=k/n,ci_low=low,ci_high=high))
                raw[s.key+':'+name]=pred
            raw[s.key+':truth']=truth
            for baseline in ('standard','correlated','v1_residual'):
                comparisons.append(dict(distance=3,profile=profile,noise=p,decoder='fullmap_v2',baseline=baseline,**paired(predictions['fullmap_v2'],predictions[baseline],truth)))
            print(json.dumps(dict(test=s.key,errors={k:int(np.sum(v!=truth)) for k,v in predictions.items()})),flush=True)
    for c in comparisons:c['bonferroni_p']=min(1.,c['mcnemar_p']*len(comparisons))
    result=dict(schema_version=1,hypothesis='Preserving the full d3 feature map instead of 2x2x2 pooling improves residual logical prediction at an explicit model-size/CPU cost.',
                adaptive_status='Designed after v1 validation and preliminary held-out failure patterns. One initialization, same data and training-step budget. New independent test stream; exploratory mechanism/capacity comparison.',
                parameters=model.parameters_count,checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=digest(checkpoint),training_seconds=elapsed,
                train_shots=len(grid),validation_shots=len(vg),validation_errors=int(best_key[0]),threshold=best_threshold,best_epoch=best_epoch,history=history,
                train_provenance=trmeta,validation_provenance=vameta,config=config,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,comparisons=comparisons,datasets=provenance)
    np.savez_compressed(art/'heldout-predictions.npz',**raw)
    (art/'followup.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
