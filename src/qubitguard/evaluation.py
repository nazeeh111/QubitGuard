"""Paired held-out evaluation, explicit CPU throughput and latency contracts."""
from pathlib import Path
import json, platform, time
import numpy as np
import torch, stim, pymatching
from .data import dataset,digest
from .simulation import Decoder,Setting
from .statistics import wilson,paired
from .training import load_model,scores
from .provenance import experiment_signature,validate_config

def prediction(model,bundle,arrays,decoder):
    grid=torch.from_numpy(decoder.grid(arrays['dets']))
    context=torch.from_numpy(arrays['context'])
    prob=scores(model,grid,context)
    pred=(prob>bundle['threshold']).astype(np.uint8)
    if bundle['variant']!='direct': pred^=arrays['standard']
    return pred,prob

def timing(fn,repeats=5):
    fn(); samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-start)/1000)
    return float(np.median(samples))

@torch.inference_mode()
def latencies(decoder,arrays,model=None,bundle=None):
    dets=arrays['dets'][:min(256,len(arrays['dets']))]; n=len(dets)
    std=lambda:decoder.predict(dets)
    corr=lambda:decoder.predict(dets,True)
    if model is None:
        result={}
        for key,corr_flag in [('standard',False),('correlated',True)]:
            fn=lambda:decoder.predict(dets,corr_flag)
            amortized=timing(fn)/n
            single=[]
            for j in range(min(32,n)):
                start=time.perf_counter_ns(); decoder.predict(dets[j:j+1],corr_flag); single.append((time.perf_counter_ns()-start)/1000)
            result[key]=dict(latency_us=amortized,single_shot_median_us=float(np.median(single)),single_shot_p95_us=float(np.percentile(single,95)),batch=n)
        return result
    variant=bundle['variant']
    def end_to_end(ds):
        # Include classical calls, feature construction and neural inference.
        from .simulation import context_features
        if variant=='direct': a=np.zeros(len(ds),np.uint8); b=a
        else:
            a=decoder.predict(ds)
            b=decoder.predict(ds,True) if variant!='no_correlation' else a
        ctx=context_features(ds,a,b,decoder.setting.noise)
        logits=model(torch.from_numpy(decoder.grid(ds)),torch.from_numpy(ctx))
        result=(logits.sigmoid().numpy()>bundle['threshold']).astype(np.uint8)
        return result if variant=='direct' else result^a
    # At full inference, no labels, cached predictions, or cached features.
    amortized=timing(lambda:end_to_end(dets))/n
    single=[]
    for j in range(min(32,n)):
        t=time.perf_counter_ns(); end_to_end(dets[j:j+1]); single.append((time.perf_counter_ns()-t)/1000)
    return dict(latency_us=amortized,single_shot_median_us=float(np.median(single)),single_shot_p95_us=float(np.percentile(single,95)),batch=n)

def evaluate(root,config):
    validate_config(config)
    root=Path(root); torch.set_num_threads(config.get('threads',4))
    rows=[]; comparisons=[]; model_meta=[]; raw={}; data_meta=[]
    for distance in config['distances']:
        expected={f'd{distance}_residual_s{s}' for s in config['training_seeds']}
        if distance in config.get('ablation_distances',config['distances']):
            expected.update(f'd{distance}_{v}_s{config["training_seeds"][0]}' for v in config['models'] if v!='residual')
            expected.add(f'd{distance}_residual_s{config["training_seeds"][0]}_single')
        paths=sorted((root/'artifacts').glob(f'd{distance}_*.pt'))
        if {p.stem for p in paths}!=expected: raise ValueError(f'Checkpoint roster differs from experiment configuration at d={distance}')
        loaded=[]
        for path in paths:
            summary=json.loads(path.with_suffix('.json').read_text())
            signature=experiment_signature(config,distance,summary['variant'],summary['seed'],summary['single_noise'])
            if summary.get('experiment_signature')!=signature or summary.get('checkpoint_sha256')!=digest(path):
                raise ValueError(f'Checkpoint provenance differs from evaluation configuration: {path}')
            model,bundle=load_model(path); name=path.stem
            if bundle['distance']!=distance or bundle['variant']!=summary['variant']: raise ValueError('Checkpoint bundle metadata mismatch')
            loaded.append((name,model,bundle))
            model_meta.append(json.loads(path.with_suffix('.json').read_text()))
        for profile in config['test_profiles']:
            for noise in config['test_noise']:
                arrays,meta=dataset(root/'data',distance,noise,profile,'test',config['test_shots_per_setting'])
                data_meta.append(meta); decoder=Decoder(Setting(distance,noise,profile))
                baseline_time=latencies(decoder,arrays)
                predictions={key:arrays[key] for key in ('standard','correlated')}
                timing_by=baseline_time
                for name,model,bundle in loaded:
                    predictions[name],prob=prediction(model,bundle,arrays,decoder)
                    timing_by[name]=latencies(decoder,arrays,model,bundle)
                    raw[f'{meta["file"]}:{name}:score']=prob.astype(np.float32)
                truth=arrays['truth']; n=len(truth)
                for name,pred in predictions.items():
                    k=int(np.sum(pred!=truth)); low,high=wilson(k,n)
                    rows.append(dict(distance=distance,noise=noise,profile=profile,basis='x',rounds=distance,decoder=name,
                                     n=n,errors=k,ler=k/n,ci_low=low,ci_high=high,**timing_by[name],dataset_sha256=meta['sha256']))
                    raw[f'{meta["file"]}:{name}:prediction']=pred
                    if name not in ('standard','correlated'):
                        for baseline in ('standard','correlated'):
                            comparisons.append(dict(distance=distance,noise=noise,profile=profile,decoder=name,baseline=baseline,n=n,
                                                    **paired(pred,arrays[baseline],truth)))
                raw[f'{meta["file"]}:truth']=truth
                print(json.dumps(dict(evaluated=meta['file'],errors={k:int(np.sum(v!=truth)) for k,v in predictions.items()})),flush=True)
    m=len(comparisons)
    for c in comparisons: c['bonferroni_p']=min(1.,c['mcnemar_p']*m)
    result=dict(schema_version=1,rows=rows,comparisons=comparisons,models=model_meta,datasets=data_meta,
                uncertainty='95% Wilson marginal intervals; conservative exact paired difference CI; exact paired binomial p with global Bonferroni adjustment',
                latency_contract='CPU, 4 threads; warmed median of 5 batches, microseconds/shot at batch 256; includes feature/grid conversion and matching dependencies for neural. Simulator, disk, HTTP, certificate excluded. Single-shot timing is sequential 32 samples, not hard-real-time.',
                environment=dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),torch=torch.__version__,stim=stim.__version__,pymatching=pymatching.__version__,threads=torch.get_num_threads()),
                claim='Exploratory CPU-scale result; no quantum hardware, threshold, real-time or superiority claim.')
    np.savez_compressed(root/'artifacts'/'heldout-predictions.npz',**raw)
    result['raw_predictions_sha256']=digest(root/'artifacts'/'heldout-predictions.npz')
    (root/'artifacts'/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def refresh_timings(root,config):
    """Re-measure saved methods after fits/tests finish; never change predictions."""
    from datetime import datetime,timezone
    root=Path(root); path=root/'artifacts'/'benchmark.json'; result=json.loads(path.read_text())
    torch.set_num_threads(config.get('threads',4)); loaded={}
    for m in result['models']:
        loaded[m['name']]=load_model(root/'artifacts'/f'{m["name"]}.pt')
    for distance in config['distances']:
        for profile in config['test_profiles']:
            for noise in config['test_noise']:
                arrays,_=dataset(root/'data',distance,noise,profile,'test',config['test_shots_per_setting'])
                dec=Decoder(Setting(distance,noise,profile)); baseline=latencies(dec,arrays)
                for row in result['rows']:
                    if row['distance']!=distance or row['profile']!=profile or row['noise']!=noise: continue
                    name=row['decoder']
                    row.update(baseline[name] if name in baseline else latencies(dec,arrays,*loaded[name]))
    result['latency_refresh']=dict(at_utc=datetime.now(timezone.utc).isoformat(),scope='No concurrent QubitGuard training/tests; background OS workload uncontrolled. Refreshed timings only; predictions/counts unchanged.')
    temporary=path.with_suffix('.partial.json'); temporary.write_text(json.dumps(result,indent=2)+'\n'); temporary.replace(path)
    return result
