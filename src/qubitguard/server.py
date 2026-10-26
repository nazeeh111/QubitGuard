"""Loopback-only research playground. Scientific outputs come from live Stim."""
from functools import lru_cache
from pathlib import Path
import json, os, time, threading
import numpy as np
import stim, torch
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from typing import Literal
from .simulation import Setting,Decoder,context_features
from .training import load_model
from .data import digest

ROOT=Path(os.environ.get('QUBITGUARD_ROOT',Path(__file__).resolve().parents[2])).resolve()
app=FastAPI(title='QubitGuard local research laboratory',version='0.1.0')
torch.set_num_threads(4)
SIMULATION_LOCK=threading.RLock()

class ShotRequest(BaseModel):
    distance: Literal[3,5,7]=3
    noise: float=Field(default=.007,ge=.0001,le=.03)
    seed: int=Field(default=42,ge=0,le=9007199254740991)
    profile: Literal['uniform','biased']='uniform'
    basis: Literal['x','z']='x'

@lru_cache(maxsize=24)
def decoder_for(distance,noise,profile,basis):
    return Decoder(Setting(distance,noise,profile,basis))

@lru_cache(maxsize=8)
def model_for(distance,mtime):
    path=ROOT/'artifacts'/f'd{distance}_residual_s17.pt'
    summary=json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=summary['checkpoint_sha256']: raise ValueError('Checkpoint integrity check failed')
    return load_model(path)

@app.get('/api/meta')
def meta():
    return dict(distances=[3,5,7],name='QubitGuard',version='0.1.0',rounds='distance',neural_basis='x',
                results='/api/results',simulation='live Stim circuit-derived DEM sampling',
                benchmarks_available=(ROOT/'artifacts'/'benchmark.json').exists())

@app.get('/api/results')
def results():
    p=ROOT/'artifacts'/'benchmark.json'
    return json.loads(p.read_text()) if p.exists() else dict(rows=[],models=[],status='Experiments running. No benchmark claims yet.')

def representative_faults(decoder,error_data):
    mechanisms=[ins for ins in decoder.dem.flattened() if ins.type=='error']
    records=[]
    for idx in np.flatnonzero(error_data):
        ins=mechanisms[int(idx)]
        parity=set()
        for target in ins.targets_copy():
            if not target.is_separator():
                key=('D' if target.is_relative_detector_id() else 'L',target.val)
                parity.symmetric_difference_update([key])
        filt=stim.DetectorErrorModel()
        filt.append('error',1.,[stim.target_relative_detector_id(v) if k=='D' else stim.target_logical_observable_id(v) for k,v in sorted(parity)])
        explained=decoder.circuit.explain_detector_error_model_errors(dem_filter=filt,reduce_to_one_representative_error=True)
        base=dict(mechanism=int(idx),detectors=[v for k,v in sorted(parity) if k=='D'],logical=int(('L',0) in parity),probability=float(ins.args_copy()[0]))
        if explained and explained[0].circuit_error_locations:
            loc=explained[0].circuit_error_locations[0]
            qubits=[]; paulis=[]
            for p in loc.flipped_pauli_product:
                target=p.gate_target
                qubits.append(int(target.value)); paulis.append('X' if target.is_x_target else 'Y' if target.is_y_target else 'Z')
            records.append(dict(**base,instruction=str(loc.instruction_targets),gate='Stim representative fault',qubits=qubits,paulis=paulis,tick=int(loc.tick_offset),explanation=str(explained[0])))
        else: records.append(dict(**base,instruction='No single circuit location available',gate='DEM mechanism',qubits=[],paulis=[],tick=0))
    return records

@app.post('/api/simulate')
def simulate(req: ShotRequest):
    # PyMatching keeps mutable decode workspace. Serialize shared cached graphs.
    with SIMULATION_LOCK:
        return _simulate(req)

def _simulate(req: ShotRequest):
    decoder=decoder_for(req.distance,req.noise,req.profile,req.basis)
    dets,obs,errors=decoder.dem.compile_sampler(seed=req.seed).sample(1,return_errors=True)
    truth=int(obs[0,0]); times={}; preds={}
    for key,corr in [('standard',False),('correlated',True)]:
        start=time.perf_counter_ns(); preds[key]=int(decoder.predict(dets,corr)[0]); times[key]=(time.perf_counter_ns()-start)/1000
    standard=decoder.correction(dets[0]); correlated=decoder.correction(dets[0],True)
    cert=dict(standard=decoder.certificate(dets[0],standard,preds['standard']),correlated=decoder.certificate(dets[0],correlated,preds['correlated']),neural=None)
    checkpoint=ROOT/'artifacts'/f'd{req.distance}_residual_s17.pt'
    supported=req.basis=='x' and checkpoint.exists() and checkpoint.with_suffix('.json').exists()
    neural=None; score=None; toggle=None
    model_meta=dict(name='FrameNet residual',supported=supported,parameters=None,training_noise=[.003,.007,.012],trained_basis='x')
    neural_edges=set()
    if supported:
        model,b=model_for(req.distance,checkpoint.stat().st_mtime_ns)
        start=time.perf_counter_ns()
        # Measure actual full prediction dependencies, not cached baselines.
        a=decoder.predict(dets); c=decoder.predict(dets,True)
        context=context_features(dets,a,c,req.noise)
        with torch.inference_mode(): score=float(model(torch.from_numpy(decoder.grid(dets)),torch.from_numpy(context)).sigmoid()[0])
        toggle=int(score>b['threshold']); neural=int(a[0])^toggle
        times['neural_total']=(time.perf_counter_ns()-start)/1000
        neural_edges=standard.symmetric_difference(decoder.logical_loop() if toggle else set())
        cert['neural']=decoder.certificate(dets[0],neural_edges,neural)
        model_meta.update(parameters=model.parameters_count,threshold=b['threshold'],seed=b['seed'],training_noise=b['train_noise'],score_kind='class-weighted residual sigmoid, uncalibrated',out_of_distribution=req.profile not in b['training_profiles'] or not min(b['train_noise'])<=req.noise<=max(b['train_noise']))
    else: times['neural_total']=None
    if not all(x for x in cert.values() if x is not None): raise HTTPException(500,'Correction consistency check failed')
    coords=decoder.circuit.get_final_qubit_coordinates()
    final_measurement=next((ins for ins in reversed(list(decoder.circuit.flattened())) if ins.name in ('M','MX','MY')),None)
    data=set(t.value for t in final_measurement.targets_copy()) if final_measurement is not None else set()
    return dict(distance=req.distance,rounds=req.distance,noise=req.noise,profile=req.profile,basis=req.basis,seed=req.seed,
                detectors=[dict(id=i,x=float(x),y=float(y),t=int(t),event=bool(dets[0,i])) for i,(x,y,t) in enumerate(decoder.coords)],
                qubits=[dict(id=int(q),x=float(xy[0]),y=float(xy[1]),role='data' if q in data else 'ancilla') for q,xy in coords.items()],
                injected_errors=representative_faults(decoder,errors[0]),
                corrections=dict(standard=decoder.edge_json(standard),correlated=decoder.edge_json(correlated),neural=decoder.edge_json(neural_edges)),
                predictions=dict(**preds,neural=neural,neural_probability=score,logical_toggle=toggle,actual=truth),
                success=dict(standard=preds['standard']==truth,correlated=preds['correlated']==truth,neural=None if neural is None else neural==truth),
                certificate=cert,model=model_meta,latency_us=times,
                trace_note='Sampled circuit-derived detector-error-model mechanisms. Physical fault locations are one Stim explanatory representative, not a unique underlying circuit trajectory. Correction edges are graphlike DEM representatives, not literal physical-qubit paths. Neural correction XORs a verified closed logical graph cycle when its learned logical-frame toggle is 1. Timing excludes sampling, certificates and HTTP; this single request is not a benchmark.',
                provenance=dict(stim_version=stim.__version__,circuit_sha256=decoder.circuit_hash))

@app.get('/')
def index(): return FileResponse(ROOT/'web'/'index.html')
@app.get('/app.js')
def js(): return FileResponse(ROOT/'web'/'app.js',media_type='application/javascript')
@app.get('/favicon.svg')
def favicon(): return FileResponse(ROOT/'web'/'favicon.svg',media_type='image/svg+xml')
@app.get('/style.css')
def css(): return FileResponse(ROOT/'web'/'style.css',media_type='text/css')
app.mount('/figures',StaticFiles(directory=ROOT/'figures'),name='figures')
