"""Run meaningful contracts and reconcile saved scientific release artifacts."""
from pathlib import Path
import argparse,hashlib,json,os,re,subprocess,sys,tempfile
import numpy as np,yaml
from qubitguard.data import dataset,digest
from qubitguard.provenance import experiment_signature,source_fingerprint
from qubitguard.simulation import Setting
from qubitguard.training import load_model
from qubitguard.fullmap import load_fullmap
from qubitguard.statistics import wilson,paired

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));args=ap.parse_args();root=args.root.resolve()
    cfg=yaml.safe_load((root/'configs/research-v1.yaml').read_text()); env=os.environ.copy();env.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'qubitguard-mpl'))
    run=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=root,env=env,capture_output=True,text=True)
    log=run.stdout+run.stderr;(root/'artifacts/test-results.txt').write_text(log)
    if run.returncode:print(log);raise SystemExit(run.returncode)
    passed=int(re.search(r'(\d+) passed',log).group(1))
    primary=json.loads((root/'artifacts/benchmark.json').read_text()); follow_dir=root/'experiments/fullmap-v2'; follow=json.loads((follow_dir/'artifacts/followup.json').read_text())
    counts={'primary_datasets':0,'primary_stored_shots':0,'followup_datasets':0,'followup_test_shots':0,'trained_checkpoints':0,'result_rows':0,'paired_comparisons':0}
    for p in (root/'data').glob('*.json'):
        m=json.loads(p.read_text());s=Setting(**m['setting']);a,_=dataset(root/'data',s.distance,s.noise,s.profile,m['split'],m['shots'],s.basis)
        assert len(a['truth'])==m['shots'];counts['primary_datasets']+=1;counts['primary_stored_shots']+=m['shots']
    for p in (root/'artifacts').glob('d*.pt'):
        m=json.loads(p.with_suffix('.json').read_text());assert digest(p)==m['checkpoint_sha256']
        assert m['experiment_signature']==experiment_signature(cfg,m['distance'],m['variant'],m['seed'],m['single_noise'])
        model,b=load_model(p);assert model.parameters_count==m['parameters'];counts['trained_checkpoints']+=1
    model,b=load_fullmap(follow_dir/'artifacts/fullmap_d3_s17.pt');assert model.parameters_count==follow['parameters'];assert digest(follow_dir/'artifacts/fullmap_d3_s17.pt')==follow['checkpoint_sha256'];counts['trained_checkpoints']+=1
    assert digest(follow_dir/'source/executed-fullmap.py')==follow['executed_source_sha256']
    for m in follow['datasets']:
        p=follow_dir/'data'/f'test_{Setting(**m["setting"]).key}.npz';assert digest(p)==m['sha256'];counts['followup_datasets']+=1;counts['followup_test_shots']+=m['shots']
    for result,folder,is_primary in [(primary,root,True),(follow,follow_dir,False)]:
        raw_path=folder/'artifacts/heldout-predictions.npz';raw=np.load(raw_path,allow_pickle=False)
        expected=result.get('raw_predictions_sha256');assert expected and digest(raw_path)==expected
        for row in result['rows']:
            s=Setting(row['distance'],row['noise'],row['profile'])
            prefix=f'test_{s.key}_n{row["n"]}.npz' if is_primary else s.key
            pred=raw[prefix+':'+row['decoder']+(':prediction' if is_primary else '')];truth=raw[prefix+':truth']
            k=int(np.sum(pred!=truth));low,high=wilson(k,len(truth))
            assert k==row['errors'] and len(truth)==row['n']
            assert np.allclose([k/len(truth),low,high],[row['ler'],row['ci_low'],row['ci_high']],rtol=0,atol=1e-12)
            counts['result_rows']+=1
        for c in result['comparisons']:
            s=Setting(c['distance'],c['noise'],c['profile']);prefix=f'test_{s.key}_n20000.npz' if is_primary else s.key
            pa=raw[prefix+':'+c['decoder']+(':prediction' if is_primary else '')];pb=raw[prefix+':'+c['baseline']+(':prediction' if is_primary else '')]
            check=paired(pa,pb,raw[prefix+':truth'])
            for key in ('wins','losses','delta_ler','delta_ci_low','delta_ci_high','mcnemar_p'):
                assert np.isclose(check[key],c[key],rtol=1e-12,atol=1e-14),key
            assert np.isclose(c['bonferroni_p'],min(1.,check['mcnemar_p']*len(result['comparisons'])))
            counts['paired_comparisons']+=1
    repro=json.loads((root/'artifacts/deterministic-reproduction.json').read_text());assert repro['exact_all_tensors'] and repro['threshold_equal'] and repro['epoch_equal']
    assert repro['original_sha256']==digest(root/'artifacts/d3_residual_s17.pt')
    audit=json.loads((root/'artifacts/initial-provenance-audit.json').read_text());assert len(audit['datasets'])==counts['primary_datasets']
    output=dict(schema_version=1,status='passed',tests=dict(passed=passed,returncode=run.returncode,warning='Starlette/httpx dependency deprecation; no failed tests',log='test-results.txt'),
                counts=counts,source_sha256=source_fingerprint(),checks=['dataset/checkpoint checksums and current compatibility signatures','retained exact circuit/DEM bytes','all saved failure counts, Wilson intervals and paired comparisons','follow-up executed source snapshot identity','checkpoint architecture and parameter counts','recorded builder retraining/replay audit consistency'],
                primary_test_shots=sum(m['shots'] for m in primary['datasets']),independent_followup_shots=counts['followup_test_shots'],
                distinctions='Full retraining and full circuit replay are separately retained builder execution records. This verification runs the contracts and reconciles retained numerical artifacts; it is not independent peer review or external-paper replication.',
                independent_review='docs/release-audit.md',numerical_artifacts_sha256=dict(primary=digest(root/'artifacts/benchmark.json'),followup=digest(follow_dir/'artifacts/followup.json')))
    (root/'artifacts/verification.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
