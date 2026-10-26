"""One command per reproducible stage; no downloads or external services at runtime."""
import argparse,json
from pathlib import Path
import yaml

def main():
    parser=argparse.ArgumentParser(prog='qubitguard')
    parser.add_argument('command',choices=['train','evaluate','run','serve','plot','verify','followup','latency'])
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--config',default='configs/research-v1.yaml')
    parser.add_argument('--distance',type=int,choices=[3,5,7])
    parser.add_argument('--variant',default='residual',choices=['residual','direct','no_time','no_correlation'])
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--port',type=int,default=8917)
    args=parser.parse_args(); root=args.root.resolve()
    cfg=yaml.safe_load((root/args.config).read_text())
    from .provenance import validate_config
    validate_config(cfg)
    if args.command in ('train','run'):
        from .training import train
        if args.command=='train':
            train(root,cfg,args.distance or 3,args.variant,args.seed)
        else:
            for d in cfg['distances']:
                for seed in cfg['training_seeds']: train(root,cfg,d,'residual',seed)
                if d in cfg.get('ablation_distances',cfg['distances']):
                    for v in cfg['models']:
                        if v!='residual': train(root,cfg,d,v,cfg['training_seeds'][0])
                    train(root,cfg,d,'residual',cfg['training_seeds'][0],single_noise=True)
    if args.command in ('evaluate','run'):
        from .evaluation import evaluate
        evaluate(root,cfg)
    if args.command in ('plot','run'):
        from .plots import plot_all
        plot_all(root)
    if args.command=='latency':
        from .evaluation import refresh_timings
        refresh_timings(root,cfg)
    if args.command=='followup':
        from .fullmap import run_followup
        run_followup(root,cfg)
    if args.command=='serve':
        import os,uvicorn
        os.environ['QUBITGUARD_ROOT']=str(root)
        uvicorn.run('qubitguard.server:app',host='127.0.0.1',port=args.port,workers=1)
    if args.command=='verify':
        from .integrity import verify_release
        result=verify_release(root)
        print(json.dumps(result))
        if result['checksum_failures']: raise SystemExit(1)

if __name__=='__main__': main()
