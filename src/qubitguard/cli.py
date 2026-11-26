"""One command per reproducible stage; no downloads or external services at runtime."""
import argparse,json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from . import __version__

def _save_shot(path, text):
    with TemporaryDirectory(prefix='.qubitguard-', dir=path.parent) as directory:
        staged = Path(directory) / 'shot.json'
        staged.write_text(text, encoding='utf-8')
        os.link(staged, path)


def main():
    parser=argparse.ArgumentParser(
        prog='qubitguard',
        description='CPU surface-code decoding: inspect a single shot or run an explicit research stage.',
        epilog='First use: serve or simulate with --root pointing to the complete source release. '
               'The wheel contains Python code; web/config/checkpoint/study files come from that root. '
               'train, run and followup create experiments. Integrity checks are unsigned consistency checks.',
    )
    parser.add_argument('--version',action='version',version=f'QubitGuard {__version__}')
    parser.add_argument('command',choices=['simulate','train','evaluate','run','serve','plot','verify','followup','latency'])
    parser.add_argument('--root',type=Path,default=Path.cwd(),help='Complete source release root, including web, configs and retained artifacts (default: current directory)')
    parser.add_argument('--config',default='configs/research-v1.yaml',help='Research-stage configuration; simulate uses its explicit circuit flags')
    parser.add_argument('--distance',type=int,choices=[3,5,7])
    parser.add_argument('--variant',default='residual',choices=['residual','direct','no_time','no_correlation'],help='Training variant; simulate uses only the supplied residual checkpoint')
    parser.add_argument('--seed',type=int,default=17,help='Training initialization seed, or sampled-shot seed for simulate (default: 17)')
    parser.add_argument('--port',type=int,default=8917)
    parser.add_argument('--noise',type=float,default=.007,help='simulate only: nominal circuit-channel probability, not a measured hardware rate')
    parser.add_argument('--profile',choices=['uniform','biased'],default='uniform',help='simulate only: circuit noise profile')
    parser.add_argument('--basis',choices=['x','z'],default='x',help='simulate only: memory basis; neural Z output is unavailable')
    parser.add_argument('--out',type=Path,help='simulate only: new JSON file in an existing directory; existing paths are preserved')
    args=parser.parse_args(); root=args.root.resolve()
    if args.command=='simulate' and (args.variant!='residual' or args.config!='configs/research-v1.yaml'):
        parser.exit(2,'qubitguard: simulate uses explicit circuit flags and the retained residual checkpoint; --config and other --variant values are research-stage controls.\n')
    if args.command=='serve' and not 1<=args.port<=65535:
        parser.exit(2,'qubitguard: port must be between 1 and 65535.\n')
    if args.out is not None and args.command!='simulate':
        parser.exit(2,'qubitguard: --out is available for simulate only.\n')
    if args.command=='verify':
        from .integrity import verify_release
        result=verify_release(root)
        print(json.dumps(result))
        if result['checksum_failures']: raise SystemExit(1)
        return
    if args.command=='simulate' and args.out is not None:
        if args.out.exists() or args.out.is_symlink():
            parser.exit(2,'qubitguard: output exists; choose a fresh JSON path.\n')
        if not args.out.parent.is_dir():
            parser.exit(2,'qubitguard: output parent directory does not exist.\n')
    if args.command in ('serve','simulate'):
        required=[root/'web/index.html',root/args.config]
        if any(not path.is_file() for path in required) or not (root/'figures').is_dir():
            parser.exit(2,'qubitguard: incomplete source root; use --root with the full source release (web, configs and artifacts).\n')
    if args.command=='simulate':
        os.environ['QUBITGUARD_ROOT']=str(root)
        from pydantic import ValidationError
        from .server import ShotRequest,simulate
        try:
            request=ShotRequest(distance=args.distance or 3,noise=args.noise,seed=args.seed,profile=args.profile,basis=args.basis)
        except ValidationError as error:
            issue=error.errors(include_url=False)[0]
            parser.exit(2,f"qubitguard: {'.'.join(map(str,issue['loc']))}: {issue['msg']}\n")
        shot=simulate(request)
        text=json.dumps({'schema':'qubitguard.shot/v1','software_version':__version__,'shot':shot},indent=2,allow_nan=False)+'\n'
        if args.out is None:
            print(text,end='')
        else:
            try:
                _save_shot(args.out, text)
            except OSError as error:
                parser.exit(2,f'qubitguard: {error}\n')
            print(str(args.out.resolve()))
        return
    import yaml
    from .provenance import validate_config
    try:
        cfg=yaml.safe_load((root/args.config).read_text())
        validate_config(cfg)
    except (OSError,yaml.YAMLError,ValueError) as error:
        parser.exit(2,f'qubitguard: {error}\n')
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
        import uvicorn
        os.environ['QUBITGUARD_ROOT']=str(root)
        print(f'QubitGuard {__version__} · http://127.0.0.1:{args.port} · source root: {root}',flush=True)
        uvicorn.run('qubitguard.server:app',host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__': main()
