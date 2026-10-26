"""Publication-exportable figures drawn only from saved experiment artifacts."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'standard':'#617a60','correlated':'#3178ac','neural':'#bd8c2d'}

def plot_all(root):
    root=Path(root); r=json.loads((root/'artifacts'/'benchmark.json').read_text()); rows=r['rows']
    out=root/'figures'; out.mkdir(exist_ok=True)
    shots_label=f'{rows[0]["n"]:,}' if len({x['n'] for x in rows})==1 else 'Variable'
    training_label=f'{r["models"][0]["train_shots"]:,}'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.bbox':'tight','figure.dpi':140})
    def save(fig,name):
        fig.savefig(out/f'{name}.svg'); fig.savefig(out/f'{name}.png'); fig.savefig(out/f'{name}.pdf'); plt.close(fig)
    for profile in ('uniform','biased'):
        fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
        for ax,d in zip(axes,(3,5,7)):
            for key,label,color in [('standard','Standard matching',COLORS['standard']),('correlated','Correlated matching',COLORS['correlated']),(f'd{d}_residual_s17','FrameNet + matching',COLORS['neural'])]:
                rs=sorted([x for x in rows if x['distance']==d and x['profile']==profile and x['decoder']==key],key=lambda x:x['noise'])
                x=np.array([z['noise'] for z in rs])*100; y=np.array([z['ler'] for z in rs]); lo=np.array([z['ci_low'] for z in rs]); hi=np.array([z['ci_high'] for z in rs])
                positive=y>0
                ax.errorbar(x[positive],y[positive],yerr=[(y-lo)[positive],(hi-y)[positive]],label=label,color=color,marker='o',markersize=4,capsize=2,linewidth=1.5)
                if np.any(~positive): ax.scatter(x[~positive],hi[~positive],marker='v',color=color,s=28)
            ax.set(yscale='log',xlabel='Circuit noise p (%)',title=f'd = {d}, rounds = {d}')
            ax.grid(alpha=.16); ax.set_ylim(1e-5,.5)
        axes[0].set_ylabel('Logical failure probability / memory shot')
        axes[1].legend(fontsize=10,loc='lower right')
        fig.suptitle(f'{"Uniform circuit noise" if profile=="uniform" else "Held-out measurement-heavy noise profile"} · {shots_label} identical shots / setting · 95% Wilson intervals (triangles: zero-failure upper bounds)',fontsize=11)
        fig.tight_layout(); save(fig,f'logical-error-{profile}')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    chosen=[x for x in rows if x['profile']=='uniform' and x['noise']==.007 and x['decoder'] in ('standard','correlated',f'd{x["distance"]}_residual_s17')]
    for j,(key,label,color) in enumerate([('standard','Standard',COLORS['standard']),('correlated','Correlated',COLORS['correlated']),('neural','FrameNet hybrid',COLORS['neural'])]):
        rs=[next(x for x in chosen if x['distance']==d and x['decoder']==(f'd{d}_residual_s17' if key=='neural' else key)) for d in (3,5,7)]
        axes[0].bar(np.arange(3)+(j-1)*.24,[x['latency_us'] for x in rs],width=.24,label=label,color=color)
        if all('single_shot_median_us' in x for x in rs):
            axes[1].plot((3,5,7),[x['single_shot_median_us'] for x in rs],label=label,color=color,marker='o')
    axes[0].set(yscale='log',xticks=np.arange(3),xticklabels=['d3','d5','d7'],ylabel='µs / shot, median warmed batch 256',title='Amortized end-to-end CPU cost')
    axes[0].legend(fontsize=11)
    axes[1].set(yscale='log',xlabel='Distance',xticks=(3,5,7),ylabel='µs / shot, sequential batch 1',title='Single-shot median latency')
    axes[1].legend(fontsize=11)
    fig.suptitle('p = 0.007 · uniform · matching + feature construction included for hybrid · 4 CPU threads',fontsize=11)
    fig.tight_layout(); save(fig,'cpu-latency')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    variants=[('d3_residual_s17','Full residual'),('d3_direct_s17','Direct'),('d3_no_time_s17','No neural time order'),('d3_no_correlation_s17','No correlated context'),('d3_residual_s17_single','Single training p')]
    for key,label in variants:
        for ax,profile in zip(axes,('uniform','biased')):
            rs=sorted([x for x in rows if x['distance']==3 and x['profile']==profile and x['decoder']==key],key=lambda x:x['noise'])
            ax.plot([x['noise']*100 for x in rs],[x['ler'] for x in rs],marker='o',markersize=3,label=label,linewidth=1.2)
            ax.set(yscale='log',xlabel='Circuit noise p (%)',title=profile+' profile'); ax.grid(alpha=.15)
    axes[0].set_ylabel('Logical failure probability / shot'); axes[1].legend(fontsize=11)
    fig.suptitle(f'd3 ablations · equal {training_label} training-shot budgets · one initialization per ablation',fontsize=11)
    fig.tight_layout(); save(fig,'ablations')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,d in zip(axes,(3,5,7)):
        for m in r['models']:
            if m['distance']==d and m['variant']=='residual' and not m['single_noise']:
                ax.plot([h['epoch'] for h in m['history']],[h['validation_ler'] for h in m['history']],label=f"seed {m['seed']}",marker='.',linewidth=1)
        ax.set(xlabel='Epoch',title=f'd = {d}'); ax.grid(alpha=.2); ax.legend(fontsize=11)
    axes[0].set_ylabel('Validation logical failure probability')
    fig.suptitle('Training seed sensitivity · checkpoint and threshold chosen only on validation',fontsize=11)
    fig.tight_layout(); save(fig,'learning-curves')
    if (root/'experiments/fullmap-v2/artifacts/followup.json').exists(): plot_followup(root)
    fields=sorted(set().union(*(x.keys() for x in rows)))
    with open(root/'artifacts'/'benchmark.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader(); writer.writerows(rows)
    fields=sorted(set().union(*(x.keys() for x in r['comparisons'])))
    with open(root/'artifacts'/'paired-comparisons.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fields); writer.writeheader(); writer.writerows(r['comparisons'])


def plot_followup(root):
    root=Path(root);r=json.loads((root/'experiments/fullmap-v2/artifacts/followup.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,profile in zip(axes,('uniform','biased')):
        for base,label,color,offset in [('correlated','vs correlated','#3178ac',-.012),('v1_residual','vs compact residual','#bd8c2d',.012)]:
            rs=sorted([x for x in r['comparisons'] if x['profile']==profile and x['baseline']==base],key=lambda x:x['noise'])
            x=np.array([z['noise'] for z in rs])*100+offset; y=np.array([z['delta_ler'] for z in rs])*100
            lo=np.array([z['delta_ci_low'] for z in rs])*100;hi=np.array([z['delta_ci_high'] for z in rs])*100
            ax.errorbar(x,y,yerr=[y-lo,hi-y],label=label,color=color,marker='o',markersize=3,capsize=2,linewidth=1)
        ax.axhline(0,color='#536158',linewidth=1,linestyle='--');ax.grid(alpha=.15)
        ax.set(xlabel='Circuit noise p (%)',title='Uniform' if profile=='uniform' else 'Measurement-heavy')
        ax.legend(fontsize=10)
    axes[0].set_ylabel('Full-map minus comparator LER\n(percentage points; lower favors full-map)')
    fig.suptitle('d3 readout revision · 280,000 new test shots · conservative paired 95% intervals',fontsize=11)
    fig.tight_layout()
    for suffix in ('png','pdf','svg'):fig.savefig(root/'figures'/f'paired-followup.{suffix}')
    plt.close(fig)
