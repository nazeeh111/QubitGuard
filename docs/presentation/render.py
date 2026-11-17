"""Render actual shot data; no simulation or model training. Matplotlib required."""
from pathlib import Path
import json,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
P=Path(__file__).resolve().parent
D=json.loads((P/'selected-toggle.json').read_text())['shot']
V=json.loads((P/'provenance.json').read_text())
assert hashlib.sha256((P.parents[1]/V['checkpoint']).read_bytes()).hexdigest()==V['checkpoint_sha256']
coords=sorted({(d['x'],d['y']) for d in D['detectors']});times=sorted({d['t'] for d in D['detectors']});lookup={(d['x'],d['y'],d['t']):d for d in D['detectors']}
assert len(D['detectors'])==24 and len(coords)==8 and len(times)==4
ink='#16231e';green='#185b42';muted='#526158';red='#a83c30';paper='#f4f5f0'
def draw(phone=False):
 w,h=(440,1150) if phone else (1000,730)
 with plt.rc_context({'font.family':['DejaVu Sans','Arial','sans-serif'],'svg.fonttype':'none','svg.hashsalt':'qubitguard-presentation','savefig.facecolor':paper}):
  fig=plt.figure(figsize=(w/100,h/100),dpi=100,facecolor=paper);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
  def txt(x,y,t,size=18,c=ink,weight='normal',ha='left'):return ax.text(x,y,t,fontsize=size*.72,color=c,weight=weight,ha=ha,va='center',zorder=4)
  def line(x1,y1,x2,y2,c='#cbd4ca',lw=1):ax.plot([x1,x2],[y1,y2],color=c,lw=lw)
  m=24 if phone else 40
  txt(m,32,'QUBITGUARD / RETAINED CHECKPOINT',12,muted,'bold')
  if phone:
   txt(m,76,'Same syndrome.',29,weight='bold');txt(m,112,'Different logical class.',29,weight='bold');txt(m,151,'d3 · X memory · p = 0.007 · seed 75',17)
  else:
   txt(m,86,'Same syndrome. Different logical class.',39,weight='bold');txt(m,130,'d3 · X memory · 3 syndrome rounds · uniform p = 0.007 · selected seed 75',20)
  y0=206 if phone else 205
  txt(m,y0,'ONE DETECTOR HISTORY',15,green,'bold')
  txt(m,y0+28,'Filled 1 = event · open 0 = no event',16)
  x0=m+66; dx=38 if phone else 48; top=y0+86
  for j,(x,y) in enumerate(coords):txt(x0+j*dx,top-30,f'{x:g},{y:g}',11,muted,ha='center')
  for i,t in enumerate(times):
   yy=top+i*43;txt(m,yy,f't = {t}',16)
   for j,(x,y) in enumerate(coords):
    xx=x0+j*dx;d=lookup.get((x,y,t))
    if d is None: txt(xx,yy,'—',16,'#879489',ha='center');continue
    ax.add_patch(Circle((xx,yy),12,facecolor=green if d['event'] else '#ffffff',edgecolor=green if d['event'] else '#a4b3a5',lw=1.2,gid=f'detector-{d["id"]}'))
    txt(xx,yy,int(d['event']),15,'white' if d['event'] else muted,ha='center').set_gid(f'bit-{d["id"]}')
  txt(m,top+168,'Column labels: detector coordinates x,y',13,muted)
  txt(m,top+191,'— = detector absent at this time boundary',13,muted)
  ty=551 if phone else 205;tx=24 if phone else 555;tw=392 if phone else 405
  txt(tx,ty,'SAME SHOT · THREE RECOVERIES',15,green,'bold')
  txt(tx,ty+33,f'Simulated true logical flip: {D["predictions"]["actual"]}',22,weight='bold')
  txt(tx,ty+66,'Decoder / graph edges',14,muted);txt(tx+tw-95,ty+66,'Pred. / parity',14,muted,ha='right')
  for i,(key,label) in enumerate([('standard','Standard matching'),('correlated','Correlated matching'),('neural','FrameNet hybrid')]):
   yy=ty+103+i*59;line(tx,yy-22,tx+tw,yy-22)
   txt(tx,yy,label,18,weight='bold');txt(tx,yy+22,f'{len(D["corrections"][key])} graphlike DEM edges',14,muted)
   pred=D['predictions'][key];good=D['success'][key]
   txt(tx+tw-15,yy,f'{pred} / valid',19,green if good else red,'bold',ha='right')
   txt(tx+tw-15,yy+22,'matches truth' if good else 'wrong logical class',13,green if good else red,ha='right')
  cy=917 if phone else 540
  txt(m,cy-33,'LEARNED TOGGLE / CLOSED LOGICAL CYCLE',12,green,'bold')
  # Exact symmetric difference between the standard and hybrid edge sets.
  pos={1:(m+65,cy+10),2:(m+220,cy+10),None:(m+142,cy+95)}
  for e in V['toggle_cycle']:
   a,b=pos[e['a']],pos[e['b']];line(*a,*b,green,2)
  for node,(x,y) in pos.items():
   ax.add_patch(Circle((x,y),20,facecolor=paper,edgecolor=green,lw=1.5,zorder=3));txt(x,y,'B' if node is None else f'D{node}',15,ha='center')
  txt(m+265,cy+18,'Parity XOR: 0',15);txt(m+265,cy+45,'Logical XOR: 1',15);txt(m+265,cy+81,'B = boundary',13,muted)
  if not phone:
   txt(555,535,'Hybrid = standard logical bit XOR learned toggle.',17)
   txt(555,565,'The toggle changes logical class, not the syndrome.',16)
   txt(555,605,'These are graph representatives, not physical paths.',15,muted)
  foot=h-53
  txt(m,foot,'Selected illustration, not a decoder-quality estimate.',14,muted)
  txt(m,foot+26,'Stim simulation · no quantum hardware' if phone else 'Stim simulation · uncalibrated neural score · no quantum hardware',13,muted)
  name='logical-class-mobile' if phone else 'logical-class'
  fig.savefig(P/(name+'.png'),dpi=100,metadata={'Software':'Matplotlib; source and input JSON are retained'})
  fig.savefig(P/(name+'.svg'),metadata={'Date':None})
  plt.close(fig)
  # Every text family, including nested tspans, gets an explicit browser fallback.
  svg=(P/(name+'.svg')).read_text()
  svg=re.sub(r'font-family:[^;\"]+',"font-family: 'DejaVu Sans', Arial, sans-serif",svg)
  svg=svg.replace('<svg ', '<svg role="img" aria-labelledby="qg-title qg-desc" ',1)
  start=svg.index('>',svg.index('<svg '))+1
  svg=svg[:start]+'<title id="qg-title">Same syndrome. Different logical class.</title><desc id="qg-desc">Actual selected simulated shot75. All three parity checks pass; standard logical0 fails while correlated and hybrid logical1 match truth. A closed odd-logical graph cycle preserves detector parity. Not a quality estimate.</desc>'+svg[start:]
  (P/(name+'.svg')).write_text(svg)
draw();draw(True)
