"""Exploratory post-hoc source-token factorization for four adaptive points. The last point here is 1.2M, not the pinned final. No new checkpoints/prompts.
One checkpoint loaded at a time. Off-diagonal prefix/readout hybrids are
inference diagnostics, not training interventions or natural checkpoints.
"""
from pathlib import Path
import json,os,sys,subprocess,numpy as np,hashlib,datetime
from prefix_model import Prefix,P,ROOT
S=ROOT/'results'/'source_states';S.mkdir(exist_ok=True)
if len(sys.argv)>1:
 name=sys.argv[1];m=Prefix(name);states=[]
 for tok in [510,15,187,535,209]:
  obs,s=m.forward([tok],return_states=True);states.append(m.ln(s['premlp'])[0])
 file=S/(name+'.npz');assert not file.exists()
 np.savez_compressed(file,normalized_pre_mlp1=np.array(states),gate=np.stack([m.w[f'gate_proj_{i}'] for i in P['neurons']]),up=np.stack([m.w[f'up_proj_{i}'] for i in P['neurons']]),down=m.down)
 print('Saved source states',name,m.check_memory(),flush=True);raise SystemExit()
files=sorted((ROOT/'results').glob('*/float32.json'),key=lambda p:10**12 if p.parent.name=='main' else int(p.parent.name.split('-')[0][4:]))
assert len(files)==4,'Wait for all four adaptive checkpoints before this exploratory analysis.'
manifest={'label':'Exploratory post-hoc diagnostic, not preregistered','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'The four adaptively selected checkpoints with unchanged frozen tokens and five frozen single tokens; no new downloads. Crossed prefix state and selected layer-1 MLP readout weights, in the tracked native coordinate basis. Off-diagonal hybrids do not claim natural or trained model behavior.','checkpoints':[p.parent.name for p in files]}
(ROOT/'source_cross_time_addendum.json').write_text(json.dumps(manifest,indent=2))
for f in files:
 name=f.parent.name
 if not (S/(name+'.npz')).exists():subprocess.run([sys.executable,__file__,name],check=True)
names=[p.parent.name for p in files];data=[np.load(S/(n+'.npz')) for n in names];states=np.stack([d['normalized_pre_mlp1'] for d in data]);gate=np.stack([d['gate'] for d in data]);up=np.stack([d['up'] for d in data]);down=np.stack([d['down'] for d in data]);g=np.einsum('wnd,std->wstn',gate,states);u=np.einsum('wnd,std->wstn',up,states);a=g*np.exp(-np.logaddexp(0,-g))*u;c=a*down[:,None,None,:]
rows=[];errors=[]
for i,wn in enumerate(names):
 for j,sn in enumerate(names):
  for ti,tok in enumerate([510,15,187,535,209]):
   for ni,neuron in enumerate(P['neurons']):rows.append({'readout_checkpoint':wn,'prefix_checkpoint':sn,'token_id':tok,'neuron':neuron,'gate':float(g[i,j,ti,ni]),'up':float(u[i,j,ti,ni]),'activation':float(a[i,j,ti,ni]),'contribution':float(c[i,j,ti,ni])})
 ref=json.loads(files[i].read_text());expected=np.array([ref['single_tokens'][str(tok)]['contribution'][0] for tok in [510,15,187,535,209]])
 errors.append(float(np.max(abs(expected-c[i,i])/np.maximum(abs(expected),1))))
import csv
with (ROOT/'deliverables'/'source_cross_time.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest['diagonal_max_scaled_errors']=dict(zip(names,errors));manifest['diagonal_validated']=max(errors)<1e-4
assert manifest['diagonal_validated']
(ROOT/'deliverables'/'source_cross_time_audit.json').write_text(json.dumps(manifest,indent=2))
for i,n in enumerate(names):
 print(n,'The old/old, last/old, old/last, last/last',c[i,i,0,0],c[-1,i,0,0],c[i,-1,0,0],c[-1,-1,0,0])
 print(n,'period old/old, last/old, old/last, last/last',c[i,i,1,0],c[-1,i,1,0],c[i,-1,1,0],c[-1,-1,1,0])
