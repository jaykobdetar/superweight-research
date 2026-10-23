"""Offline combination of saved source states from the initial and adaptive cohorts."""
from pathlib import Path
import json,numpy as np,csv
C=Path(__file__).resolve().parents[1];B=C.parent
files=[*sorted((B/'phase2/results').glob('*/float32.json')),*sorted((B/'phase3/results').glob('*/float32.json'))];files.sort(key=lambda p:10**12 if p.parent.name=='main' else int(p.parent.name.split('-')[0][4:]));assert len(files)==12
names=[f.parent.name for f in files];data=[np.load(f.parent.parent/'source_states'/(f.parent.name+'.npz')) for f in files];h=np.stack([d['normalized_pre_mlp1'] for d in data]);g=np.einsum('wnd,std->wstn',np.stack([d['gate'] for d in data]),h);u=np.einsum('wnd,std->wstn',np.stack([d['up'] for d in data]),h);a=g*np.exp(-np.logaddexp(0,-g))*u;c=a*np.stack([d['down'] for d in data])[:,None,None,:];rows=[];errs=[];toks=[510,15,187,535,209];ns=[1710,4820,5755,8041]
for i,wn in enumerate(names):
 ref=json.loads(files[i].read_text());exp=np.array([ref['single_tokens'][str(t)]['contribution'][0] for t in toks]);errs.append(float(np.max(abs(exp-c[i,i])/np.maximum(abs(exp),1))))
 for j,sn in enumerate(names):
  for ti,t in enumerate(toks):
   for ni,n in enumerate(ns):rows.append({'readout_checkpoint':wn,'prefix_checkpoint':sn,'token_id':t,'neuron':n,'gate':float(g[i,j,ti,ni]),'up':float(u[i,j,ti,ni]),'activation':float(a[i,j,ti,ni]),'contribution':float(c[i,j,ti,ni])})
assert max(errs)<1e-4
with (C/'deliverables/source_cross_time.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(C/'deliverables/source_cross_time_audit.json').write_text(json.dumps({'diagonal_validated':True,'diagonal_max_scaled_errors':dict(zip(names,errs)),'scope':'Exploratory native-basis combinations, not natural checkpoints or training interventions. Initial eight and four adaptive points remain separate cohorts.'},indent=2))
for i in [names.index('step512000-tokens1073B'),names.index('step800000-tokens1677B')]:
 for j in [names.index('step512000-tokens1073B'),names.index('step800000-tokens1677B')]:print('Readout',names[i],'prefix',names[j],'single newline',c[i,j,2,0])
