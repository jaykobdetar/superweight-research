"""Post-hoc native-coordinate readout/state compatibility, no new downloads.
Same frozen probes and interventions. Hybrid values are not natural checkpoints.
"""
from pathlib import Path
import sys,json,datetime,os,subprocess,hashlib,csv
import numpy as np
C=Path(__file__).resolve().parents[1];B=C.parent;D=C/'results/query_states';D.mkdir(parents=True,exist_ok=True)
A=C/'query_compatibility_addendum.json'
if '--freeze' in sys.argv:
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':'Exploratory after observing the 800k functional result, before computing the hybrid diagnostic','checkpoints':[{'phase':'phase2','name':'step120000-tokens251B'},{'phase':'phase3','name':'step256000-tokens536B'},{'phase':'phase3','name':'step512000-tokens1073B'},{'phase':'phase3','name':'step800000-tokens1677B'},{'phase':'phase3','name':'step1200000-tokens2515B'},{'phase':'phase2','name':'main'}],'conditions':['newline_U','newline_R','newline_R_edge_H14','newline_R_edge_H0'],'scope':'Reuse four original fixed panels and existing prefix tensors, one checkpoint at a time. Cross normalized pre-MLP1 query states with selected gate/up/down rows. Native-coordinate diagnostic, not training intervention or natural intermediate model. No new prompt or download.'}
 assert not A.exists();A.write_text(json.dumps(plan,indent=2));print('Frozen diagnostic',hashlib.sha256(A.read_bytes()).hexdigest());raise SystemExit()
plan=json.loads(A.read_text())
if len(sys.argv)>1:
 phase,name=sys.argv[1:3];sys.path.insert(0,str(B/phase/'checks'));from prefix_model import Prefix,P
 m=Prefix(name);hs=[];labels=[];maxerr=0;ref=json.loads((B/phase/'results'/name/'float32.json').read_text());by={(o['panel'],o['condition']):o for o in ref['observations']}
 for panel in P['panels']:
  ids=[209 if i in (187,535) else i for i in panel['token_ids']];ids[0]=510;ids[128]=187
  for cond in plan['conditions']:
   x=ids.copy();x[64]=209 if cond=='newline_U' else 187;kw={}
   if cond=='newline_R_edge_H14':kw={'edges':[(14,128,64)]}
   if cond=='newline_R_edge_H0':kw={'edges':[(0,128,64)]}
   o,s=m.forward(x,return_states=True,**kw);h=m.ln(s['premlp'])[-1];hs.append(h);labels.append([panel['name'],cond]);a=np.array(o['contribution'][-1]);b=np.array(by[panel['name'],cond]['contribution'][-1]);maxerr=max(maxerr,float(np.max(abs(a-b)/np.maximum(abs(b),1))))
 assert maxerr<1e-4
 f=D/(name+'.npz');assert not f.exists();np.savez_compressed(f,normalized_pre_mlp1=np.stack(hs),gate=np.stack([m.w[f'gate_proj_{i}'] for i in P['neurons']]),up=np.stack([m.w[f'up_proj_{i}'] for i in P['neurons']]),down=m.down)
 (D/(name+'.json')).write_text(json.dumps({'phase':phase,'name':name,'commit':ref['revision'],'labels':labels,'neuron_ids':P['neurons'],'recomputed_natural_result_max_scaled_error':maxerr,'rss_bytes':m.check_memory()},indent=2));print('Saved',name,'RSS',m.check_memory(),flush=True);raise SystemExit()
for ck in plan['checkpoints']:
 if not (D/(ck['name']+'.npz')).exists():subprocess.run([sys.executable,__file__,ck['phase'],ck['name']],check=True)
names=[ck['name'] for ck in plan['checkpoints']];arrays=[np.load(D/(n+'.npz')) for n in names];metas=[json.loads((D/(n+'.json')).read_text()) for n in names];assert all(m['labels']==metas[0]['labels'] for m in metas)
h=np.stack([x['normalized_pre_mlp1'] for x in arrays]);g=np.einsum('wnd,std->wstn',np.stack([x['gate'] for x in arrays]),h);u=np.einsum('wnd,std->wstn',np.stack([x['up'] for x in arrays]),h);act=g*np.exp(-np.logaddexp(0,-g))*u;c=act*np.stack([x['down'] for x in arrays])[:,None,None,:];rows=[];errs=[]
for i,ck in enumerate(plan['checkpoints']):
 ref=json.loads((B/ck['phase']/'results'/ck['name']/'float32.json').read_text());by={(o['panel'],o['condition']):o for o in ref['observations']};expected=np.array([by[tuple(k)]['contribution'][-1] for k in metas[i]['labels']]);errs.append(float(np.max(abs(expected-c[i,i])/np.maximum(abs(expected),1))))
 for j,prefix in enumerate(names):
  for k,(panel,cond) in enumerate(metas[0]['labels']):
   for n,neuron in enumerate(metas[0]['neuron_ids']):rows.append({'readout_checkpoint':names[i],'prefix_checkpoint':prefix,'panel':panel,'condition':cond,'neuron':neuron,'gate':float(g[i,j,k,n]),'up':float(u[i,j,k,n]),'activation':float(act[i,j,k,n]),'contribution':float(c[i,j,k,n])})
assert max(errs)<1e-4
with (C/'deliverables/query_cross_time.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
audit={'diagonal_max_scaled_errors':dict(zip(names,errs)),'diagonal_validated':True,'max_rss_bytes':max(m['rss_bytes'] for m in metas),'scope':plan['scope'],'addendum_sha256':hashlib.sha256(A.read_bytes()).hexdigest()};(C/'deliverables/query_cross_time_audit.json').write_text(json.dumps(audit,indent=2))
for i in [2,3,5]:
 for j in [2,3,5]:
  vals=c[i,j,:,0];print('Readout',names[i],'prefix',names[j],{cond:float(np.median([vals[k] for k,l in enumerate(metas[0]['labels']) if l[1]==cond])) for cond in plan['conditions']})
