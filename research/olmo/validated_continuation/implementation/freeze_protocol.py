"""Outcome-blind fresh input panel and archived validation target selection."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P));from reconstruct_tokenizer import load_window
M=json.loads((P/'input_manifest.json').read_text());old=json.loads((P/'inputs/recovered_window_manifest.json').read_text())
seed=20261023;rng=np.random.default_rng(seed);records=[];arrays={}
for corpus,count in [('dubois',6),('hobbes',5),('optimizers',5),('wiki_validation',8)]:
 name='wiki_validation_partial_ids.npy' if corpus=='wiki_validation' else corpus+'_ids.npy'
 ids=np.load(P/'inputs'/name);selected=[]
 excluded=[(x['start'],x['start']+x['length']) for x in old if x['corpus']==corpus]
 # Exclude first/last5% of each public stream to reduce headers/footers.
 candidates=np.arange(int(len(ids)*.05),min(int(len(ids)*.95),len(ids)-256),dtype=np.int64)
 rng.shuffle(candidates)
 for s in candidates:
  s=int(s)
  if any(s<b and s+256>a for a,b in excluded+[(t,t+256) for t in selected]):continue
  x=ids[s:s+256];ps=np.flatnonzero(x==187)
  if corpus=='wiki_validation':
   if len(ps):continue
   p=int(records[len(selected)%16]['p'])
  else:
   if not len(ps) or not 16<=ps[0]<=96:continue
   p=int(ps[0])
  assert x[0]!=187 and x[p-1]!=187
  key=f'{corpus}_{len(selected):02d}';arrays[key]=x.copy();selected.append(s)
  records.append({'id':key,'corpus':corpus,'start':s,'length':256,'p':p,'ordinary_control':p-1,'p_is_newline':bool(x[p]==187),'source0_token':int(x[0]),'token_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'primary_target_start':p+2,'primary_target_end_inclusive':255})
  if len(selected)==count:break
 assert len(selected)==count,(corpus,len(selected))
validation=[];d=P/'repo/results/pulse_continuation/confirmation_run';ix=json.loads((d/'per_token_index.json').read_text());archive=np.load(d/'per_token_numeric.npz')
for corpus,w in [('wiki_validation',0),('dubois',1),('hobbes',1),('optimizers',0),('dubois',0)]:
 start=M['confirmation'][corpus]['evaluation_starts'][w];ids=load_window(corpus,start)
 for trial in ['clean','remove_source']:
  cond=f'confirm_{corpus}_natural__{trial}';r=next(x for x in ix if x['condition']==cond and x['window']==w)
  assert hashlib.sha256(ids.tobytes()).hexdigest()==r['token_sha256']
  key=f'validation_{corpus}_{w}_{trial}';arrays[key+'_ids']=ids[:128].copy();arrays[key+'_archived_loss']=archive[str(r['index'])+'_loss'][:126].copy()
  validation.append({'id':key,'corpus':corpus,'window':w,'trial':trial,'length':128,'target_positions':[2,127],'archive_full_token_sha256':r['token_sha256'],'prefix_token_sha256':hashlib.sha256(ids[:128].tobytes()).hexdigest()})
protocol={'created_unix':time.time(),'selection_seed':seed,'selection_uses_model_outputs':False,'fresh_windows':records,'validation':validation,'conditions':['native','mask0','maskp','mask0p','maskr','mask0r'],'mask_layers':list(range(2,16)),'mask_queries':'strictly later than each masked source','primary_estimand':'mean_paired_target_NLL((mask0p-maskp)-(mask0-native)) for targets p+2..255, window-weighted; positive means p substitutes for source0','negative_control_estimand':'same with ordinary r=p-1; compare interaction difference on identical p+2..255 targets','no_newline_control':'8 Wiki windows with no token187; pseudo-p taken from treatment positions','subsets':['before_p_broadcast:targets2..p+1','after_p_broadcast:targets p+2..255','early_after: p+2..min(p+33,255)','late_after: max(p+34,2)..255'],'validation_gate':{'mean_abs_window_NLL_difference_max':0.02,'per_target_MAE_max':0.05,'per_target_p95_abs_difference_max':0.15,'max_absolute_target_difference_reported':True,'note':'FP32 is not bitwise FP16/SDPA replication. Both clean and source-removal effects must agree; failed thresholds trigger diagnosis, not scientific interpretation.'},'resources':{'rss_target_bytes':1073741824,'rss_hard_ceiling_bytes':4000000000,'blas_threads':2,'sequence_length':256,'simultaneous_heavy_jobs':1},'stop_rules':['Validate native and source-removal per-target archived outputs before interpreting fresh model results.','Record all selected windows, including null/opposite-sign effects; no output-based selection.','Any resource-cap or acquisition denial stops that operation and is archived.']}
p=R/'protocols/factorial_protocol.json';assert not p.exists(),'Frozen protocol already exists'
p.write_text(json.dumps(protocol,indent=2));np.savez_compressed(R/'protocols/frozen_inputs.npz',**arrays)
print('Frozen',len(records),'fresh windows and',len(validation),'128-token validation comparisons')
print([(x['corpus'],x['p']) for x in records]);print('protocol_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
