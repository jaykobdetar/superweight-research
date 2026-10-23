"""Run all frozen conditions, incremental immutable outputs, fixed validation gate."""
from pathlib import Path
import argparse,hashlib,json,resource,time,csv
import numpy as np
from streamed_olmo import StreamedOlmo
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--limit',type=int);args=ap.parse_args()
resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
validation=json.loads((R/'results/validation_fp32.json').read_text());assert validation['complete'] and validation['all_pass'] and len(validation['rows'])==10,'Full frozen validation gate must pass before scientific runs'
protocol=json.loads((R/'protocols/factorial_protocol.json').read_text());inputs=np.load(R/'protocols/frozen_inputs.npz');model=StreamedOlmo()
O=R/'results/factorial';O.mkdir(exist_ok=True)
code_sha=hashlib.sha256((R/'streamed_olmo.py').read_bytes()).hexdigest();assert code_sha==validation['code_sha256'],'Model changed after validation'
start=time.time();records=protocol['fresh_windows'][args.start:args.start+args.limit if args.limit else None]
for item in records:
 ids=inputs[item['id']];assert hashlib.sha256(ids.tobytes()).hexdigest()==item['token_sha256'];p=item['p'];r=item['ordinary_control']
 conditions={'native':(),'mask0':(0,),'maskp':(p,),'mask0p':(0,p),'maskr':(r,),'mask0r':(0,r),'post_mask0':(0,),'post_mask0p':(0,p),'post_maskr':(r,),'post_mask0r':(0,r)}
 losses={};argmaxes={};prefix_hashes={}
 for name,sources in conditions.items():
  path=O/(item['id']+'__'+name+'.npz');meta=O/(item['id']+'__'+name+'.json')
  if path.exists():
   receipt=json.loads(meta.read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['npz_sha256']
   assert receipt['code_sha256']==code_sha and receipt['token_sha256']==item['token_sha256']
   with np.load(path) as result:losses[name]=result['loss'].copy();argmaxes[name]=result['argmax'].copy()
   prefix_hashes[name]=receipt['prefix_hashes']
   continue
  captured_values={}
  def capture_values(layer,context,probs,values):
   captured_values[f'native_V_L{layer}']=values[:,[0,p,r],:].copy()
   return context
  result=model.forward(ids,mask_sources=sources,attention_hook=capture_values if name=='native' else None,diagnostic_sources=(0,p,r),mask_query_start=p+1 if name.startswith('post_') else None,prefix_hash_end=p)
  saved={k:result[k] for k in ['loss','argmax','target_logit','log_normalizer']}
  saved.update(captured_values)
  for layer,values in result['attention'].items():
   for source,v in values.items():saved[f'attention_L{layer}_source{source}']=v
  np.savez_compressed(path,**saved)
  receipt={**item,'condition':name,'mask_sources':sources,'seconds':result['seconds'],'peak_rss':result['peak_rss'],'code_sha256':code_sha,'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prefix_hashes':result['prefix_hashes']}
  meta.write_text(json.dumps(receipt,indent=2));losses[name]=result['loss'];argmaxes[name]=result['argmax'];prefix_hashes[name]=result['prefix_hashes']
  print('CONDITION',item['id'],name,'seconds',round(result['seconds'],3),'rss_MiB',round(result['peak_rss']/2**20,1),flush=True)
  del result,saved
 for name in conditions:
  if name.startswith('post_') or name=='maskp':
   assert prefix_hashes[name]==prefix_hashes['native'],('Prefix state changed',item['id'],name)
   assert np.array_equal(losses[name][:p+1],losses['native'][:p+1]),('Pre-cut target changed',item['id'],name)
 pos=np.arange(1,len(ids));scopes={'all_scored':pos>=2,'before':(pos>=2)&(pos<p+2),'after':pos>=p+2,'early_after':(pos>=p+2)&(pos<=p+33),'late_after':pos>=p+34}
 rows=[]
 for scope,mask in scopes.items():
  if not mask.any():continue
  means={name:float(loss[mask].mean()) for name,loss in losses.items()}
  interaction=(losses['mask0p']-losses['maskp'])-(losses['mask0']-losses['native'])
  post_interaction=(losses['post_mask0p']-losses['maskp'])-(losses['post_mask0']-losses['native'])
  post_ordinary=(losses['post_mask0r']-losses['post_maskr'])-(losses['post_mask0']-losses['native'])
  ordinary=(losses['mask0r']-losses['maskr'])-(losses['mask0']-losses['native'])
  row={**item,'scope':scope,'targets':int(mask.sum()),'native_nll':means['native'],'mask0_delta':means['mask0']-means['native'],'maskp_delta':means['maskp']-means['native'],'mask0p_delta':means['mask0p']-means['native'],'maskr_delta':means['maskr']-means['native'],'mask0r_delta':means['mask0r']-means['native'],'interaction_p':float(interaction[mask].mean()),'interaction_r':float(ordinary[mask].mean()),'interaction_difference':float((interaction-ordinary)[mask].mean())}
  row.update(post_mask0_delta=float((losses['post_mask0']-losses['native'])[mask].mean()),post_mask0p_delta=float((losses['post_mask0p']-losses['native'])[mask].mean()),post_interaction_p=float(post_interaction[mask].mean()),post_interaction_r=float(post_ordinary[mask].mean()),post_interaction_difference=float((post_interaction-post_ordinary)[mask].mean()))
  for name in conditions:
   row[name+'_argmax_agreement']=float((argmaxes[name][mask]==argmaxes['native'][mask]).mean())
  rows.append(row)
 (O/(item['id']+'__summary.json')).write_text(json.dumps(rows,indent=2))
 print('WINDOW',json.dumps(next(x for x in rows if x['scope']=='after')),flush=True)
print('RUN_FINISHED','seconds',time.time()-start,flush=True)
