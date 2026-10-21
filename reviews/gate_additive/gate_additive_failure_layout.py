"""Measured production failure helper on full-size reachable payload classes.
No model, tokenizer or GPU. Independent-storage upper envelopes retain every
production field; no impossible gate+reconstruction+singleton+suffix union.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1')
import json,resource,sys,tempfile,time,io,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(1)
import gate_additive_failure as F
from run_transfer import sha
L=range(4,12);V=50257;Q=511

def gates():return {l:{k:torch.zeros(12,Q)for k in('b','mass','target_b','target_mass')}for l in L}
def contexts():return {l:{k:torch.zeros(1,128,768)for k in('current','delivered')}for l in L}
def norms():return {l:{k:torch.zeros(Q,dtype=torch.float32 if k=='s'else torch.float64)for k in('s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm')}for l in L}
def source():return {'kv':{l:{k:torch.zeros(768)for k in('k','v')}for l in L},'q':{l:torch.zeros(12,64)for l in L}}
def metrics(n=511):return {'loss':torch.ones(min(n,127)if n==128 else n),'kl':torch.zeros(n),'kl_p':torch.zeros(n),'pred':torch.zeros(n,dtype=torch.long),'agreement_p':torch.ones(n,dtype=torch.bool),'final_hidden_sha256':'f'*64}
def logits(n=511,dtype=torch.float32):
 a=torch.zeros(n,V,dtype=dtype);a[min(n-2,300),-1]=1
 return a

def bundle(n=511):return {'hidden':torch.zeros(1,512,768),'gates':gates(),'source':source(),'prefix_contexts':contexts(),
 'current_contexts':{l:torch.zeros(1,512,768)for l in L},'localization_norms':norms(),'z':metrics(n)}
def accumulator():return SimpleNamespace(layers=list(L),loss_p=torch.ones(Q,dtype=torch.float64),
 single_losses={l:torch.ones(Q,dtype=torch.float64)for l in L},targets=torch.zeros(Q,dtype=torch.long),
 additive=logits(dtype=torch.float64),base=logits())
def common():
 ident={'corpus':'optimizers_new','window':3,'identity':'bos','length':512,'trial':'orthogonal_control_all','phase':'suffix_control'}
 return {'last_forward_identity':dict(ident),'active_record_target':dict(ident),'active_check_target':dict(ident),
 'current_tensor_identity':dict(ident),'calibration_sha256':'f'*64,'direction_sha256':'f'*64,
 'gate':None,'reconstruction':None,'current_gates':gates(),'current_contexts':contexts(),
 'current_source':source()['kv'],'current_hidden':torch.zeros(1,512,768),'current_localization_norms':norms(),
 'readout_check':dict(ident),'accumulator':F.accumulator_state(accumulator()),'singleton_failure':None,'suffix_failure':None,
 'control_computation_metrics':None,'control_computation_logits':None,'pending_reference_logits':F.logit_pair(logits(128)),
 'pending_record_metrics':None}
def qkv():return {**{k:torch.zeros(1,12,512,64)for k in('q','k','v')},'mask':torch.zeros(1,1,512,512)}
def gate_error():return {'layer':11,'mode':'orthogonal_control_all','calibration_sha256':'f'*64,'direction_sha256':'f'*64,
 'state':gates()[11],'check':{'finite':False},'norms':norms()[11],'raw_inputs':qkv(),
 'stable_mixture':torch.zeros(12,511,64),'actual_context':torch.zeros(1,512,768),'delivered_context':torch.zeros(1,512,768)}
def reconstruction():return {'layer':11,**qkv(),'actual_context':torch.zeros(1,512,768),
 'expected_context':torch.zeros(1,512,768),'max_abs_error':1.,'threshold':1e-4}

started=time.time();results={};source_hashes={n:sha(ROOT/n)for n in('gpt2_gate_additive.py','gate_additive_failure.py')}
with tempfile.TemporaryDirectory(prefix='gate-additive-failure-layout-')as td:
 p=Path(td)
 for case in('gate_forward_accumulator','reconstruction_forward_accumulator','singleton_record_accumulator',
              'suffix_comparison_accumulator','suffix_metrics_accumulator','additive_readout_accumulator'):
  payload=common()
  if case=='gate_forward_accumulator':payload['gate']=gate_error()
  elif case=='reconstruction_forward_accumulator':payload['reconstruction']=reconstruction()
  elif case=='singleton_record_accumulator':
   a,b=bundle(),bundle();payload['singleton_failure']=F.singleton_state({'check':{'layer':11},'reference':a,'trial':b});payload['pending_record_metrics']=metrics();del a,b
  elif case=='suffix_comparison_accumulator':
   a,b=bundle(),bundle(128);left,right=logits(128),logits(128);right[64,123]=2
   payload['suffix_failure']=F.suffix_state({'check':{'logits_exact':False},'reference':a,'control':b,
    'reference_logits':left,'control_logits':right,'original_tokens':torch.zeros(512,dtype=torch.long),'control_tokens':torch.ones(512,dtype=torch.long)})
   assert 64 in payload['suffix_failure']['logits']['query_rows'].tolist()
   assert payload['suffix_failure']['logits']['per_query_max_abs'][64]==2
   del a,b,left,right
  elif case=='suffix_metrics_accumulator':
   bad=logits(128);bad[64,123]=float('nan');payload['control_computation_metrics']=metrics(128);payload['control_computation_logits']=F.logit_pair(bad)
   assert 64 in payload['control_computation_logits']['query_rows'].tolist() and payload['control_computation_logits']['nonfinite_query_indices']==[64]
   del bad
  elif case=='additive_readout_accumulator':payload['pending_reference_logits']=None
  dest=p/(case+'.pt');n=F.save_capsule(dest,payload);assert dest.stat().st_size==n
  loaded=torch.load(dest,weights_only=True,map_location='cpu',mmap=True)
  assert loaded.keys()==payload.keys()and loaded['current_hidden'].shape==(1,512,768)
  # Actual helper has no lossy conversions: selected matrices preserve dtype.
  assert loaded['accumulator']['logits']['left'].dtype==torch.float64
  assert loaded['accumulator']['logits']['right'].dtype==torch.float32
  assert len(loaded['accumulator']['logits']['per_query_max_abs'])==511
  results[case]={'serialized_bytes':n,'cap_bytes':F.CAPSULE_LIMIT,'margin_bytes':F.CAPSULE_LIMIT-n,'loaded_and_shape_checked':True}
  del loaded,payload;dest.unlink()
 # Exercise rejection before file write. This is intentionally not a valid production payload.
 reject=p/'oversized.pt';bad={'oversized':torch.zeros(F.CAPSULE_LIMIT//4+1)}
 try:F.save_capsule(reject,bad)
 except AssertionError as err:
  assert str(err).startswith("('Failure capsule exceeds frozen limit'");assert not reject.exists();rejection=str(err)
 else:raise AssertionError('oversized capsule accepted')
 del bad
result={'status':'passed','actual_model_forwards':0,'gpu_used':False,'source_hashes':source_hashes,'cases':results,
 'largest_serialized_bytes':max(r['serialized_bytes']for r in results.values()),'capsule_limit_bytes':F.CAPSULE_LIMIT,
 'oversize_rejected_before_write':True,'oversize_message':rejection,'max_error_row_preserved':True,'first_nonfinite_row_preserved':True,
 'scope':'Reachable failure classes with full independent-storage upper envelopes, maximum eight-layer accumulated metrics and pending prefix reference. Gate/reconstruction/singleton/suffix failures are not combined because each raises immediately. Some upper envelopes retain completed fields that would be absent in an earlier-layer failure; no field is omitted for size.',
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.time()-started,
 'temporary_files_removed':not p.exists(),'source_hashes_unchanged':all(sha(ROOT/n)==h for n,h in source_hashes.items())}
assert result['peak_rss_kib']<3*1024**2 and result['source_hashes_unchanged']
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
