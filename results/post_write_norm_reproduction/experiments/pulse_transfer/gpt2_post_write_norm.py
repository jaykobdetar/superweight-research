"""Prospective T20 segmented runner. HOLD: literal reviewed launch freeze.

Imports frozen T19 scoring/forward helpers; does not execute the T19 study.
No weights may load until new input, hook, source and full writer reviews pass.
"""
import argparse,contextlib,copy,hashlib,json,platform,resource,shutil,subprocess,time
from importlib.metadata import version
from pathlib import Path
import torch
from gate_native_value import fingerprint
from gate_write_norm import GateWriteNorm,runtime_behavior
from gpt2_gate_native_value import metric_rows,model_forward,F_DIGEST,METADATA_PINS
from post_write_norm_storage import (ARMS,EvidenceWriter,NORMAL_LIMIT,FAILURE_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,READOUT_ROWS,
 byte_exact,nested_fp,summarize_diag,empty_payload,trim_gate,trim_gate_norm,failure_bundle,logit_rows,source_closure,sha,json_write,read_npz)

ROOT=Path(__file__).resolve().parent
PROPOSAL_SHA='e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff'
INPUT_ADDENDUM_SHA='2f458ad30f0d92bfd1a1b031e0c0f8e5bcd775a90208d1333ebad2b81216ce1c'
RESOURCE_ADDENDUM_SHA='adc8aadae813549fb2a8c808df8f8c84a4635fde4eca6b120dd25cd18658c213'
NATIVE_HOOK_SHA='b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd'
T18_MANIFEST_SHA='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060'
BOUNDARIES=('000','111')
SEGMENTS={'A':('wiki_validation','utilitarianism'),'B':('ethics','optimizers_new')}
CORPORA=tuple(c for names in SEGMENTS.values() for c in names)
SNAPSHOT_INPUT=('wiki_validation',0,'natural')
INPUT_HASHES={'data_manifest.json': '7f0e18f2015b0ccad19cdf3be157ba5fa8d575eaeebb68550f570d9ff0a037cc', 'exclusion_inventory.json': '29fda514236794304fb67d092c278ec4db930d4c6df199808773d96b64a5ca80', 'historical_manifest_scan.json': 'db372ba1f73902c496a5f49bc5360aef797f1e6ce58216e92105efb2a5968230', 'historical_manifest_scan_review.json': 'db70e141b3ea6c663ae204e5ba433aec8b9c5d39c58949a38a24638bf55900cb', 'preflight.json': '7340e7dfd8223dce27bd2aa4e0ec4c4f3af62e7d4f9472b029a31a9722c091c5', 'start_identity_summary.json': 'b876051f9f9a9b92696a58b91e83b2d68be9d80aa6c85daa1e91ff3da92b6661', 'tokens.pt': '42c6b1a061df53e018465997670f89f95e27612ce36a42b3d81658791f7101b1'}
STORAGE_READY=True
GROUP_ALLOWANCES={'ordinary':2221085,'snapshot':100795903}
PINNED_VERSIONS={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'}

def capture(probe,hidden,metrics):
 probe.finish()
 return {'metrics':metrics,'hidden':hidden,'source':copy.deepcopy(probe.source_states),'gates':copy.deepcopy(probe.gate_cache),
  'gate_norms':copy.deepcopy(probe.localization_norms),'write_norms':copy.deepcopy(probe.write_norms),
  'snapshots':copy.deepcopy(probe.snapshots),'write_snapshot':copy.deepcopy(probe.write_snapshot),
  'prefix_contexts':copy.deepcopy(probe.prefix_contexts),'prefix_writes':probe.prefix_writes.clone(),
  'returned_write':probe._returned_write.clone(),
  'operations':{'native_diag':summarize_diag(probe.diag),'native_gate_checks':copy.deepcopy(probe.gate_checks),
   'projection_records':copy.deepcopy(probe.projection_records),'context_records':copy.deepcopy(probe.context_records),
   'mask_records':copy.deepcopy(probe.mask_records),'source_fingerprints':nested_fp(probe.source_states),
   'gate_fingerprints':nested_fp(probe.gate_cache),'prefix_context_fingerprints':nested_fp(probe.prefix_contexts),
   'write_checks':copy.deepcopy(probe.write_checks),'write_cache_key':copy.deepcopy(probe.cache_key),
   'projection_checks':probe.projection_checks,'context_checks':probe.context_checks,'query_checks':probe.query_checks}}

def pack(payload,arm,bundle,logits,tokens,*,suffix=False):
 metrics=dict(bundle['metrics']);metrics['final_hidden_sha256']=fingerprint(bundle['hidden'])['sha256']
 if suffix:
  payload['suffix'][arm]=metrics;payload['suffix_gates'][arm]=trim_gate(bundle['gates']);payload['suffix_gate_norms'][arm]=trim_gate_norm(bundle['gate_norms'])
  payload['suffix_write_norms'][arm]=bundle['write_norms'];payload['suffix_operations'][arm]=bundle['operations']
 else:
  for key,source in (('metrics',metrics),('gates',bundle['gates']),('gate_norms',bundle['gate_norms']),('write_norms',bundle['write_norms']),('operations',bundle['operations'])):payload[key][arm]=source
  if tuple(payload['metadata'][k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT and arm in ARMS:
   # write_snapshot already owns the complete native incoming QKV/mask/gate/
   # context reference. Do not serialize a second duplicate native snapshot.
   assert byte_exact(bundle['write_snapshot']['incoming_reference'],bundle['snapshots'][10])
   payload['snapshots'][arm]=bundle['write_snapshot']
   payload['readouts'][arm]={'hidden':bundle['hidden'],'query_rows':torch.tensor(READOUT_ROWS),
    'logits':logits[list(READOUT_ROWS)].detach().cpu().clone(),'targets':tokens[[q+1 for q in READOUT_ROWS]],
    'token_sha256':fingerprint(tokens)['sha256'],'hidden_fingerprint':fingerprint(bundle['hidden'])}

def identity_check(left,right,ll,rr,kind):
 result={'kind':kind,'logits_exact':byte_exact(ll,rr),'hidden_exact':byte_exact(left['hidden'],right['hidden'])}
 for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):result[key+'_exact']=byte_exact(left['metrics'][key],right['metrics'][key])
 if left['returned_write'] is not None and right['returned_write'] is not None:result['write_exact']=byte_exact(left['returned_write'],right['returned_write'])
 return result

def suffix_check(left,right,ll,rr):
 check={'kind':'suffix','logits_exact':byte_exact(ll[:128],rr[:128]),'hidden_exact':byte_exact(left['hidden'][:,:128],right['hidden'][:,:128]),
  'source_exact':byte_exact(left['source'],right['source']),'gate_exact':byte_exact(trim_gate(left['gates']),trim_gate(right['gates'])),
  'gate_norm_exact':byte_exact(trim_gate_norm(left['gate_norms']),trim_gate_norm(right['gate_norms'])),
  'write_norm_exact':byte_exact({k:v[:128] for k,v in left['write_norms'].items()},{k:v[:128] for k,v in right['write_norms'].items()}),
  'context_exact':byte_exact(left['prefix_contexts'],right['prefix_contexts']),
  'write_exact':byte_exact(left['prefix_writes'],right['prefix_writes']),
  'mask_exact':byte_exact(left['operations']['mask_records'],right['operations']['mask_records'])}
 for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
  n=127 if key=='loss' else 128;check[key+'_exact']=byte_exact(left['metrics'][key][:n],right['metrics'][key])
 return check

def passed(check):return all(v for k,v in check.items() if k.endswith(('_exact','_passed')) and isinstance(v,bool))

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('config','inputs','calibration-reference','out'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--segment',choices=tuple(SEGMENTS),required=True);ap.add_argument('--previous-segment',type=Path);a=ap.parse_args()
 assert (a.previous_segment is None)==(a.segment=='A'),'B requires completed A predecessor; A accepts no predecessor'
 for source in (a.inputs,a.calibration_reference,*(() if a.previous_segment is None else (a.previous_segment,))):assert not a.out.resolve().is_relative_to(source.resolve()),'Output inside immutable input/reference/predecessor'
 writer=EvidenceWriter(a.out,outstanding_other_caps=NORMAL_LIMIT*2)
 started=time.time();model=probe=identity_probe=None;attempts=returned_count=processed=main_count=suffix_count=anchor_count=0
 active={};last_attempt={};last_returned={};last_processed={};tokens=None;latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started';payload=None;group_name=None;group_committed=True
 pair_left=pair_right=pair_logits=None
 meta={'study':'post_write_norm','status':'preparing','phase':'setup','segment':a.segment,'execution_type':'planned_segment',
  'model_loaded':False,'model_load_attempted':False,'model_object_returned':False,'calibration_forwards':0,'normal_limit_bytes':NORMAL_LIMIT,'failure_limit_bytes':FAILURE_LIMIT,'disk_reserve_bytes':DISK_RESERVE,
  'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},'boundaries':list(BOUNDARIES),'arms':list(ARMS),
  'expected_model_returns':388,'expected_boundary_shards':32,'started_utc':time.strftime('%FT%TZ',time.gmtime())}
 def progress():return {'phase':meta['phase'],'forward_attempts':attempts,'forward_passes':returned_count,'fully_processed_forwards':processed,
  'evaluation_forwards':main_count,'suffix_forwards':suffix_count,'anchor_forwards':anchor_count,'active':active,
  'last_attempted':last_attempt,'last_returned':last_returned,'last_processed':last_processed}
 def resources():return {'wall_seconds':time.time()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
  'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}
 def resources_guard(stage):
  observed=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  assert observed<=3*1024**2,('CPU peak RSS budget',stage,observed,3*1024**2)
 def checkpoint(*,final=False,failure=False):
  meta.update(progress(),**resources())
  if not failure:resources_guard('checkpoint')
  writer.checkpoint(progress(),manifest=meta,final=final,failure=failure)
 def check(value,left=None,right=None,ll=None,rr=None):
  nonlocal pair_left,pair_right,pair_logits
  payload['checks'].append(value)
  if not passed(value):
   pair_left,pair_right=left,right;pair_logits=logit_rows(ll,rr) if ll is not None else None
   raise AssertionError(value)
 def ev(x,boundary,arm,c_logits=None,b_logits=None,*,suffix=False,unhooked=False,anchor=False,observe_only=False):
  nonlocal attempts,returned_count,processed,last_attempt,last_returned,last_processed,tokens,latest_logits,latest_hidden,latest_metrics,scoring_status,main_count,suffix_count,anchor_count
  assert x.dtype==torch.int64 and x.shape==(512,);tokens=x.clone();latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started'
  writer.journal({'event':'forward_intent','next_attempt':attempts+1,'identity':active,'token_fingerprint':fingerprint(x)})
  if not unhooked:probe.reset(boundary,arm,variant='suffix' if suffix else 'main',token_sha256=fingerprint(x)['sha256'],snapshot=True,observe_only=observe_only)
  def invoked():
   nonlocal attempts,last_attempt
   writer.journal({'event':'forward_attempt','attempt':attempts+1,'identity':active});attempts+=1;last_attempt=dict(active)
  def returned():
   nonlocal returned_count,last_returned,main_count,suffix_count,anchor_count
   returned_count+=1;last_returned=dict(active)
   if anchor:anchor_count+=1
   elif suffix:suffix_count+=1
   else:main_count+=1
   writer.journal({'event':'model_returned','attempt':attempts,'completed':returned_count,'identity':active})
  profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if returned_count==0 else contextlib.nullcontext()
  with torch.inference_mode():
   with profile as profiler:output=model_forward(model,x,returned,invoked)
   if returned_count==1:
    events=[{'operator':e.key,'count':e.count} for e in profiler.key_averages() if 'scaled_dot_product' in e.key];meta['sdpa_profiler_events']=events
    assert any('_scaled_dot_product' in e['operator'] for e in events),'Expected observed SDPA backend'
   latest_logits=output.logits[0].detach().clone();latest_hidden=output.hidden_states[-1].detach().cpu().clone();del output
   scoring_status='scoring_started_not_completed';metrics=metric_rows(latest_logits,x,c_logits,b_logits,suffix=suffix);latest_metrics=metrics;scoring_status='completed'
  if unhooked:bundle={'metrics':metrics,'hidden':latest_hidden,'source':{},'gates':{},'gate_norms':{},'write_norms':{},'snapshots':{},'write_snapshot':{},'prefix_contexts':{},'prefix_writes':None,'returned_write':None,'operations':{}}
  else:bundle=capture(probe,latest_hidden,metrics)
  resources_guard('after_capture_before_processed')
  processed+=1;last_processed=dict(active);writer.journal({'event':'forward_completed','attempt':attempts,'completed':returned_count,'processed':processed,'identity':active,'hidden_fingerprint':fingerprint(latest_hidden),'logits_fingerprint':fingerprint(latest_logits)})
  return bundle,latest_logits
 try:
  checkpoint();assert INPUT_HASHES and STORAGE_READY and GROUP_ALLOWANCES,'Launch blocked: independent T20 input/storage/source freeze incomplete'
  writer.guard('complete_planned_caps_before_model')
  assert sha(ROOT/'GPT2_POST_WRITE_NORM_MATCHING_PROPOSAL.md')==PROPOSAL_SHA and sha(ROOT/'GPT2_POST_WRITE_NORM_MATCHING_INPUT_ADDENDUM.md')==INPUT_ADDENDUM_SHA
  assert sha(ROOT/'GPT2_POST_WRITE_NORM_MATCHING_RESOURCE_ADDENDUM.md')==RESOURCE_ADDENDUM_SHA
  assert sha(ROOT/'gate_native_value.py')==NATIVE_HOOK_SHA
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items())
  pre=json.loads((a.inputs/'preflight.json').read_text());assert pre['status']=='passed' and pre['model_forwards']==0 and pre['evaluation_passages']==16 and pre['evaluation_inputs']==32
  dm=json.loads((a.inputs/'data_manifest.json').read_text());items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu')
  assert len(items)==16 and {(i['corpus'],i['window']) for i in items}=={(c,w) for c in CORPORA for w in range(4)}
  for item in items:
   record=next(r for r in dm['passages'] if (r['corpus'],r['window'])==(item['corpus'],item['window']))
   assert item['tokens'].dtype==torch.int64 and item['tokens'].shape==(512,) and fingerprint(item['tokens'])['sha256']==record['window_token_sha256']
   bos=item['tokens'].clone();bos[0]=dm['literals']['start_special']['id'];assert fingerprint(bos)['sha256']==record['bos_window_token_sha256']
  for name,h in INPUT_HASHES.items():shutil.copyfile(a.inputs/name,a.out/name);assert sha(a.out/name)==h
  ref=a.calibration_reference;assert sha(ref/'manifest.json')==dm['calibration_reference_manifest_sha256']==T18_MANIFEST_SHA
  refmeta=json.loads((ref/'manifest.json').read_text());assert refmeta['status']=='completed' and refmeta['calibration_map_digests']['F']==F_DIGEST
  for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):
   if name!='manifest.json':assert sha(ref/name)==dm['calibration_reference_files'][name]
  (a.out/'calibration_reference').mkdir(exist_ok=True)
  for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):shutil.copyfile(ref/name,a.out/'calibration_reference'/name);assert sha(a.out/'calibration_reference'/name)==sha(ref/name)
  maps=read_npz(ref/'calibration_maps.npz');directions=read_npz(ref/'calibration_directions.npz')
  from gpt2_gate_subspace import maps_digest
  from run_transfer import modelhash
  from native_prefix_diagnostic import precision_settings
  assert maps_digest(maps['F'])==F_DIGEST;alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
  closure=source_closure(ROOT,['gpt2_post_write_norm.py','gate_write_norm.py','post_write_norm_storage.py','post_write_norm_inputs.py','post_write_norm_selector.py','analyze_post_write_norm.py','GPT2_POST_WRITE_NORM_MATCHING_PROPOSAL.md','GPT2_POST_WRITE_NORM_MATCHING_INPUT_ADDENDUM.md','GPT2_POST_WRITE_NORM_MATCHING_RESOURCE_ADDENDUM.md','gpt2_layer_writes.py','merge_post_write_norm.py'])
  for name,h in closure.items():shutil.copyfile(ROOT/name,a.out/'frozen_sources'/name);assert sha(a.out/'frozen_sources'/name)==h
  cfg=json.loads(a.config.read_text())
  for name,h in METADATA_PINS.items():
   assert sha(ROOT/name)==h;dest=a.out/'metadata_dependencies'/name.removeprefix('../').replace('/','__');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);assert sha(dest)==h
  old=json.loads((ROOT/'gpt2_run/manifest.json').read_text());spec=json.loads((ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json').read_text())['gpt2'];configured=cfg['models']['gpt2'];modelpath=Path(configured['path'])
  assert configured['model_id']==old['model_id']==refmeta['model_id']==spec['model_id'] and configured['revision']==old['revision']==refmeta['revision']==spec['revision']
  for name,entry in spec['checkpoint_files'].items():assert sha(modelpath/name)==entry['sha256']
  torch.set_num_threads(4);torch.manual_seed(20261021);settings=precision_settings();assert settings==refmeta['precision_settings_before']
  runtime={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda};assert runtime==PINNED_VERSIONS
  arithmetic_runtime=runtime_behavior();freeze=hashlib.sha256(json.dumps({'sources':closure,'inputs':INPUT_HASHES,'metadata':METADATA_PINS,'runtime':runtime,'precision':settings},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  meta.update(source_hashes=closure,input_dependencies=INPUT_HASHES,metadata_dependencies=METADATA_PINS,freeze_digest=freeze,model_id=old['model_id'],revision=old['revision'],dtype='float32',attention='sdpa',runtime_versions=runtime,precision_settings_before=settings,arithmetic_runtime=arithmetic_runtime,calibration_map_digest=F_DIGEST,calibration_fit=False,physical_execution_lengths=[512])
  previous_binding=None
  if a.segment=='B':
   from merge_post_write_norm import validate,EQUAL_FIELDS
   previous_manifest,previous_inventory=validate(a.previous_segment,'A')
   for name in EQUAL_FIELDS:assert previous_manifest[name]==meta[name],('A predecessor differs',name)
   assert previous_manifest['model_hash_before']==previous_manifest['model_hash_after']==old['model_hash_before'],'A predecessor parameter identity differs from pinned checkpoint'
   previous_binding={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
   meta['previous_segment_A']=previous_binding
   (a.out/'previous_segment_A').mkdir()
   for name in ('manifest.json','numeric_inventory.json'):shutil.copyfile(a.previous_segment/name,a.out/'previous_segment_A'/name);assert sha(a.out/'previous_segment_A'/name)==previous_binding[name.replace('.json','_sha256')]
   # Only a verified complete predecessor releases its unused future cap.
   writer.outstanding_other_caps=NORMAL_LIMIT
  checkpoint()
  resources_guard('before_model_load')
  meta['host_mem_available_before_model_kib']=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
  meta['host_mem_available_minimum_bytes']=4*1024**3
  assert meta['host_mem_available_before_model_kib']*1024>=meta['host_mem_available_minimum_bytes'],'Pre-model host headroom below4GiB'
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu,('Competing GPU computation',gpu)
  meta['gpu_compute_processes_before_load']=gpu;meta['gpu_hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
  from transformers import AutoModelForCausalLM
  torch.cuda.reset_peak_memory_stats();meta['model_load_attempted']=True
  model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True);meta['model_object_returned']=True;model=model.eval().requires_grad_(False)
  meta['model_loaded']=True;resources_guard('after_model_load')
  before=modelhash(model);assert before==old['model_hash_before'];versions={name:p._version for name,p in model.named_parameters()};meta['model_hash_before']=before
  probe=GateWriteNorm(model,alpha,u,v,F_DIGEST,freeze)
  for item in items:
   if item['corpus'] not in SEGMENTS[a.segment]:continue
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'corpus':item['corpus'],'window':item['window'],'identity':identity,'length':512,'token_sha256':fingerprint(x)['sha256']};rotated=x.clone();rotated[128:]=x[128:].roll(1);c_bundle=c_logits=None
    for boundary in BOUNDARIES:
     group_name=f"{base['corpus']}__{base['window']}__{identity}__{boundary}";snapshot=tuple(base[k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT
     writer.guard('boundary:'+group_name,GROUP_ALLOWANCES['snapshot' if snapshot else 'ordinary']+METADATA_ALLOWANCE)
     payload=empty_payload({**base,'boundary':boundary});group_committed=False;probe.start_input(base,boundary);own_base=own_logits=None;meta['phase']='evaluation'
     for arm in ARMS:
      active={**base,'boundary':boundary,'arm':arm,'phase':'main'};trial,ll=ev(x,boundary,arm,c_logits,own_logits)
      if arm=='baseline':
       own_base,own_logits=trial,ll;payload['source_reference']=trial['source']
       if boundary=='000':c_bundle,c_logits=trial,ll
      trial['metrics']=metric_rows(ll,x,c_logits,own_logits);pack(payload,arm,trial,ll,x)
      if boundary=='000' and arm=='baseline' and identity=='natural' and item['window']==0:
       saved_active=dict(active)
       for anchor in ('native_repeat','native_identity'):
        active={**base,'boundary':boundary,'arm':anchor,'phase':'anchor'}
        if anchor=='native_repeat':
         with probe.unhooked():candidate,cl=ev(x,'000','baseline',c_logits,c_logits,unhooked=True,anchor=True)
        else:
         from gpt2_layer_writes import LayerWrites
         selection=json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected'];identity_probe=LayerWrites(model,selection,[447,138]);identity_probe.reset_layers(E=False,blocked=[],special='identity')
         try:candidate,cl=ev(x,'000','baseline',c_logits,c_logits,anchor=True,observe_only=True)
         finally:identity_probe.close();identity_probe=None
        pack(payload,anchor,candidate,cl,x);check(identity_check(c_bundle,candidate,c_logits,cl,anchor),c_bundle,candidate,c_logits,cl);del candidate,cl
       active=saved_active
      if arm!='baseline':check({'kind':'source_q0','arm':arm,'source_exact':byte_exact(trial['source'],own_base['source'])},own_base,trial,own_logits,ll)
      if arm=='current':check({'arm':arm,**identity_check(own_base,trial,own_logits,ll,'current')},own_base,trial,own_logits,ll)
      active={**base,'boundary':boundary,'arm':arm,'phase':'suffix'};altered,al=ev(rotated,boundary,arm,c_logits,own_logits,suffix=True);pack(payload,arm,altered,al,rotated,suffix=True)
      check({'arm':arm,**suffix_check(trial,altered,ll,al)},trial,altered,ll[:128],al[:128]);del altered,al
      if arm!='baseline':del trial,ll
     assert probe.order_index==12 and set(payload['suffix'])==set(ARMS)
     expected=set(ARMS)|({'native_repeat','native_identity'} if boundary=='000' and identity=='natural' and item['window']==0 else set());assert set(payload['metrics'])==expected
     writer.save('inputs/'+group_name+'.npz',payload);group_committed=True;payload=None;own_base=own_logits=latest_logits=None;checkpoint();print('POST_WRITE_BOUNDARY_DONE',group_name,returned_count,flush=True)
     if boundary=='000':c_bundle=None
    c_bundle=c_logits=None
  assert attempts==returned_count==processed==388 and main_count==suffix_count==192 and anchor_count==4 and len(writer.inventory)==32
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items()) and all(sha(ROOT/name)==h for name,h in closure.items()) and all(sha(ROOT/name)==h for name,h in METADATA_PINS.items())
  assert sha(ref/'manifest.json')==T18_MANIFEST_SHA and all(sha(ref/name)==dm['calibration_reference_files'][name] for name in ('calibration_maps.npz','calibration_directions.npz'))
  assert maps_digest(maps['F'])==F_DIGEST and precision_settings()==settings and runtime_behavior()==arithmetic_runtime
  if previous_binding is not None:
   assert validate(a.previous_segment,'A')==(previous_manifest,previous_inventory)
   assert previous_binding=={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
  after=modelhash(model);assert after==before and versions=={name:p._version for name,p in model.named_parameters()}
  meta.update(status='completed',phase='completed',model_hash_after=after,parameter_versions_unchanged=True,precision_settings_after=precision_settings(),all_final_guards_passed=True,completed_utc=time.strftime('%FT%TZ',time.gmtime()));checkpoint(final=True)
 except BaseException as error:
  preservation={}
  try:
   if payload is not None and not group_committed:preservation['partial_boundary']=writer.save('inputs/'+group_name+'__partial.npz',payload)
  except BaseException as save_error:preservation['partial_boundary_error']=repr(save_error)
  if probe is not None and probe.failure is None:
   try:probe.capture_failure(error)
   except BaseException as capture_error:preservation['hook_capture_error']=repr(capture_error)
  capsule={'error_type':type(error).__name__,'error':str(error),'progress':progress(),'current_attempt_tokens':tokens,'current_input_identity':active,
   'hook_failure':None if probe is None else probe.failure,'left':failure_bundle(pair_left),'right':failure_bundle(pair_right),
   'comparison_logits':pair_logits,'latest_logits':logit_rows(latest_logits),'current_hidden':latest_hidden,'current_metrics':latest_metrics,'scoring_status':scoring_status,
   'uncommitted_boundary':payload if 'partial_boundary_error' in preservation else None}
  try:preservation['failure_capsule']=writer.save('failure_state.npz',capsule,failure=True)
  except BaseException as save_error:preservation['failure_capsule_error']=repr(save_error)
  meta.update(progress(),**resources(),status='failed',phase='failed',failure_type=type(error).__name__,failure_message=str(error),preservation=preservation)
  try:checkpoint(final=True,failure=True)
  except BaseException as save_error:
   if writer.owns():json_write(a.out/'checkpoint_failure.json',{'error':repr(save_error),'known_progress':progress()})
  raise
 finally:
  if probe is not None:probe.close()
  if identity_probe is not None:identity_probe.close()
if __name__=='__main__':main()
