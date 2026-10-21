"""Create an explicit T19 composite after both bounded recovery segments pass.

Copies bytes to a fresh directory, never hardlinks/replaces/deletes evidence.
The compatibility forward_passes field counts retained scheduled scientific
records. Actual original+validation+new invocation totals are separate.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
from gpt2_gate_native_value_recovery import (ROOT,COUNTS,CHECKPOINT,CONTINUITY,ORIGINAL_RUNNER_SHA,
 FORENSIC_SHA,PREFIX_SHA,PROPOSAL_SHA,read,put,sha,copy_exact,journal,verify_original,verify_segment,compare_npz_payload)
RESERVE=2*1024**3

def merge(a):
 assert not a.out.exists(),'Fresh immutable composite directory required'
 cp,old,f=verify_original(a.original,a.forensic,a.prefix)
 am,ai,aj=verify_segment(a.segment_a,'A');bm,bi,bj=verify_segment(a.segment_b,'B')
 assert am['original_source_hashes']==bm['original_source_hashes']==cp['source_hashes']
 assert am['input_dependencies']==bm['input_dependencies']==cp['input_dependencies']
 assert am['calibration_reference_sha256']==bm['calibration_reference_sha256']==cp['calibration_reference_sha256']
 assert am['model_hash_before']==bm['model_hash_before']==cp['model_hash_before']
 assert am['precision_settings_before']==bm['precision_settings_before']==cp['precision_settings_before']
 for field in ('runtime_versions','model_id','revision','dtype','attention','metadata_dependencies'):
  assert am[field]==bm[field]==cp[field],('Recovery provenance mismatch',field)
 assert am['calibration_map_digest']==bm['calibration_map_digest']==cp['calibration_map_digest']
 for segment,m,base in [('A',am,a.segment_a),('B',bm,a.segment_b)]:
  ctx=read(base/'frozen_sources/recovery_context.json');proof=read(base/'frozen_sources/recovery_ast_proof.json')
  assert ctx['segment']==segment and ctx['adapter_sha256']==m['recovery_adapter_sha256']
  assert m['source_hashes']['gpt2_gate_native_value.py']==proof['adapted_runner_sha256']
  assert proof['original_runner_sha256']==ORIGINAL_RUNNER_SHA and proof['complete_boundary_condition_and_suffix_loop_ast_exact']
  assert m['adapted_runner_ast_proof_sha256']==sha(base/'frozen_sources/recovery_ast_proof.json')
  for name,h in cp['source_hashes'].items():
   if name!='gpt2_gate_native_value.py':assert m['source_hashes'][name]==h
 assert read(a.segment_b/'frozen_sources/recovery_context.json')['segment_a_manifest_sha256']==sha(a.segment_a/'manifest.json')
 replay=a.segment_a/'inputs'/(CONTINUITY+'.npz');reference=a.original/'inputs'/(CONTINUITY+'.npz')
 comparison=compare_npz_payload(reference,replay);assert comparison['passed']
 gate=read(a.segment_a/'continuity_gate.json');assert gate['reference_sha256']==sha(reference) and gate['replay_sha256']==sha(replay)
 selected=[]
 for n,v in old.items():selected.append((a.original,n,v,'original_committed'))
 for base,inv,label in [(a.segment_a,ai,'segment_A_new'),(a.segment_b,bi,'segment_B_new')]:
  for n,v in inv.items():
   if Path(n).stem==CONTINUITY:continue
   selected.append((base,n,v,label))
 assert len(selected)==32 and len({n for _,n,_,_ in selected})==32
 roster={f'inputs/{c}__{w}__{i}.npz' for c in ('wiki_validation','utilitarianism','ethics','optimizers_new') for w in range(4) for i in ('natural','bos')}
 assert {n for _,n,_,_ in selected}==roster
 projected=sum(v['bytes'] for _,_,v,_ in selected)+32*1024**2
 assert projected<=512*1024**2 and shutil.disk_usage(a.out.parent).free>=RESERVE+projected
 a.out.mkdir(parents=True);a._created_out=True;inventory={};origins={}
 for base,n,v,label in selected:
  copy_exact(base/n,a.out/n,v['sha256']);inventory[n]=v;origins[n]={'origin':label,'source_directory':str(base.resolve()),**v}
 for n,h in cp['source_hashes'].items():copy_exact(a.original/'frozen_sources'/n,a.out/'frozen_sources'/n,h)
 for n,h in cp['input_dependencies'].items():copy_exact(a.original/n,a.out/n,h)
 for n in ('calibration_maps.npz','calibration_directions.npz'):copy_exact(a.original/n,a.out/n)
 for name,h in cp['metadata_dependencies'].items():
  sanitized=name.removeprefix('../').replace('/','__')
  copy_exact(a.original/'metadata_dependencies'/sanitized,a.out/'metadata_dependencies'/sanitized,h)
 dm=read(a.original/'data_manifest.json')
 assert sha(a.calibration_reference/'manifest.json')==cp['calibration_reference_sha256']==dm['calibration_reference_manifest_sha256']
 calibration_reference_files={}
 for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):
  h=cp['calibration_reference_sha256'] if name=='manifest.json' else dm['calibration_reference_files'][name]
  copy_exact(a.calibration_reference/name,a.out/'execution_provenance/calibration_reference'/name,h)
  calibration_reference_files[name]=h
 for label,base in [('original',a.original),('segment_A',a.segment_a),('segment_B',a.segment_b)]:
  dest=a.out/'execution_provenance'/label
  copy_exact(base/'journal.jsonl',dest/'journal.jsonl')
  if label=='original':
   copy_exact(base/CHECKPOINT/'manifest.json',dest/'last_checkpoint_manifest.json')
   copy_exact(base/CHECKPOINT/'numeric_inventory.json',dest/'numeric_inventory.json')
  else:
   copy_exact(base/'manifest.json',dest/'manifest.json');copy_exact(base/'numeric_inventory.json',dest/'numeric_inventory.json')
   for n in ('gpt2_gate_native_value.py','gpt2_gate_native_value_recovery.py','GPT2_NATIVE_VALUE_RECOVERY_PROPOSAL.md','recovery_context.json','recovery_ast_proof.json'):
    copy_exact(base/'frozen_sources'/n,dest/n)
 copy_exact(a.segment_a/'continuity_gate.json',a.out/'execution_provenance/continuity_gate.json')
 copy_exact(a.forensic,a.out/'execution_provenance/original_interruption.json',FORENSIC_SHA)
 copy_exact(a.prefix,a.out/'execution_provenance/retained_prefix_bytes.json',PREFIX_SHA)
 actual=1110+am['forward_passes']+bm['forward_passes'];assert actual==2272
 def categories(base):
  _,_,events=journal(base/'journal.jsonl');out={'main':0,'suffix':0,'anchors':0}
  for e in events:
   if e['event']=='model_returned':out[{'evaluation':'main','suffix_control':'suffix','anchor':'anchors'}[e['identity']['phase']]]+=1
  return out
 actual_categories={label:categories(base) for label,base in [('original',a.original),('A',a.segment_a),('B',a.segment_b)]}
 totals={k:sum(d[k] for d in actual_categories.values()) for k in ('main','suffix','anchors')}
 assert totals=={'main':1130,'suffix':1130,'anchors':12} and sum(totals.values())==actual
 resource_keys=('wall_seconds','peak_process_rss_kib','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')
 m=copy.deepcopy(cp);m.update(status='completed',phase='completed',execution_type='verified_composite_after_interruption',execution_description='Interrupted original prefix plus two bounded recovery segments; never one continuous execution',
  forward_passes=2184,forward_attempts=2184,evaluation_forwards=1088,suffix_forwards=1088,anchor_forwards=8,
  forward_counter_scope='Retained scheduled scientific records for frozen analyzer compatibility; not cumulative process/model invocations.',
  retained_scheduled_forwards=2184,actual_cumulative_model_returns=actual,actual_cumulative_model_attempts=1110+am['forward_attempts']+bm['forward_attempts'],actual_return_categories_by_execution=actual_categories,actual_return_category_totals=totals,
  continuity_validation_forwards=70,discarded_uncommitted_original_returns=18,original_model_returns=1110,original_fully_processed_forwards=1109,
  original_final_guards_available=False,original_missing_final_guards=f['missing_guards'],
  parameter_versions_unchanged=None,model_hash_after=None,precision_settings_after=None,
  model_loaded=None,wall_seconds=None,peak_process_rss_kib=None,peak_gpu_allocated_bytes=None,peak_gpu_reserved_bytes=None,
  original_observed_partial_checkpoint_resources={k:cp.get(k) for k in resource_keys},segment_complete_resources={'A':{k:am.get(k) for k in resource_keys},'B':{k:bm.get(k) for k in resource_keys}},
  cumulative_runtime_and_original_final_peak_unknown=True,source_hashes_scope='Original20algorithmic source freeze; actual adapted execution hashes separately recorded for both segments.',
  composite_construction_model_forwards=0,completed_utc=time.strftime('%FT%TZ',time.gmtime()),
  segment_final_guards_passed={'A':True,'B':True},segment_manifests={'A':sha(a.segment_a/'manifest.json'),'B':sha(a.segment_b/'manifest.json')},
  executed_segment_source_hashes={'A':am['source_hashes'],'B':bm['source_hashes']},
  exact_continuity_passed=True,continuity_scope='Same-machine replay compatibility; cannot recover original missing after-run parameter/version/precision checks.',
  merger_sha256=sha(__file__),recovery_proposal_sha256=PROPOSAL_SHA,calibration_reference_artifact_files=calibration_reference_files,
  original_resource_measurements_scope='Inherited pre-interruption checkpoint only, not aggregate or final peak over all processes.')
 put(a.out/'numeric_inventory.json',inventory);put(a.out/'execution_provenance/shard_origins.json',origins)
 put(a.out/'manifest.json',m)
 assert shutil.disk_usage(a.out).free>=RESERVE
 print('COMPOSITE_NATIVE_VALUE',2184,'retained',actual,'actual returns',a.out,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('original','forensic','prefix','segment-a','segment-b','calibration-reference','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a._created_out=False
 try:merge(a)
 except BaseException as e:
  if a._created_out and a.out.is_dir() and not (a.out/'merge_failure.json').exists():put(a.out/'merge_failure.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'new_model_forwards':0})
  raise
if __name__=='__main__':main()
