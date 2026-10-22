"""T23 evaluation SOURCE ONLY HOLD. No execution until separate reviewed activation.
Native000 fixed PP/PC/CP/CC18arms; reuses frozen scoring and immutable writer.
"""
import argparse,contextlib,copy,hashlib,json,platform,resource,shutil,subprocess,time
from importlib.metadata import version
from pathlib import Path
import torch
from gate_native_value import fingerprint
from gate_write_norm import runtime_behavior
from gate_calibration_transfer import GateCalibrationTransfer,CELLS,ARMS
from gpt2_gate_native_value import metric_rows,model_forward,F_DIGEST,METADATA_PINS
from l11_key_value_donor_storage import (EvidenceWriter,READOUT_ROWS,METADATA_ALLOWANCE,
 byte_exact,nested_fp,summarize_diag,trim_gate,trim_gate_norm,source_closure,sha,json_write,read_npz,
 current_rss_bytes,owned_cpu_bytes,WORKSPACE_ALLOWANCE,NORMALIZATION_ALLOWANCE)
ROOT=Path(__file__).resolve().parent
SOURCE_READY=True
HOOK_SHA='decf50e505235bf2f37cc48f14b89e0397eb6f2604a979e2cb8a32bf091c3bb1'
INPUT_HASHES={'tokens.pt': '23e11fa6564d4d29cb847432aff97a60d3dc762042fe625535d6db0207f15e9a', 'data_manifest.json': '4696bb86cb666351a00d3b68f9005088a8b9563816a50478a86455f2279f8505', 'exclusion_inventory.json': 'afe1cd68e15848b6c5b619e75b541adb0145060a37601fa4a506a6909af12ba6', 'known_history_chain.json': '55fd60df741c3eadc81487cf77b4fe77e68a4ac59968433af8a577b451be3258', 'known_history_review.json': 'ef971da48a0f3c8e344b27e668146248484d25140c002adb42bb7c3fb1aaf3a0', 'start_identity_summary.json': 'b876051f9f9a9b92696a58b91e83b2d68be9d80aa6c85daa1e91ff3da92b6661', 'preflight.json': '21bd63963ce17bdb3df52ad327abb01790a9bdaccf887d572d8802370048f492'}
RESOURCE_CONTRACT_PATH='review/t23_evaluation_runtime_contract_v1.json'
RESOURCE_CONTRACT_SHA='a5b822b2f1b1f051c5393de3c35dd9bf62a566106dbdcfd905bde7906a982c6d'
WRITER_SHA='132ad35ff5b240eac0a41392a8a04f584d9383d0f3dd82fc4e7e566916a6e7eb'
PROPOSAL_SHA='a78e9678ffceefc85abcccc2a11cd750754efd1adb7738caa03d508f957c2e7d'
CALIBRATION_ACCEPTANCE_SHA='87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300'
CALIBRATION_MANIFEST_SHA='f71a3c3b8af737f240b9d559b0786c90501f89dae7ef3a2fba2efacc004f6b2a'
CALIBRATION_CE_RECORD={'file':'fits/CE.npz','sha256':'31c9db95912c8e317f77d0814a5ee94bdd66a25566bea48245495519d2b5ccd9','map_digest':'2c26d9c65918f298a3f933986438bd5daf805b539270ffe15b0993e98cbf418d'}
T18_MANIFEST_SHA='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060'
BOUNDARIES=('000',)
SEGMENTS={'A':('wiki_validation','utilitarianism'),'B':('ethics','optimizers_new')}
CORPORA=tuple(c for names in SEGMENTS.values() for c in names)
MATCHED=tuple(a for a in ARMS if '_write_matched_early' in a)
SNAPSHOT_INPUT=('wiki_validation',0,'natural')
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')
PINNED_VERSIONS={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'}

def empty_payload(metadata):
 return {'schema':'calibration_world_evaluation_explicit_v1','metadata':metadata,**{k:{} for k in ('metrics','suffix','gates','suffix_gates','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms','operations','suffix_operations','readouts','snapshots')},'checks':[],'retirement_receipts':[]}

def known_header_guard(writer,owners,contract,stage):
 # Source-reviewed complete failure union already includes GPU handoff leaves.
 # CPU _owned_cpu handoff attaches existing storage; do not count a second raw
 # clone for it. Four held runner bundles are bounded separately (64MiB).
 rss=current_rss_bytes();owned=owned_cpu_bytes(owners());base=max(0,rss-owned)
 raw=contract['failure_header_raw_upper_bytes'];largest=contract['largest_identity_bytes']
 unencoded=contract['unencoded_live_owner_allowance_bytes']
 assert raw==740495216 and largest==512*50257*4 and unencoded==64*1024**2
 assert contract['non_evidence_limit_bytes']==2240*1024**2
 # R0 + evidence unique storage(raw upper + unencoded owners) + retained
 # contiguous copies(raw upper) + largest normalization + largest byte string.
 predicted=base+2*raw+2*largest+WORKSPACE_ALLOWANCE+NORMALIZATION_ALLOWANCE+unencoded
 record={'event':'T23_known_header_admission','stage':stage,'current_rss_bytes':rss,'owned_CPU_bytes':owned,
  'non_evidence_observed_bytes':base,'non_evidence_limit_bytes':contract['non_evidence_limit_bytes'],
  'failure_header_raw_upper_bytes':raw,'largest_identity_bytes':largest,
  'unencoded_live_owner_allowance_bytes':unencoded,'owned_storage_upper_bytes':raw+unencoded,
  'predicted_bytes':predicted,'rss_limit_bytes':4*1024**3,
  'passed':base<=contract['non_evidence_limit_bytes'] and owned<=raw+unencoded and predicted<=4*1024**3}
 writer.journal(record);assert record['passed'],record
 return record

def capture(probe,hidden,metrics,*,selected):
 assert set(probe.timing_current)==set(TIMING_NODES)
 operations={'native_diag':summarize_diag(probe.diag),'native_gate_checks':copy.deepcopy(probe.gate_checks),
  'projection_records':copy.deepcopy(probe.projection_records),'context_records':copy.deepcopy(probe.context_records),
  'mask_records':copy.deepcopy(probe.mask_records),'source_fingerprints':nested_fp(probe.source_states),
  'gate_fingerprints':nested_fp(probe.gate_cache),'prefix_context_fingerprints':nested_fp(probe.prefix_contexts),
  'timing_checks':copy.deepcopy(probe.timing_checks),'timing_fingerprints':nested_fp(probe.timing_current),
  'write_checks':copy.deepcopy(probe.write_checks),'write_cache_key':copy.deepcopy(probe.cache_key),
  'cell_records':copy.deepcopy(probe.cell_records),'output_fields':copy.deepcopy(probe.output_fields),
  'projection_checks':probe.projection_checks,'context_checks':probe.context_checks,'query_checks':probe.query_checks}
 bundle={'metrics':metrics,'hidden':hidden,'source':copy.deepcopy(probe.source_states),'gates':copy.deepcopy(probe.gate_cache),
  'gate_norms':copy.deepcopy(probe.localization_norms),'write_norms':copy.deepcopy(probe.write_norms),
  'prefix_contexts':copy.deepcopy(probe.prefix_contexts),'prefix_writes':probe.prefix_writes.clone(),
  'returned_write':probe._returned_write.clone(),'operations':operations,
  'write_snapshot':{},'shared_snapshot':None}
 probe.finish()
 if selected:
  bundle['write_snapshot']=probe.panel_snapshot()
  if probe.arm=='baseline':bundle['shared_snapshot']=probe.shared_snapshot
 return bundle

def pack(payload,arm,bundle,logits,tokens,*,suffix=False):
 metrics=dict(bundle['metrics']);metrics['final_hidden_sha256']=fingerprint(bundle['hidden'])['sha256']
 if suffix:
  payload['suffix'][arm]=metrics;payload['suffix_gates'][arm]=trim_gate(bundle['gates']);payload['suffix_gate_norms'][arm]=trim_gate_norm(bundle['gate_norms'])
  payload['suffix_write_norms'][arm]=bundle['write_norms'];payload['suffix_operations'][arm]=bundle['operations']
 else:
  for k,v in (('metrics',metrics),('gates',bundle['gates']),('gate_norms',bundle['gate_norms']),('write_norms',bundle['write_norms']),('operations',bundle['operations'])):payload[k][arm]=v
  if tuple(payload['metadata'][k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT and arm in ARMS:
   if arm=='baseline':payload['shared_snapshot']=bundle['shared_snapshot']
   payload['snapshots'][arm]=bundle['write_snapshot']
   payload['readouts'][arm]={'hidden':bundle['hidden'],'query_rows':torch.tensor(READOUT_ROWS),
    'logits':logits[list(READOUT_ROWS)].detach().cpu().clone(),'targets':tokens[[q+1 for q in READOUT_ROWS]],
    'token_sha256':fingerprint(tokens)['sha256'],'hidden_fingerprint':fingerprint(bundle['hidden'])}

def failure_bundle(bundle):
 if bundle is None:return None
 return {k:bundle[k] for k in ('metrics','hidden','source','gates','gate_norms','write_norms','prefix_contexts','prefix_writes','returned_write','operations') if k in bundle}

def suffix_operands(left,right):
 def side(b):
  return {'hidden':b['hidden'][:,:128],'source':b['source'],'gates':trim_gate(b['gates']),'gate_norms':trim_gate_norm(b['gate_norms']),
   'write_norms':{k:v[:128] for k,v in b['write_norms'].items()},'prefix_contexts':b['prefix_contexts'],'prefix_writes':b['prefix_writes'],
   'metrics':{k:v[:127 if k=='loss' else 128] for k,v in b['metrics'].items() if isinstance(v,torch.Tensor)},'mask_records':b['operations']['mask_records']}
 return {'expected':side(left),'actual':side(right)}

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
 for name in ('config','inputs','calibration-reference','calibration-study','out'):ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--segment',choices=tuple(SEGMENTS),required=True);ap.add_argument('--previous-segment',type=Path);a=ap.parse_args()
 assert (a.previous_segment is None)==(a.segment=='A'),'B requires completed A predecessor; A accepts no predecessor'
 for source in (a.inputs,a.calibration_reference,*(() if a.previous_segment is None else (a.previous_segment,))):assert not a.out.resolve().is_relative_to(source.resolve()),'Output inside immutable input/reference/predecessor'
 assert SOURCE_READY and RESOURCE_CONTRACT_SHA and HOOK_SHA and INPUT_HASHES,'SOURCE HOLD: separate input/source/resource/execution approval required'
 contract=json.loads((ROOT/RESOURCE_CONTRACT_PATH).read_text());assert sha(ROOT/RESOURCE_CONTRACT_PATH)==RESOURCE_CONTRACT_SHA
 assert contract['status']=='reviewed_execution_ready' and contract['model_execution_approved'] is True
 normal_limit=contract['normal_limit_bytes'];failure_limit=contract['failure_limit_bytes'];reserve=contract['disk_reserve_bytes']
 assert all(isinstance(v,int) and v>0 for v in (normal_limit,failure_limit,reserve))
 writer=EvidenceWriter(a.out,normal_limit=normal_limit,failure_limit=failure_limit,reserve=reserve,outstanding_other_caps=normal_limit*2)
 started=time.time();model=probe=identity_probe=None;attempts=observed_count=returned_count=processed=main_count=suffix_count=anchor_count=0
 active={};last_attempt={};last_returned={};last_processed={};tokens=None;latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started';payload=None;group_name=None;group_committed=True
 pair_left=pair_right=pair_logits=None
 raw_output_owner=None;last_unhooked=False
 runner_owners={}
 control_counts={k:0 for k in ('current','suffix','matched_later','source_q0','anchors','retired_cells')}
 observed_identity={};observed_current=False;observation_origins={'hook_model_end':0,'runner_after_return':0}
 meta={'study':'calibration_world_evaluation','status':'preparing','phase':'setup','segment':a.segment,'execution_type':'planned_segment',
  'model_loaded':False,'model_load_attempted':False,'model_object_returned':False,'calibration_forwards':0,'normal_limit_bytes':normal_limit,'failure_limit_bytes':failure_limit,'disk_reserve_bytes':reserve,
  'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},'boundaries':list(BOUNDARIES),'arms':list(ARMS),
  'expected_model_returns':580,'expected_boundary_shards':16,'started_utc':time.strftime('%FT%TZ',time.gmtime())}
 def progress():return {'phase':meta['phase'],'forward_attempts':attempts,'forward_passes':returned_count,'fully_processed_forwards':processed,
  'evaluation_forwards':main_count,'suffix_forwards':suffix_count,'anchor_forwards':anchor_count,'active':active,
  'model_outputs_observed':observed_count,'observation_origins':dict(observation_origins),'last_output_observed':observed_identity,'control_counts':dict(control_counts),'last_attempted':last_attempt,'last_returned':last_returned,'last_processed':last_processed}
 def resource_owners():
  # Explicit existing CPU producer owners, never model parameters/GPU baseline
  # logits and never a cloned second failure cache. Backing intervals are used
  # only for RAM admission; serialization still emits every occurrence.
  owned=[payload,runner_owners,tokens,latest_hidden,latest_metrics,pair_left,pair_right,pair_logits]
  if probe is not None:owned.extend(probe.resource_owners())
  return owned
 writer.set_resource_owners(resource_owners)
 def resources():return {'wall_seconds':time.time()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
  'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}
 def resources_guard(stage):
  observed=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  assert observed<=4*1024**2,('CPU peak RSS budget',stage,observed,4*1024**2)
 def checkpoint(*,final=False,failure=False):
  meta.update(progress(),**resources())
  if not failure:resources_guard('checkpoint')
  writer.checkpoint(progress(),manifest=meta,final=final,failure=failure)
 def check(value,left=None,right=None,ll=None,rr=None,*,comparison='generic'):
  nonlocal pair_left,pair_right,pair_logits
  payload['checks'].append(value)
  if not passed(value):
   if comparison=='suffix':pair_left,pair_right=suffix_operands(left,right),None
   elif comparison=='identity':
    pair_left={'hidden':left['hidden'],'metrics':left['metrics'],'returned_write':left['returned_write']}
    pair_right={'hidden':right['hidden'],'metrics':right['metrics'],'returned_write':right['returned_write']}
   elif value.get('kind')=='source_q0':pair_left,pair_right=left['source'],right['source']
   else:pair_left,pair_right=failure_bundle(left),failure_bundle(right)
   pair_logits=None if ll is None else {'left':ll.detach().cpu().clone(),'right':None if rr is None else rr.detach().cpu().clone(),'scope':comparison,'comparison_query_slice':[0,128] if comparison=='suffix' else [0,512],'right_full_actual_readout':rr is not None and len(rr)==512}
   raise AssertionError(value)
 def ev(x,boundary,arm,c_logits=None,b_logits=None,*,suffix=False,unhooked=False,anchor=False,observe_only=False,selected=False):
  nonlocal attempts,returned_count,processed,last_attempt,last_returned,last_processed,tokens,latest_logits,latest_hidden,latest_metrics,scoring_status,main_count,suffix_count,anchor_count,observed_current,observed_count,observed_identity,raw_output_owner,last_unhooked
  assert x.dtype==torch.int64 and x.shape==(512,);tokens=x.clone();latest_logits=latest_hidden=latest_metrics=None;scoring_status='not_started';observed_current=False;raw_output_owner=None;last_unhooked=unhooked
  known_header_guard(writer,resource_owners,contract,'before_forward:'+str(attempts+1))
  writer.journal({'event':'forward_intent','next_attempt':attempts+1,'identity':active,'token_fingerprint':fingerprint(x)})
  if not unhooked:probe.reset(boundary,arm,variant='suffix' if suffix else 'main',token_sha256=fingerprint(x)['sha256'],snapshot=selected,observe_only=observe_only)
  def output_observed(event):
   nonlocal observed_count,observed_identity,observed_current
   assert not observed_current,'Duplicate actual model-end observation'
   assert event['arm']==arm and event['variant']==('suffix' if suffix else 'main') and event['actual_token_sha256']==fingerprint(x)['sha256']
   origin='runner_after_return' if unhooked else 'hook_model_end'
   observed_current=True;observed_count+=1;observation_origins[origin]+=1;observed_identity=dict(active)
   writer.journal({'event':'model_output_observed','attempt':attempts,'observed':observed_count,'observer_origin':origin,'identity':active,'hook_event':event})
  if not unhooked:probe.set_output_observer(output_observed)
  def invoked():
   nonlocal attempts,last_attempt
   writer.journal({'event':'forward_attempt','attempt':attempts+1,'identity':active});attempts+=1;last_attempt=dict(active)
  def returned():
   nonlocal returned_count,last_returned,main_count,suffix_count,anchor_count
   returned_count+=1;last_returned=dict(active)
   if anchor:anchor_count+=1
   elif suffix:suffix_count+=1
   else:main_count+=1
  profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if returned_count==0 else contextlib.nullcontext()
  with torch.inference_mode():
   with profile as profiler:
    raw_output_owner=model_forward(model,x,returned,invoked)
    # First actions after actual return bind complete actual owners. Even
    # profiler/journal/allocation failure now has reached-output evidence.
    latest_logits=raw_output_owner.logits[0].detach()
    latest_hidden=raw_output_owner.hidden_states[-1].detach()
   writer.journal({'event':'model_returned','attempt':attempts,'completed':returned_count,'identity':active})
   if unhooked:output_observed({'arm':arm,'variant':'suffix' if suffix else 'main','actual_token_sha256':fingerprint(x)['sha256'],'scope':'Actual output captured by runner AFTER model_returned; no model-end hook installed'})
   else:assert observed_current,'Hooked call returned without owned model-end observation'
   if returned_count==1:
    events=[{'operator':e.key,'count':e.count} for e in profiler.key_averages() if 'scaled_dot_product' in e.key];meta['sdpa_profiler_events']=events
    assert any('_scaled_dot_product' in e['operator'] for e in events),'Expected observed SDPA backend'
   actual_hidden=latest_hidden
   latest_hidden=actual_hidden.to('cpu',copy=True) if unhooked else probe.current_hidden
   if not byte_exact(latest_hidden,actual_hidden):
    probe.capture_failure('Actual returned hidden differs from independently owned hidden',
     actual_returned_hidden=actual_hidden,expected_observed_hidden=latest_hidden,
     reference_scope='unhooked_cpu_copy' if unhooked else 'hook_observer')
    raise AssertionError('Actual returned/observed hidden mismatch')
   raw_output_owner=None;del actual_hidden
   scoring_status='scoring_started_not_completed';metrics=metric_rows(latest_logits,x,c_logits,b_logits,suffix=suffix);latest_metrics=metrics;scoring_status='completed'
  if unhooked:bundle={'metrics':metrics,'hidden':latest_hidden,'source':{},'gates':{},'gate_norms':{},'write_norms':{},'snapshots':{},'write_snapshot':{},'prefix_contexts':{},'prefix_writes':None,'returned_write':None,'operations':{}}
  else:bundle=capture(probe,latest_hidden,metrics,selected=selected)
  resources_guard('after_capture_before_processed')
  processed+=1;last_processed=dict(active);writer.journal({'event':'forward_completed','attempt':attempts,'completed':returned_count,'processed':processed,'identity':active,'hidden_fingerprint':fingerprint(latest_hidden),'logits_fingerprint':fingerprint(latest_logits)})
  return bundle,latest_logits
 try:
  checkpoint()
  writer.guard('complete_planned_caps_before_model')
  assert sha(ROOT/'GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md')==PROPOSAL_SHA
  assert sha(ROOT/'gate_calibration_transfer.py')==HOOK_SHA
  assert sha(ROOT/'l11_key_value_donor_storage.py')==WRITER_SHA
  acceptance_path=ROOT/'review/t23_calibration_actual_result_independent_acceptance_v1.json'
  assert sha(acceptance_path)==CALIBRATION_ACCEPTANCE_SHA
  acceptance=json.loads(acceptance_path.read_text());assert acceptance['status']=='passed'
  calmeta=json.loads((a.calibration_study/'manifest.json').read_text())
  assert sha(a.calibration_study/'manifest.json')==CALIBRATION_MANIFEST_SHA
  assert calmeta['status']=='completed' and calmeta['all_final_guards_passed'] is True
  assert calmeta['model_hash_before']==calmeta['model_hash_after'] and calmeta['calibration_forwards']==16
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items()) and INPUT_HASHES==contract['audited_input_hashes']
  pre=json.loads((a.inputs/'preflight.json').read_text());assert pre['status']=='passed' and pre['model_forwards']==0 and pre['evaluation_passages']==16 and pre['evaluation_inputs']==32
  dm=json.loads((a.inputs/'data_manifest.json').read_text())
  assert dm['T23_calibration_manifest_sha256']==CALIBRATION_MANIFEST_SHA and dm['T23_calibration_acceptance_sha256']==CALIBRATION_ACCEPTANCE_SHA
  items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu')
  assert len(items)==16 and {(i['corpus'],i['window']) for i in items}=={(c,w) for c in CORPORA for w in range(4)}
  for item in items:
   record=next(r for r in dm['passages'] if (r['corpus'],r['window'])==(item['corpus'],item['window']))
   assert item['tokens'].dtype==torch.int64 and item['tokens'].shape==(512,) and fingerprint(item['tokens'])['sha256']==record['window_token_sha256']
   bos=item['tokens'].clone();bos[0]=dm['literals']['start_special']['id'];assert fingerprint(bos)['sha256']==record['bos_window_token_sha256']
  for name,h in INPUT_HASHES.items():shutil.copyfile(a.inputs/name,a.out/name);assert sha(a.out/name)==h
  ref=a.calibration_reference;assert sha(ref/'manifest.json')==dm['calibration_reference_manifest_sha256']==T18_MANIFEST_SHA
  refmeta=json.loads((ref/'manifest.json').read_text());assert refmeta['status']=='completed' and refmeta['calibration_map_digests']['F']==F_DIGEST
  for name in ('calibration_maps.npz','calibration_directions.npz'):assert sha(ref/name)==dm['calibration_reference_files'][name]
  (a.out/'calibration_reference').mkdir()
  for name in ('manifest.json','calibration_maps.npz','calibration_directions.npz'):shutil.copyfile(ref/name,a.out/'calibration_reference'/name);assert sha(a.out/'calibration_reference'/name)==sha(ref/name)
  # PE center and pair come ONLY from original F, not its contemporary refit.
  maps=read_npz(ref/'calibration_maps.npz');directions=read_npz(ref/'calibration_directions.npz')
  from gpt2_gate_subspace import maps_digest
  from run_transfer import modelhash
  from native_prefix_diagnostic import precision_settings
  assert maps_digest(maps['F'])==F_DIGEST
  ce_record=calmeta['fit_records']['CE'];assert ce_record==CALIBRATION_CE_RECORD
  assert sha(a.calibration_study/ce_record['file'])==ce_record['sha256']
  ce=read_npz(a.calibration_study/ce_record['file']);assert ce['world']=='CE' and ce['schema']=='T23_calibration_fit_v1' and maps_digest(ce['maps'])==ce['map_digest']==ce_record['map_digest']
  assert sha(a.calibration_study/'PE_continuity.json')==calmeta['PE_continuity_sha256']
  calibration_files={'manifest.json':CALIBRATION_MANIFEST_SHA,'PE_continuity.json':calmeta['PE_continuity_sha256'],ce_record['file']:ce_record['sha256']}
  calibration_files['acceptance.json']=CALIBRATION_ACCEPTANCE_SHA
  for name,h in calibration_files.items():
   source=acceptance_path if name=='acceptance.json' else a.calibration_study/name
   dest=a.out/'calibration_study'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==h
  cells={cell:{'alpha':(maps['F'] if cell[0]=='P' else ce['maps'])[10]['alpha'],
               'u':(directions['F'] if cell[1]=='P' else ce['directions'])[10]['u'],
               'v':(directions['F'] if cell[1]=='P' else ce['directions'])[10]['v']} for cell in CELLS}
  fit_digests={'PE':F_DIGEST,'CE':ce_record['map_digest']}
  closure=source_closure(ROOT,['gpt2_calibration_world_evaluation.py','gate_calibration_transfer.py','calibration_world_inputs.py','analyze_calibration_world_evaluation.py','merge_calibration_world_evaluation.py','l11_key_value_donor_storage.py','GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md',RESOURCE_CONTRACT_PATH])
  assert {n:h for n,h in closure.items() if n not in (Path(__file__).name,RESOURCE_CONTRACT_PATH)}==contract['reviewed_source_hashes'],'Exact independently reviewed source closure required'
  for name,h in closure.items():
   dest=a.out/'frozen_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);assert sha(dest)==h
  cfg=json.loads(a.config.read_text())
  for name,h in METADATA_PINS.items():
   assert sha(ROOT/name)==h;dest=a.out/'metadata_dependencies'/name.removeprefix('../').replace('/','__');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);assert sha(dest)==h
  old=json.loads((ROOT/'gpt2_run/manifest.json').read_text());spec=json.loads((ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json').read_text())['gpt2'];configured=cfg['models']['gpt2'];modelpath=Path(configured['path'])
  assert configured['model_id']==old['model_id']==refmeta['model_id']==calmeta['model_id']==spec['model_id'] and configured['revision']==old['revision']==refmeta['revision']==calmeta['revision']==spec['revision']
  for name,entry in spec['checkpoint_files'].items():assert sha(modelpath/name)==entry['sha256']
  torch.set_num_threads(4);torch.manual_seed(20261021);settings=precision_settings();assert settings==refmeta['precision_settings_before']==calmeta['precision_settings_before']==calmeta['precision_settings_after']
  runtime={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda};assert runtime==PINNED_VERSIONS
  arithmetic_runtime=runtime_behavior();freeze=hashlib.sha256(json.dumps({'sources':closure,'inputs':INPUT_HASHES,'metadata':METADATA_PINS,'runtime':runtime,'precision':settings},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  meta.update(execution_contract_sha256=RESOURCE_CONTRACT_SHA,source_hashes=closure,input_dependencies=INPUT_HASHES,metadata_dependencies=METADATA_PINS,freeze_digest=freeze,model_id=old['model_id'],revision=old['revision'],dtype='float32',attention='sdpa',runtime_versions=runtime,precision_settings_before=settings,arithmetic_runtime=arithmetic_runtime,calibration_map_digest=F_DIGEST,calibration_fit=False,physical_execution_lengths=[512],calibration_study_files=calibration_files,fit_digests=fit_digests,calibration_acceptance_sha256=CALIBRATION_ACCEPTANCE_SHA,raw_panel_contract={'shared_inputs':1,'write_states':list(ARMS),'readout_states':list(ARMS),'readout_rows':list(READOUT_ROWS),'timing_samples':0},slice_contract={'earlier':[1,127],'later':[127,512]})
  previous_binding=None
  if a.segment=='B':
   from merge_calibration_world_evaluation import validate,EQUAL_FIELDS
   previous_manifest,previous_inventory=validate(a.previous_segment,'A')
   for name in EQUAL_FIELDS:assert previous_manifest[name]==meta[name],('A predecessor differs',name)
   assert previous_manifest['model_hash_before']==previous_manifest['model_hash_after']==old['model_hash_before'],'A predecessor parameter identity differs from pinned checkpoint'
   previous_binding={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
   meta['previous_segment_A']=previous_binding
   (a.out/'previous_segment_A').mkdir()
   for name in ('manifest.json','numeric_inventory.json'):shutil.copyfile(a.previous_segment/name,a.out/'previous_segment_A'/name);assert sha(a.out/'previous_segment_A'/name)==previous_binding[name.replace('.json','_sha256')]
   # Only a verified complete predecessor releases its unused future cap.
   writer.outstanding_other_caps=normal_limit
  checkpoint()
  resources_guard('before_model_load');meta['header_admission_before_model']=known_header_guard(writer,resource_owners,contract,'before_model_load')
  meta['host_mem_available_before_model_kib']=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
  meta['host_mem_available_minimum_bytes']=6*1024**3
  assert meta['host_mem_available_before_model_kib']*1024>=meta['host_mem_available_minimum_bytes'],'Pre-model host headroom below6GiB'
  gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not gpu,('Competing GPU computation',gpu)
  meta['gpu_compute_processes_before_load']=gpu;meta['gpu_hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
  from transformers import AutoModelForCausalLM
  torch.cuda.reset_peak_memory_stats();meta['model_load_attempted']=True
  model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True);meta['model_object_returned']=True;model=model.eval().requires_grad_(False)
  meta['model_loaded']=True;resources_guard('after_model_load');meta['header_admission_after_model']=known_header_guard(writer,resource_owners,contract,'after_model_load_before_first_forward')
  before=modelhash(model);assert before==old['model_hash_before'];versions={name:p._version for name,p in model.named_parameters()};meta['model_hash_before']=before
  probe=GateCalibrationTransfer(model,cells,fit_digests,freeze,CALIBRATION_ACCEPTANCE_SHA)
  for item in items:
   if item['corpus'] not in SEGMENTS[a.segment]:continue
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'corpus':item['corpus'],'window':item['window'],'identity':identity,'length':512,'token_sha256':fingerprint(x)['sha256']};rotated=x.clone();rotated[128:]=x[128:].roll(1);c_bundle=c_logits=None
    for boundary in BOUNDARIES:
     group_name=f"{base['corpus']}__{base['window']}__{identity}__{boundary}";snapshot=tuple(base[k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT
     writer.guard('boundary:'+group_name,contract['group_allowances']['snapshot' if snapshot else 'ordinary']+METADATA_ALLOWANCE)
     payload=empty_payload({**base,'boundary':boundary});group_committed=False;probe.start_input(base,boundary);own_base=own_logits=own_suffix_base=own_suffix_logits=None;meta['phase']='evaluation'
     for arm in ARMS:
      active={**base,'boundary':boundary,'arm':arm,'phase':'main'};trial,ll=ev(x,boundary,arm,c_logits,own_logits,selected=snapshot)
      if arm=='baseline':
       own_base,own_logits=trial,ll;runner_owners['main_baseline']=own_base;payload['source_reference']=trial['source']
       if boundary=='000':c_bundle,c_logits=trial,ll
      runner_owners['main_trial']=trial
      trial['metrics']=metric_rows(ll,x,c_logits,own_logits);pack(payload,arm,trial,ll,x)
      if boundary=='000' and arm=='baseline' and identity=='natural' and item['window']==0:
       saved_active=dict(active)
       for anchor in ('native_repeat','native_identity'):
        active={**base,'boundary':boundary,'arm':anchor,'phase':'anchor'}
        if anchor=='native_repeat':
         with probe.unhooked():candidate,cl=ev(x,'000','baseline',c_logits,c_logits,unhooked=True,anchor=True)
        else:
         from gpt2_layer_writes import LayerWrites
         selection=json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected'];identity_probe=LayerWrites(model,selection,[447,138]);identity_probe.reset_layers(E=False,blocked=[],special='identity');probe.allow_identity_hooks(identity_probe.handles)
         try:candidate,cl=ev(x,'000','baseline',c_logits,c_logits,anchor=True,observe_only=True)
         finally:
          identity_probe.close();identity_probe=None
          if not probe.failed:probe.allow_identity_hooks(())
        pack(payload,anchor,candidate,cl,x);check(identity_check(c_bundle,candidate,c_logits,cl,anchor),c_bundle,candidate,c_logits,cl,comparison='identity');control_counts['anchors']+=1;del candidate,cl
       active=saved_active
      if arm!='baseline':
       check({'kind':'source_q0','arm':arm,'variant':'main','source_exact':byte_exact(trial['source'],own_base['source'])},own_base,trial);control_counts['source_q0']+=1
      if arm=='current':check({'arm':arm,'variant':'main',**identity_check(own_base,trial,own_logits,ll,'current')},own_base,trial,own_logits,ll,comparison='identity');control_counts['current']+=1
      if arm in MATCHED:
       assert all(trial['operations']['timing_checks'][node]['byte_exact'] is True and trial['operations']['timing_checks'][node]['query_slice']==[127,None] for node in TIMING_NODES);control_counts['matched_later']+=1
      active={**base,'boundary':boundary,'arm':arm,'phase':'suffix'};altered,al=ev(rotated,boundary,arm,c_logits,own_logits,suffix=True);pack(payload,arm,altered,al,rotated,suffix=True)
      if arm=='baseline':
       own_suffix_base,own_suffix_logits=altered,al;runner_owners['suffix_baseline']=own_suffix_base
      if arm!='baseline':
       check({'kind':'source_q0','arm':arm,'variant':'suffix','source_exact':byte_exact(altered['source'],own_suffix_base['source'])},own_suffix_base,altered);control_counts['source_q0']+=1
      if arm=='current':check({'arm':arm,'variant':'suffix',**identity_check(own_suffix_base,altered,own_suffix_logits,al,'current')},own_suffix_base,altered,own_suffix_logits,al,comparison='identity');control_counts['current']+=1
      if arm in MATCHED:
       assert all(altered['operations']['timing_checks'][node]['byte_exact'] is True and altered['operations']['timing_checks'][node]['query_slice']==[127,None] for node in TIMING_NODES);control_counts['matched_later']+=1
      check({'arm':arm,**suffix_check(trial,altered,ll,al)},trial,altered,ll[:128],al,comparison='suffix');control_counts['suffix']+=1
      if arm!='baseline':
       runner_owners.pop('main_trial',None);del altered,al,trial,ll
      if arm.endswith('_orth_write_matched_early'):
       cell=arm.split('_',1)[0];writer.journal({'event':'cell_controls_completed','cell':cell,'order_index':probe.order_index,'identity':base})
       receipt=probe.retire_cell(cell);payload['retirement_receipts'].append(receipt);writer.journal({'event':'cell_retired','identity':base,'receipt':receipt});control_counts['retired_cells']+=1
     assert probe.order_index==36 and set(payload['suffix'])==set(ARMS)
     expected=set(ARMS)|({'native_repeat','native_identity'} if boundary=='000' and identity=='natural' and item['window']==0 else set());assert set(payload['metrics'])==expected
     writer.save('inputs/'+group_name+'.npz',payload);group_committed=True;payload=None;runner_owners.clear();own_base=own_logits=own_suffix_base=own_suffix_logits=latest_logits=None;checkpoint();print('CALIBRATION_WORLD_EVALUATION_DONE',group_name,returned_count,flush=True)
     if boundary=='000':c_bundle=None
    c_bundle=c_logits=None
  assert attempts==returned_count==processed==observed_count==580 and main_count==suffix_count==288 and anchor_count==4 and len(writer.inventory)==16
  assert observation_origins=={'hook_model_end':578,'runner_after_return':2}
  assert control_counts=={'current':32,'suffix':288,'matched_later':256,'source_q0':544,'anchors':4,'retired_cells':64}
  assert all(sha(a.inputs/name)==h for name,h in INPUT_HASHES.items()) and all(sha(ROOT/name)==h for name,h in closure.items()) and all(sha(ROOT/name)==h for name,h in METADATA_PINS.items())
  assert sha(ref/'manifest.json')==T18_MANIFEST_SHA and all(sha(ref/name)==dm['calibration_reference_files'][name] for name in ('calibration_maps.npz','calibration_directions.npz'))
  assert maps_digest(maps['F'])==F_DIGEST and maps_digest(ce['maps'])==fit_digests['CE'] and precision_settings()==settings and runtime_behavior()==arithmetic_runtime
  assert all(sha(a.out/'calibration_study'/n)==h for n,h in calibration_files.items()) and sha(acceptance_path)==CALIBRATION_ACCEPTANCE_SHA
  if previous_binding is not None:
   assert validate(a.previous_segment,'A')==(previous_manifest,previous_inventory)
   assert previous_binding=={'manifest_sha256':sha(a.previous_segment/'manifest.json'),'numeric_inventory_sha256':sha(a.previous_segment/'numeric_inventory.json')}
  after=modelhash(model);assert after==before and versions=={name:p._version for name,p in model.named_parameters()}
  meta.update(status='completed',phase='completed',model_hash_after=after,parameter_versions_unchanged=True,precision_settings_after=precision_settings(),all_final_guards_passed=True,completed_utc=time.strftime('%FT%TZ',time.gmtime()));checkpoint(final=True)
 except BaseException as error:
  error_text=str(error);error_type=type(error).__name__;error.__traceback__=None
  preservation={}
  if probe is not None and probe.failure is None:
   try:probe.capture_failure(error)
   except BaseException as capture_error:preservation['hook_capture_error']=repr(capture_error)
  try:
   if payload is not None and not group_committed:preservation['partial_boundary']=writer.save('inputs/'+group_name+'__partial.npz',payload)
  except BaseException as save_error:preservation['partial_boundary_error']=repr(save_error)
  capsule={'error_type':error_type,'error':error_text,'progress':progress(),'current_attempt_tokens':tokens,'current_input_identity':active,
   'hook_failure':None if probe is None else probe.failure,'left':pair_left,'right':pair_right,
   'comparison_logits':pair_logits,
   'current_actual_readout':None if pair_logits is not None or latest_logits is None else latest_logits.detach().to('cpu',copy=True),
   'current_hidden':latest_hidden.detach().to('cpu',copy=True) if latest_hidden is not None and (probe is None or (last_unhooked and pair_logits is None)) else None,
   'current_metrics':latest_metrics,'scoring_status':scoring_status,
   'uncommitted_boundary':payload if 'partial_boundary_error' in preservation else None}
  try:preservation['failure_capsule']=writer.save('failure_state.npz',capsule,failure=True)
  except BaseException as save_error:preservation['failure_capsule_error']=repr(save_error)
  preservation['journal_preservation_errors']=writer.failure_journal_errors
  meta.update(progress(),**resources(),status='failed',phase='failed',failure_type=error_type,failure_message=error_text,preservation=preservation)
  try:checkpoint(final=True,failure=True)
  except BaseException as save_error:
   if writer.owns():json_write(a.out/'checkpoint_failure.json',{'error':repr(save_error),'known_progress':progress()})
  raise
 finally:
  if probe is not None:probe.close()
  if identity_probe is not None:identity_probe.close()
if __name__=='__main__':main()
