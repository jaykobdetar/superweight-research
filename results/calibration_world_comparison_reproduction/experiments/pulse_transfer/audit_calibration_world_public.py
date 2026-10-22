"""T23 NumPy-only exact restricted-observation reanalysis SOURCE HOLD.

No weights/corpora/tokenizer/Torch. Optional --raw compares approved scalar
observations and typed ledgers, never fits, launches or reconstructs omitted
activations. Recorded agreement is not regenerated from prediction IDs.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())

def public_rows(records,backend):
 """Use the same frozen reductions; no fabricated prediction array."""
 import numpy as np
 rows=[];anchors=[];gates=[];writes=[]
 expected={(c,w,i,'000') for c in backend.CORPORA for w in range(4) for i in ('natural','bos')};assert set(records)==expected
 fields=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')
 for (c,w,i,b),r in sorted(records.items()):
  baseline=r['metrics']['baseline']
  for arm,z in r['metrics'].items():
   assert backend.exact(z['kl_c'],z['kl_boundary'])
   if arm=='baseline':assert np.count_nonzero(z['kl_boundary'])==0
   if arm=='current' or arm not in backend.ARMS:assert all(backend.exact(z[name],baseline[name]) for name in fields)
   if arm in backend.ARMS:
    for name in fields:assert backend.exact(r['suffix'][arm][name],z[name][:(127 if name=='loss' else 128)])
   for view,(lo,hi) in backend.VIEWS.items():
    row={'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),'kl_baseline':float(z['kl_boundary'][lo:hi].astype(np.float64).mean()),'agreement_baseline':float(z['agreement_boundary'][lo:hi].astype(np.float64).mean())}
    (rows if arm in backend.ARMS else anchors).append(row)
   if arm not in backend.ARMS:continue
   gate=r['gate_norms'][arm][10];suffix=r['suffix_gate_norms'][arm][10]
   for name,a in gate.items():
    if name!='s':assert np.all(a>=0)
    assert backend.exact(suffix[name],a[:127])
   for view,(lo,hi) in backend.GATE_VIEWS.items():gates.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a if name=='source_v_norm' else a[lo:hi],'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in gate.items()}})
   vals=r['write_norms'][arm];svals=r['suffix_write_norms'][arm]
   for name,a in vals.items():
    if name!='signed_target_residual':assert np.all(a>=0) and np.all(svals[name]>=0)
    if name=='lambda':assert np.all(a<=1) and np.all(svals[name]<=1)
    assert backend.exact(svals[name][:128],a[:128])
   if arm in ('baseline','current'):assert all(np.count_nonzero(vals[n])==np.count_nonzero(svals[n])==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'))
   assert all(vals[n][0]==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'))
   for view,(lo,hi) in backend.VIEWS.items():writes.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a[lo:hi],'scored_query_positions') for name,a in vals.items()}})
 assert (len(rows),len(anchors),len(gates),len(writes))==(1728,24,1728,1728)
 return {'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gates,'write_norm_window_summaries':writes,'matching_quality':backend.matching_rows(records),**backend.summarize_rows(rows)}

def audit(package,code,raw=None):
 import numpy as np
 sys.path.insert(0,str(code));import calibration_world_portable as portable
 assert Path(portable.__file__).resolve()==(code/'calibration_world_portable.py').resolve();portable.require_release()
 backend=portable.backend_module(code)
 export=read(package/'export.json');assert set(export)=={'schema','canonical_run_sha256','analysis_sha256','raw_inventory_sha256','audit_private_commitments','adapter_sha256','raw_shards','main_records','suffix_records','numeric_arrays','count_contract','omitted','model_forwards','new_full_model_replay'}
 assert export['schema']=='T23_restricted_numeric_v1' and export['adapter_sha256']==sha(portable.__file__)
 assert export['canonical_run_sha256']==portable.CANONICAL['manifest.json'] and export['analysis_sha256']==portable.CANONICAL[portable.ANALYSIS_NAME] and export['raw_inventory_sha256']==portable.CANONICAL['numeric_inventory.json']
 assert export['count_contract']==portable.COUNT_CONTRACT and export['raw_shards']==32 and export['main_records']==584 and export['suffix_records']==576 and export['model_forwards']==0 and export['new_full_model_replay']=='unperformed'
 assert export['audit_private_commitments']=={k:v['private_sha256'] for k,v in portable.AUDITS.items()}
 expected=portable.numeric_members();inventory=read(package/'numeric_export_inventory.json')
 assert set(inventory)=={'schema','files'} and inventory['schema']=='T23_restricted_numeric_inventory_v1' and set(inventory['files'])==expected-{'numeric_export_inventory.json'}
 assert portable.base.scan_package(package,portable.base.expected_dirs(expected))==expected
 for name,entry in inventory['files'].items():portable.base.relative_member(name);assert entry=={'sha256':sha(package/name),'bytes':(package/name).stat().st_size}
 for name in ('export.json','execution.json',portable.ANALYSIS_NAME,'index.json'):portable.safe(read(package/name))
 execution=read(package/'execution.json')
 assert set(execution)=={'schema','canonical_private_manifest_sha256','retained_scheduled_evaluation_forwards','main','suffix','anchors','separate_completed_calibration_forwards','calibration_manifest_sha256','calibration_audit_sha256','calibration_acceptance_sha256','source_hashes','original_segment_manifests','model_hash','runtime_versions','precision_settings','fresh_model_replay'}
 assert execution['schema']=='T23_public_execution_v1' and execution['canonical_private_manifest_sha256']==export['canonical_run_sha256'] and execution['retained_scheduled_evaluation_forwards']==1160 and execution['source_hashes']==portable.FROZEN_SOURCE_HASHES
 assert (execution['main'],execution['suffix'],execution['anchors'],execution['separate_completed_calibration_forwards'])==(576,576,8,16)
 assert execution['calibration_manifest_sha256']==portable.CALIBRATION_MANIFEST_SHA and execution['calibration_audit_sha256']==portable.CALIBRATION_AUDIT_SHA and execution['calibration_acceptance_sha256']==portable.CALIBRATION_ACCEPTANCE_SHA
 for kind in portable.AUDITS:portable.assert_public_audit(read(package/'audits'/(kind+'.json')),kind)
 analysis=read(package/portable.ANALYSIS_NAME);assert sha(package/portable.ANALYSIS_NAME)==export['analysis_sha256'] and analysis['scope']['analyzer_sha256']==sha(backend.__file__)
 index=read(package/'index.json');names={f'{c}__{w}__{i}__000.npz' for c in backend.CORPORA for w in range(4) for i in ('natural','bos')}
 assert len(index)==32 and {x['file'] for x in index}==names
 if raw is not None:portable.canonical_run(raw,shards=True);raw_inventory=read(raw/'numeric_inventory.json')
 records={};array_count=raw_count=0
 for item in index:
  assert set(item)=={'file','export_sha256','ledger_sha256','raw_sha256','metadata','arrays'};portable.safe(item);name=item['file'];path=package/'inputs'/name
  assert item['raw_sha256']==analysis['scope']['source_hashes'][name] and sha(path)==item['export_sha256'] and sha(path.with_suffix('.json'))==item['ledger_sha256']
  ledger=read(path.with_suffix('.json'));portable.assert_public_ledger(ledger);assert ledger['metadata']==item['metadata'];key=tuple(item['metadata'][k] for k in ('corpus','window','identity','boundary'));assert key not in records
  record={g:{} for g in portable.GROUPS};record.update(metadata=item['metadata'],operations=ledger['operations'],suffix_operations=ledger['suffix_operations']);records[key]=record
  with np.load(path,allow_pickle=False) as saved:
   assert len(saved.files)==len(set(saved.files)) and set(saved.files)==set(item['arrays'])
   for field in saved.files:
    parts=field.split('__');group,arm=parts[:2];assert group in portable.GROUPS and arm in (*backend.ARMS,'native_repeat','native_identity')
    if group in ('gate_norms','suffix_gate_norms'):
     assert len(parts)==4 and parts[2]=='10';record[group].setdefault(arm,{10:{}})[10][parts[3]]=saved[field]
    else:assert len(parts)==3;record[group].setdefault(arm,{})[parts[2]]=saved[field]
   if key[1]==0 and key[2]=='natural':
    for group in ('gate_norms','write_norms'):record[group]['native_repeat']={}
   arrays=portable.safe_arrays(record,backend);assert set(saved.files)==set(arrays) and all(backend.exact(saved[n],a) for n,a in arrays.items());array_count+=len(arrays)
   if raw is not None:
    original=backend.load_npz(raw/'inputs'/name);assert raw_inventory['inputs/'+name]['sha256']==item['raw_sha256'];original_arrays=portable.safe_arrays(original,backend)
    assert set(original_arrays)==set(arrays) and all(backend.exact(a,original_arrays[n]) for n,a in arrays.items())
    assert json.loads(json.dumps(portable.scalar_ledger(original)))==ledger;raw_count+=len(arrays);del original,original_arrays
 result=public_rows(records,backend)
 assert set(analysis)==set(result)|{'scope'}
 for name,value in result.items():assert value==analysis[name],('Exact public statistic',name)
 assert array_count==export['numeric_arrays'] and len(result['matching_quality'])==256
 return {'schema':'T23_public_numeric_reanalysis_v1','status':'passed','numeric_arrays':array_count,'raw_arrays_exact_compared':raw_count,'window_rows':1728,'window_scalar_metric_entries':5184,'anchor_rows':24,'summaries':432,'contrasts':2304,'paired_starts':1152,'primary_comparisons':64,'matching_pairs':256,'all_published_statistics_exact':True,'export_sha256':sha(package/'export.json'),'analysis_sha256':export['analysis_sha256'],'scope':'Recorded five per-target observations plus gate scalar/norm and write traces; agreement not regenerated from omitted IDs. Same frozen bootstrap formulas. Runtime fingerprints/flags are provenance, not primitive reconstruction; fresh1160model replay unperformed.','model_forwards':0,'GPU_used':False,'torch_imported':'torch' in sys.modules}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--code',type=Path,default=ROOT);parser.add_argument('--raw',type=Path);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args();assert not a.report.exists()
 started=time.monotonic()
 try:result=audit(a.package,a.code,a.raw)
 except BaseException as error:
  import traceback
  result={'status':'failed','error':repr(error),'traceback':traceback.format_exc(),'model_forwards':0}
 result.update(auditor_source_sha256=sha(__file__),wall_seconds=time.monotonic()-started,peak_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 assert result['peak_RSS_kib']*1024<=3*1024**3
 with a.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'report_sha256':sha(a.report),'wall_seconds':result['wall_seconds'],'peak_RSS_kib':result['peak_RSS_kib']}));return 0 if result['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
