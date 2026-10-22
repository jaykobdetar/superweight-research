"""T22 NumPy-only restricted-array audit and exact public statistical replay.

No weights, Torch, source corpora, IDs or raw activations. Observed agreement
arrays are checked, not regenerated from omitted prediction IDs. Optional --raw
compares every allowed array and typed ledger against the canonical private
shards. This is numerical reanalysis, never a second 776-forward model replay.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
FIELDS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()

def public_rows(records,backend):
 """Frozen T22 aggregate formulas; deliberately no fabricated pred observation."""
 rows=[];anchors=[];gate_rows=[];write_rows=[];l11_rows=[]
 for (c,w,i,b),record in sorted(records.items()):
  metrics=record['metrics'];baseline=metrics['baseline']
  for arm,z in metrics.items():
   assert exact(z['kl_c'],z['kl_boundary'])
   if arm=='baseline':assert np.count_nonzero(z['kl_boundary'])==0
   if arm=='current' or arm not in backend.ARMS:assert all(exact(z[name],baseline[name]) for name in FIELDS)
   if arm in backend.ARMS:
    for name in FIELDS:assert exact(record['suffix'][arm][name],z[name][:127 if name=='loss' else 128])
   for view,(lo,hi) in backend.VIEWS.items():
    row={'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),'kl_baseline':float(z['kl_boundary'][lo:hi].astype(np.float64).mean()),'agreement_baseline':float(z['agreement_boundary'][lo:hi].astype(np.float64).mean())}
    (rows if arm in backend.ARMS else anchors).append(row)
   if arm not in backend.ARMS:continue
   gate=record['gate_norms'][arm][10]
   for name,a in gate.items():
    assert exact(record['suffix_gate_norms'][arm][10][name],a[:127])
    if name!='s':assert (a>=0).all()
   for view,(lo,hi) in backend.GATE_VIEWS.items():gate_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a if name=='source_v_norm' else a[lo:hi],'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in gate.items()}})
   values=record['write_norms'][arm]
   for name,a in values.items():
    sa=record['suffix_write_norms'][arm][name];assert exact(sa[:128],a[:128])
    if name!='signed_target_residual':assert (a>=0).all() and (sa>=0).all()
    if name=='lambda':assert (a<=1).all() and (sa<=1).all()
   if arm in ('baseline','current')+backend.RECIPIENTS:assert all(np.count_nonzero(values[n])==np.count_nonzero(record['suffix_write_norms'][arm][n])==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'))
   assert all(values[n][0]==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'))
   for view,(lo,hi) in backend.VIEWS.items():write_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a[lo:hi],'scored_query_positions') for name,a in values.items()}})
   values=record['l11_norms'][arm]
   for name,a in values.items():
    sa=record['suffix_l11_norms'][arm][name];assert (a>=0).all() and (sa>=0).all() and exact(sa[:128],a[:128])
   for view,(lo,hi) in backend.VIEWS.items():l11_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':11,'values':{name:backend.aggregate(a[lo:hi],'scored_query_positions') for name,a in values.items()},'active_early_values':{name:backend.aggregate(a[1:127],'active_E_queries1..126') for name,a in values.items()}})
 assert tuple(map(len,(rows,anchors,gate_rows,write_rows,l11_rows)))==(1152,24,1152,1152,1152)
 return {'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gate_rows,'write_norm_window_summaries':write_rows,'l11_norm_window_summaries':l11_rows,'matching_quality':backend.matching_rows(records),**backend.summarize_rows(rows)}

def audit(package,code,raw=None):
 sys.path.insert(0,str(code));import analyze_l11_key_value_donor as backend
 import l11_key_value_donor_portable as portable
 assert Path(backend.__file__).resolve()==(code/'analyze_l11_key_value_donor.py').resolve() and sha(backend.__file__)==portable.FROZEN_SOURCE_HASHES['analyze_l11_key_value_donor.py']
 assert Path(portable.__file__).resolve()==(code/'l11_key_value_donor_portable.py').resolve()
 exports=read(package/'export.json');assert exports['schema']=='T22_restricted_numeric_v1'
 assert exports['canonical_run_sha256']==portable.CANONICAL['manifest.json'] and exports['analysis_sha256']==portable.CANONICAL['l11_key_value_donor_analysis.json'] and exports['raw_inventory_sha256']==portable.CANONICAL['numeric_inventory.json']
 assert exports['adapter_sha256']==sha(portable.__file__) and exports['audit_private_commitments']=={k:v['sha256'] for k,v in portable.AUDITS.items()}
 expected=portable.numeric_members();inventory=read(package/'numeric_export_inventory.json')
 assert set(inventory)=={'schema','files'} and inventory['schema']=='T22_restricted_numeric_inventory_v1' and set(inventory['files'])==expected-{'numeric_export_inventory.json'}
 assert portable.base.scan_package(package,portable.base.expected_dirs(expected))==expected
 for name,entry in inventory['files'].items():portable.base.relative_member(name);assert entry=={'sha256':sha(package/name),'bytes':(package/name).stat().st_size}
 for name in ('export.json','execution.json','l11_key_value_donor_analysis.json','index.json'):portable.safe(read(package/name))
 execution=read(package/'execution.json');assert execution['canonical_private_manifest_sha256']==exports['canonical_run_sha256'] and execution['retained_scheduled_forwards']==776 and execution['actual_cumulative_acquisition']==803 and execution['source_hashes']==portable.FROZEN_SOURCE_HASHES
 for kind,spec in portable.AUDITS.items():
  view=read(package/'audits'/(kind+'.json'));portable.assert_public_audit(view,kind)
  assert view['schema']==spec['schema'] and view['status']=='passed' and view['run_manifest_sha256']==exports['canonical_run_sha256'] and view['private_report_sha256']==spec['sha256'] and view['auditor_sha256']==spec['auditor_sha256']
  assert view['analysis_sha256']==exports['analysis_sha256'] if kind=='numerical' else view['inventory_sha256']==exports['raw_inventory_sha256']
 analysis=read(package/'l11_key_value_donor_analysis.json');assert sha(package/'l11_key_value_donor_analysis.json')==exports['analysis_sha256'] and analysis['scope']['analyzer_sha256']==sha(backend.__file__)
 index=read(package/'index.json');names={f'{c}__{w}__{i}__000.npz' for c in backend.CORPORA for w in range(4) for i in ('natural','bos')};assert len(index)==32 and {x['file'] for x in index}==names
 if raw is not None:portable.canonical_run(raw,shards=True);raw_inventory=read(raw/'numeric_inventory.json')
 records={};array_count=0;raw_count=0
 for item in index:
  portable.safe(item);name=item['file'];path=package/'inputs'/name
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
    for group in ('gate_norms','write_norms','l11_norms'):record[group]['native_repeat']={}
   expected_arrays=portable.safe_arrays(record,backend);assert set(saved.files)==set(expected_arrays)
   assert all(exact(saved[name],a) for name,a in expected_arrays.items());array_count+=len(expected_arrays)
   if raw is not None:
    original=backend.load_npz(raw/'inputs'/name);assert raw_inventory['inputs/'+name]['sha256']==item['raw_sha256']
    original_arrays=portable.safe_arrays(original,backend);assert set(original_arrays)==set(expected_arrays)
    assert all(exact(a,original_arrays[field]) for field,a in expected_arrays.items());raw_count+=len(expected_arrays)
    assert json.loads(json.dumps(portable.scalar_ledger(original)))==ledger
    del original,original_arrays
  # Record arrays remain owned NumPy values, with no model or Torch import.
 result=public_rows(records,backend)
 for name,value in result.items():assert value==analysis[name],('Exact public statistical reanalysis',name)
 assert array_count==exports['numeric_arrays'] and len(result['matching_quality'])==64
 return {'schema':'T22_public_numeric_reanalysis_v1','status':'passed','numeric_arrays':array_count,'raw_arrays_exact_compared':raw_count,'window_rows':1152,'anchor_rows':24,'summaries':288,'contrasts':1368,'paired_starts':684,'primary_comparisons':128,'matching_pairs':64,'all_published_statistics_exact':True,'export_sha256':sha(package/'export.json'),'analysis_sha256':exports['analysis_sha256'],'scope':'Recorded safe metrics/norms and inherited frozen bootstrap formulas. Agreement not recomputed from omitted IDs; runtime activation fingerprints not primitive regeneration; second model replay unperformed.','model_forwards':0,'GPU_used':False,'torch_imported':'torch' in sys.modules}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--code',type=Path,default=ROOT);parser.add_argument('--raw',type=Path);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
 assert not a.report.resolve().is_relative_to(a.package.resolve()),'Report outside immutable package'
 started=time.monotonic();result={'status':'running'}
 try:result=audit(a.package,a.code,a.raw)
 except BaseException as error:result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  with a.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps(result,indent=2))
if __name__=='__main__':main()
