"""NumPy-only T21 public safe-array audit/reanalysis, no models/network/Torch.

Checks exact membership, SHA/shape/dtype/bytes, recomputes every published
loss/KL/agreement/gate/write statistic,192matching-quality records and all
paired bootstraps from the safe arrays. Optional--raw independently compares
all safe exported arrays to the original private shards without exposing IDs.
Agreement arrays are published observations; omitted prediction IDs cannot be
independently reconstructed from this public package. No scientific replay. Timing fingerprints/control flags remain runtime provenance; raw timing activation rows are excluded.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
FIELDS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def audit(package,code,raw=None):
 sys.path.insert(0,str(code));import analyze_query_write_timing as backend
 import query_write_timing_portable as portable
 exports=read(package/'export.json');assert exports['schema']=='T21_restricted_numeric_v1' and exports['canonical_run_sha256']==portable.CANONICAL['manifest.json'] and exports['analysis_sha256']==portable.CANONICAL['query_write_timing_analysis.json'] and exports['raw_inventory_sha256']==portable.CANONICAL['numeric_inventory.json']
 assert exports['adapter_sha256']==sha(code/'query_write_timing_portable.py') and exports['audit_private_commitments']=={k:v['sha256'] for k,v in portable.AUDITS.items()}
 assert sha(package/'execution.json')==exports['canonical_run_sha256'],'Canonical complete execution bytes'
 for name in ('export.json','execution.json','query_write_timing_analysis.json'):
  portable.safe(read(package/name))
 for kind,spec in portable.AUDITS.items():
  view=read(package/'audits'/(kind+'.json'));portable.safe(view)
  assert view['schema']==spec['schema'] and view['status']=='passed' and view['run_manifest_sha256']==exports['canonical_run_sha256'] and view['auditor_sha256']==sha(package/'audits'/(kind+'.py'))
  assert view['analysis_sha256']==exports['analysis_sha256'] if kind=='numerical' else view['inventory_sha256']==exports['raw_inventory_sha256']
 inventory=read(package/'numeric_export_inventory.json');assert set(inventory)=={'schema','files'} and inventory['schema']=='T21_restricted_numeric_inventory_v1'
 names={f'{c}__{w}__{i}__{b}.npz' for c in backend.CORPORA for w in range(4) for i in ('natural','bos') for b in backend.BOUNDARIES}
 expected={'export.json','execution.json','index.json','query_write_timing_analysis.json','numeric_export_inventory.json'}|{f'audits/{kind}.{ext}' for kind in ('numerical','projection','readout') for ext in ('json','py')}|{f'inputs/{name[:-4]}.{ext}' for name in names for ext in ('npz','json')}
 assert set(inventory['files'])==expected-{'numeric_export_inventory.json'},'Fixed safe numeric membership required'
 for name in inventory['files']:portable.base.relative_member(name)
 assert portable.base.scan_package(package,portable.base.expected_dirs(expected))==expected
 for n,entry in inventory['files'].items():assert entry=={'sha256':sha(package/n),'bytes':(package/n).stat().st_size}
 index=read(package/'index.json');assert len(index)==32;analysis=read(package/'query_write_timing_analysis.json');assert sha(package/'query_write_timing_analysis.json')==exports['analysis_sha256'] and sha(code/'analyze_query_write_timing.py')==analysis['scope']['analyzer_sha256']
 names={f'{c}__{w}__{i}__{b}.npz' for c in backend.CORPORA for w in range(4) for i in ('natural','bos') for b in backend.BOUNDARIES};assert {r['file'] for r in index}==names
 records={};array_count=0;raw_compared=0
 if raw is not None:
  assert sha(raw/'manifest.json')==exports['canonical_run_sha256'] and sha(raw/'numeric_inventory.json')==exports['raw_inventory_sha256'];raw_inv=read(raw/'numeric_inventory.json')
 for item in index:
  portable.safe(item);name=item['file'];assert item['raw_sha256']==analysis['scope']['source_hashes'][name];path=package/'inputs'/name;assert sha(path)==item['export_sha256'] and sha(path.with_suffix('.json'))==item['ledger_sha256'];ledger=read(path.with_suffix('.json'));portable.safe(ledger);assert ledger['metadata']==item['metadata'];key=tuple(item['metadata'][k] for k in ('corpus','window','identity','boundary'));assert key not in records;records[key]={g:{} for g in portable.GROUPS};records[key].update(operations=ledger['operations'],suffix_operations=ledger['suffix_operations'])
  with np.load(path,allow_pickle=False) as saved:
   assert set(saved.files)==set(item['arrays']) and len(saved.files)==len(set(saved.files));original=None
   if raw is not None:
    original_path=raw/'inputs'/name;assert raw_inv['inputs/'+name]['sha256']==item['raw_sha256']==sha(original_path);original=backend.load_npz(original_path)
   for field in saved.files:
    parts=field.split('__');group,arm=parts[:2];assert group in portable.GROUPS and arm in (*backend.ARMS,'native_repeat','native_identity');a=saved[field];assert a.dtype in (np.dtype('float32'),np.dtype('float64'),np.dtype('bool')) and np.isfinite(a).all()
    if group in ('metrics','suffix'):
     assert len(parts)==3 and parts[2] in FIELDS;metric=parts[2];n=127 if group=='suffix' and metric=='loss' else 128 if group=='suffix' else 511;assert a.shape==(n,) and a.dtype==(np.bool_ if metric.startswith('agreement') else np.float32);records[key][group].setdefault(arm,{})[metric]=a.copy();source=original[group][arm][metric] if original is not None else None
    elif group in ('gate_norms','suffix_gate_norms'):
     assert len(parts)==4 and parts[2]=='10' and parts[3] in backend.GATE_KEYS;metric=parts[3];n=12 if metric=='source_v_norm' else 127 if group=='suffix_gate_norms' else 511;assert a.shape==(n,) and a.dtype==(np.float32 if metric=='s' else np.float64);records[key][group].setdefault(arm,{10:{}})[10][metric]=a.copy();source=original[group][arm][10][metric] if original is not None else None
    else:
     assert len(parts)==3 and parts[2] in backend.WRITE_KEYS['baseline' if arm=='native_identity' else arm];metric=parts[2];assert a.shape==(512,) and a.dtype==np.float64;records[key][group].setdefault(arm,{})[metric]=a.copy();source=original[group][arm][metric] if original is not None else None
    if source is not None:assert exact(a,source);raw_compared+=1
    array_count+=1
  if original is not None:assert json.loads(json.dumps(portable.scalar_ledger({k:original[k] for k in ('metadata','checks','operations','suffix_operations')})))==ledger
  # Closed expected array universe, including optional nativeidentity diagnostics.
  for group in ('metrics','suffix'):
   expected_arms=set(backend.ARMS)|({'native_repeat','native_identity'} if group=='metrics' and key[3]=='000' and key[1]==0 and key[2]=='natural' else set());assert set(records[key][group])==expected_arms and all(set(v)==set(FIELDS) for v in records[key][group].values())
  for group in ('gate_norms','suffix_gate_norms'):
   expected_arms=set(backend.ARMS)|({'native_identity'} if group=='gate_norms' and key[3]=='000' and key[1]==0 and key[2]=='natural' else set());assert set(records[key][group])==expected_arms and all(set(v)=={10} and set(v[10])==backend.GATE_KEYS for v in records[key][group].values())
  for group in ('write_norms','suffix_write_norms'):
   expected_arms=set(backend.ARMS)|({'native_identity'} if group=='write_norms' and key[3]=='000' and key[1]==0 and key[2]=='natural' else set());assert set(records[key][group])==expected_arms and all(set(v)==backend.WRITE_KEYS['baseline' if a=='native_identity' else a] for a,v in records[key][group].items())
 rows=[];anchors=[];gate_rows=[];write_rows=[]
 for (c,w,i,b),record in sorted(records.items()):
  metrics=record['metrics'];baseline=metrics['baseline'];native=records[c,w,i,'000']['metrics']['baseline']
  for arm,z in metrics.items():
   if arm=='current' or arm not in backend.ARMS:assert all(exact(z[k],baseline[k]) for k in FIELDS)
   if arm.endswith('_later'):assert all(exact(z[k][:127],baseline[k][:127]) for k in FIELDS)
   if arm in backend.ARMS:
    for k in FIELDS:assert exact(record['suffix'][arm][k],z[k][:127 if k=='loss' else 128])
   for view,(lo,hi) in backend.VIEWS.items():
    row={'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),'delta_nll_c':float((z['loss'][lo:hi].astype(np.float64)-native['loss'][lo:hi].astype(np.float64)).mean()),'delta_nll_boundary':float((z['loss'][lo:hi].astype(np.float64)-baseline['loss'][lo:hi].astype(np.float64)).mean()),**{k:float(z[k][lo:hi].astype(np.float64).mean()) for k in ('kl_c','kl_boundary','agreement_c','agreement_boundary')}};(rows if arm in backend.ARMS else anchors).append(row)
   if arm not in backend.ARMS:continue
   gate=record['gate_norms'][arm][10]
   for name,a in gate.items():assert exact(record['suffix_gate_norms'][arm][10][name],a[:127])
   for view,(lo,hi) in backend.GATE_VIEWS.items():gate_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a if name=='source_v_norm' else a[lo:hi],'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in gate.items()}})
   vals=record['write_norms'][arm]
   for name,a in vals.items():assert exact(record['suffix_write_norms'][arm][name][:128],a[:128])
   for view,(lo,hi) in backend.VIEWS.items():write_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{name:backend.aggregate(a[lo:hi],'scored_query_positions') for name,a in vals.items()}})
 result={'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gate_rows,'write_norm_window_summaries':write_rows,'matching_quality':backend.matching_rows(records),**backend.summarize_rows(rows)}
 for key,value in result.items():assert value==analysis[key],('Public exact reanalysis',key)
 assert array_count==exports['numeric_arrays'];return {'status':'passed','numeric_arrays':array_count,'raw_arrays_exact_compared':raw_compared,'rows':len(rows),'anchors':len(anchors),'summaries':len(result['summaries']),'contrasts':len(result['contrasts']),'paired_start_interactions':len(result['paired_start_interactions']),'matching_quality':len(result['matching_quality']),'all_published_statistics_exact':True,'export_sha256':sha(package/'export.json'),'analysis_sha256':sha(package/'query_write_timing_analysis.json'),'scope':'Safe numerical evidence/reanalysis only. No model replay or reconstruction of omitted IDs/activations.','model_forwards':0,'gpu_used':False,'torch_imported':'torch' in sys.modules}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True);parser.add_argument('--code',type=Path,default=ROOT);parser.add_argument('--raw',type=Path);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args();assert not a.report.resolve().is_relative_to(a.package.resolve()),'Report must be outside immutable package';started=time.time();result={'status':'running'}
 try:result=audit(a.package,a.code,a.raw)
 except BaseException as error:result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  result.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  with a.report.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
  print(json.dumps(result,indent=2))
if __name__=='__main__':main()
