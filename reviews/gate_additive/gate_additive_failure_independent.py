"""Independent bounded-capsule selection/serialization tests; no model/GPU."""
from pathlib import Path
import sys,tempfile,json,hashlib,resource
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P));torch.set_num_threads(1)
import gate_additive_failure as f
from gpt2_gate_subspace_v2 import fingerprint
x=torch.zeros(128,50257);y=x.clone();y[77,40100]=5.
a=f.logit_pair(x,y);assert 77 in a['query_rows'].tolist()and a['per_query_max_abs'][77]==5
idx=a['query_rows'].tolist().index(77);assert a['right'][idx,40100]==5 and torch.equal(a['left'],x[a['query_rows']])
assert a['left_full_fingerprint']==fingerprint(x)and a['right_full_fingerprint']==fingerprint(y)
y[42,8]=float('nan');b=f.logit_pair(x,y);assert b['nonfinite_query_indices']==[42]and 42 in b['query_rows'].tolist()
assert torch.isnan(b['right'][b['query_rows'].tolist().index(42),8]);assert torch.isinf(b['per_query_max_abs'][42])
with tempfile.TemporaryDirectory(prefix='additive-capsule-independent-')as td:
 p=Path(td)/'capsule.pt';n=f.save_capsule(p,{'finite':a,'nonfinite':b});assert n==p.stat().st_size and n<f.CAPSULE_LIMIT
 q=torch.load(p,weights_only=True,map_location='cpu');assert torch.equal(q['finite']['right'],a['right'])
 old=f.CAPSULE_LIMIT;f.CAPSULE_LIMIT=1024
 try:
  try:f.save_capsule(Path(td)/'oversize.pt',{'tensor':torch.zeros(1024)})
  except AssertionError:pass
  else:raise AssertionError('oversized capsule accepted')
  assert not(Path(td)/'oversize.pt').exists()
 finally:f.CAPSULE_LIMIT=old
shape_rejected=False
try:f.logit_pair(x,y[:127])
except AssertionError:shape_rejected=True
assert shape_rejected
out={'status':'passed','module_sha256':hashlib.sha256((P/'gate_additive_failure.py').read_bytes()).hexdigest(),'checks':['preserves exact maximum-discrepancy row and vocabulary value','retains first nonfinite row and all-query error maxima','both full-tensor fingerprints match','exact serialization/load and measured byte count','oversized payload rejected before file creation','shape mismatch rejected'],'selected_pair_serialization_bytes':n,'not_a_full_production_capsule_size_test':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_loaded':False,'gpu_used':False}
assert out['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
