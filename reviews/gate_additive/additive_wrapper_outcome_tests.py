"""Narrow child-result/status and calibration-identity tests; no subprocess/model."""
import json,sys,tempfile,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import reproduce_additive as W
cases=[]
with tempfile.TemporaryDirectory(prefix='additive-child-status-')as td:
 p=Path(td);assert W.child_outcome(p)['model_forwards']is None;cases.append('absent child remains unknown')
 completed={'status':'completed','phase':'completed','forward_passes':1992,'forward_attempts':1992,'records':1000,'suffix_forwards':992,'calibration_forwards':0,
  'model_hash_before':W.MODEL_HASH,'model_hash_after':W.MODEL_HASH,'parameter_versions_unchanged':True,'calibration_maps_unchanged':True,'calibration_maps_tensor_digest':W.MAP_DIGEST,
  'peak_gpu_allocated_bytes':100,'peak_gpu_reserved_bytes':200}
 W.save(p/'manifest.json',completed);W.save(p/'audit.json',{'status':'passed','forwards':1992,'records':1000})
 success=W.verify_completed_child(p);assert success['status']=='completed'and success['model_forwards']==1992 and success['calibration_forwards']==0 and success['model_loaded']is True and success['gpu_used']is True
 cases.append('verified child success fills completed counters and resource facts')
 W.save(p/'manifest.json',{'status':'failed','forward_passes':101,'forward_attempts':102,'calibration_forwards':0,'peak_gpu_allocated_bytes':77,'peak_gpu_reserved_bytes':88})
 failed=W.child_outcome(p);assert failed['model_forwards']==101 and failed['forward_attempts']==102 and failed['model_loaded']is True and failed['gpu_used']is True
 cases.append('partial failed child preserves actual returned/attempt counts')
 W.save(p/'manifest.json',{'status':'failed','forward_passes':0,'forward_attempts':0,'model_loaded':False,'peak_gpu_allocated_bytes':0,'peak_gpu_reserved_bytes':0})
 pre=W.child_outcome(p);assert pre['model_forwards']==0 and pre['model_loaded']is False and pre['gpu_used']is False and pre['calibration_forwards']is None
 cases.append('preload failure does not invent calibration count')
 (p/'manifest.json').write_text('{');assert W.child_outcome(p)['child_manifest_status']=='unreadable';cases.append('partial JSON remains explicit unknown')
 # Same six substantive artifacts; preserve each actual operations-schema digest.
 accepted={}
 for folder in('gpt2_gate_fixed512_run','gpt2_gate_calibration_rebuild'):
  cal=ROOT/folder;m=W.read(cal/'manifest.json');files=m['source_hashes'];cm,hashes=W.calibration_valid(cal,files)
  accepted[folder]={'manifest_sha256':W.sha(cal/'manifest.json'),'operations_sha256':hashes['calibration_operations.json'],'source_dependencies':files}
  assert all(hashes[n]==h for n,h in W.CAL_SHA.items()if n!='calibration_operations.json')
 cases.append('both exact reviewed calibration operation schemas accepted without inference')
 missing=dict(accepted['gpt2_gate_calibration_rebuild']['source_dependencies']);missing.pop('rebuild_gate_calibration.py')
 try:W.calibration_valid(ROOT/'gpt2_gate_calibration_rebuild',missing)
 except AssertionError:cases.append('missing rebuild source dependency rejected')
 else:raise AssertionError('missing source accepted')
result={'status':'passed','cases':cases,'accepted_calibration_references':accepted,'wrapper_sha256':W.sha(ROOT/'reproduce_additive.py'),
 'model_forwards':0,'gpu_used':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'scope':'Direct child-result helper fixtures and real saved calibration-hash validation only; no child subprocess, tokenizer or model run.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='accepted_calibration_references'},indent=2))
