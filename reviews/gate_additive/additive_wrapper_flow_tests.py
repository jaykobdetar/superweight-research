"""Actual wrapper main with stubbed child commands, no inference/tokenization.
Uses already verified small input metadata. Both success and partial failure
exercise the final persisted parent state, not merely result helper functions.
"""
import json,sys,tempfile,subprocess,resource,shutil
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import reproduce_additive as W
source=ROOT/'additive_replay_preparation';cfg=W.read(source/'inputs.local.json')
files=W.read(source/'source_inventory.json')['files'];files['reproduce_additive.py']=W.sha(ROOT/'reproduce_additive.py')
for n,h in W.read(ROOT/'gpt2_gate_calibration_rebuild/manifest.json')['source_hashes'].items():files[n]=h
parent=W.read(ROOT/'gpt2_run/manifest.json');actualsha=W.sha
weightfiles={str(Path(cfg['models']['gpt2']['path'])/n):v['sha256']for n,v in parent['checkpoint_files'].items()}
def fixture_sha(p):return weightfiles.get(str(p))or actualsha(p)
completed={'status':'completed','phase':'completed','forward_passes':1992,'forward_attempts':1992,'records':1000,'suffix_forwards':992,'calibration_forwards':0,
 'model_hash_before':W.MODEL_HASH,'model_hash_after':W.MODEL_HASH,'parameter_versions_unchanged':True,'calibration_maps_unchanged':True,'calibration_maps_tensor_digest':W.MAP_DIGEST,
 'peak_gpu_allocated_bytes':100,'peak_gpu_reserved_bytes':200}
results={}
with tempfile.TemporaryDirectory(prefix='additive-wrapper-flow-')as td:
 temp=Path(td);manifest=temp/'sources.json';W.save(manifest,{'files':files})
 for mode in('success','partial_failure'):
  work=temp/mode;calls=[]
  def invoke(cmd,**kwargs):
   script=Path(cmd[1]).name;calls.append(script)
   if script=='prepare_additive_replay_inputs.py':
    out=Path(cmd[cmd.index('--out')+1]);shutil.copytree(source/'inputs_v2',out)
   elif script=='gpt2_gate_additive.py':
    out=Path(cmd[cmd.index('--out')+1]);out.mkdir()
    if mode=='success':W.save(out/'manifest.json',completed)
    else:
     W.save(out/'manifest.json',{'status':'failed','forward_passes':101,'forward_attempts':102,'calibration_forwards':0,'peak_gpu_allocated_bytes':77,'peak_gpu_reserved_bytes':88})
     raise subprocess.CalledProcessError(17,cmd)
   elif script=='analyze_gate_additive.py':W.save(Path(cmd[2])/'audit.json',{'status':'passed','forwards':1992,'records':1000})
   else:raise AssertionError(('Unexpected child',cmd))
   return subprocess.CompletedProcess(cmd,0)
  argv=['wrapper','--config',str(source/'inputs.local.json'),'--metadata',str(ROOT/'additive_replay_metadata'),'--calibration-reference',str(ROOT/'gpt2_gate_calibration_rebuild'),
   '--parent-metadata',str(ROOT/'gpt2_run'),'--source-dir',str(ROOT),'--source-manifest',str(manifest),'--work',str(work),'--execute']
  with patch.object(sys,'argv',argv),patch.object(W.subprocess,'run',invoke),patch.object(W,'sha',fixture_sha):
   try:W.main()
   except subprocess.CalledProcessError:assert mode=='partial_failure'
   else:assert mode=='success'
  state=W.read(work/'additive_preparation.json');expected=1992 if mode=='success'else 101
  assert state['status']==('completed'if mode=='success'else'failed')and state['model_forwards']==expected
  assert state['calibration_forwards']==0 and state['model_loaded']is True and state['gpu_used']is True
  assert state['calibration_operations_schema']=='reviewed_eight_forward_rebuild'
  assert state['calibration_files_sha256']['calibration_operations.json']==W.REBUILD_OPERATIONS_SHA
  assert calls==['prepare_additive_replay_inputs.py','gpt2_gate_additive.py']+(['analyze_gate_additive.py']if mode=='success'else[])
  assert (work/'additive_reproduction.json').exists()==(mode=='success')
  results[mode]={'persisted_status':state['status'],'forward_passes_from_stub':state['model_forwards'],'attempts_from_stub':state['forward_attempts'],'child_calls':calls}
result={'status':'passed','cases':results,'wrapper_sha256':actualsha(ROOT/'reproduce_additive.py'),'actual_model_forwards':0,'tokenizer_calls':0,'gpu_used':False,
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Child commands and checkpoint hashes are stubs for wrapper bookkeeping; file sizes and staged code/calibration identities remain real. Not a model reproduction.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
