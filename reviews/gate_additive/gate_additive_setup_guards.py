"""Zero-forward configuration-failure and frozen resource-bound checks."""
import importlib,sys,tempfile,json,shutil,hashlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gpt2_gate_additive as g
root=Path(__file__).resolve().parent
results=[]
for label,content in [('missing',None),('malformed','{invalid')]:
 d=root/('gate_additive_setup_'+label);d.mkdir(exist_ok=False)
 cfg=d/'config.json'
 if content is not None:cfg.write_text(content)
 args=['run','--config',str(cfg),'--inputs',str(d/'unused_inputs'),'--calibration-reference',str(d/'unused_reference'),'--out',str(d/'output')]
 with patch.object(sys,'argv',args),patch.object(g.shutil,'disk_usage',return_value=shutil._ntuple_diskusage(20*1024**3,0,20*1024**3)),patch.object(g.AutoModelForCausalLM,'from_pretrained',side_effect=AssertionError('UNEXPECTED MODEL LOAD'))as model:
  try:g.main()
  except (FileNotFoundError,json.JSONDecodeError)as error:expected=type(error).__name__
  else:raise AssertionError('Malformed setup accepted')
  assert model.call_count==0
 m=json.loads((d/'output/manifest.json').read_text())
 assert m['status']=='failed'and m['forward_attempts']==m['forward_passes']==m['records']==0 and not m['model_loaded']and m['failure_type']==expected
 results.append({'case':label,'manifest':str((d/'output/manifest.json').relative_to(root)),'failure_type':expected,'model_loads':0,'forwards':0})
assert g.NORMAL_OUTPUT_LIMIT==2046820352
assert [g.next_group_allowance(*x)for x in [('wiki_validation',0,'natural'),('mill',0,'natural'),('wiki_validation',0,'bos'),('mill',1,'natural')]]==[146000000,104000000,58000000,58000000]
result={'status':'passed','cases':results,'runner_sha256':hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),'model_loads':0,'forwards':0,'disk_usage_mocked':True,'normal_cap':g.NORMAL_OUTPUT_LIMIT}
(root/'gate_additive_setup_guards.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
