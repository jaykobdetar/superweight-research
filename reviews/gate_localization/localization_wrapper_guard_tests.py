"""CPU orchestration/error fixtures. Every subprocess is stubbed; no tokenizer/model."""
from pathlib import Path
import tempfile,sys,json,hashlib,importlib.util,contextlib,io,resource,copy
BASE=Path(__file__).resolve().parents[2];REPO=BASE/'superweight-causal-circuits';sys.path.insert(0,str(REPO/'scripts'))
spec=importlib.util.spec_from_file_location('local_wrapper',REPO/'scripts/run_gate_localization.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
calbase=json.loads((BASE/'pulse_transfer/gpt2_gate_calibration_rebuild/manifest.json').read_text());studybase=json.loads((BASE/'pulse_transfer/gpt2_gate_localization_run/manifest.json').read_text());auditbase=json.loads((BASE/'pulse_transfer/gpt2_gate_localization_run/audit.json').read_text())
def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
results=[]
def case(label,change=None,reference=False,prepare=False):
 with tempfile.TemporaryDirectory(prefix='localization-wrapper-fixture-')as td:
  root=Path(td);repo=root/'repo';work=root/'work';dest=work/'pulse_transfer';calls=[]
  identity=copy.deepcopy(json.loads((REPO/'manifests/localization_reproduction.json').read_text()));identity['metadata_sha256']={'evidence.json':digest(b'{}')};identity['calibration_raw_sha256']=digest(b'raw');identity['calibration_maps_sha256']=digest(b'maps')
  meta=repo/'results/gate_localization/input_reconstruction/evidence.json';meta.parent.mkdir(parents=True);meta.write_bytes(b'{}');save(repo/'manifests/localization_reproduction.json',identity)
  cm=copy.deepcopy(calbase);cm.update({k:identity[k]for k in ('calibration_raw_sha256','calibration_maps_sha256','calibration_maps_tensor_digest')});sm=copy.deepcopy(studybase);au=copy.deepcopy(auditbase)
  if change:change(cm,sm,au)
  def buildcal(out):
   out.mkdir(parents=True,exist_ok=True)
   for n in identity['calibration_dependency_files']:(out/n).write_bytes(b'{}')
   (out/'calibration_raw.pt').write_bytes(b'raw');(out/'calibration_maps.pt').write_bytes(b'maps');save(out/'manifest.json',cm)
  def stub(command,**kw):
   script=Path(command[1]).name;calls.append(script)
   if script=='run_gate_study.py':dest.mkdir(parents=True);save(dest/'inputs.local.json',{})
   elif script=='rebuild_gate_calibration.py':buildcal(Path(command[command.index('--out')+1]))
   elif script=='prepare_localization_replay_inputs.py':save(Path(command[command.index('--out')+1])/'reconstruction.json',{'status':'passed','model_forwards':0})
   elif script=='gpt2_gate_localization.py':save(Path(command[command.index('--out')+1])/'manifest.json',sm)
   elif script=='analyze_gate_localization.py':save(Path(command[2])/'audit.json',au)
   else:raise AssertionError(('unexpected subprocess',script))
  m.ROOT=repo;m.subprocess.run=stub;sys.argv=['run_gate_localization.py','--config',str(root/'private.json'),'--work',str(work)]
  if reference:
   buildcal(root/'reference');sys.argv+=['--calibration-reference',str(root/'reference')]
  if prepare:sys.argv+=['--prepare-only']
  try:
   with contextlib.redirect_stdout(io.StringIO()):m.main()
   outcome='accepted'
  except (AssertionError,KeyError,ValueError)as e:outcome='rejected'
  completed=(work/'localization_reproduction.json').exists();entry={'case':label,'outcome':outcome,'calls':calls,'completion_written':completed}
  if completed:entry['accounting']=json.loads((work/'localization_reproduction.json').read_text())['new_total_forwards']
  results.append(entry)
case('prepare only',prepare=True);case('fresh eight-forward calibration');case('explicit calibration reference',reference=True)
case('failed rebuilt calibration',lambda c,s,a:c.update(status='failed'))
case('wrong rebuilt calibration forwards',lambda c,s,a:c.update(forward_passes=7))
case('evaluation performed in calibration',lambda c,s,a:c.update(evaluation_forwards=1))
case('calibration parameter hash changed',lambda c,s,a:c.update(model_hash_after='f'*64))
case('calibration parameter versions changed',lambda c,s,a:c.update(parameter_versions_unchanged=False))
case('failed explicit reference',lambda c,s,a:c.update(status='failed'),reference=True)
case('failed study',lambda c,s,a:s.update(status='failed'))
case('wrong main record count',lambda c,s,a:s.update(records=937))
case('wrong suffix forward count',lambda c,s,a:s.update(suffix_forwards=929))
case('study parameter hash changed',lambda c,s,a:s.update(model_hash_after='f'*64))
case('wrong analyzer count',lambda c,s,a:a.update(records=937))
case('failed analyzer',lambda c,s,a:a.update(status='checking'))
failures=[r['case']for i,r in enumerate(results)if r['outcome']!=('accepted'if i<3 else'rejected')]
assert results[0]['calls']==['run_gate_study.py']and not results[0]['completion_written']
assert results[1].get('accounting')==1876 and results[2].get('accounting')==1868
out={'status':'passed'if not failures else'guard_gaps','failures':failures,'checks':results,'wrapper_sha256':digest((REPO/'scripts/run_gate_localization.py').read_bytes()),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'subprocesses_stubbed':True,'model_loaded':False,'gpu_used':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
