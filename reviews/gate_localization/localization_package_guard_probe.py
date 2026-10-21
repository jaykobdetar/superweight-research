"""CPU adversarial probe of portable NumPy auditor; cloned export only."""
from pathlib import Path
import tempfile,shutil,sys,json,hashlib,importlib.util,contextlib,io,resource
REPO=Path(__file__).resolve().parents[2]/'superweight-causal-circuits';sys.path.insert(0,str(REPO/'scripts'))
spec=importlib.util.spec_from_file_location('localization_portable',REPO/'scripts/audit_localization_results.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
checks=[]
with tempfile.TemporaryDirectory(prefix='localization-portable-guards-')as td:
 p=Path(td);(p/'manifests').mkdir();shutil.copyfile(REPO/'manifests/localization_numeric_export.json',p/'manifests/localization_numeric_export.json');shutil.copytree(REPO/'results/gate_localization',p/'results/gate_localization');mod.ROOT=p
 def run():
  sys.argv=['audit_localization_results.py','--output',str(p/'out.json')]
  try:
   with contextlib.redirect_stdout(io.StringIO()):mod.main()
   return 'accepted'
  except (AssertionError,KeyError,ValueError):return 'rejected'
 def rewrite(name,change):
  target=p/'results/gate_localization'/name;original=target.read_bytes();data=json.loads(original);change(data);target.write_text(json.dumps(data));m=p/'manifests/localization_numeric_export.json';oldm=m.read_bytes();md=json.loads(oldm);md['exported_files'][name]=hashlib.sha256(target.read_bytes()).hexdigest();m.write_text(json.dumps(md));return target,original,m,oldm
 checks.append({'case':'valid original export','result':run()})
 cases=[('summary.json','duplicate summary/missing group',lambda a:a.__setitem__(0,a[1])),('segment_summary.json','invalid later segment name',lambda a:next(r for r in a if r['segment']=='later_128_511').__setitem__('segment','INVALID_SEGMENT')),('paired_summary.json','duplicate paired summary/missing group',lambda a:a.__setitem__(0,a[1]))]
 cases += [('paired_contrasts.json','duplicate contrast/missing input',lambda a:a.__setitem__(0,a[1])),('nonadditivity.json','duplicate nonadditivity/missing input',lambda a:a.__setitem__(0,a[1])),('nonadditivity_summary.json','duplicate nonadditivity group',lambda a:a.__setitem__(0,a[1])),('per_token_index.json','wrong reference baseline',lambda a:a[0].__setitem__('baseline_key',a[0]['p_baseline_key'])),('conditions.json','unknown condition',lambda a:a.__setitem__(0,'UNKNOWN')),('manifest.json','failed manifest',lambda a:a.__setitem__('status','failed')),('audit.json','stale checking audit',lambda a:a.__setitem__('status','checking'))]
 for name,label,change in cases:
  t,old,m,oldm=rewrite(name,change);checks.append({'case':label,'result':run(),'refreshed_export_hash':True});t.write_bytes(old);m.write_bytes(oldm)
assert checks[0]['result']=='accepted' and all(r['result']=='rejected'for r in checks[1:]),checks
result={'status':'passed','checks':checks,'auditor_sha256':hashlib.sha256((REPO/'scripts/audit_localization_results.py').read_bytes()).hexdigest(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_loaded':False,'gpu_used':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
