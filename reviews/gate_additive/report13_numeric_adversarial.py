"""Run only after the reviewed numeric export exists. NumPy-only; no raw model data.
Copies one small numerical export to a disposable directory, never mutating the
repository. Every corrupted file's declared hash is refreshed so tests exercise
semantic guards beyond the outer checksum. Requires production to have completed.
"""
import argparse,contextlib,importlib.util,io,json,resource,shutil,sys,tempfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent/'superweight-causal-circuits'
sys.path.insert(0,str(REPO/'scripts'))
from audit_gate_results import sha
spec=importlib.util.spec_from_file_location('audit_additive_candidate',ROOT/'audit_additive_results.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def read(p):return json.loads(p.read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=REPO);a=ap.parse_args();repo=a.repo
 source=repo/'results/gate_additive';export=repo/'manifests/additive_numeric_export.json'
 assert export.is_file(),'Reviewed export is not available; do not fabricate completed results'
 assert read(source/'manifest.json')['status']=='completed'and read(source/'audit.json')['status']=='passed'
 started=time.time();results=[]
 with tempfile.TemporaryDirectory(prefix='report13-numeric-adversarial-')as td:
  r=Path(td);p=r/'results/gate_additive';shutil.copytree(source,p);(r/'manifests').mkdir();shutil.copyfile(export,r/'manifests/additive_numeric_export.json');M.ROOT=r
  out=r/'audit.json'
  def invoke():
   old=sys.argv;sys.argv=['audit','--output',str(out)]
   try:
    with contextlib.redirect_stdout(io.StringIO()):M.main()
   finally:sys.argv=old
  invoke();baseline=read(out);assert baseline['status']=='passed'
  def run_case(name,file,change):
   target=p/file;original=target.read_bytes();manifest=read(export)
   try:
    change(target)
    manifest['exported_files'][file]=sha(target);dump(r/'manifests/additive_numeric_export.json',manifest)
    try:invoke()
    except (AssertionError,KeyError,ValueError)as e:
     assert read(out)['status']=='checking','Old passed result survived rejection'
     results.append({'name':name,'rejected':True,'error_type':type(e).__name__,'error':str(e)[:300]})
    else:raise AssertionError(('Corruption accepted',name))
   finally:target.write_bytes(original);shutil.copyfile(export,r/'manifests/additive_numeric_export.json')
  def mutate_json(fn):
   def change(path):d=read(path);fn(d);dump(path,d)
   return change
  def mutate_npz(fn):
   def change(path):
    with np.load(path,allow_pickle=False)as z:d={k:z[k].copy()for k in z.files}
    fn(d);np.savez_compressed(path,**d)
   return change
  run_case('duplicate record universe','windows.jsonl',lambda p:p.write_text('\n'.join([p.read_text().splitlines()[0]]+p.read_text().splitlines()[:-1])+'\n'))
  run_case('missing primary array','per_token_numeric.npz',mutate_npz(lambda d:d.pop('0_kl')))
  run_case('unexpected token array','per_token_numeric.npz',mutate_npz(lambda d:d.update(unexpected_tokens=np.zeros(512,dtype=np.int64))))
  run_case('incorrect boolean dtype','per_token_numeric.npz',mutate_npz(lambda d:d.update({'0_matches_p':d['0_matches_p'].astype(np.int64)})))
  ci=next(e['index']for e in read(p/'per_token_index.json')if e['key'].endswith('__current'))
  def swap_control_targets(d):
   x=d[str(ci)+'_loss'];j=next(j for j in range(1,126)if x[j]!=x[0]);x[0],x[j]=x[j],x[0]
  run_case('control per-target drift with unchanged full and segment means','per_token_numeric.npz',mutate_npz(swap_control_targets))
  run_case('changed additive source identity','additive_index.json',mutate_json(lambda d:d[0].update(source_sha256='0'*64)))
  def shift_losses(d):
   for key in ('loss_p','loss_joint','loss_additive','singleton_losses'):d['0_'+key]+=0.01
  run_case('internally consistent FP64 ledger detached from FP32','additive_numeric.npz',mutate_npz(shift_losses))
  run_case('broken J/R identity','additive_numeric.npz',mutate_npz(lambda d:d['0_J'].__setitem__(1,d['0_J'][1]+0.01)))
  run_case('incorrect paired interval','paired_summary.json',mutate_json(lambda d:d[0].update(ci95_delta_nll=[100.,101.])))
  run_case('missing segment universe','segment_summary.json',mutate_json(lambda d:d.pop()))
  run_case('incorrect manifest forward count','manifest.json',mutate_json(lambda d:d.update(forward_passes=1991)))
  # An extra file is outside the fixed29-member schema even if it is harmless text.
  extra=p/'extra.json';dump(extra,{})
  try:
   try:invoke()
   except AssertionError:results.append({'name':'unexpected export member','rejected':True})
   else:raise AssertionError('Extra export member accepted')
  finally:extra.unlink()
  invoke();assert read(out)['status']=='passed';assert sha(export)==sha(r/'manifests/additive_numeric_export.json')
 result={'status':'passed','baseline':baseline,'corruptions':results,'auditor_sha256':sha(ROOT/'audit_additive_results.py'),
  'export_manifest_sha256':sha(export),'no_model_or_raw_tensor_load':True,'gpu_used':False,
  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.time()-started,
  'scope':'Reaggregates only exported numerical scalars/agreements. Does not independently reconstruct logits, gate hooks or the original private input history.'}
 assert result['peak_rss_kib']<3*1024**2
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
