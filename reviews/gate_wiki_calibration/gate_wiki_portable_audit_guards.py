"""CPU adversarial tests for standalone NumPy T18 audit; originals untouched."""
import sys,json,copy,tempfile,shutil,os,resource,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];repo=R.parent/'superweight-causal-circuits';sys.path.insert(0,str(repo/'scripts'))
import audit_wiki_calibration_results as a
start=time.time();checks=[]
def reject(name,fn):
 try:fn()
 except(AssertionError,KeyError,ValueError):checks.append(name)
 else:raise AssertionError('accepted '+name)
run=repo/'results/gate_wiki_calibration';f=run/'inputs/gate__ethics__0__512__bos.npz'
with np.load(f,allow_pickle=False)as z:base={k:z[k]for k in z.files}
a.validate_arrays(base,'ethics','bos',0)
for name,mut in [('missing_array',lambda x:x.pop('C__loss')),('unknown_array',lambda x:x.__setitem__('tokens',np.zeros(512,dtype='int64'))),('wrong_dtype',lambda x:x.__setitem__('P__loss',x['P__loss'].astype('float64'))),('wrong_shape',lambda x:x.__setitem__('P__loss',x['P__loss'][:-1])),('nonfinite',lambda x:x['remove_FF_all__loss'].__setitem__(3,float('nan'))),('copy_permutation_same_mean',lambda x:x.__setitem__('current__loss',np.r_[x['current__loss'][:1],x['current__loss'][1:127][::-1],x['current__loss'][127:]])),('suffix_permutation_same_mean',lambda x:x.__setitem__('suffix__remove_FF_all__loss',x['suffix__remove_FF_all__loss'][::-1])),('norm_mismatch',lambda x:x['norm__remove_OO_L10__10__hypothetical_pc1_norm'].__setitem__(2,1000.)),('negative_norm',lambda x:x['norm__remove_OO_L10__10__mass_delta_norm'].__setitem__(3,-1.)),('baseline_boolean',lambda x:x['C__agreement_c'].__setitem__(5,False))]:
 d={k:v.copy()for k,v in base.items()};mut(d);reject(name,lambda:a.validate_arrays(d,'ethics','bos',0))
reject('duplicate_rows',lambda:a.keyed([{'x':1},{'x':1}],('x',),{(1,)}));reject('missing_rows',lambda:a.keyed([],('x',),{(1,)}));reject('unknown_view',lambda:a.keyed([{'view':'unknown'}],('view',),{('full',)}))
# Full valid pass; each end-to-end mutation updates export hashes deliberately,
# so rejection tests semantics rather than merely stale file hashes.
passed=a.audit(repo)
with tempfile.TemporaryDirectory(prefix='t18-public-audit-',dir=R/'review')as td:
 t=Path(td);(t/'manifests').mkdir();shutil.copytree(run,t/'results/gate_wiki_calibration',copy_function=os.link);mp=t/'manifests/wiki_calibration_numeric_export.json';original=(repo/'manifests/wiki_calibration_numeric_export.json').read_bytes();mp.write_bytes(original)
 def mutate_file(name,change):
  p=t/'results/gate_wiki_calibration'/name;data=json.loads((run/name).read_text());change(data);p.unlink();p.write_text(json.dumps(data));meta=json.loads(original);meta['exported_files'][name]=a.sha(p);mp.write_text(json.dumps(meta))
 def reset(name):
  p=t/'results/gate_wiki_calibration'/name;p.unlink();os.link(run/name,p);mp.write_bytes(original)
 cases=[('wrong_model','manifest.json',lambda d:d.__setitem__('model_id','other')),('wrong_revision','manifest.json',lambda d:d.__setitem__('revision','0'*40)),('wrong_model_digest','manifest.json',lambda d:d.update(model_hash_before='0'*64,model_hash_after='0'*64)),('wrong_map_digest','manifest.json',lambda d:d['calibration_map_digests'].__setitem__('F','0'*64)),('wrong_readout_digest','manifest.json',lambda d:d['readout']['weight'].__setitem__('sha256','0'*64)),('wrong_bootstrap_scope','wiki_factorial_analysis.json',lambda d:d['statistical_scope'].__setitem__('bootstrap_seed',1)),('wrong_contrast_scope','wiki_factorial_analysis.json',lambda d:d['contrasts'][0].__setitem__('scope','unconditional')),('wrong_layout_adapter','saved_state_audit.json',lambda d:d['layout_replay'].__setitem__('adapter_sha256','0'*64)),('wrong_layout_policy','saved_state_audit.json',lambda d:d['layout_replay'].__setitem__('tolerances_changed',True)),('failed_manifest','manifest.json',lambda d:d.__setitem__('status','failed')),('duplicate_index','index.json',lambda d:d.append(copy.deepcopy(d[0]))),('missing_summary','wiki_factorial_analysis.json',lambda d:d['summary'].pop()),('duplicate_contrast','wiki_factorial_analysis.json',lambda d:d['contrasts'].append(copy.deepcopy(d['contrasts'][0]))),('unknown_segment','wiki_factorial_analysis.json',lambda d:d['window_metrics'][0].__setitem__('view','invalid')),('missing_exact_flag','inputs/gate__ethics__0__512__bos.json',lambda d:d['suffix_checks'][0].pop('logits_exact')),('duplicate_check','inputs/gate__ethics__0__512__bos.json',lambda d:d['checks'].append(copy.deepcopy(d['checks'][0]))),('missing_gate_error','inputs/gate__ethics__0__512__bos.json',lambda d:d['operations'][1]['gate_checks']['4'].pop('formula_error'))]
 for name,path,change in cases:
  mutate_file(path,change);reject(name,lambda:a.audit(t));reset(path)
passed['guard_checks']=checks;passed['guard_count']=len(checks);passed['fixture_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;passed['fixture_wall_seconds']=time.time()-start
out=R/'review/gate_wiki_portable_audit_guards.json';out.write_text(json.dumps(passed,indent=2));print(json.dumps(passed,indent=2))
