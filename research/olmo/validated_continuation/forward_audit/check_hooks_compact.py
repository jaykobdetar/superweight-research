"""Independent compact-column hook semantics and fixed-factorial audit; no full weights."""
from pathlib import Path
import numpy as np,json,importlib.util,hashlib
R=Path(__file__).resolve().parent;P=Path('/workspace/shared/olmo_signal_production/continuation');spec=importlib.util.spec_from_file_location('hooks',P/'alternate_writer_hooks.py');m=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=m;spec.loader.exec_module(m)
z=np.load(P.parent/'results/mlp1_replay.npz');c=np.load(P.parent/'inputs/compact_down_projection.npz');ids=c['neuron_ids'];C=c['columns'];lookup={int(j):k for k,j in enumerate(ids)};D,N=2048,8192
class Transpose:
 __array_priority__=10000
 def __rmatmul__(self,a):return a[:,ids]@C.T
class CompactDown:
 shape=(D,N);T=Transpose()
 def __getitem__(self,key):
  rows,js=key
  return np.stack([C[:,lookup[int(j)]] if int(j) in lookup else np.zeros(D,dtype=C.dtype) for j in js],axis=1)[rows]
down=CompactDown();axis=np.full(D,-1/D);axis[1764]+=1;axis/=np.linalg.norm(axis);act=z['activation'][[0,2,7,11,15]].copy();h=z['normalized_pre_mlp1'][[0,2,7,11,15]].copy();native=act[:,ids]@C.T;native[[0,2,4]]=np.arange(D,dtype='f4')[None,:]/1000;before=[a.copy() for a in [act,h,native]];pos=np.array([1,3]);checks=[]
for name,group in [('source_negative',m.SOURCE_NEGATIVE_GROUP),('newline_positive',m.NEWLINE_POSITIVE_GROUP),('h14_query',m.H14_QUERY_GROUP)]:
 results={}
 for sp in m.group_factorial(group,positions=tuple(pos),prefix=name):
  fn=m.make_hook(down,sp);got=fn(1,h,act,native);results[sp.name]=got
  coeff=act[pos][:,ids].astype('f8').copy()
  if sp.keep is not None:coeff[:,[k for k,j in enumerate(ids) if j not in sp.keep]]=0
  else:coeff[:,[lookup[j] for j in sp.drop]]=0
  expected=coeff@C.astype('f8').T
  if sp.partial_group:
   gw=act[pos][:,list(group)].astype('f8')@C[:,[lookup[j] for j in group]].astype('f8').T;ax=(gw@axis)[:,None]*axis;expected-=ax if sp.remove_part=='axis' else gw-ax
  if sp.addback:expected+=act[pos][:,list(sp.addback)].astype('f8')@C[:,[lookup[j] for j in sp.addback]].astype('f8').T
  err=float(abs(got[pos]-expected).max());local=bool(np.array_equal(got[[0,2,4]],native[[0,2,4]]));unchanged=all(np.array_equal(a,b) for a,b in zip([act,h,native],before));other=fn(0,h,act,native) is native and fn(2,h,act,native) is native
  assert err<2e-4 and local and unchanged and other;checks.append({'group':name,'intervention':sp.name,'max_abs_error':err,'nonselected_exact':local,'inputs_unchanged':unchanged,'other_layers_identity':other})
 add=float(abs(results[name+'_same_world_addback'][pos]-results['drop1710'][pos]).max());checks.append({'group':name,'same_world_addback_vs_primary_drop_max_abs':add})
 # Audit raw-mean complement identity separately.
 gw=act[pos][:,list(group)].astype('f8')@C[:,[lookup[j] for j in group]].astype('f8').T;ax=(gw@axis)[:,None]*axis;comp=gw-ax
 checks.append({'group':name,'axis_mean_max_abs':float(abs(ax.mean(-1)).max()),'complement_mean_vs_raw_mean_max_abs':float(abs(comp.mean(-1)-gw.mean(-1)).max()),'complement_dot_axis_max_abs':float(abs(comp@axis).max()),'sum_reconstructs_raw_max_abs':float(abs(ax+comp-gw).max())})
rows=json.loads((P/'results/alternate_gate_up_factorials.json').read_text());labels=z['labels'].tolist();maxerr=0
for r in rows:
 if r['panel']=='source_The_vs_newline':ir,ie=0,2
 else:ir=labels.index([r['panel'],'newline_R']);ie=labels.index([r['panel'],'newline_R_edge_H14'])
 j=r['neuron'];g=float(z['gate'][ir if r['gate_donor']=='R' else ie,j]);u=float(z['up'][ir if r['up_donor']=='R' else ie,j]);col=C[:,lookup[j]].astype('f8');w=(col[1764]-col.mean())/np.sqrt(1-1/D);v=g/(1+np.exp(-g))*u*w;maxerr=max(maxerr,abs(v-r['axis_write']))
assert maxerr<1e-10
out={'scope':__doc__,'hook_sha256':hashlib.sha256((P/'alternate_writer_hooks.py').read_bytes()).hexdigest(),'analyze_controls_sha256':hashlib.sha256((P/'analyze_controls.py').read_bytes()).hexdigest(),'existing_validation_sha256':hashlib.sha256((P/'results/hook_validation.json').read_bytes()).hexdigest(),'compact_sparse_model_caveat':'Hook tests use nine saved columns with all other columns zero, so they verify semantics and locality, not the full actual model intervention.','tests':checks,'fixed_factorial_rows':len(rows),'fixed_factorial_max_abs_recompute_error':maxerr}
(R/'hook_compact_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='tests'},indent=2));print('max hook error',max(r.get('max_abs_error',0) for r in checks));print('addback',[(r['group'],r['same_world_addback_vs_primary_drop_max_abs']) for r in checks if 'same_world_addback_vs_primary_drop_max_abs' in r])
