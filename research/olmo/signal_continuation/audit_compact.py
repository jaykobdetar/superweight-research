"""No-inference, standalone numerical audit. Requires only NumPy and this bundle.
This author-provided verification is NOT an additional independent audit.
Run: OPENBLAS_NUM_THREADS=1 python audit_compact.py
"""
from pathlib import Path
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent; C=R/'continuation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((R/'EXPORT_DERIVATION.json').read_text()); checked=0
for entry in d['entries']:
 if entry.get('excluded'):continue
 p=R/entry['path'];assert sha(p)==entry['export_sha256'],str(p)
 if 'retained_arrays' in entry:
  with np.load(p,allow_pickle=False) as z:
   for key,ref in entry['retained_arrays'].items():
    a=z[key];assert list(a.shape)==ref['shape'] and str(a.dtype)==ref['dtype']
    assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==ref['sha256'];checked+=1
protocol=json.loads((C/'downstream/frozen_protocol.json').read_text())
summary=json.loads((C/'results/downstream_summary.json').read_text())
interactions=json.loads((C/'results/downstream_interactions.json').read_text())
deriv={x['path']:x for x in d['entries']}
checks=[];means={};max_metric_error=0
for panel in protocol['panels']:
 name=panel['name'];g=panel['group_name'];later=panel['first_later187'];values={}
 treatments=[t['name'] for t in panel['treatments']]
 if panel['source_token']==187:treatments += [g+'_axis_remove_with1710',g+'_complement_remove_with1710']
 for t in treatments:
  folder='axis_controls' if '_axis_remove_' in t or '_complement_remove_' in t else 'downstream'
  rel='continuation/'+folder+'/'+name+'__'+t+'.npz';p=R/rel
  receipt=json.loads(p.with_suffix('.json').read_text())
  assert receipt['result_sha256']==deriv[rel]['original_sha256']
  with np.load(p) as z:loss=z['loss'].copy();agree=z['argmax_agrees_with_native'].copy();state=z['source_post_l1'][0].astype('f8')
  assert loss.shape==(255,) and np.isfinite(loss).all();values[t]=loss
  q=np.arange(len(loss));masks={'after_source':q>=1,'source_query_only':q==0,'before_next_newline':(q>=1)&(q<later),'after_next_newline':q>later}
  state-=state.mean();u=np.full(2048,-1/2048);u[1764]+=1;u/=np.linalg.norm(u);alignment=float(state@u/np.linalg.norm(state))
  for scope,mask in masks.items():
   ref=next(x for x in summary if x['panel']==name and x['treatment']==t and x['scope']==scope)
   got={'mean_nll':float(loss[mask].mean()),'argmax_agreement':float(agree[mask].mean()),'source_alignment':alignment}
   for key,value in got.items():max_metric_error=max(max_metric_error,abs(value-ref[key]))
   assert ref['query_count']==int(mask.sum())
  means[name+'__'+t]=float(loss[1:].mean())
 for left,right in [('native_no_hook','reconstruct_native'),('drop1710',g+'_same_world_addback')]:
  e=abs(values[left]-values[right]);assert e.max()<1e-3 and e.mean()<1e-4
  checks.append({'panel':name,'left':left,'right':right,'max_error':float(e.max()),'mean_absolute_error':float(e.mean())})
 for scope,mask in masks.items():
  ref=next(x for x in interactions if x['panel']==name and x['scope']==scope)
  got=(values[g+'_drop_with1710']-values['drop1710']-values[g+'_drop_only']+values['native_no_hook'])[mask].mean()
  max_metric_error=max(max_metric_error,abs(float(got)-ref['primary_by_group_interaction']))
  if panel['source_token']==187:
   got=(values[g+'_drop_with1710']-values[g+'_axis_remove_with1710']-values[g+'_complement_remove_with1710']+values['drop1710'])[mask].mean()
   max_metric_error=max(max_metric_error,abs(float(got)-ref['axis_by_complement_interaction']))
assert max_metric_error<1e-12
# Recompute all80 primary writes directly from stored activations and pinned scalar.
w=np.load(R/'compact_weight_slices.npz');sc=np.float32(w['l1_row1764'][1710]);donors=json.loads((C/'m0_donors/donor_summary.json').read_text())
max_write=0
for row in donors:
 stem=row['panel']+'__'+row['world']+'__'+row['treatment'];p=C/'m0_donors'/(stem+'.npz')
 with np.load(p) as z:got=float(np.float32(z['l1_activations'][1710])*sc)
 max_write=max(max_write,abs(got-row['primary_raw_write']))
assert max_write==0 and len(donors)==80
# Independent source-horizon reference comparison from compact old losses.
with np.load(R/'archived_reference/matched_losses.npz') as a:
 archive={}
 for old,new in [('clean','native_no_hook'),('delete_column','drop1710')]:
  with np.load(C/'downstream'/('dubois__32768__natural187__'+new+'.npz')) as z:err=z['loss'][1:]-a[old]
  archive[new]={'mean_delta':float(err.mean()),'max_abs':float(abs(err).max())}
result={'all_pass':True,'retained_arrays_verified':checked,'downstream_forward_count':len(means),'numerical_controls':checks,'maximum_saved_metric_error':max_metric_error,'primary_donor_writes_checked':len(donors),'maximum_primary_write_error':max_write,'matched_archived_validation':archive,'mean_nll_after_source':means,'scope':'Author-provided compact verification, no new model inference and no added independent audit claim.'}
print(json.dumps(result,indent=2))
