"""Small independent arithmetic and isolated-hook audit; no model forwards."""
import ast, hashlib, json
from pathlib import Path
import numpy as np
P=Path('/workspace/shared/olmo_signal_production/continuation')
S=Path('/workspace/shared/olmo_computational_role/downstream')
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=np.load(P/'h14_prefix/traces.npz'); c=np.load(P/'h14_prefix/m0_feature_contributions.npz')
A=c['feature_rows'];coeff=c['coefficients'];names=c['feature_names'].tolist();panels=c['panels'].tolist()
rows=json.loads((P/'h14_prefix/feature_component_decomposition.json').read_text())
gold=np.load(P.parent/'results/mlp1_replay.npz');labels=gold['labels'].tolist()
receipt=json.loads((P/'h14_prefix/receipt.json').read_text())
out={'scope':'Compact numerical/state review and synthetic isolated hooks; no model forwards, no downloads.'}
assert receipt['model_code_sha256']==sha(S/'streamed_olmo.py')
assert receipt['collector_sha256']==sha(P/'collect_h14_prefix.py')
cache=[]
for p in panels:
 for cond in ('newline_U','newline_R','newline_R_edge_H14','newline_R_edge_H0'):
  norm=z[p+'__'+cond+'__l1_normalized'];err=float(abs(norm-gold['normalized_pre_mlp1'][labels.index([p,cond])]).max())
  entry=next(x for x in receipt['checks'] if x['panel']==p and x['condition']==cond)
  assert err==entry['normalized_cache_max_error']
  if cond.startswith('newline_R_edge'):
   baseline=next(x for x in receipt['checks'] if x['panel']==p and x['condition']=='newline_R')
   assert entry['prefix_hashes']==baseline['prefix_hashes']
  cache.append({'panel':p,'condition':cond,'max_error':err})
out['cache_rechecks']=cache
results=[];contrib=[]
for p in panels:
 get=lambda world,k:z[p+'__'+('newline_R' if world=='R' else 'newline_R_edge_H14')+'__'+k].astype(np.float64)
 xr,xe=get('R','l1_pre_mlp'),get('E','l1_pre_mlp')
 sr,se=[np.sqrt(np.dot(x-x.mean(),x-x.mean())/len(x)+1e-5) for x in (xr,xe)]
 nr,ne=A@xr,A@xe
 scale=(1/sr+1/se)/2
 components={label:A@(get('E',k)-get('R',k))*scale for label,k in [('A0','l0_pre_mlp'),('M0','m0_output'),('A1','a1_output')]}
 components['normalization_interaction']=(ne+nr)/2*(1/se-1/sr)
 total=ne/se-nr/sr
 neuron=np.einsum('j,kj->kj',get('E','m0_activations')-get('R','m0_activations'),coeff)*scale
 contrib.append(neuron)
 diff=max(abs(float(row[key])-float(components[key][names.index(row['feature'])])) for row in rows if row['panel']==p for key in components)
 details=[]
 for k in (0,3,5):
  order=np.argsort(-abs(neuron[k]));assert order[0]==1091
  details.append({'feature':names[k],'neuron1091_contribution':float(neuron[k,1091]),'M0_sum':float(components['M0'][k]),'total_feature_change':float(total[k]),'1091_rank':int(np.where(order==1091)[0][0]+1),'top4':order[:4].tolist()})
 results.append({'panel':p,'max_saved_component_error':diff,'max_total_component_closure':float(abs(sum(components.values())-total).max()),'max_neuron_m0_closure':float(abs(neuron.sum(axis=1)-components['M0']).max()),'max_symmetric_product_identity_error':float(abs((ne-nr)*scale+(ne+nr)/2*(1/se-1/sr)-total).max()),'std_R':float(sr),'std_E':float(se),'carry_A0_difference_norm':float(np.linalg.norm(get('E','l0_pre_mlp')-get('R','l0_pre_mlp'))),'selected':details})
out['decomposition']=results
assert max(x['max_saved_component_error'] for x in results)<1e-12
assert np.max(abs(np.stack(contrib)-c['contributions']))<1e-12
selected=(1091,3419,6535,4798,2522,7249)
w=np.memmap(S/'weights/model.layers.0.mlp.down_proj.weight.f32',mode='r',dtype='<f4',shape=(2048,8192))
cols=np.array(w[:,selected],dtype=np.float64);w._mmap.close()
out['selected_coefficients']={'neurons':selected,'float32_elements_read':int(cols.size),'max_independent_error':float(abs(A@cols-coeff[:,selected]).max())}
assert out['selected_coefficients']['max_independent_error']<1e-12
selection=json.loads((P/'h14_prefix/decomposition_checks.json').read_text())
for label,k in [('primary_gate',0),('backup3950_up',3),('backup4786_up',5)]:
 assert np.argsort(-abs(np.stack(contrib)[:,k]).mean(axis=0))[:4].tolist()==selection['candidate_top4_groups'][label]
# Frozen donor input identities and exact transformations, independent from freeze().
protocol=json.loads((P/'m0_donors/protocol.json').read_text());frozen=np.load(P/'m0_donors/inputs.npz')
windows=np.load(S.parent/'inputs/recovered_windows.npz')
assert protocol['runner_sha256']==sha(P/'run_m0_donors.py')
assert protocol['trace_sha256']==sha(P/'h14_prefix/traces.npz')
assert protocol['group_selection_sha256']==sha(P/'h14_prefix/decomposition_checks.json')
for panel in protocol['panels']:
 name=panel['name']
 if panel['split']=='discovery':expected=z[name+'__newline_R__input_ids']
 else:
  src=windows[panel['source_window']]
  assert hashlib.sha256(src.tobytes()).hexdigest()==panel['source_full_sha256']
  expected=src[:129].copy();expected[np.isin(expected,[187,535])]=209;expected[[0,64,128]]=[510,187,187]
 for world in ('R','E'):
  k=name+'__'+world;assert np.array_equal(frozen[k],expected)
  assert hashlib.sha256(frozen[k].tobytes()).hexdigest()==protocol['input_sha256'][k]
assert protocol['group']==selection['candidate_top4_groups']['primary_gate']
out['frozen_inputs']={'panels':8,'identical_R_E_ids':True,'length':129,'intervention_passes':8*2*5,'heldout_native_passes':4*2,'total_new_prefix_passes':88}
# Extract only the local closure's AST. Do not run/import the parent runner.
tree=ast.parse((P/'run_m0_donors.py').read_text());node=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='hook')
compiled=compile(ast.Module(body=[node],type_ignores=[]),'<isolated donor hook>','exec')
rng=np.random.default_rng(742);acts=rng.normal(size=(129,8192)).astype(np.float32);weight=rng.normal(size=(7,8192)).astype(np.float32)
native=rng.normal(size=(129,7)).astype(np.float32);donor={'m0_activations':rng.normal(size=8192).astype(np.float32),'m0_output':rng.normal(size=7).astype(np.float32)}
checks=[]
for label,group in protocol['treatments'].items():
 env={'np':np,'group':group,'donor':donor,'w':weight};exec(compiled,env);hook=env['hook']
 actbefore=acts.copy();before=native.copy();y=hook(0,None,acts,native)
 a=acts[128].copy()
 if group is None:expected=donor['m0_output']
 else:a[group]=donor['m0_activations'][group];expected=a@weight.T
 assert np.array_equal(y[128],expected)
 assert np.array_equal(y[:128],native[:128]) and np.array_equal(acts,actbefore) and np.array_equal(native,before)
 assert hook(1,None,acts,native) is native
 checks.append({'treatment':label,'target_formula_exact':True,'other_positions_exact':True,'input_activations_unchanged':True,'other_layer_identity':True})
out['isolated_donor_hook']=checks
# Validate the actual wrapper's current dependencies without executing it.
validation=json.loads((S/'results/validation_fp32.json').read_text());acq=json.loads((S/'results/acquisition_complete.json').read_text())
assert validation['complete'] and validation['all_pass'] and len(validation['rows'])==10
assert all(x['pass'] and not x['half_round'] for x in validation['rows'])
assert validation['code_sha256']==sha(S/'streamed_olmo.py')
assert acq['all_complete'] and acq['revision']==protocol['checkpoint']
out['execution_gate']={'all_checks_currently_pass':True,'rows':10,'validation_sha256':sha(S/'results/validation_fp32.json'),'model_sha256':sha(S/'streamed_olmo.py'),'acquisition_sha256':sha(S/'results/acquisition_complete.json'),'wrapper_sha256':sha(P/'execute_validated_study.py'),'original_frozen_protocol_sha256':sha(P/'downstream/frozen_protocol.json'),'does_not_grant_CPU_slot':True}
(O/'m0_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
