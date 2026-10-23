"""Read-only compact checksum, donor-direction and sparse output-delta audit."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path('/workspace/shared/olmo_signal_production/continuation');R=P/'m0_donors';O=Path(__file__).resolve().parent
S=Path('/workspace/shared/olmo_computational_role/downstream')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads((R/'protocol.json').read_text());identity=json.loads((R/'run_identity.json').read_text());tr=np.load(P/'h14_prefix/traces.npz')
assert identity['model_sha256']==sha(S/'streamed_olmo.py') and identity['protocol_sha256']==sha(R/'protocol.json')
w=np.memmap(S/'weights/model.layers.0.mlp.down_proj.weight.f32',mode='r',dtype='<f4',shape=(2048,8192));cols=np.array(w[:,p['group']],dtype=np.float64);w._mmap.close()
checks=[];n=0
for panel in p['panels']:
 name=panel['name'];native={}
 for world in ('R','E'):
  if panel['split']=='discovery':
   prefix=name+'__'+('newline_R' if world=='R' else 'newline_R_edge_H14')+'__'
   native[world]={k:tr[prefix+k] for k in ('m0_activations','m0_output','l1_normalized')}
  else:
   stem=name+'__'+world+'__native';z=np.load(R/(stem+'.npz'));j=json.loads((R/(stem+'.json')).read_text())
   assert j['result_sha256']==sha(R/(stem+'.npz')) and j['run_identity']==identity;n+=1
   native[world]={k:z[k] for k in ('m0_activations','m0_output','l1_normalized')}
 for world in ('R','E'):
  donor=native['E' if world=='R' else 'R'];local=native[world]
  sham=np.load(R/(name+'__'+world+'__sham_reproject.npz'))
  sham_error=float(abs(sham['l1_normalized']-local['l1_normalized']).max())
  assert sham_error<5e-4
  for label,group in p['treatments'].items():
   stem=name+'__'+world+'__'+label;z=np.load(R/(stem+'.npz'));j=json.loads((R/(stem+'.json')).read_text())
   assert j['result_sha256']==sha(R/(stem+'.npz')) and j['run_identity']==identity;n+=1
   assert j['panel']==name and j['world']==world and j['treatment']==label
   assert np.array_equal(z['m0_activations'],local['m0_activations'])
   if group is None:
    err=float(abs(z['m0_output']-donor['m0_output']).max());assert err==0
   else:
    indices=[p['group'].index(jj) for jj in group]
    delta=donor['m0_activations'][group].astype(np.float64)-local['m0_activations'][group].astype(np.float64)
    expected=sham['m0_output'].astype(np.float64)+cols[:,indices]@delta
    err=float(abs(z['m0_output']-expected).max());assert err<1e-5,(stem,err)
   checks.append({'panel':name,'world':world,'treatment':label,'sparse_delta_output_max_error':err,'sham_normalized_max_error':sham_error})
out={'verified_result_receipts':n,'intervention_results':len(checks),'same_native_pre_hook_activations_all':True,'full_output_donor_exact':True,'max_sparse_delta_error':max(x['sparse_delta_output_max_error'] for x in checks),'max_sham_normalized_error':max(x['sham_normalized_max_error'] for x in checks),'checks':checks}
(O/'donor_result_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
