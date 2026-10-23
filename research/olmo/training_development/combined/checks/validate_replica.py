"""Compare a fresh authorized replay with preserved original results, ignoring timings/RSS."""
from pathlib import Path
import json,numpy as np
B=Path(__file__).resolve().parents[2];rows=[]
keys=['gate','up','activation','contribution','l0_attention_q_to_0_p_q','l1_attention_q_to_0_p_q','pre_l1_centered_norm','pre_mlp1_centered_norm','l0_H14_context_norm_q','l1_H12_context_norm_q']
for phase in ['phase2','phase3']:
 R=B/phase;P=json.loads((R/'protocol.json').read_text());assert (R/'results_recorded').exists(),'Preserve original results first'
 for ref in P['checkpoints']:
  for mode in ['float32','fp16_approx']:
   old=json.loads((R/'results_recorded'/ref['name']/(mode+'.json')).read_text());new=json.loads((R/'results'/ref['name']/(mode+'.json')).read_text());assert old['revision']==new['revision']==ref['targetCommit'];assert old['protocol_sha256']==new['protocol_sha256'];a={(o['panel'],o['condition']):o for o in old['observations']};b={(o['panel'],o['condition']):o for o in new['observations']};assert a.keys()==b.keys();errors=[]
   for k in a:
    for v in keys:
     x=np.array(a[k][v]);y=np.array(b[k][v]);assert x.shape==y.shape;errors.append(float(np.max(abs(x-y)/np.maximum(abs(x),1))))
   for tok in old['single_tokens']:
    for v in ['gate','up','activation','contribution']:
     x=np.array(old['single_tokens'][tok][v]);y=np.array(new['single_tokens'][tok][v]);errors.append(float(np.max(abs(x-y)/np.maximum(abs(x),1))))
   tol=1e-4 if mode=='float32' else 1e-2;err=max(errors);assert err<=tol,(phase,ref['name'],mode,err,tol);rows.append({'phase':phase,'checkpoint':ref['name'],'mode':mode,'max_scaled_error':err,'tolerance':tol})
out=B/'combined/deliverables/replica_validation.json';out.write_text(json.dumps({'all_checks_passed':True,'comparisons':rows,'scope':'Numerical replica comparison, not a new historical-transfer or full-framework-equivalence audit.'},indent=2));print('Validated',len(rows),'replica datasets')
