from pathlib import Path
import json,numpy as np,hashlib
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'protocol.json').read_text());a=json.loads((ROOT/'results/main/fp16_approx.json').read_text());b=json.loads((ROOT/'results/main/float32.json').read_text());checks=[]
for token,ref in [('510',P['validation_gate']['The_source_pulse']),('15',P['validation_gate']['period_source_pulse'])]:
 v=a['single_tokens'][token]['contribution'][0][0];err=abs(v-ref)/abs(ref);checks.append({'check':'FP16-style source pulse '+token,'observed':v,'reference':ref,'relative_error':err,'pass':err<=.01})
checks.append({'check':'causal prefix float32','error':b['causal_prefix_max_scaled_error'],'pass':b['causal_prefix_max_scaled_error']<=.0001})
by={(r['panel'],r['condition']):r for r in b['observations']};fractions=[];values={c:[] for c in ['newline_U','newline_R','newline_R_edge_H14','newline_R_edge_H0']}
for p in P['panels']:
 name=p['name'];r=by[(name,'empty_isolated')];e=np.array(r['activation']);err=float(np.max(abs(e[0]-e[-1])/(1+abs(e[0]))));checks.append({'check':'self-only same token '+name,'error':err,'pass':err<=.0001})
 for c in values:values[c].append(abs(by[(name,c)]['contribution'][-1][0]))
 def v(c):return by[(name,c)]['contribution'][-1][0]
 fractions.append((v('empty_patch_H12')-v('empty_ordinary'))/(v('empty_isolated')-v('empty_ordinary')))
for c,vs in values.items():
 med=float(np.median(vs));checks.append({'check':'mature '+c,'median_abs_pulse':med,'all_values':vs,'pass':med>100 if c in ['newline_U','newline_R_edge_H14'] else med<1})
med=float(np.median(fractions));checks.append({'check':'mature H12 restoration','median_fraction':med,'all_values':fractions,'pass':med>=.5})
result={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'passed':all(c['pass'] for c in checks),'checks':checks,'scope':'Numerical source-pulse references, code/formula parity and exact invariants plus qualitative replication on new fixed probes. Does not execute the full Transformers/CUDA reference implementation; no bit-exact SDPA claim.'}
(ROOT/'results/endpoint_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert result['passed'],'Endpoint validation failed: do not interpret developmental trajectory'
