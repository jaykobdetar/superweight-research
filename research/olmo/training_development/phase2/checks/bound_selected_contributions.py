"""Prompt-independent upper bound for a scalar SwiGLU contribution in real arithmetic.
For non-affine LayerNorm h, mean(h)=0 and ||h||^2<=d. Center g/u rows.
|SiLU(g.h)*(u.h)| <= |(g.h)*(u.h)| <= d/2 (||g||||u||+|g.u|).
This bounds amplitude, not downstream importance or lesion damage.
"""
from pathlib import Path
import json,csv,numpy as np
R=Path(__file__).resolve().parents[1];P=json.loads((R/'protocol.json').read_text());rows=[]
for f in sorted((R/'sources').glob('*/acquisition.json')):
 m=json.loads(f.read_text());name=f.parent.name;down=np.fromfile(R.parent/'sources/trajectory'/name/'down_row1764.f32',dtype='<f4')
 for n in P['neurons']:
  vs=[]
  for kind in ['gate','up']:
   t=m['tensors'][f'{kind}_proj_{n}'];v=np.fromfile(f.parent/t['file'],dtype='<f4').astype(np.float64);v-=v.mean();vs.append(v)
  g,u=vs;prod=np.linalg.norm(g)*np.linalg.norm(u);dot=float(g@u);bound=abs(float(down[n]))*1024*(prod+abs(dot))
  rows.append({'checkpoint':name,'commit':m['commit'],'neuron':n,'down_weight':float(down[n]),'centered_gate_norm':float(np.linalg.norm(g)),'centered_up_norm':float(np.linalg.norm(u)),'gate_up_dot':dot,'absolute_scalar_contribution_upper_bound':float(bound)})
with (R/'deliverables'/'global_amplitude_bounds.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for r in rows:
 if r['checkpoint']=='main' or r['neuron']==8041:print(r['checkpoint'],r['neuron'],r['absolute_scalar_contribution_upper_bound'])
