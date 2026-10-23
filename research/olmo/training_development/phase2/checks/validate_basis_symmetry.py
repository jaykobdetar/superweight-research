"""Offline analytic-symmetry sanity check, added after the eight-point study.
No downloads or training. Transform the pinned prefix by a Householder
reflection R with R1=1; check its hidden-state equivariance and unchanged
selected MLP activations. This does not execute rotated later layers/logits.
"""
from pathlib import Path
import json,numpy as np,hashlib,os,resource
from prefix_model import Prefix,P,ROOT
m=Prefix('main');tests={'single_The':[510],'single_period':[15],**{p['name']:p['token_ids'] for p in P['panels']}};before={}
for name,ids in tests.items():before[name]=m.forward(ids,return_states=True)
d=2048;rng=np.random.default_rng(20261022);z=rng.normal(size=d);z-=z.mean();z/=np.linalg.norm(z);q=np.ones(d)/d+np.sqrt(1-1/d)*z;e=np.zeros(d);e[1764]=1;v=e-q;v/=np.linalg.norm(v);v=v.astype(np.float32)
def right(a):return a-2*(a@v)[...,None]*v
rss=[]
for name,W in list(m.w.items()):
 if name.endswith('self_attn.o_proj.weight') or name=='model.layers.0.mlp.down_proj.weight':
  m.w[name]=np.asarray(W)-2*v[:,None]*(v@W)[None,:]
 else:m.w[name]=right(np.asarray(W))
 rss.append(m.check_memory())
for i,E in list(m.embedding.items()):m.embedding[i]=right(E)
results=[]
for name,ids in tests.items():
 o,s=m.forward(ids,return_states=True);bo,bs=before[name];a=bs['activation'];b=s['activation'];rot=right(bs['premlp']);r={'probe':name,'activation_max_scaled_error':float(np.max(abs(a-b)/np.maximum(abs(a),1))),'premlp_equivariance_max_scaled_error':float(np.max(abs(rot-s['premlp'])/np.maximum(abs(rot),1))),'attention_l0_max_absolute_error':float(np.max(abs(np.array(o['l0_attention_q_to_0_p_q'])-np.array(bo['l0_attention_q_to_0_p_q']))))};results.append(r)
assert max(r['activation_max_scaled_error'] for r in results)<1e-3
assert max(r['attention_l0_max_absolute_error'] for r in results)<1e-4
assert max(r['premlp_equivariance_max_scaled_error'] for r in results)<1e-4
out={'label':'Post-hoc numerical sanity check of an analytic architecture symmetry','checkpoint':'d7cbab742d80589e714b1a2d7f838dcd21cbe143','reflection_vector_sum':float(v.sum()),'reflection_vector_squared_norm':float(v@v),'transformed_basis_direction_max_absolute_coordinate':float(abs(q).max()),'memory_after_transforms_max_rss':max(rss),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'tests':results,'passed':True,'scope':'Selected activations and prefix hidden-state equivariance only; the down scalar was not rotated, so its unchanged numeric contribution is not claimed as a physical scalar in a rotated model. Global function preservation is an algebraic statement with every residual input/output projection and final unembedding transformed consistently. No rotated full-model logits or training were run.'}
(ROOT/'deliverables/basis_symmetry_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
