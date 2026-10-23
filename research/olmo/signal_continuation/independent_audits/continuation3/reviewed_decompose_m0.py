"""Exact additive pre-LN feature numerators, with explicit norm interaction.

Ranks L0 neuron activation changes, not model parameters by size. This defines
candidate groups; it does not establish their necessity before donor tests.
"""
from pathlib import Path
import csv,hashlib,json,resource
import numpy as np

R=Path(__file__).resolve().parent;O=R/'h14_prefix'
z=np.load(O/'traces.npz')
gate=np.memmap(R.parent/'tensors/gate_proj.f32',mode='r',dtype='<f4',shape=(8192,2048))
up=np.memmap(R.parent/'tensors/up_proj.f32',mode='r',dtype='<f4',shape=(8192,2048))
wpath=Path('/workspace/shared/olmo_computational_role/downstream/weights/model.layers.0.mlp.down_proj.weight.f32')
w=np.memmap(wpath,mode='r',dtype='<f4',shape=(2048,8192))
feature_names=[f'{kind}_{j}' for j in (1710,3950,4786) for kind in ('gate','up')]
A=np.array([row[j] for j in (1710,3950,4786) for row in (gate,up)],dtype=np.float64)
A-=A.mean(axis=1,keepdims=True)
coeff=np.zeros((6,8192))
for start in range(0,2048,128):coeff+=A[:,start:start+128]@w[start:start+128].astype(np.float64)
components=[];top=[];contributions=[];checks=[]
panels=('water_cycle','library','bread','programming')
for panel in panels:
    r=lambda name: z[panel+'__newline_R__'+name].astype(np.float64)
    e=lambda name: z[panel+'__newline_R_edge_H14__'+name].astype(np.float64)
    xr,xe=r('l1_pre_mlp'),e('l1_pre_mlp')
    sr=np.sqrt(np.mean((xr-xr.mean())**2)+1e-5);se=np.sqrt(np.mean((xe-xe.mean())**2)+1e-5)
    # Symmetric product difference: Delta(n/s)=Delta(n)*mean(1/s)+mean(n)*Delta(1/s).
    reciprocal_mean=.5*(1/sr+1/se)
    deltas={'A0':e('l0_pre_mlp')-r('l0_pre_mlp'),
            'M0':e('m0_output')-r('m0_output'),
            'A1':e('a1_output')-r('a1_output')}
    vals={name:(A@delta)*reciprocal_mean for name,delta in deltas.items()}
    vals['normalization_interaction']=.5*(A@xe+A@xr)*(1/se-1/sr)
    delta_feature=A@xe/se-A@xr/sr
    closure=sum(vals.values())-delta_feature
    neuron=(e('m0_activations')-r('m0_activations'))[None,:]*coeff*reciprocal_mean
    contributions.append(neuron)
    m0_error=neuron.sum(axis=1)-vals['M0']
    checks.append(dict(panel=panel,std_R=float(sr),std_E=float(se),
        max_component_closure_error=float(abs(closure).max()),
        max_m0_neuron_closure_error=float(abs(m0_error).max())))
    assert abs(closure).max()<1e-4 and abs(m0_error).max()<1e-4
    for k,name in enumerate(feature_names):
        components.append(dict(panel=panel,feature=name,total_change=float(delta_feature[k]),
            **{part:float(v[k]) for part,v in vals.items()}))
        order=np.argsort(-abs(neuron[k]))[:20]
        for rank,j in enumerate(order,1):
            top.append(dict(panel=panel,feature=name,rank=rank,neuron=int(j),
                activation_R=float(r('m0_activations')[j]),activation_E=float(e('m0_activations')[j]),
                coefficient=float(coeff[k,j]),signed_contribution=float(neuron[k,j])))

for filename,rows in [('feature_component_decomposition',components),('m0_neurons_top20',top)]:
    (O/(filename+'.json')).write_text(json.dumps(rows,indent=2))
    with (O/(filename+'.csv')).open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
contributions=np.stack(contributions)
sets={}
for name,k in [('primary_gate',0),('backup3950_up',3),('backup4786_up',5)]:
    order=np.argsort(-np.abs(contributions[:,k]).mean(axis=0))
    sets[name]=[int(x) for x in order[:4]]
result=dict(scope='Signed additive numerator decomposition with symmetric norm interaction. Not mediation percentages or necessity claims.',
    feature_names=feature_names,panels=panels,checks=checks,candidate_top4_groups=sets,
    selection_rule='Top4 by mean absolute neuron contribution across4 discovery panels; groups frozen before donor tests.',
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    traced_states_sha256=hashlib.sha256((O/'traces.npz').read_bytes()).hexdigest())
(O/'decomposition_checks.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(O/'m0_feature_contributions.npz',coefficients=coeff,contributions=contributions,
                    feature_rows=A,feature_names=np.array(feature_names),panels=np.array(panels))
print(json.dumps(result,indent=2))
for panel in panels:
    print('PANEL',panel)
    for row in components:
        if row['panel']==panel:print(row)
    for name,k in [('primary_gate',0),('backup3950_up',3),('backup4786_up',5)]:
        print(name,[x for x in top if x['panel']==panel and x['feature']==feature_names[k]][:4])
