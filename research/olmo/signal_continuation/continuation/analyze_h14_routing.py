"""Head14-only layer0 routing replay; no transformer/LM-head forward.

Checks the changed context against captured full-prefix residual differences.
Only selected Q/K/V rows and O columns are read, and existing shared weights reused.
"""
from pathlib import Path
import json
import numpy as np

R=Path(__file__).resolve().parent;W=Path('/workspace/shared/olmo_computational_role/downstream/weights')
z=np.load(R/'h14_prefix/traces.npz');H=14;D=2048;Q=128;P=64
embedding=np.memmap(W/'model.embed_tokens.weight.f32',mode='r',dtype='<f4',shape=(50304,D))
weights={}
for kind in 'qkv':
    m=np.memmap(W/('model.layers.0.self_attn.'+kind+'_proj.weight.f32'),mode='r',dtype='<f4',shape=(D,D))
    weights[kind]=np.array(m[H*128:(H+1)*128]);m._mmap.close()
m=np.memmap(W/'model.layers.0.self_attn.o_proj.weight.f32',mode='r',dtype='<f4',shape=(D,D))
wo=np.array(m[:,H*128:(H+1)*128]);m._mmap.close()
m=np.memmap(W/'model.layers.0.mlp.gate_proj.weight.f32',mode='r',dtype='<f4',shape=(8192,D))
gate=np.array(m[1091],dtype=np.float64);gate-=gate.mean();m._mmap.close()
m=np.memmap(W/'model.layers.0.mlp.up_proj.weight.f32',mode='r',dtype='<f4',shape=(8192,D))
up=np.array(m[1091],dtype=np.float64);up-=up.mean();m._mmap.close()
rows=[]
factorials=[]
for panel in ('water_cycle','library','bread','programming'):
    ids=z[panel+'__newline_R__input_ids'];x=np.array(embedding[ids]);x-=x.mean(axis=1,keepdims=True)
    x/=np.sqrt(np.mean(x*x,axis=1,keepdims=True)+np.float32(1e-5))
    q,k,v=[np.clip(x@weights[a].T,-8,8) for a in 'qkv']
    assert np.array_equal(v[P],v[Q]),'Identical layer0 token IDs must have identical unpositioned V'
    inv=np.float32(1)/(np.float32(10000)**(np.arange(0,128,2,dtype=np.float32)/np.float32(128)))
    frequency=np.arange(len(ids),dtype=np.float32)[:,None]*inv
    phase=np.concatenate([frequency,frequency],axis=1);co=np.cos(phase);si=np.sin(phase)
    rotate=lambda a:np.concatenate([-a[:,64:],a[:,:64]],axis=1)
    q=q*co+rotate(q)*si;k=k*co+rotate(k)*si
    score=q[Q]@k.T*np.float32(128**-.5)
    prob=np.exp(score-score.max());prob/=prob.sum(dtype=np.float32)
    blocked=prob.copy();blocked[P]=0;blocked/=blocked.sum(dtype=np.float32)
    context=prob@v;newcontext=blocked@v
    predicted=wo@(newcontext-context)
    actual=z[panel+'__newline_R_edge_H14__l0_pre_mlp']-z[panel+'__newline_R__l0_pre_mlp']
    error=float(np.max(abs(predicted-actual)));assert error<5e-5
    before=z[panel+'__newline_R__l0_pre_mlp'].astype(np.float64)
    after=z[panel+'__newline_R_edge_H14__l0_pre_mlp'].astype(np.float64)
    stdr=np.sqrt(np.var(before)+1e-5);stde=np.sqrt(np.var(after)+1e-5)
    n_r=float(gate@before);n_e=float(gate@after)
    # Query-local K/V factorial in the space-at-p recipient. Other heads and
    # all other query positions remain in that recipient world.
    sp=np.array(embedding[209]);sp-=sp.mean();sp/=np.sqrt(np.mean(sp*sp)+np.float32(1e-5))
    sk=np.clip(sp@weights['k'].T,-8,8);sv=np.clip(sp@weights['v'].T,-8,8)
    sk=sk*co[P]+np.concatenate([-sk[64:],sk[:64]])*si[P]
    space_score=float(q[Q]@sk*np.float32(128**-.5))
    score_u=score.copy();score_u[P]=space_score
    pu=np.exp(score_u-score_u.max());pu/=pu.sum(dtype=np.float32)
    vu=v.copy();vu[P]=sv;context_u=pu@vu
    pre_u=z[panel+'__newline_U__l0_pre_mlp'].astype(np.float64)
    for key_newline in (False,True):
        for value_newline in (False,True):
            ss=score.copy() if key_newline else score_u.copy()
            pp=np.exp(ss-ss.max());pp/=pp.sum(dtype=np.float32)
            vv=v if value_newline else vu
            pre=pre_u+(wo@(pp@vv-context_u)).astype(np.float64)
            centered=pre-pre.mean();hn=centered/np.sqrt(np.mean(centered*centered)+1e-5)
            gg=float(gate@hn);uu=float(up@hn)
            factorials.append(dict(panel=panel,key_donor='newline' if key_newline else 'space',
                value_donor='newline' if value_newline else 'space',source_mass=float(pp[P]),
                gate1091=gg,up1091=uu,activation1091=float(gg*np.exp(-np.logaddexp(0,-gg))*uu)))
    rows.append(dict(panel=panel,prior_newline_mass=float(prob[P]),self_newline_mass=float(prob[Q]),
        self_newline_mass_after_block=float(blocked[Q]),source0_mass_after_block=float(blocked[0]),
        prior_and_current_newline_values_identical=True,
        native_score_prior=float(score[P]),native_score_self=float(score[Q]),
        newline_prior_to_self_attention_ratio=float(prob[P]/prob[Q]),
        space_score_at_prior_position=space_score,
        top_after_block=[dict(position=int(i),token=int(ids[i]),probability=float(blocked[i])) for i in np.argsort(-blocked)[:8]],
        head_context_norm=float(np.linalg.norm(context)),blocked_context_norm=float(np.linalg.norm(newcontext)),
        full_prefix_attention_difference_max_error=error,
        gate1091_R=n_r/stdr,gate1091_E=n_e/stde,
        gate1091_numerator_R=n_r,gate1091_numerator_E=n_e,
        numerator_change_from_head14=float(gate@predicted),
        numerator_change_from_full_prefix=n_e-n_r))
result=dict(scope='Head14-only exact-formula replay plus agreement with captured full-prefix residual difference; no later forward.',
    rows=rows,query_local_key_value_factorials=factorials,
    factorial_scope='Counterfactual H14 K/V content for source64 at query128 only, with the space-at64 recipient residual and all other heads fixed. Exact M0-1091 scalar computation from the modified pre-M0 residual, not a later-model or all-query key intervention.',
    conclusion='The previous and current newline have identical layer0 value vectors. Removing the previous edge does not make H14 read its identical current-newline value; it redistributes attention toward non-newline content. The selected-head output alone accounts for the observed pre-M0 residual change.')
(R/'h14_prefix/head14_routing.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
