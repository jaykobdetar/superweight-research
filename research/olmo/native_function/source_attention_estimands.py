"""Small, model-independent reference implementation for proposed local tests.

This validates estimands only. It is NOT an OLMo model experiment or a tested
transformers hook. Pass pre-softmax causal head logits and post-clipping values
from a real implementation; keep native RoPE, clipping, masks and projections.
"""
import json
from pathlib import Path
import numpy as np

def softmax(x):
    z=x-np.max(x,axis=-1,keepdims=True);e=np.exp(z);return e/e.sum(axis=-1,keepdims=True)

def local_branches(logits,values,source):
    """logits [Q,K], values [K,D]; queries must be strictly after source.

    Source-free conditional mean is recomputed from logits, not divided by
    1-alpha, so the definition remains stable when alpha rounds to 1.
    """
    a=softmax(logits);without=logits.copy();without[:,source]=-np.inf
    b=softmax(without);alpha=a[:,source:source+1]
    mu=b@values;injection=alpha*values[source][None,:]
    return {'alpha':alpha,'conditional_non_source':mu,
        'native':a@values,
        'attenuation_and_value':(1-alpha)*mu+injection,
        'attenuation_only':(1-alpha)*mu,
        'value_only':mu+injection,
        'neither':mu}

def value_donor_fixed_routing(logits,values,source,donor_value):
    a=softmax(logits);return a@values+a[:,source:source+1]*(donor_value-values[source])[None,:]

def match_added_write_radius(reference,residual,rng):
    """Randomize write direction preserving its mean, norm and residual dot.

    Applied in residual-space AFTER W_O, not to raw V norm. Thus the resulting
    residual mean and centered variance are also matched. Diagnostic control,
    not an on-manifold donor or a claim of a semantic/value-matched direction.
    """
    r=np.asarray(residual,dtype=float);v=np.asarray(reference,dtype=float)
    one=np.ones_like(r)/np.sqrt(len(r));rc=r-r.mean()
    assert np.linalg.norm(rc)>0
    basis=np.stack([one,rc/np.linalg.norm(rc)])
    parallel=(basis@v)@basis
    perp=v-parallel;rand=rng.normal(size=r.shape);rand-=(basis@rand)@basis
    return parallel+np.linalg.norm(perp)*rand/np.linalg.norm(rand)

def test():
    rng=np.random.default_rng(20261023);logits=rng.normal(size=(7,11));v=rng.normal(size=(11,32));s=2
    b=local_branches(logits,v,s);checks={}
    checks['native_reconstruction_max_abs_error']=float(np.max(np.abs(b['native']-b['attenuation_and_value'])))
    vz=v.copy();vz[s]=0
    checks['zeroV_closure_max_abs_error']=float(np.max(np.abs(softmax(logits)@vz-b['attenuation_only'])))
    l=logits.copy();l[:,s]=-np.inf
    checks['mask_closure_max_abs_error']=float(np.max(np.abs(softmax(l)@v-b['neither'])))
    checks['equal_donor_max_abs_error']=float(np.max(np.abs(value_donor_fixed_routing(logits,v,s,v[s])-b['native'])))
    dv=rng.normal(size=32);vd=v.copy();vd[s]=dv
    checks['value_swap_closure_max_abs_error']=float(np.max(np.abs(value_donor_fixed_routing(logits,v,s,dv)-softmax(logits)@vd)))
    # The two local components add exactly before W_O / residual / nonlinearities.
    # This never implies additive final loss effects.
    checks['local_factorial_interaction_max_abs_error']=float(np.max(np.abs(b['attenuation_and_value']-b['attenuation_only']-b['value_only']+b['neither'])))
    r=rng.normal(size=32);w=rng.normal(size=32);wc=match_added_write_radius(w,r,rng)
    checks['norm_matched_write_error']=float(abs(np.linalg.norm(w)-np.linalg.norm(wc)))
    checks['norm_matched_resulting_residual_error']=float(abs(np.linalg.norm(r+w)-np.linalg.norm(r+wc)))
    checks['norm_matched_residual_dot_error']=float(abs(np.dot(r,w)-np.dot(r,wc)))
    checks['norm_matched_resulting_residual_mean_error']=float(abs(np.mean(r+w)-np.mean(r+wc)))
    checks['norm_matched_resulting_residual_variance_error']=float(abs(np.var(r+w)-np.var(r+wc)))
    assert max(checks.values())<1e-12
    return {'scope':'Synthetic algebra and control validation only. No pretrained model inference.', 'passed':True,'checks':checks}

if __name__=='__main__':
    r=test();print(json.dumps(r,indent=2));(Path(__file__).resolve().parent/'analysis/estimand_unit_tests.json').write_text(json.dumps(r,indent=2)+'\n')
