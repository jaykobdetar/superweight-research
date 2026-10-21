"""Bounded failure evidence, preserving exact selected rows and full failing layer."""
import io
import torch
from gpt2_gate_subspace_v2 import fingerprint

CAPSULE_LIMIT=64*1024**2
ROWS=(0,1,31,63,126,127,254,510)

def logit_pair(a,b=None):
    if a is None:return None
    assert a.ndim==2 and (b is None or b.shape==a.shape)
    errors=[];bad=[]
    for start in range(0,len(a),32):
        x=a[start:start+32].detach().cpu().double()
        y=None if b is None else b[start:start+32].detach().cpu().double()
        finite=torch.isfinite(x).all(-1)if y is None else torch.isfinite(x).all(-1)&torch.isfinite(y).all(-1)
        bad.extend((~finite).nonzero().flatten().add(start).tolist())
        d=x.abs()if y is None else(x-y).abs()
        errors.append(torch.nan_to_num(d,nan=float('inf'),posinf=float('inf'),neginf=float('inf')).max(-1).values)
    errors=torch.cat(errors);indices=sorted({r for r in ROWS if r<len(a)}|{int(errors.argmax())}|set(bad[:1]))
    return {'query_rows':torch.tensor(indices),'left':a[indices].detach().cpu().clone(),
            'right':None if b is None else b[indices].detach().cpu().clone(),
            'left_full_fingerprint':fingerprint(a),'right_full_fingerprint':None if b is None else fingerprint(b),
            'per_query_max_abs':errors,'nonfinite_query_indices':bad,
            'scope':'Frozen rows plus first nonfinite and maximum-discrepancy row; full per-query maxima. Full arrays are not retained.'}

def accumulator_state(a):
    if a is None:return None
    return {'layers':list(a.layers),'loss_p':a.loss_p,'single_losses':a.single_losses,
            'targets':a.targets,'logits':logit_pair(a.additive,a.base)}

def singleton_state(failure):
    if failure is None:return None
    l=failure['check']['layer'];out={'check':failure['check']}
    for side in('reference','trial'):
        b=failure[side];out[side]={'gate':b['gates'][l],'context':b['current_contexts'][l],
            'norms':b['localization_norms'][l],'hidden':b['hidden'],'metrics':b['z']}
    return out

def suffix_state(failure):
    if failure is None:return None
    out={'check':failure['check'],'original_tokens':failure['original_tokens'],'control_tokens':failure['control_tokens'],
         'logits':logit_pair(failure['reference_logits'],failure['control_logits'])}
    for side in('reference','control'):
        b=failure[side];out[side]={'hidden':b['hidden'][:,:128].clone(),
            'gates':{l:{k:v[:,:127].clone()for k,v in d.items()}for l,d in b['gates'].items()},
            'source':b['source'],'prefix_contexts':b['prefix_contexts'],'metrics':b['z']}
    return out

def save_capsule(path,payload):
    # Measure exact serialization before writing; preserve metadata if an unexpected
    # payload violates the cap rather than silently truncating evidence.
    buffer=io.BytesIO();torch.save(payload,buffer);n=buffer.tell()
    assert n<=CAPSULE_LIMIT,('Failure capsule exceeds frozen limit',n,CAPSULE_LIMIT)
    path.write_bytes(buffer.getbuffer());return n
