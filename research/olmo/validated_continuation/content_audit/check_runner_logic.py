"""Actual function bodies, synthetic arrays and temporary audit JSON only."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
R=Path('/workspace/shared/olmo_computational_role/content_donors');O=Path(__file__).resolve().parent
src=ast.parse((R/'run_content_donors.py').read_text())
def extract(name):
 node=next(n for n in ast.walk(src) if isinstance(n,ast.FunctionDef) and n.name==name)
 return compile(ast.Module(body=[node],type_ignores=[]),'<audited '+name+'>','exec')
env={'np':np};exec(extract('kwargs'),env);kw=env['kwargs'];p=4
for world,sources in [('native',()),('postboundary_mask0',(0,)),('postboundary_maskp',(p,)),('postboundary_mask0p',(0,p))]:
 d=kw({'p':p},world);assert d['mask_sources']==sources and d['mask_query_start']==p+1 and d['prefix_hash_end']==p and list(d['mask_layers'])==list(range(2,16))
# Correct versus foil and correct-label NLL are oriented independently of common observed A.
class Map(np.ndarray):
 class Close:
  def close(self):pass
 _mmap=Close()
class M:
 def __init__(self,W):self.W=W
 def ln(self,x):
  c=x-x.mean(-1,keepdims=True);return c/np.sqrt((c*c).mean(-1,keepdims=True)+np.float32(1e-5))
 def mmap(self,key):return self.W.view(Map)
rng=np.random.default_rng(660);N,D,V=9,12,17;W=rng.normal(size=(V,D)).astype(np.float32);x=rng.normal(size=(N,D)).astype(np.float32);model=M(W)
q=N-2;labels=[3,11];logits=model.ln(x)@W.T
lq=logits[q].astype(np.float64);normalizer=np.max(lq)+np.log(np.exp(lq-np.max(lq)).sum())
result={'states':{'15':{'post_layer':x}},'target_logit':np.zeros(N-1),'log_normalizer':np.zeros(N-1),'loss':np.zeros(N-1),'argmax':np.zeros(N-1,dtype=int),'seconds':0.,'peak_rss':0,'prefix_hashes':['example']*16}
result['target_logit'][q]=logits[q,labels[0]];result['log_normalizer'][q]=normalizer;result['loss'][q]=normalizer-lq[labels[0]];result['argmax'][q]=lq.argmax()
exec(extract('score'),env);scores=[]
for ci in (0,1):
 out=env['score'](model,result,{'query_pos':q,'label_ids':labels},ci)
 assert abs(out['correct_minus_foil']-(lq[labels[ci]]-lq[labels[1-ci]]))<2e-6
 assert abs(out['correct_label_loss']-(normalizer-lq[labels[ci]]))<2e-6
 assert out['observed_target_loss']==result['loss'][q]
 scores.append(out['correct_minus_foil'])
assert scores[0]==-scores[1]
# Real closure body: all-layer routing frozen but current V retained; single layer differs.
H,N,D=3,9,4;rec={'p':4};P=rng.random((16,H,N,N),dtype=np.float32);P/=P.sum(-1,keepdims=True)
probs=rng.random((H,N,N),dtype=np.float32);probs/=probs.sum(-1,keepdims=True)
v=rng.normal(size=(H,N,D)).astype(np.float32);base={'P':P};donor={'vp':rng.normal(size=(16,H,D)).astype(np.float32)}
context=rng.normal(size=(N,H*D)).astype(np.float32)
exec(compile((R/'donor_algebra.py').read_text(),'<donor algebra>','exec'),env)
hooknode=next(n for n in ast.walk(src) if isinstance(n,ast.FunctionDef) and n.name=='intervene')
node=next(n for n in ast.walk(hooknode) if isinstance(n,ast.FunctionDef) and n.name=='hook')
code=compile(ast.Module(body=[node],type_ignores=[]),'<actual content hook>','exec')
checks=[]
for scope in ['frozen_all','single_layer_endogenous']:
 layers=set(range(2,16)) if scope=='frozen_all' else {8};e=dict(env,scope=scope,layers=layers,base=base,rec=rec,source=4,donor=donor,dkey='vp',condition='mismatched',diagnostics=[])
 exec(code,e)
 for layer in [0,1,2,8,15]:
  usedprobs=P[layer] if scope=='single_layer_endogenous' and layer==8 else probs
  y=e['hook'](layer,context,usedprobs,v)
  if scope=='frozen_all' or layer==8:
   expected=env['replay_context'](P[layer],v)
   if layer in layers:expected=expected+env['replacement_delta'](P[layer],v,4,donor['vp'][layer],5)
   assert np.array_equal(y,expected)
  else:assert y is context
  checks.append({'scope':scope,'layer':layer,'formula_and_scope_exact':True})
amend=json.loads((R/'preoutput_amendment_01.json').read_text());assert amend['expanded_default_counts']=={'recipient_baselines':64,'donor_baselines':24,'interventions':256,'total':344}
out={'actual_mask_kwargs_all_four_worlds':True,'manual_score_both_orientations_correct':True,'common_A_observed_loss_not_confused_with_correct_B_loss':True,'hook_checks':checks,'amended_default_counts':amend['expanded_default_counts'],'no_learned_weights_or_forwards':True}
(O/'runner_logic_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
