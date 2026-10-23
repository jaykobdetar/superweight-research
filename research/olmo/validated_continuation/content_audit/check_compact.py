"""Independent token-manifest and small value-donor algebra checks, no learned weights."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
R=Path('/workspace/shared/olmo_computational_role/content_donors');O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((R/'frozen_manifest.json').read_text());assert sha(R/'frozen_manifest.json')==(R/'frozen_manifest.sha256').read_text().strip()
tok=json.loads((R.parent/'inputs/tokenizer.json').read_text());assert sha(R.parent/'inputs/tokenizer.json')==m['tokenizer_sha256']
inv={v:k for k,v in tok['model']['vocab'].items()}
base=list(range(33,127))+list(range(161,173))+list(range(174,256));other=[i for i in range(256) if i not in base]
dec={chr(v):v for v in base};dec.update({chr(256+i):v for i,v in enumerate(other)})
def decode(ids):return bytes(dec[c] for i in ids for c in inv[i]).decode('utf8')
checks=[]
for rec in m['sequences']:
 sequences=rec['recipients']+rec['paraphrases']+[rec['irrelevant']]
 for seq in sequences:
  ids=seq['ids'];assert decode(ids)==seq['text'] and len(ids)==rec['n']
  assert ids[0]==510 and ids[rec['p']]==187 and ids.count(187)==1
  assert ids[rec['p']+1:]==rec['recipients'][0]['ids'][rec['p']+1:]
  assert ids[rec['query_pos']+1]==rec['label_ids'][0] and rec['query_pos']==rec['n']-2
  if 'sha256' in seq:assert hashlib.sha256(str(ids).encode('ascii')).hexdigest()==seq['sha256']
 for pair in (rec['recipients'],rec['paraphrases']):
  changed=np.flatnonzero(np.array(pair[0]['ids'])!=pair[1]['ids']).tolist();assert changed==[rec['fact_pos']]
  assert [x['ids'][rec['fact_pos']] for x in pair]==rec['label_ids']
 assert rec['p']-rec['fact_pos']==2
 checks.append({'pair_id':rec['pair_id'],'n':rec['n'],'p':rec['p'],'q':rec['query_pos'],'fact_pos':rec['fact_pos'],'one_preboundary_token_reciprocal_change':True,'all_five_texts_decode_exactly':True,'common_appended_label_A_future_to_query':True})
spec=importlib.util.spec_from_file_location('audited_donor_algebra',R/'donor_algebra.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
rng=np.random.default_rng(105032);H,N,D=4,9,6;p=4
logits=rng.normal(size=(H,N,N));logits[:,np.triu_indices(N,1)[0],np.triu_indices(N,1)[1]]=-np.inf
P=np.exp(logits-logits.max(-1,keepdims=True));P=(P/P.sum(-1,keepdims=True)).astype(np.float32)
V=rng.normal(size=(H,N,D)).astype(np.float32);donor=rng.normal(size=(H,D)).astype(np.float32)
# Independent head/query/source loops, not a rearranged call of the production algebra.
ref_context=np.zeros((N,H*D),np.float64);ref_delta=np.zeros_like(ref_context)
for h in range(H):
 for q in range(N):
  for s in range(N):ref_context[q,h*D:(h+1)*D]+=float(P[h,q,s])*V[h,s].astype(np.float64)
  if q>p:ref_delta[q,h*D:(h+1)*D]=float(P[h,q,p])*(donor[h].astype(np.float64)-V[h,p].astype(np.float64))
ctx=a.replay_context(P,V);delta=a.replacement_delta(P,V,p,donor,p+1)
ctxerr=float(abs(ctx-ref_context).max());deltaerr=float(abs(delta-ref_delta).max());assert ctxerr<1e-6 and deltaerr<1e-6
assert np.array_equal(delta[:p+1],np.zeros_like(delta[:p+1]))
assert np.array_equal(a.replacement_delta(P,V,p,V[:,p],p+1),np.zeros_like(delta))
W=rng.normal(size=(H*D,11)).astype(np.float32);ordinary=a.replacement_delta(P,V,p-2,donor,p+1)
matched,diag=a.match_output_norm(ordinary,delta,lambda z:z@W)
normerr=float(abs(np.linalg.norm(matched.astype(np.float64)@W.astype(np.float64),axis=1)-np.linalg.norm(delta.astype(np.float64)@W.astype(np.float64),axis=1)).max());assert normerr<1e-5
try:a.match_output_norm(np.zeros_like(delta),delta,lambda z:z@W)
except ValueError:zero_rejected=True
else:raise AssertionError('Zero control was not rejected')
maskP=P.copy();maskP[:,p+1:,0]=0
assert np.array_equal(a.replacement_delta(maskP,V,0,donor,p+1),np.zeros_like(delta))
assert np.array_equal(a.replay_context(maskP,V)[:p+1],ctx[:p+1])
# The actual p-from-source0 branch must replace source p, not source0's attention column.
ref_source0_delta=a.replacement_delta(P,V,p,V[:,0],p+1)
assert np.any(ref_source0_delta[p+1:]!=0)
out={'scope':'Token bytes and synthetic value algebra only; no model/tokenizer installation, no learned forward.','manifest_sha256':sha(R/'frozen_manifest.json'),'manifest_checks':checks,'original_two_world_baseline_passes':len(checks)*2*2,'amended_four_world_baseline_passes':len(checks)*2*4,'algebra':{'context_independent_loop_max_error':ctxerr,'delta_independent_loop_max_error':deltaerr,'post_WO_FP64_norm_max_error':normerr,'zero_direction_rejected':zero_rejected,'strict_prefix_and_mask0_null_exact':True,'p_from_source0_is_nontrivial_at_p':True,'reported_norm_diagnostics':diag}}
(O/'compact_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
