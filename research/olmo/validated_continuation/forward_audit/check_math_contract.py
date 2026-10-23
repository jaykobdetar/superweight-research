"""Small synthetic checks of the audited equations, not a model/native-framework replay."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;rng=np.random.default_rng(20261023);D,H,L=32,4,11;HD=D//H;F=64;V=71
x=rng.normal(size=(L,D)).astype(np.float64);W={k:rng.normal(size=s) for k,s in {'q':(D,D),'k':(D,D),'v':(D,D),'o':(D,D),'gate':(F,D),'up':(F,D),'down':(D,F),'lm':(V,D)}.items()}
def ln(z):
 c=z-z.mean(-1,keepdims=True);return c/np.sqrt((c*c).mean(-1,keepdims=True)+1e-5)
def linear_chunks(z,w,block=7):return np.concatenate([z@w[i:i+block].T for i in range(0,len(w),block)],axis=-1)
def rope(q):
 inv=10000.**(-np.arange(0,HD,2,dtype=np.float64)/HD);f=np.arange(L)[:,None]*inv;ang=np.concatenate([f,f],axis=-1)[None]
 return q*np.cos(ang)+np.concatenate([-q[...,HD//2:],q[...,:HD//2]],axis=-1)*np.sin(ang)
y=ln(x);q,k,v=[np.clip(y@W[t].T,-8,8).reshape(L,H,HD).transpose(1,0,2) for t in ['q','k','v']];q0=q.copy();q=rope(q);k=rope(k)
s=q@k.transpose(0,2,1)/np.sqrt(HD);bad=np.arange(L)[None,:]>np.arange(L)[:,None];s[:,bad]=-np.inf
# Include one surgical edge block with at least one allowed key left.
s[2,8,3]=-np.inf;p=np.exp(s-s.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);ctx=(p@v).transpose(1,0,2).reshape(L,D)
# Query/key blocked attention with online stable softmax, preserving global positions.
c2=np.zeros((H,L,HD));seen=np.zeros((H,L),dtype=bool)
for head in range(H):
 for t in range(L):
  M=-np.inf;S=0.;A=np.zeros(HD)
  for lo in range(0,L,3):
   hi=min(L,lo+3);ss=q[head,t]@k[head,lo:hi].T/np.sqrt(HD);allowed=np.arange(lo,hi)<=t
   if head==2 and lo<=3<hi and t==8:allowed[3-lo]=False
   if not allowed.any():continue
   ss=ss[allowed];vv=v[head,lo:hi][allowed];m=max(M,float(ss.max()));old=0. if not np.isfinite(M) else np.exp(M-m);e=np.exp(ss-m);A=A*old+e@vv;S=S*old+e.sum();M=m;seen[head,t]=True
  c2[head,t]=A/S
ctx2=c2.transpose(1,0,2).reshape(L,D)
z=x+ctx@W['o'].T;a=ln(z);gg=a@W['gate'].T;uu=a@W['up'].T;act=gg/(1+np.exp(-gg))*uu;state=z+act@W['down'].T;hn=ln(state);logits=hn@W['lm'].T
# Complete vocabulary-streamed exact logsumexp and target extraction.
targets=rng.integers(0,V,size=L);M=np.full(L,-np.inf);S=np.zeros(L);target=np.empty(L)
for lo in range(0,V,9):
 hi=min(V,lo+9);b=hn@W['lm'][lo:hi].T;m=np.maximum(M,b.max(-1));S=S*np.exp(M-m)+np.exp(b-m[:,None]).sum(-1);M=m;hit=(targets>=lo)&(targets<hi);target[hit]=b[np.where(hit)[0],targets[hit]-lo]
stream_nll=M+np.log(S)-target;maxlog=logits.max(-1);direct_nll=maxlog+np.log(np.exp(logits-maxlog[:,None]).sum(-1))-logits[np.arange(L),targets]
analytic=np.array([1.,2.,3.,4.]);expected=(analytic-2.5)/np.sqrt(1.25+1e-5)
checks={'description':__doc__,'seed':20261023,'dimensions':{'D':D,'H':H,'L':L,'F':F,'V':V},'population_ln_analytic_max_error':float(abs(ln(analytic)-expected).max()),'ln_mean_max_abs':float(abs(y.mean(-1)).max()),'rope_norm_max_error':float(abs((q*q).sum(-1)-(q0*q0).sum(-1)).max()),'post_rope_can_exceed_clip':bool(np.max(abs(q))>8),'max_post_rope_coordinate':float(np.max(abs(q))),'future_probability_max':float(p[:,bad].max()),'blocked_edge_probability':float(p[2,8,3]),'softmax_rows_sum_max_error':float(abs(p.sum(-1)-1).max()),'stream_attention_max_error':float(abs(ctx2-ctx).max()),'output_row_chunk_linear_max_error':float(abs(linear_chunks(hn,W['lm'])-logits).max()),'full_vocab_stream_nll_max_error':float(abs(stream_nll-direct_nll).max()),'all_rows_have_allowed_key':bool(seen.all())}
assert checks['population_ln_analytic_max_error']==0 and checks['future_probability_max']==0 and checks['blocked_edge_probability']==0 and checks['post_rope_can_exceed_clip'] and checks['all_rows_have_allowed_key'];assert max(checks[k] for k in checks if k.endswith('max_error'))<1e-10
(R/'math_contract_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
