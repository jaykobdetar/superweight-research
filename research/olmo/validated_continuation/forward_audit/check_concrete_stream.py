"""Synthetic tests of actual StreamedOlmo methods. No pretrained weights loaded."""
from pathlib import Path
import numpy as np,json,importlib.util,hashlib
R=Path(__file__).resolve().parent;P=Path('/workspace/shared/olmo_computational_role/downstream/streamed_olmo.py')
spec=importlib.util.spec_from_file_location('audited_stream',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);S=m.StreamedOlmo
s=S.__new__(S);s.half=False;s.check=lambda:0
rng=np.random.default_rng(80411710);L,D,H,HD=9,2048,16,128;x=rng.normal(size=(L,D)).astype('f4')
pre={c:rng.normal(scale=5,size=(L,D)).astype('f4') for c in 'qkv'}
def project(_,key):
 for c in 'qkv':
  if key.endswith(c+'_proj.weight'):return pre[c].copy()
 if key.endswith('o_proj.weight'):return _.copy()
 raise ValueError(key)
s.linear=project;mask=[0,4];out,diag=s.attention(x,2,mask_sources=mask,diagnostic_sources=mask)
# Independent reference: transpose into token/head layout and rotate each pair.
Q,K,V=[np.clip(pre[c],-8,8).reshape(L,H,HD).copy() for c in 'qkv'];freq=np.float32(1)/(np.float32(10000)**(np.arange(0,128,2,dtype=np.float32)/np.float32(128)))
for z in [Q,K]:
 a=z[:,:,:64].copy();b=z[:,:,64:].copy();theta=np.arange(L,dtype='f4')[:,None]*freq;co=np.cos(theta)[:,None,:];si=np.sin(theta)[:,None,:];z[:,:,:64]=a*co-b*si;z[:,:,64:]=b*co+a*si
ref=np.empty((L,H,HD),dtype='f4');rp=np.zeros((H,L,L),dtype='f4')
for h in range(H):
 for q in range(L):
  valid=[k for k in range(q+1) if k not in mask or k==q];sco=(Q[q,h]@K[valid,h].T)*np.float32(HD**-.5);p=np.exp(sco-sco.max());p/=p.sum(dtype='f4');ref[q,h]=p@V[valid,h];rp[h,q,valid]=p
# Actual streaming LM-head method with tiny synthetic vocab, actual hidden width.
WV=rng.normal(scale=.1,size=(71,D)).astype('<f4');wp=R/'synthetic_head.f32';WV.tofile(wp);s.mmap=lambda key:np.memmap(wp,mode='r',dtype='<f4',shape=WV.shape);ids=rng.integers(0,len(WV),size=L).tolist();result=s.logits_summary(x,ids,vocab_chunk=13)
c=x-x.mean(-1,keepdims=True,dtype='f4');norm=c/np.sqrt((c*c).mean(-1,keepdims=True,dtype='f4')+np.float32(1e-5));logits=norm[:-1]@WV.T;l=logits.astype('f8');mx=l.max(-1);ls=mx+np.log(np.exp(l-mx[:,None]).sum(-1));nll=ls-l[np.arange(L-1),ids[1:]]
checks={'description':__doc__,'implementation_file':str(P),'implementation_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'attention_independent_loop_max_abs_error':float(abs(out-ref.reshape(L,D)).max()),'source_probability_max_abs_error':max(float(abs(diag[str(k)]-rp[:,:,k]).max()) for k in mask),'source0_masked_max_probability':float(diag['0'][:,1:].max()),'source4_masked_max_probability':float(diag['4'][:,5:].max()),'source0_self_probability_min':float(diag['0'][:,0].min()),'lm_stream_nll_max_abs_error':float(abs(result['loss']-nll).max()),'lm_stream_argmax_equal':bool(np.array_equal(result['argmax'],logits.argmax(-1))),'no_pretrained_weight_files_read':True}
assert checks['attention_independent_loop_max_abs_error']<.0005;assert checks['source_probability_max_abs_error']<1e-4;assert checks['source0_masked_max_probability']==checks['source4_masked_max_probability']==0;assert checks['source0_self_probability_min']==1;assert checks['lm_stream_nll_max_abs_error']<1e-5 and checks['lm_stream_argmax_equal']
(R/'concrete_stream_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
