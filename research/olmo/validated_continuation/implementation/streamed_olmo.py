"""Memory-bounded NumPy OLMo forward, exact pinned FP32 tensors.

Single sequence, no KV cache or autograd. Weight mmap is closed after each
projection; only compact activations survive. half_round approximates FP16
storage/operation rounding, not CUDA/SDPA bit equivalence.
"""
from pathlib import Path
import json,os,time,resource,hashlib
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parent
class StreamedOlmo:
 def __init__(self,half_round=False,weights_root=ROOT,cap_bytes=4_000_000_000):
  self.root=Path(weights_root);self.half=half_round;self.cap=cap_bytes
  self.plan=json.loads((self.root/'protocols/acquisition_plan.json').read_text())
  self.meta={j['key']:j for j in self.plan['jobs']}
  self.config=json.loads((self.root/'sources/config.json').read_text())
  assert self.config['hidden_size']==2048 and self.config['num_attention_heads']==16
  assert self.config['num_key_value_heads']==16 and self.config['num_hidden_layers']==16
  assert not self.config['tie_word_embeddings'] and self.config['clip_qkv']==8.0
  assert self.config['hidden_act']=='silu' and self.config['rope_theta']==10000.0
  assert self.config['intermediate_size']==8192 and self.config['vocab_size']==50304
  assert not self.config['attention_bias'] and self.config['attention_dropout']==0.0
  self.proc=psutil.Process();self.peak_rss=0
 def check(self):
  rss=self.proc.memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
  assert rss<self.cap,('RSS cap exceeded',rss)
  return rss
 def r(self,x):
  return x.astype(np.float16).astype(np.float32) if self.half else np.asarray(x,dtype=np.float32)
 def mmap(self,key):
  m=self.meta[key];assert Path(m['file']).stat().st_size==m['size']
  return np.memmap(m['file'],mode='r',dtype='<f4',shape=tuple(m['shape']))
 def ln(self,x):
  centered=x-x.mean(axis=-1,keepdims=True,dtype=np.float32)
  return self.r(centered/np.sqrt((centered*centered).mean(axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-5)))
 def linear(self,x,key):
  w=self.mmap(key)
  try:
   # Even in half mode the BLAS multiplication accumulates in FP32.
   if self.half:
    wh=self.r(w);y=self.r(x@wh.T);del wh
   else:y=x@w.T
   self.check();return y
  finally:w._mmap.close()
 def embed(self,ids):
  w=self.mmap('model.embed_tokens.weight')
  try:return self.r(np.array(w[np.asarray(ids,dtype=np.int64)],copy=True))
  finally:w._mmap.close()
 def attention(self,x,layer,mask_sources=(),attention_hook=None,diagnostic_sources=(),mask_query_start=None):
  n=len(x);prefix=f'model.layers.{layer}.self_attn.'
  q,k,v=[np.clip(self.linear(x,prefix+c+'_proj.weight'),-8,8).reshape(n,16,128).transpose(1,0,2) for c in 'qkv']
  inv=np.float32(1)/(np.float32(10000)**(np.arange(0,128,2,dtype=np.float32)/np.float32(128)))
  freqs=np.arange(n,dtype=np.float32)[:,None]*inv
  emb=np.concatenate([freqs,freqs],axis=-1)
  co=self.r(np.cos(emb))[None];si=self.r(np.sin(emb))[None]
  def rotate(z):return np.concatenate([-z[:,:,64:],z[:,:,:64]],axis=-1)
  # In true half eager RoPE, products and sum have separate half rounding.
  if self.half:
   q=self.r(self.r(q*co)+self.r(rotate(q)*si));k=self.r(self.r(k*co)+self.r(rotate(k)*si))
  else:q=q*co+rotate(q)*si;k=k*co+rotate(k)*si
  scores=(q@k.transpose(0,2,1))*np.float32(128**-0.5)
  i,j=np.triu_indices(n,k=1);scores[:,i,j]=-np.inf
  for s in mask_sources:
   assert 0<=s<n
   scores[:,max(s+1,mask_query_start or 0):,s]=-np.inf
  probs=np.exp(scores-scores.max(axis=-1,keepdims=True))
  probs/=probs.sum(axis=-1,keepdims=True,dtype=np.float32)
  diagnostics={str(s):probs[:,:,s].copy() for s in diagnostic_sources}
  context=self.r((probs@v).transpose(1,0,2).reshape(n,2048))
  if attention_hook is not None:
   context=attention_hook(layer,context,probs,v)
   assert context.shape==(n,2048) and np.isfinite(context).all()
  out=self.linear(context,prefix+'o_proj.weight')
  return out,diagnostics
 def logits_summary(self,x,ids,vocab_chunk=2048):
  x=self.ln(x)[:-1];targets=np.asarray(ids[1:],dtype=np.int64);n=len(x)
  maxlog=np.full(n,-np.inf,dtype=np.float64);sumexp=np.zeros(n,dtype=np.float64)
  argmax=np.zeros(n,dtype=np.int64);argval=np.full(n,-np.inf,dtype=np.float32)
  targetlog=np.full(n,np.nan,dtype=np.float64)
  w=self.mmap('lm_head.weight')
  try:
   for lo in range(0,w.shape[0],vocab_chunk):
    hi=min(lo+vocab_chunk,w.shape[0]);wc=self.r(w[lo:hi]) if self.half else w[lo:hi]
    logits=self.r(x@wc.T);del wc
    assert np.isfinite(logits).all(), 'Nonfinite logits'
    localmax=logits.max(axis=-1);la=logits.argmax(axis=-1)
    take=localmax>argval;argmax[take]=lo+la[take];argval=np.maximum(argval,localmax)
    newmax=np.maximum(maxlog,localmax)
    sumexp=sumexp*np.exp(maxlog-newmax)+np.exp(logits.astype(np.float64)-newmax[:,None]).sum(axis=-1)
    maxlog=newmax
    hit=(targets>=lo)&(targets<hi);rows=np.flatnonzero(hit)
    targetlog[rows]=logits[rows,targets[rows]-lo]
    del logits;self.check()
   assert np.isfinite(targetlog).all() and np.isfinite(sumexp).all()
   return {'loss':maxlog+np.log(sumexp)-targetlog,'argmax':argmax,'target_logit':targetlog,'log_normalizer':maxlog+np.log(sumexp)}
  finally:w._mmap.close()
 def forward(self,ids,mask_sources=(),mask_layers=range(2,16),mlp_hook=None,attention_hook=None,capture_layers=(),diagnostic_sources=(),stop_layer=None,progress=False,mask_query_start=None,prefix_hash_end=None):
  ids=np.asarray(ids,dtype=np.int64);assert ids.ndim==1 and len(ids)>=2 and len(ids)<=512
  start=time.time();x=self.embed(ids);states={};diag={};times=[];prefix_hashes=[]
  for layer in range(16):
   t=time.time();out,attn_diag=self.attention(self.ln(x),layer,mask_sources if layer in mask_layers else (),attention_hook,diagnostic_sources,mask_query_start)
   x=self.r(x+out);del out
   normalized=self.ln(x);p=f'model.layers.{layer}.mlp.'
   gate=self.linear(normalized,p+'gate_proj.weight');up=self.linear(normalized,p+'up_proj.weight')
   silu=self.r(gate*np.exp(-np.logaddexp(np.float32(0),-gate)))
   activations=self.r(silu*up);del gate,up,silu
   output=self.linear(activations,p+'down_proj.weight')
   if mlp_hook is not None:
    output=mlp_hook(layer,normalized,activations,output)
    assert output.shape==x.shape and np.isfinite(output).all()
   if layer in capture_layers:
    states[str(layer)]={'normalized':normalized.copy(),'activations':activations.copy(),'mlp_output':output.copy(),'pre_mlp':x.copy()}
   x=self.r(x+output);del normalized,activations,output
   assert np.isfinite(x).all(), ('Nonfinite hidden state',layer)
   if layer in capture_layers:states[str(layer)]['post_layer']=x.copy()
   if prefix_hash_end is not None:prefix_hashes.append(hashlib.sha256(x[:prefix_hash_end+1].tobytes()).hexdigest())
   diag[str(layer)]=attn_diag;times.append(time.time()-t);rss=self.check()
   if progress:print('LAYER',layer,'seconds',round(times[-1],3),'rss_MiB',round(rss/2**20,1),flush=True)
   if stop_layer==layer:return {'hidden':x,'states':states,'attention':diag,'layer_seconds':times,'seconds':time.time()-start,'peak_rss':self.peak_rss,'prefix_hashes':prefix_hashes}
  result=self.logits_summary(x,ids)
  result.update(states=states,attention=diag,layer_seconds=times,seconds=time.time()-start,peak_rss=self.peak_rss,prefix_hashes=prefix_hashes)
  return result
