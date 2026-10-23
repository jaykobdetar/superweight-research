'''CPU NumPy port of the needed transformers 4.57.6 OLMo prefix.
No downloaded code executes. No model training, autograd, GPU, full layer-1 MLP or later layers.
FP16 mode approximates storage/operation rounding, not a claim of CUDA SDPA bit equivalence.
'''
from pathlib import Path
import numpy as np,json,psutil,os
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'protocol.json').read_text())
class Prefix:
 def __init__(self,name,half=False,manifest_file="acquisition.json"):
  assert os.environ.get('CUDA_VISIBLE_DEVICES','')==''
  self.name=name;self.half=half;self.dir=ROOT/'sources'/name;self.meta=json.loads((self.dir/manifest_file).read_text());self.w={}
  for key,t in self.meta['tensors'].items():
   a=np.memmap(self.dir/t['file'],mode='r',dtype='<f4',shape=tuple(t['shape']));self.w[key]=self.r(a) if half else a
  self.embedding={}
  for t in self.meta['embedding_groups']:
   a=np.memmap(self.dir/t['file'],mode='r',dtype='<f4',shape=(t['hi']-t['lo']+1,2048))
   for i in P['embedding_ids']:
    if t['lo']<=i<=t['hi']:self.embedding[i]=self.r(np.array(a[i-t['lo']]))
  self.down=np.fromfile(ROOT.parent/'sources/trajectory'/name/'down_row1764.f32',dtype='<f4')[P['neurons']];self.down=self.r(self.down)
  self.check_memory()
 def check_memory(self):
  rss=psutil.Process().memory_info().rss
  assert rss<P['resources']['process_rss_cap_bytes'],f'RSS cap exceeded: {rss}'
  return rss
 def r(self,x):return x.astype(np.float16).astype(np.float32) if self.half else np.asarray(x,dtype=np.float32)
 def ln(self,x):
  c=x-np.mean(x,axis=-1,keepdims=True,dtype=np.float32)
  return self.r(c/np.sqrt(np.mean(c*c,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-5)))
 def linear(self,x,key):return self.r(x@self.w[key].T)
 def silu(self,g):return self.r(g*np.exp(-np.logaddexp(np.float32(0),-g)))
 def attention(self,x,layer,isolated=False,edges=(),patch=None):
  n=len(x);pre=f'model.layers.{layer}.self_attn.'
  q,k,v=[np.clip(self.linear(x,pre+t+'_proj.weight'),-8,8).reshape(n,16,128).transpose(1,0,2) for t in 'qkv']
  inv=np.float32(1)/(np.float32(10000)**(np.arange(0,128,2,dtype=np.float32)/np.float32(128)));freq=np.arange(n,dtype=np.float32)[:,None]*inv;emb=np.concatenate((freq,freq),axis=-1);co=np.cos(emb)[None];si=np.sin(emb)[None]
  def rotate(z):return np.concatenate((-z[:,:,64:],z[:,:,:64]),axis=-1)
  q=self.r(q*co+rotate(q)*si);k=self.r(k*co+rotate(k)*si)
  scores=(q@k.transpose(0,2,1))*np.float32(128**-.5);scores[:,np.triu_indices(n,k=1)[0],np.triu_indices(n,k=1)[1]]=-np.inf
  if isolated:scores[:,-1,:-1]=-np.inf
  for h,dest,source in edges:scores[h,dest,source]=-np.inf
  probs=np.exp(scores-np.max(scores,axis=-1,keepdims=True));probs/=np.sum(probs,axis=-1,keepdims=True,dtype=np.float32)
  context=self.r((probs@v).transpose(1,0,2).reshape(n,2048))
  if patch is not None:
   head,donor=patch;context[-1,head*128:(head+1)*128]=donor[head*128:(head+1)*128]
  output=self.linear(context,pre+'o_proj.weight')
  return output,context,probs
 def forward(self,ids,isolated_layers=(),edges=(),mlp0_q_zero=False,patch_l1=None,return_states=False):
  x=np.stack([self.embedding[i] for i in ids]);n=len(x)
  z,c0,a0=self.attention(self.ln(x),0,0 in isolated_layers,edges);post0=self.r(x+z);m=self.ln(post0)
  g=self.linear(m,'model.layers.0.mlp.gate_proj.weight');u=self.linear(m,'model.layers.0.mlp.up_proj.weight');a=self.r(self.silu(g)*u);m0=self.linear(a,'model.layers.0.mlp.down_proj.weight')
  if mlp0_q_zero:m0[-1]=0
  x1=self.r(post0+m0);z,c1,a1=self.attention(self.ln(x1),1,1 in isolated_layers,patch=patch_l1);premlp=self.r(x1+z);normalized=self.ln(premlp)
  gate=np.stack([self.r(normalized@self.w[f'gate_proj_{i}']) for i in P['neurons']],axis=-1);up=np.stack([self.r(normalized@self.w[f'up_proj_{i}']) for i in P['neurons']],axis=-1);activation=self.r(self.silu(gate)*up);contribution=activation*self.down
  pos=sorted(set([0,min(64,n-1),n-1]));centered=premlp-np.mean(premlp,axis=-1,keepdims=True)
  obs={'positions':pos,'gate':gate[pos].tolist(),'up':up[pos].tolist(),'activation':activation[pos].tolist(),'contribution':contribution[pos].tolist(),'l0_attention_q_to_0_p_q':a0[:,-1,pos].tolist(),'l1_attention_q_to_0_p_q':a1[:,-1,pos].tolist(),'pre_l1_centered_norm':np.linalg.norm(x1[pos]-x1[pos].mean(axis=-1,keepdims=True),axis=-1).tolist(),'pre_mlp1_centered_norm':np.linalg.norm(centered[pos],axis=-1).tolist(),'l0_H14_context_norm_q':float(np.linalg.norm(c0[-1,14*128:15*128])),'l1_H12_context_norm_q':float(np.linalg.norm(c1[-1,12*128:13*128])),'rss_bytes':self.check_memory()}
  if return_states:return obs,{'l1_context_q':c1[-1].copy(),'activation':activation.copy(),'premlp':premlp.copy()}
  return obs
