"""Correctly cited same-layer two-column factorial, source0 only."""
import itertools,json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
class Pair:
 def __init__(self,model):
  self.m=model.model.layers[1].mlp.down_proj;self.state=('clean','clean');self.alt=None;self.cache={};self.handle=self.m.register_forward_hook(self.hook)
 def set(self,a,b):
  self.state=(a,b);self.alt=None
  if self.state!=('clean','clean'):
   self.alt=self.m.weight.detach().clone()
   for c,s in zip((1710,8041),self.state):
    if s=='row':self.alt[1764,c]=0
    elif s=='column':self.alt[:,c]=0
 def hook(self,m,a,o):
  z=o
  if self.alt is not None:z=o.clone();z[:,0]=F.linear(a[0],self.alt,m.bias)[:,0]
  self.cache={'activations':a[0][0,0,[1710,8041]].detach().cpu().clone(),'weights':m.weight[1764,[1710,8041]].detach().cpu().clone(),'before':o[0,0].detach().cpu().clone(),'after':z[0,0].detach().cpu().clone()};return z
 def close(self):self.handle.remove()
def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not(out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);p=Pair(ctx['model']);sets=torch.load(ROOT/'upstream_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');raw={};checks=[]
 old=torch.load(ROOT/'column_factorial_run/per_token.pt',weights_only=True,map_location='cpu');old={k:{a['window']:a for a in v} for k,v in old.items()}
 meta={'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'SAME_LAYER_PAIR.md'),'model_hash_before':ctx['original_hash'],'targets':[[1,1764,1710],[1,1764,8041]],'tokens_parent_sha256':sha(ROOT/'upstream_run/tokens.pt'),'score_start':1,'seed':20261021,'dtype':'float16','started_utc':time.strftime('%FT%TZ',time.gmtime())};savej(out/'manifest.json',meta)
 try:
  for c,ws in sets.items():
   for i,x in enumerate(ws):
    p.set('clean','clean');b,_,bl=r.evaluate(x,keep=True);bc={k:v.clone() for k,v in p.cache.items()}
    for a,s in itertools.product(('clean','row','column'),repeat=2):
     t='clean' if a==s=='clean' else 'first_'+a+'_second_'+s
     if a==s=='clean':z=b;cap=bc
     else:p.set(a,s);z,_,_=r.evaluate(x,baseline=bl);cap={k:v.clone() for k,v in p.cache.items()}
     assert torch.equal(cap['activations'],bc['activations']);g='pair_'+c;r.record(g,t,i,x,z,b,{'first':a,'second':s,'source_position':0},1);raw[f'{g}__{t}__{i}']=cap
     if s=='clean':
      ref=old[f'first_{c}_natural__'+{'clean':'clean','row':'delete_row','column':'delete_column'}[a]][i];ch={'group':g,'trial':t,'window':i,'loss_exact':torch.equal(z['loss'][1:],ref['loss']),'pred_exact':torch.equal(z['pred'][1:],ref['pred'])};checks.append(ch);assert ch['loss_exact'] and ch['pred_exact'],ch
    del bl;r.checkpoint();torch.save(raw,out/'traces.pt');savej(out/'checks.json',checks);log(f'Same layer pair {c} {i+1}/4;{r.count}forwards')
  meta.update(forward_passes=r.count,wall_seconds=time.time()-start,completed_utc=time.strftime('%FT%TZ',time.gmtime()),model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);savej(out/'manifest.json',meta)
 finally:p.close();r.close()
