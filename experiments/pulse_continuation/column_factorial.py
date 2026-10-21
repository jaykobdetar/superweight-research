"""Split one MLP neuron's down-column into dominant row and its complement."""
import json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from empty_history import Masks
from upstream import Heads

class Column:
 def __init__(self,model,mask):
  self.mask=mask;self.mode='clean';self.module=model.model.layers[1].mlp.down_proj;self.weight=self.module.weight.detach().clone();self.weight[:,1710]=0;self.handle=self.module.register_forward_hook(self.hook)
 def hook(self,m,args,o):
  if self.mode=='clean':return o
  alt=F.linear(args[0],self.weight,m.bias);z=o.clone();p=self.mask.p
  if self.mode=='delete_row':z[:,p,1764]=alt[:,p,1764]
  elif self.mode=='delete_complement':z[:,p]=alt[:,p];z[:,p,1764]=o[:,p,1764]
  elif self.mode=='delete_column':z[:,p]=alt[:,p]
  else:raise ValueError(self.mode)
  return z
 def close(self):self.handle.remove()

class Trace:
 def __init__(self,model,mask):
  self.mask=mask;self.cache={};self.handles=[];layer=model.model.layers[2]
  self.handles.append(layer.input_layernorm.register_forward_hook(self.norm))
  for k in ('k','v'):self.handles.append(getattr(layer.self_attn,k+'_proj').register_forward_hook(lambda m,a,o,k=k:self.proj(k,o)))
 def norm(self,m,args,o):self.cache={'input':args[0][0,[0,self.mask.p]].detach().cpu().clone(),'normalized':o[0,[0,self.mask.p]].detach().cpu().clone()};return o
 def proj(self,k,o):self.cache[k]=o[0,[0,self.mask.p]].clamp(-8,8).detach().cpu().clone();return o
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);mask=Masks(ctx['model']);heads=Heads(ctx['model'],mask);column=Column(ctx['model'],mask);trace=Trace(ctx['model'],mask)
 sets=torch.load(ROOT/'upstream_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');saved={};checks=[]
 torch.save(ctx['model'].model.layers[1].mlp.down_proj.weight[:,1710].detach().cpu().clone(),out/'down_column1710.pt')
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'COLUMN_FACTORIAL.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'upstream_run/tokens.pt'),'intervention':'column1710 split into row1764 and complementary2047 coordinates; target token only','source0_score_start':1,'boundary_score_start':'p+1; same254 target IDs','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,world,patch=None,mode='clean',state='clean',bl=None,keep=False):
  mask.mode='L0_L1' if world=='isolated' else 'explicit';heads.patch=patch or {};column.mode=mode;z,_,ll=r.evaluate(x,state,source_position=mask.p,baseline=bl,keep=keep)
  return z,ll,{l:d.clone() for l,d in heads.cache.items()},{**trace.cache,'intermediate':mask.trace.clone()}
 def panel(x,g,i,world,patch,b,bl,bt,ss,spec):
  r.record(g,'clean',i,x,b,b,spec,ss);saved[f'{g}__clean__{i}']=bt
  for mode in ('delete_row','delete_complement','delete_column'):
   z,_,_,t=ev(x,world,patch,mode,bl=bl);assert torch.equal(t['intermediate'],bt['intermediate']);r.record(g,mode,i,x,z,b,{**spec,'component':mode},ss);saved[f'{g}__{mode}__{i}']=t
   if mode=='delete_row':zr=z
  z,_,_,t=ev(x,world,patch,state='deleted',bl=bl);check={'group':g,'window':i,'row_implementation_loss_exact':torch.equal(z['loss'],zr['loss']),'row_implementation_pred_exact':torch.equal(z['pred'],zr['pred']),'intermediate_unchanged':torch.equal(t['intermediate'],bt['intermediate'])};checks.append(check);assert all(check[k] for k in ('row_implementation_loss_exact','row_implementation_pred_exact','intermediate_unchanged')),check
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    mask.p=0
    for identity,tid in [('natural',None),('The',510),('period',15)]:
     x=xx.clone()
     if tid is not None:x[0]=tid
     b,bl,_,bt=ev(x,'ordinary',keep=True);panel(x,f'first_{c}_{identity}',i,'ordinary',None,b,bl,bt,1,{'position':0,'identity':identity});del bl
    for identity,tid in [('The',510),('period',15)]:
     for p in (128,256):
      mask.p=p;x=xx[256-p:].clone();x[0]=tid;x[p]=tid
      ob,ol,oh,ot=ev(x,'ordinary',keep=True);ib,il,ih,it=ev(x,'isolated',keep=True);hp={'heads':[(1,12)],'donor':ih};hb,hl,_,ht=ev(x,'ordinary',hp,keep=True)
      for name,world,patch,b,bl,bt in [('ordinary','ordinary',None,ob,ol,ot),('isolated','isolated',None,ib,il,it),('induced','ordinary',hp,hb,hl,ht)]:
       panel(x,f'boundary_{c}_{identity}_{p}_{name}',i,world,patch,b,bl,bt,p+1,{'position':p,'identity':identity,'world':name,'source0':'intact'})
      del ol,il,hl
    r.checkpoint();torch.save(saved,out/'source_traces.pt');savej(out/'checks.json',checks);log(f'Column factorial {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:trace.close();column.close();heads.close();mask.close();r.close()
