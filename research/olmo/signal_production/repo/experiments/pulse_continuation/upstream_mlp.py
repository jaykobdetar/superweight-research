"""Causal scalar MLP-factor and H12 routing tests after upstream localization."""
import json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from empty_history import Masks
from upstream import Heads

class Routes(Heads):
 def __init__(self,model,mask):
  super().__init__(model,mask);self.mode='none';self.values=None
  self.handles.append(model.model.layers[1].self_attn.v_proj.register_forward_hook(self.vhook))
  self.handles.append(model.model.layers[1].self_attn.register_forward_pre_hook(self.attn,with_kwargs=True))
 def vhook(self,m,args,o):self.values=o[0,[0,self.mask.p],12*128:13*128].clamp(-8,8).detach().clone()
 def hook(self,l,args):
  if l==1 and self.mode in ('value0','valuep','zero'):
   x=args[0].clone();x[:,self.mask.p,12*128:13*128]=torch.zeros_like(self.values[0]) if self.mode=='zero' else self.values[0 if self.mode=='value0' else 1];args=(x,)
  return super().hook(l,args)
 def attn(self,m,args,k):
  if self.mode!='selfmask':return None
  k=dict(k);m=k['attention_mask'].expand(1,16,-1,-1).clone();m[:,12,self.mask.p,:self.mask.p]=-torch.inf;k['attention_mask']=m;return args,k

class Factors:
 def __init__(self,model,mask):
  self.mask=mask;self.patch={};self.cache={};self.handles=[];layer=model.model.layers[1]
  self.handles.append(layer.post_attention_layernorm.register_forward_hook(self.norm))
  for name in ('gate','up'):self.handles.append(getattr(layer.mlp,name+'_proj').register_forward_hook(lambda m,a,o,name=name:self.factor(name,o)))
 def norm(self,m,args,o):
  self.cache={'norm_input':args[0][0,[0,self.mask.p]].detach().cpu().clone(),'norm_output':o[0,[0,self.mask.p]].detach().cpu().clone()};return o
 def factor(self,name,o):
  if name in self.patch:o=o.clone();o[0,self.mask.p,1710]=self.patch[name].to(o.device)
  self.cache[name]=o[0,[0,self.mask.p],1710].detach().cpu().clone();return o
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);mask=Masks(ctx['model']);route=Routes(ctx['model'],mask);factors=Factors(ctx['model'],mask)
 source=torch.load(ROOT/'upstream_run/tokens.pt',weights_only=True,map_location='cpu');sets=source['evaluation'];torch.save(sets,out/'tokens.pt');raw={};obs=[];checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'UPSTREAM_MLP.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'upstream_run/tokens.pt'),'selection':'L1/H12 frozen from preceding development screen; no fit','factor_patch':'only neuron1710 of gate/up projections, before SiLU/product; whole down-projection column changes','source_deletion':'boundary row1764 contribution only; source0 intact','score_start':'boundary+1,254 targets','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,world='ordinary',heads=None,factor=None,mode='none',state='clean',bl=None,keep=False):
  mask.mode='explicit' if world=='ordinary' else 'L0_L1';route.patch=heads or {};route.mode=mode;factors.patch=factor or {};z,_,logits=r.evaluate(x,state,source_position=mask.p,baseline=bl,keep=keep)
  cache={**factors.cache,'intermediate':mask.trace.clone(),'head_context':route.cache[1].clone(),'clipped_H12_values':route.values.cpu().clone()};return z,logits,cache,{l:a.clone() for l,a in route.cache.items()}
 def record(g,t,i,x,z,b,cache,spec):
  r.record(g,t,i,x,z,b,spec,mask.p+1);key=f'{g}__{t}__{i}';raw[key]=cache
  gate=cache['gate'][1];up=cache['up'][1];inter=cache['intermediate'][1,1710];calc=torch.nn.functional.silu(gate)*up
  cpu_reconstruction_difference=float(calc.float()-inter.float())
  obs.append({'group':g,'trial':t,'window':i,'gate':float(gate),'up':float(up),'intermediate':float(inter),'pulse':float(inter.float()*mask.weight),'source_pulse':float(cache['intermediate'][0,1710].float()*mask.weight),'cpu_factor_reconstruction_exact':torch.equal(calc,inter),'cpu_factor_reconstruction_difference':cpu_reconstruction_difference,'token_sha256':tokenhash(x)})
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity,tid in [('The',510),('period',15)]:
     for p in (128,256):
      x=xx[256-p:].clone();mask.p=p;x[0]=tid;x[p]=tid;g=f'upstream_{c}_{identity}_{p}'
      b,bl,bc,bh=ev(x,keep=True);a,_,ac,ah=ev(x,'isolated',bl=bl);refs={'ordinary':bc,'isolated':ac};headrepair={'heads':[(1,12)],'donor':ah};headdamage={'heads':[(1,12)],'donor':bh}
      for name,z,cache in [('ordinary',b,bc),('isolated',a,ac)]:record(g,name,i,x,z,b,cache,{'world':name})
      for world,donor in [('ordinary',ac),('isolated',bc)]:
       for kinds in [('gate',),('up',),('gate','up')]:
        patch={k:donor[k][1] for k in kinds};z,_,cc,_=ev(x,world,factor=patch,bl=bl);t=world+'_swap_'+'_'.join(kinds);record(g,t,i,x,z,b,cc,{'recipient':world,'factor_patch':list(kinds),'donor':'other world'})
        if len(kinds)==2:
         assert torch.equal(cc['intermediate'][1,1710],donor['intermediate'][1,1710]);checks.append({'group':g,'window':i,'world':world,'donor_intermediate_exact':True})
      # Explicit H12-absent follow-up controls, chosen before this run.
      for world,donor in [('ordinary',ah),('isolated',bh)]:
       for name,hs in [('H0',[(1,0)]),('H3',[(1,3)]),('H15',[(1,15)]),('complement31',[(l,h) for l in (0,1) for h in range(16) if (l,h)!=(1,12)])]:
        z,_,cc,_=ev(x,world,heads={'heads':hs,'donor':donor},bl=bl);record(g,world+'_'+name,i,x,z,b,cc,{'recipient':world,'head_patch':'other world','heads':hs,'followup_control':'explicitly excludes L1/H12'})
      for world,hpatch in [('ordinary',headrepair),('isolated',headdamage)]:
       z,_,cc,_=ev(x,world,heads=hpatch,bl=bl);record(g,world+'_H12',i,x,z,b,cc,{'recipient':world,'head_patch':'other world L1/H12'})
      for kinds in [('gate',),('up',),('gate','up')]:
       z,_,cc,_=ev(x,heads=headrepair,factor={k:bc[k][1] for k in kinds},bl=bl);record(g,'induced_restore_'+'_'.join(kinds),i,x,z,b,cc,{'recipient':'ordinary + isolated H12','factor_patch':list(kinds),'donor':'ordinary'})
      z,_,cc,_=ev(x,heads=headrepair,state='deleted',bl=bl);record(g,'ordinary_H12_delete_boundary',i,x,z,b,cc,{'head_patch':'isolated H12','source':'boundary contribution deleted'})
      routes={}
      for mode in ('value0','valuep','zero','selfmask'):
       z,_,cc,_=ev(x,mode=mode,bl=bl);routes[mode]=cc;record(g,'H12_'+mode,i,x,z,b,cc,{'recipient':'ordinary','routing':mode})
      checks.append({'group':g,'window':i,'self_context_vs_mask_intermediate_maxdiff':float((routes['valuep']['intermediate'].float()-routes['selfmask']['intermediate'].float()).abs().max()),'self_context_vs_mask_pulse_diff':float((routes['valuep']['intermediate'][1,1710].float()-routes['selfmask']['intermediate'][1,1710].float())*mask.weight)})
      del bl
    r.checkpoint();torch.save(raw,out/'factor_donors.pt');savej(out/'factor_observations.json',obs);savej(out/'checks.json',checks);log(f'Upstream MLP {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:factors.close();route.close();mask.close();r.close()
