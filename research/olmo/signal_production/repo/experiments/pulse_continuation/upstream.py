"""Development-selected causal head patches for upstream pulse production."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from empty_history import Masks

class Heads:
 def __init__(self,model,mask):
  self.mask=mask;self.patch={};self.cache={};self.handles=[]
  for l in (0,1):self.handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(lambda m,a,l=l:self.hook(l,a)))
 def hook(self,l,args):
  x=args[0];heads=self.patch.get('heads',[])
  if any(ll==l for ll,h in heads):
   x=x.clone();d=self.patch['donor'][l].to(x.device)
   for ll,h in heads:
    if ll==l:x[:,self.mask.p,h*128:(h+1)*128]=d[:,h*128:(h+1)*128]
  self.cache[l]=x[:,self.mask.p].detach().cpu().clone();return (x,)
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);mask=Masks(ctx['model']);heads=Heads(ctx['model'],mask)
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');fresh=torch.load(ROOT/'confirmation_run/tokens.pt',weights_only=True,map_location='cpu')
 dev=[old['corpora']['wiki'][s:s+512].clone() for s in range(18432,26624,1024)];sets={c:d['evaluation'][:4] for c,d in fresh.items()};torch.save({'development':dev,'evaluation':sets},out/'tokens.pt')
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'UPSTREAM.md'),'model_hash_before':ctx['original_hash'],'development_parent_sha256':sha(ROOT.parent/'pulse_causal_tranche/run/tokens.pt'),'heldout_parent_sha256':sha(ROOT/'confirmation_run/tokens.pt'),'source':'L1 down_proj[1764,1710] boundary only when deleted; source0 intact','selection':'signed fraction of ordinary-to-isolated pulse gap, mean over8 development windows','score_start':'boundary+1; same254 targets across prefix lengths','dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta);allheads=[(l,h) for l in (0,1) for h in range(16)];screen=[];donors={};traces={};obs=[];checks=[]
 def ev(x,world,patch=None,state='clean',baseline=None,keep=False):
  mask.mode='explicit' if world=='ordinary' else 'L0_L1';heads.patch=patch or {};z,_,bl=r.evaluate(x,state,source_position=mask.p,baseline=baseline,keep=keep)
  return z,bl,{l:a.clone() for l,a in heads.cache.items()},mask.trace.clone()
 def observe(group,trial,i,x,z,b,trace,refs,spec):
  r.record(group,trial,i,x,z,b,spec,mask.p+1);key=f'{group}__{trial}__{i}';traces[key]=trace
  pulse=float(trace[1,1710].float()*mask.weight);p0=float(refs['ordinary'][1,1710].float()*mask.weight);p1=float(refs['isolated'][1,1710].float()*mask.weight)
  rec={'group':group,'trial':trial,'window':i,'pulse':pulse,'pulse_gap_fraction':(pulse-p0)/(p1-p0),'ordinary_pulse':p0,'isolated_pulse':p1,'source_pulse':float(trace[0,1710].float()*mask.weight),'intermediate_source_boundary_equal':torch.equal(trace[0],trace[1]),'token_sha256':tokenhash(x)}
  obs.append(rec)
 try:
  for i,xx in enumerate(dev):
   x=xx.clone();mask.p=256;x[0]=510;x[256]=510;_,_,dc,t0=ev(x,'ordinary');_,_,ic,t1=ev(x,'isolated');den=float((t1[1,1710].float()-t0[1,1710].float()));assert abs(den)>1
   donors['dev_'+str(i)]={'ordinary':dc,'isolated':ic,'intermediates':{'ordinary':t0,'isolated':t1}}
   for l,h in allheads:
    _,_,_,t=ev(x,'ordinary',{'heads':[(l,h)],'donor':ic});screen.append({'window':i,'layer':l,'head':h,'gap_fraction':float((t[1,1710].float()-t0[1,1710].float()))/den,'pulse':float(t[1,1710].float()*mask.weight)})
  scores={(l,h):float(np.mean([s['gap_fraction'] for s in screen if s['layer']==l and s['head']==h])) for l,h in allheads};ranked=sorted(allheads,key=lambda a:(-scores[a],a));rng=np.random.default_rng(20261021);frozen={}
  for k in (1,2,4,8,16,32):
   chosen=ranked[:k];frozen[f'top{k}']=chosen
   if k<32:frozen[f'control{k}']=[(l,int(h)) for l in (0,1) for h in rng.choice(16,sum(ll==l for ll,h in chosen),replace=False)]
  frozen['L0']=[(0,h) for h in range(16)];frozen['L1']=[(1,h) for h in range(16)]
  savej(out/'screen.json',screen);savej(out/'frozen_sets.json',{'sets':frozen,'ranking':[{'layer':l,'head':h,'mean_gap_fraction':scores[l,h]} for l,h in ranked]});log('Upstream head sets frozen before evaluation')
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity,tid in [('The',510),('period',15)]:
     for p in (128,256):
      x=xx[256-p:].clone();mask.p=p;x[0]=tid;x[p]=tid;group=f'upstream_{c}_{identity}_{p}'
      b,bl,bc,bt=ev(x,'ordinary',keep=True);a,al,ac,at=ev(x,'isolated',baseline=bl,keep=True);refs={'ordinary':bt,'isolated':at};donors[f'{group}_{i}']={'ordinary':bc,'isolated':ac,'intermediates':refs}
      assert torch.equal(at[0],at[1]),'Self-only same-ID intermediate control failed'
      for name,z,t in [('ordinary',b,bt),('isolated',a,at)]:observe(group,name,i,x,z,b,t,refs,{'world':name,'prefix':p,'identity':identity})
      for world,donor,ref in [('ordinary',ac,at),('isolated',bc,bt)]:
       for name,hs in frozen.items():
        patch={'heads':hs,'donor':donor};z,_,_,t=ev(x,world,patch,baseline=bl);trial=world+'_'+name;observe(group,trial,i,x,z,b,t,refs,{'recipient':world,'heads':hs,'prefix':p,'identity':identity})
        if name=='top32':
         target=a if world=='ordinary' else b
         check={'group':group,'window':i,'recipient':world,'intermediate_exact':torch.equal(t,ref),'loss_exact':torch.equal(z['loss'],target['loss']),'pred_exact':torch.equal(z['pred'],target['pred'])};checks.append(check);assert all(check[k] for k in ('intermediate_exact','loss_exact','pred_exact')),check
        if world=='ordinary' and name in ('top2','top8','top32'):
         dz,_,_,dt=ev(x,world,patch,'deleted',baseline=bl);assert torch.equal(dt,t);observe(group,trial+'_delete_boundary',i,x,dz,b,dt,refs,{'recipient':world,'heads':hs,'delete':'boundary pulse only','prefix':p,'identity':identity})
      for world in ('ordinary','isolated'):
       z,_,_,t=ev(x,world,state='deleted',baseline=bl);observe(group,world+'_delete_boundary',i,x,z,b,t,refs,{'world':world,'delete':'boundary pulse only','prefix':p,'identity':identity})
      del bl,al
    r.checkpoint();savej(out/'pulse_observations.json',obs);torch.save(traces,out/'source_intermediates.pt');torch.save(donors,out/'head_donors.pt');log(f'Upstream {c} {i+1}/4; {r.count} forwards')
  savej(out/'closure_checks.json',checks);meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:heads.close();mask.close();r.close()
