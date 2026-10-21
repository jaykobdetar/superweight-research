"""Separate missing fixed source values from the downstream gate changes they induce."""
import resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from quiet_anchor import SourceMask
from global_gates import Gates

class DonorGates(Gates):
 def context(self,l,inputs):
  if not getattr(self,'donor_active',False):return super().context(l,inputs)
  current=self.probabilities(l);del self.gpu[l];a=self.donor[l].to(inputs[0].device) if l in self.donor_layers else current
  o=inputs[0];n=o.shape[1];u=o[:,1:].reshape(1,n-1,16,128).float();vv=self.mean[l]['v'].to(o.device).reshape(1,1,16,128).float();aa=a[:,1:].T[None,:,:,None]
  result=o.clone();result[:,1:]=((1-aa)*u+aa*vv).reshape(1,n-1,2048).to(o.dtype);return (result,)

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 sets=torch.load(ROOT/'residual_paths_run/tokens.pt',weights_only=True,map_location='cpu');packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki'];cal=torch.load(ROOT/'global_gates_run/calibration.pt',weights_only=True,map_location='cpu')
 mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 zero={l:{'k':d['k'],'v':torch.zeros_like(d['v'])} for l,d in mean.items()};worlds={'mean':mean,'zero':zero};layers=list(range(2,16))
 r=Run(model,out);r.local.close();r.local=Paths(model);mask=SourceMask(model);gate=DonorGates(model,mean);donors={};checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'tokens_sha256':sha(ROOT/'residual_paths_run/tokens.pt'),'mean_cache_sha256':sha(ROOT/'cache_run/calibration.pt'),'gate_calibration_sha256':sha(ROOT/'global_gates_run/calibration.pt'),
  'worlds':'all source K0 clamped to Wiki means L2..15; source V0 clamped to means or zero at all those layers',
  'donors':'per-window full causal source attention gates captured independently in each native world before interventions; diagnostic oracles, not fitted replacements',
  'patch':'source-excluded native conditional mixture combined with current or donor source gate and recipient fixed V0; queries and other-token states remain recipient-dependent',
  'layer_sets':{'all':layers,'early':list(range(2,6)),'middle':list(range(6,12)),'late':list(range(12,16)),'L2':[2],'L6':[6]},
  'controls':'current-recipient reconstruction; same-world donor gates; separately frozen Wiki/Hume global gates',
  'scope':'feedforward dependencies across layers; the filename does not imply a literal recurrent feedback loop','score_start':1,'dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta);torch.save({c:d['evaluation'] for c,d in sets.items()},out/'tokens.pt')
 try:
  for c,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    mask.layers=[];gate.donor_active=False;gate.reset();b,_,bl=r.evaluate(x,keep=True);group='gate_paths_'+c;r.record(group,'clean',i,x,b,b,{'reference':'native model'},1)
    native={};source={}
    for name,w in worlds.items():
     gate.mean=w;gate.reset('calibrate');z,_,_=r.evaluate(x,patch={'type':'cut','layers':layers,'donor':w},baseline=bl);native[name]=z;source[name]=gate.cache
     r.record(group,name+'_native',i,x,z,b,{'source_values':name,'gates':'native'},1)
    donors[c,i]=source;mask.layers=layers
    for world,w in worlds.items():
     gate.mean=w;other='zero' if world=='mean' else 'mean';patch={'type':'cut','layers':layers,'donor':w}
     variants=[('current',None,[]),('same_all',source[world],layers)]+[('cross_'+name,source[other],ls) for name,ls in meta['layer_sets'].items()]
     for name,donor,ls in variants:
      gate.reset('oracle');gate.donor_active=donor is not None;gate.donor=donor;gate.donor_layers=ls
      z,_,_=r.evaluate(x,patch=patch,baseline=bl);r.record(group,world+'_'+name,i,x,z,b,{'source_values':world,'donor':other if name.startswith('cross') else world,'gate_layers':ls,'gates':name},1)
      if name in ('current','same_all'):checks.append({'corpus':c,'window':i,'world':world,'control':name,'delta_nll_to_native':float((z['loss'][1:]-native[world]['loss'][1:]).double().mean()),'agreement_to_native':float(z['pred'][1:].eq(native[world]['pred'][1:]).double().mean())})
      if world=='zero' and name=='cross_all':
       zd,_,_=r.evaluate(x,'deleted',patch,baseline=bl);assert all(torch.equal(z[k][1:],zd[k][1:]) for k in ('loss','pred','kl'))
       r.record(group,'zero_cross_all_deleted',i,x,zd,b,{'source_values':'zero','gate_donor':'mean','source_state':'deleted'},1)
     gate.donor_active=False
     for fit in ('wiki','hume'):
      gate.reset('global',cal[fit]);z,_,_=r.evaluate(x,patch=patch,baseline=bl)
      r.record(group,world+'_global_'+fit,i,x,z,b,{'source_values':world,'global_calibration':fit},1)
    del bl;r.checkpoint();torch.save(donors,out/'native_gate_donors.pt');savej(out/'reconstruction_checks.json',checks);log(f'Gate paths {c} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:gate.close();mask.close();r.close()
