"""Replace query-dependent source attention with calibrated scalar gates and fixed values."""
import json,math,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,rotate_half
from mediation import Paths
from quiet_anchor import SourceMask

class Gates:
 def __init__(self,model,mean):
  self.mean=mean;self.handles=[];self.reset()
  for l in range(2,16):
   a=model.model.layers[l].self_attn
   self.handles.append(a.register_forward_pre_hook(lambda m,args,kw,l=l:self.attn(l,kw),with_kwargs=True))
   for k in ('q','k'):self.handles.append(getattr(a,k+'_proj').register_forward_hook(lambda m,i,o,l=l,k=k:self.proj(l,k,o)))
   self.handles.append(a.o_proj.register_forward_pre_hook(lambda m,i,l=l:self.context(l,i)))
 def reset(self,mode='off',cal=None,snapshot=False):self.mode=mode;self.cal=cal;self.gpu={};self.cache={};self.saved={};self.snapshot=snapshot
 def attn(self,l,kw):
  if self.mode in ('calibrate','oracle'):self.gpu[l]={'cos':kw['position_embeddings'][0],'sin':kw['position_embeddings'][1]}
 def proj(self,l,k,o):
  if self.mode in ('calibrate','oracle'):self.gpu[l][k]=o.detach()
 def probabilities(self,l):
  c=self.gpu[l];q=c['q'].clamp(-8,8).reshape(1,-1,16,128).transpose(1,2);k=c['k'].clamp(-8,8).reshape(1,-1,16,128).transpose(1,2)
  co=c['cos'][:,None];si=c['sin'][:,None]
  q=(q*co+rotate_half(q)*si).to(q.dtype).float()[0];k=(k*co+rotate_half(k)*si).to(k.dtype).float()[0];n=q.shape[1];parts=[]
  for s in range(0,n,64):
   e=min(n,s+64);scores=q[:,s:e]@k.transpose(1,2)/math.sqrt(128)
   future=torch.arange(n,device=q.device)[None,:]>torch.arange(s,e,device=q.device)[:,None];scores.masked_fill_(future[None],-torch.inf)
   parts.append(scores.softmax(-1)[:,:,0])
  return torch.cat(parts,1)
 def context(self,l,inputs):
  if self.mode=='off':return None
  o=inputs[0];n=o.shape[1]
  if self.mode in ('calibrate','oracle'):
   a=self.probabilities(l);self.cache[l]=a.detach().cpu()
   if self.snapshot:self.saved[l]={k:v.detach().cpu().clone() for k,v in self.gpu[l].items()}
   del self.gpu[l]
   if self.mode=='calibrate':return None
  elif self.mode in ('global','bias'):a=self.cal[l]['global'].to(o.device)[:,None].expand(-1,n)
  elif self.mode=='binned':
   a=torch.ones((16,n),device=o.device)
   for j,(s,e) in enumerate(zip([1,16,64,128],[16,64,128,512])):a[:,s:e]=self.cal[l]['binned'][:,j,None].to(o.device)
  elif self.mode=='zero':a=torch.zeros((16,n),device=o.device)
  elif self.mode=='one':a=torch.ones((16,n),device=o.device)
  else:raise ValueError(self.mode)
  u=o.reshape(1,n,16,128).float();vv=self.mean[l]['v'].to(o.device).reshape(1,1,16,128).float();aa=a.T[None,:,:,None]
  y=u+aa*vv if self.mode=='bias' else (1-aa)*u+aa*vv
  result=o.clone();result[:,1:]=y.reshape(1,n,2048)[:,1:].to(o.dtype)
  return (result,)
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 sets=torch.load(ROOT/'residual_paths_run/tokens.pt',weights_only=True,map_location='cpu');old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
 packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki'];mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 r=Run(model,out);r.local.close();r.local=Paths(model);mask=SourceMask(model);gate=Gates(model,mean);layers=list(range(2,16));patch={'type':'cut','layers':layers,'donor':mean}
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'frozen_mean_cache_sha256':sha(ROOT/'cache_run/calibration.pt'),'evaluation_tokens_sha256':sha(ROOT/'residual_paths_run/tokens.pt'),
  'calibration':'separate8 Wiki and8 Hume old development windows; native mean-KV world, arithmetic source mass means, no optimization/outcome selection',
  'equation':'Y=(1-a)*u+a*v0; u is native SDPA with source0 masked for queries1+, v0 is fixed Wiki mean; all source K/V fixed at layers2..15',
  'global':'one scalar per layer/head, mean across calibration windows and queries1..511',
  'binned':'four query-position bins1:16,16:64,64:128,128:512, independently averaged over calibration windows',
  'bias':'u+mean(a)*v0; deliberately omits attenuation of nonzero-key mixture',
  'oracle_control':'recompute current recipient full causal source mass using clipped/FP16-rotated Q/K, then combine native conditional u and mean v; expose reconstruction error',
  'evaluation':'same8 heldout512-token windows/corpus as quiet/value panels; no new fitting on outcomes','score_start':1,'dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta);calibration={};calmass={};checks=[]
 try:
  for c in ('wiki','hume'):
   caches=[];xs=[];mask.layers=[]
   for i,s in enumerate(old['starts'][c]['development']):
    x=old['corpora'][c][s:s+512].clone();xs.append(x);gate.reset('calibrate',snapshot=i==0);r.evaluate(x,patch=patch);caches.append(gate.cache)
    if i==0:torch.save(gate.saved,out/f'calibration_qk_snapshot_{c}.pt')
   calmass[c]=caches;calibration[c]={}
   for l in layers:
    a=torch.stack([z[l] for z in caches]);calibration[c][l]={'global':a[:,:,1:].mean((0,2)),
     'binned':torch.stack([a[:,:,s:e].mean((0,2)) for s,e in zip([1,16,64,128],[16,64,128,512])],1)}
   torch.save(xs,out/f'calibration_tokens_{c}.pt')
  torch.save(calibration,out/'calibration.pt');torch.save(calmass,out/'calibration_source_mass.pt');log('Global gates frozen before evaluation')
  torch.save({c:d['evaluation'] for c,d in sets.items()},out/'tokens.pt')
  for c,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    mask.layers=[];gate.reset();b,_,bl=r.evaluate(x,keep=True);group='gates_'+c;r.record(group,'clean',i,x,b,b,{'reference':'native model with explicit causal mask'},1)
    bm,_,_=r.evaluate(x,patch=patch,baseline=bl);r.record(group,'mean_KV',i,x,bm,b,{'reference':'native mean source cache'},1)
    mask.layers=layers
    variants=[('oracle','oracle',None),('remove_source','zero',None),('constant_value_only','one',None)]
    for fit in ('wiki','hume'):
     for mode in ('global','binned','bias'):variants.append((fit+'_'+mode,mode,calibration[fit]))
    for name,mode,cal in variants:
     prior=None
     for state in ('clean','deleted'):
      gate.reset(mode,cal);z,_,_=r.evaluate(x,state,patch,baseline=bl)
      if prior is not None:assert all(torch.equal(z[k][1:],prior[k][1:]) for k in ('loss','pred','kl'))
      prior=z;r.record(group,name+'_'+state,i,x,z,b,{'operation':mode,'calibration':name.split('_')[0] if cal else None,'source_state':state},1)
      if name=='oracle' and state=='clean':
       checks.append({'corpus':c,'window':i,'oracle_mean_delta_nll':float((z['loss'][1:]-bm['loss'][1:]).double().mean()),
        'oracle_max_abs_token_loss_delta':float((z['loss'][1:]-bm['loss'][1:]).abs().max()),'oracle_argmax_agreement_to_mean':float(z['pred'][1:].eq(bm['pred'][1:]).double().mean())})
    del bl;r.checkpoint();log(f'Global gates {c} {i+1}/8; {r.count} forwards')
  savej(out/'reconstruction_checks.json',checks)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:gate.close();mask.close();r.close()
