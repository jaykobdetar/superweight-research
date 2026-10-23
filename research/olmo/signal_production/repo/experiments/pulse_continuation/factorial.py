"""Cross-corpus Q/K/V factorial and source-normalization interventions."""
import hashlib,itertools,json,math,resource,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.olmo.modeling_olmo import rotate_half
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'pulse_causal_tranche'))
from run_tranche import sha,savej,tokenhash,log
from single_projection_core import hash_model

class Source:
 def __init__(self,model):
  self.module=model.model.layers[1].mlp.down_proj;self.state='clean';self.position=0
  self.weight=self.module.weight.detach().clone();self.weight[1764,1710]=0
  self.handle=self.module.register_forward_hook(self.hook)
 def hook(self,m,inputs,o):
  if self.state=='clean':return o
  z=o.clone();v=F.linear(inputs[0],self.weight,m.bias);z[:,self.position,1764]=v[:,self.position,1764]
  return z
 def close(self):self.handle.remove()

class Local:
 def __init__(self,model):
  self.model=model;self.patch={};self.cache={};self.capture=False;self.handles=[]
  layer=model.model.layers[2];a=layer.self_attn
  self.handles.append(layer.input_layernorm.register_forward_hook(self.norm))
  self.handles.append(a.register_forward_pre_hook(self.attn,with_kwargs=True))
  for k in ('q','k','v'):self.handles.append(getattr(a,k+'_proj').register_forward_hook(lambda m,i,o,k=k:self.proj(k,o)))
  self.handles.append(a.o_proj.register_forward_pre_hook(self.context))
 def reset(self,patch=None,capture=False):self.patch=patch or {};self.capture=capture;self.cache={}
 def norm(self,m,inputs,o):
  p=self.patch;x=inputs[0]
  if p.get('type')=='norm':
   o=o.clone();d=p['donor'];mode=p['mode'];other=d['norm'].to(o.device)
   if mode=='coordinate':o[:,0,1764]=other[:,0,1764]
   elif mode=='complement':
    c=o[:,0,1764].clone();o[:,0]=other[:,0];o[:,0,1764]=c
   elif mode=='full':o[:,0]=other[:,0]
   elif mode=='donor_scale':
    xx=x[:,0].float();sigma=d['input'][:,0].float().var(-1,unbiased=False).add(1e-5).sqrt().to(o.device)
    o[:,0]=((xx-xx.mean(-1,keepdim=True))/sigma[:,None]).to(o.dtype)
   else:raise ValueError(mode)
  if self.capture:self.cache['input']=x[:,:1].detach().cpu().clone();self.cache['norm']=o[:,:1].detach().cpu().clone()
  return o
 def attn(self,m,args,kwargs):
  self.rope=kwargs['position_embeddings'];return None
 def proj(self,k,o):
  p=self.patch
  if p.get('type')=='qkv' and k in p['kinds']:
   o=o.clone();pos=slice(None) if k=='q' else slice(0,1);h=p['head'];sl=slice(h*128,(h+1)*128)
   o[...,pos,sl]=p['donor'][k][...,pos,sl].to(o.device)
  if self.capture:self.cache[k]=o.detach().cpu().clone()
  return o
 def context(self,m,inputs):
  if self.capture:
   self.cache['context']=inputs[0].detach().cpu().clone()
   self.cache['cos']=self.rope[0].detach().cpu().clone();self.cache['sin']=self.rope[1].detach().cpu().clone()
  return None
 def close(self):
  for h in self.handles:h.remove()

def diagnostics(cache,head=15):
 sl=slice(head*128,(head+1)*128);co=cache['cos'][0];si=cache['sin'][0]
 q=cache['q'][0,:,sl].clamp(-8,8);k=cache['k'][0,:,sl].clamp(-8,8);v=cache['v'][0,:,sl].clamp(-8,8)
 q=(q*co+rotate_half(q)*si).float();k=(k*co+rotate_half(k)*si).float();v=v.float()
 score=(q@k.T)/(128**.5);score.masked_fill_(torch.ones_like(score,dtype=torch.bool).triu(1),-torch.inf)
 prob=score.softmax(-1);expected=prob@v;actual=cache['context'][0,:,sl].float()
 err=float((expected-actual).norm()/actual.norm().clamp_min(1e-12))
 # Conditional mixture over nonzero keys is defined only for queries>=1.
 u=score[1:,1:].softmax(-1)@v[1:]
 result={'head':head,'reconstruction_relative_error':err,'key0_mass_mean':float(prob[1:,0].mean()),
  'key0_mass_median':float(prob[1:,0].median()),'key0_mass_late':float(prob[min(128,len(prob)-1):,0].mean()),
  'value0_norm':float(v[0].norm()),'key0_norm':float(k[0].norm()),'other_values_rms_norm':float(v[1:].square().sum(-1).mean().sqrt()),
  'conditional_mixture_rms_norm':float(u.square().sum(-1).mean().sqrt()),
  'source_residual_std':float(cache['input'][0,0].float().std(unbiased=False)),'normalized_source_coordinate':float(cache['norm'][0,0,1764])}
 assert err<.002,err
 return result,{'attention0':prob[:,0],'conditional_mixture':u,'value0':v[0],'actual_context':actual}

class Run:
 def __init__(self,model,out):self.model=model;self.out=out;self.source=Source(model);self.local=Local(model);self.count=0;self.raw={};self.conditions={}
 def evaluate(self,x,state='clean',patch=None,capture=False,source_position=0,baseline=None,keep=False):
  self.source.state=state;self.source.position=source_position;self.local.reset(patch,capture)
  with torch.inference_mode():
   ids=x[None].cuda();logits=self.model(input_ids=ids,use_cache=False).logits[0,:-1]
   losses=[];kls=[]
   for s in range(0,len(x)-1,128):
    e=min(s+128,len(x)-1);lp=logits[s:e].float().log_softmax(-1)
    losses.append(-lp.gather(-1,ids[0,s+1:e+1,None]).squeeze(-1).cpu())
    if baseline is None:kls.append(torch.zeros(e-s))
    else:
     lb=baseline[s:e].float().log_softmax(-1);kls.append((lb.exp()*(lb-lp)).sum(-1).cpu())
   z={'loss':torch.cat(losses),'pred':logits.argmax(-1).cpu(),'kl':torch.cat(kls)}
   assert torch.isfinite(z['loss']).all() and torch.isfinite(z['kl']).all()
  self.count+=1
  return z,self.local.cache,logits if keep else None
 def record(self,group,trial,i,x,z,b,spec,score_start=0):
  key=group+'__'+trial;self.conditions.setdefault(key,spec);z={k:v[score_start:] for k,v in z.items()};b={k:v[score_start:] for k,v in b.items()}
  self.raw.setdefault(key,[]).append({'window':i,'token_sha256':tokenhash(x),**z})
  rec={'group':group,'trial':trial,'window':i,'token_sha256':tokenhash(x),'input_tokens':len(x),'score_start':score_start,'scored_tokens':len(z['loss']),
       'nll':float(z['loss'].double().mean()),'baseline_nll':float(b['loss'].double().mean()),'agreement':float(z['pred'].eq(b['pred']).double().mean()),'kl_clean_to_condition':float(z['kl'].double().mean())}
  with (self.out/'windows.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 def checkpoint(self):
  torch.save(self.raw,self.out/'per_token.pt');savej(self.out/'conditions.json',self.conditions);savej(self.out/'progress.json',{'forwards':self.count,'conditions':len(self.raw),'utc':time.strftime('%FT%TZ',time.gmtime())})
 def close(self):self.source.close();self.local.close()

def prepare(out):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');records={};manifest={}
 for corpus in ('wiki','hume'):
  data=old['corpora'][corpus];starts=[s+2048 for s in old['starts'][corpus]['evaluation'][:8]]
  excluded=[]
  for split in ('calibration','development','evaluation'):
   for s in old['starts'][corpus][split]:excluded.append((s,s+1024))
  for s in starts:
   assert all(s+512<=a or s-512>=b for a,b in excluded),(corpus,s)
  records[corpus]=[]
  for s in starts:records[corpus].append({'natural':data[s:s+512].clone(),'prefix':data[s-512:s].clone(),'start':s})
  manifest[corpus]={'starts':starts,'excluded_tranche1_spans':excluded,'new_span_per_window':'[start-512,start+512); disjoint from tranche1 declared spans'}
 savej(out/'data_manifest.json',manifest);torch.save(records,out/'tokens.pt');return records

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists()
 records=prepare(out);start=time.time();torch.cuda.reset_peak_memory_stats();model=ctx['model'];r=Run(model,out)
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_revision':ctx['model_path'].name,'model_hash_before':ctx['original_hash'],
  'script_sha256':sha(__file__),'worker_sha256':sha(ROOT/'worker.py'),'source':'L1/r1764/c1710 position0 only, except explicit later-boundary negative controls',
  'new_heldout_windows_per_corpus':8,'prefix_lengths':[0,16,128,512],'device':'single RTX3060Ti','dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta)
 subsets=['','q','k','v','qk','qv','kv','qkv']
 try:
  for corpus,windows in records.items():
   for i,w in enumerate(windows):
    x=w['natural'];b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl)
    group='natural_'+corpus
    for trial,z in [('clean',b),('deleted',d)]:r.record(group,trial,i,x,z,b,{'state':trial})
    metrics={};small={}
    for state,cache in [('clean',bc),('deleted',dc)]:
     for h in (15,0):
      mm,ss=diagnostics(cache,h);metrics[f'{state}_H{h}']=mm;small[f'{state}_H{h}']=ss
    # Local decomposition residuals can be verified independently from saved tensors.
    torch.save({'clean':bc,'deleted':dc,'decomposition':small},out/f'local_{corpus}_{i}.pt')
    with (out/'diagnostics.jsonl').open('a') as f:f.write(json.dumps({'corpus':corpus,'window':i,'metrics':metrics})+'\n')
    assert torch.equal(bc['q'][:,1:],dc['q'][:,1:]),'Later query changed before source broadcast'
    for head in (15,0):
     for direction,state,donor,ref in [('damage','clean',dc,b),('rescue','deleted',bc,d)]:
      for kinds in subsets:
       if not kinds:z=ref
       else:z,_,_=r.evaluate(x,state,{'type':'qkv','head':head,'kinds':kinds,'donor':donor},baseline=bl)
       if kinds=='q':assert torch.equal(z['loss'],ref['loss']) and torch.equal(z['pred'],ref['pred'])
       r.record(group,f'H{head}_{kinds or "none"}_{direction}',i,x,z,b,{'head':head,'kinds':kinds,'direction':direction,'q_positions':'all','kv_positions':'zero'})
    for direction,state,donor in [('damage','clean',dc),('rescue','deleted',bc)]:
     for mode in ('coordinate','complement','full','donor_scale'):
      z,cc,_=r.evaluate(x,state,{'type':'norm','mode':mode,'donor':donor},capture=True,baseline=bl)
      r.record(group,'norm_'+mode+'_'+direction,i,x,z,b,{'type':'source_L2_input_layernorm','mode':mode,'direction':direction})
      mm,_=diagnostics(cc,15)
      with (out/'norm_diagnostics.jsonl').open('a') as f:f.write(json.dumps({'corpus':corpus,'window':i,'mode':mode,'direction':direction,**mm})+'\n')
    del bl,bc,dc
    # Prefix panel fixes source identity while varying the available real history.
    suffix=w['natural'].clone();suffix[0]=510
    for p in (0,16,128,512):
     if p:x=torch.cat([w['prefix'][-p:],suffix]);x[0]=510
     else:x=suffix.clone()
     b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl)
     group=f'prefix_{corpus}_{p}'
     for trial,z in [('clean',b),('deleted',d)]:r.record(group,trial,i,x,z,b,{'state':trial,'prefix':p},p)
     for direction,state,donor in [('damage','clean',dc),('rescue','deleted',bc)]:
      for kinds in ('q','k','v','kv','qkv'):
       z,_,_=r.evaluate(x,state,{'type':'qkv','head':15,'kinds':kinds,'donor':donor},baseline=bl)
       r.record(group,f'H15_{kinds}_{direction}',i,x,z,b,{'kinds':kinds,'head':15,'direction':direction,'source_position':0,'prefix':p},p)
     if p:
      z,_,_=r.evaluate(x,'deleted',source_position=p,baseline=bl)
      r.record(group,'delete_later_same_token',i,x,z,b,{'source_position':p,'prefix':p,'original_position0':'intact'},p)
     del bl,bc,dc
    r.checkpoint();log(f'Factorial {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(model),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
