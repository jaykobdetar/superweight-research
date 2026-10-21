"""Separate source-coordinate content from normalization scale across layers."""
import resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model

class Norms:
 def __init__(self,model):
  self.handles=[];self.reset()
  for l,layer in enumerate(model.model.layers):self.handles.append(layer.input_layernorm.register_forward_hook(lambda m,i,o,l=l:self.hook(l,i,o)))
 def reset(self,patch=None,capture=False):self.patch=patch or {};self.capture=capture;self.cache={}
 def hook(self,l,inputs,o):
  x=inputs[0];p=self.patch
  if l in p.get('layers',[]):
   o=o.clone();d=p['donor'][l];mode=p['mode'];don=d['norm'].to(o.device)
   if mode=='full':o[:,0]=don[:,0]
   elif mode=='coordinate':o[:,0,1764]=don[:,0,1764]
   elif mode=='complement':
    old=o[:,0,1764].clone();o[:,0]=don[:,0];o[:,0,1764]=old
   elif mode=='scale':
    xx=x[:,0].float();o[:,0]=((xx-xx.mean(-1,keepdim=True))/d['sigma'].to(o.device)[:,None]).to(o.dtype)
   else:raise ValueError(mode)
  if self.capture:
   xx=x[:,:1].detach().float().cpu();self.cache[l]={'input':xx,'norm':o[:,:1].detach().cpu().clone(),'sigma':xx[:,0].var(-1,unbiased=False).add(1e-5).sqrt()}
  return o
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=torch.load(ROOT/'gate_run/tokens.pt',weights_only=True,map_location='cpu')
 r=Run(ctx['model'],out);r.local.close();r.local=Norms(ctx['model']);start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'tokens_sha256':sha(ROOT/'gate_run/tokens.pt'),'evaluation':'same gate-batch evaluation windows, exploratory cross-layer mechanism followup',
  'calibration':'8 WikiText and 8 Hume development windows; mean normalized vector and arithmetic mean source sigma, no heldout fitting',
  'source':'L1/r1764/c1710 at0 only','scale':'recipient centered source input divided by donor sqrt(population variance+1e-5)',
  'positions':'source0 at attention input layernorm only; final output and post-attention MLP norms untouched','score_start':1,
  'seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 try:
  cals={}
  for corpus in ('wiki','hume'):
   cs=[]
   for x in sets[corpus]['calibration']:
    _,c,_=r.evaluate(x,capture=True);cs.append(c)
   cals[corpus]={l:{'norm':torch.stack([c[l]['norm'].float() for c in cs]).mean(0).half(),'sigma':torch.stack([c[l]['sigma'] for c in cs]).mean(0)} for l in range(16)}
  torch.save(cals,out/'calibration.pt');log('Source normalization calibration frozen')
  for corpus,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl);group='natural_'+corpus
    torch.save({'clean':bc,'deleted':dc},out/f'norms_{corpus}_{i}.pt')
    for name,z in [('clean',b),('deleted',d)]:r.record(group,name,i,x,z,b,{'state':name},1)
    for state,donor,direction,ref in [('clean',dc,'damage',d),('deleted',bc,'rescue',b)]:
     for layername,layers in [('L2',[2]),('all_2_15',list(range(2,16)))]:
      for mode in ('coordinate','complement','scale','full'):
       z,_,_=r.evaluate(x,state,{'layers':layers,'donor':donor,'mode':mode},baseline=bl)
       r.record(group,f'{layername}_{mode}_{direction}',i,x,z,b,{'layers':layers,'mode':mode,'direction':direction},1)
       if layername=='all_2_15' and mode=='full':assert torch.equal(z['loss'][1:],ref['loss'][1:]) and torch.equal(z['pred'][1:],ref['pred'][1:])
     for cal in ('wiki','hume'):
      for mode in ('coordinate','complement','scale','full'):
       z,_,_=r.evaluate(x,state,{'layers':list(range(2,16)),'donor':cals[cal],'mode':mode},baseline=bl)
       r.record(group,f'constant_{cal}_{mode}_{state}',i,x,z,b,{'layers':list(range(2,16)),'mode':mode,'state':state,'calibration':cal},1)
    del bl,bc,dc;r.checkpoint();log(f'Normalization {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
