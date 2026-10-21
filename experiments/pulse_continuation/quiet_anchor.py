"""Test a global zero-value routing anchor against content-bearing constant source caches."""
import resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths

class SourceMask:
 def __init__(self,model):
  self.handles=[];self.layers=[];self.explicit=True;self.masks={}
  for l,layer in enumerate(model.model.layers):self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k,l=l:self.hook(l,a,k),with_kwargs=True))
 def hook(self,l,args,kwargs):
  if not self.explicit:return None
  x=kwargs.get('hidden_states',args[0] if args else None);n=x.shape[1]
  if n not in self.masks:
   m=torch.zeros((n,n),device=x.device,dtype=x.dtype);m.masked_fill_(torch.ones((n,n),device=x.device,dtype=torch.bool).triu(1),-torch.inf);self.masks[n]=m[None,None]
  mask=self.masks[n]
  if l in self.layers:mask=mask.clone();mask[:,:,1:,0]=-torch.inf
  k=dict(kwargs);k['attention_mask']=mask;return args,k
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 sets=torch.load(ROOT/'residual_paths_run/tokens.pt',weights_only=True,map_location='cpu');torch.save({c:d['evaluation'] for c,d in sets.items()},out/'tokens.pt')
 packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki']
 mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 zero={l:{k:torch.zeros_like(v) for k,v in d.items()} for l,d in mean.items()}
 r=Run(model,out);r.local.close();r.local=Paths(model);mask=SourceMask(model);checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'tokens_sha256':sha(ROOT/'residual_paths_run/tokens.pt'),'calibration_sha256':sha(ROOT/'cache_run/calibration.pt'),
  'evaluation':'same8 heldout512-token windows/corpus from residual-path batch, frozen Wiki means, no new fit or head selection',
  'source_cache_modes':'all heads at layers2..15, key native/mean/zero crossed with value native/mean/zero except native/native baseline',
  'mean_definition':'arithmetic mean of clipped native first-token caches from prior8 Wiki development contexts; native projection hook followed by normal clip',
  'source_removal':'mask source0 only for queries1+ at specified layers; source0 keeps its self attention, early layers otherwise unchanged',
  'explicit_control':'all conditions share explicit native causal4D mask; ordinary automatic-mask clean/deleted checked separately',
  'score_start':1,'source':'L1/r1764/c1710 contribution at0 only','dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta)
 try:
  for c,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    mask.explicit=True;mask.layers=[];b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl);group='quiet_'+c
    torch.save({'clean':bc,'deleted':dc},out/f'source_cache_{c}_{i}.pt')
    for name,z in [('clean',b),('deleted',d)]:r.record(group,name,i,x,z,b,{'state':name,'mask':'explicit'},1)
    mask.explicit=False
    for state,ref in [('clean',b),('deleted',d)]:
     z,_,_=r.evaluate(x,state,baseline=bl);assert all(torch.equal(z[k][1:],ref[k][1:]) for k in ('loss','pred','kl'))
     r.record(group,'native_mask_'+state,i,x,z,b,{'state':state,'mask':'native'},1)
    mask.explicit=True
    for km in ('native','mean','zero'):
     for vm in ('native','mean','zero'):
      if km==vm=='native':continue
      kinds=('k' if km!='native' else '')+('v' if vm!='native' else '');donor={l:{} for l in range(16)}
      for k,mode in [('k',km),('v',vm)]:
       if mode!='native':
        for l in range(16):donor[l][k]=(mean if mode=='mean' else zero)[l][k]
      prior=None
      for state in ('clean','deleted'):
       z,_,_=r.evaluate(x,state,{'type':'cut','layers':list(range(2,16)),'kinds':kinds,'donor':donor},baseline=bl)
       if prior is not None and kinds=='kv':assert all(torch.equal(z[k][1:],prior[k][1:]) for k in ('loss','pred','kl'))
       prior=z;r.record(group,f'K_{km}_V_{vm}_{state}',i,x,z,b,{'state':state,'keys':km,'values':vm,'layers':list(range(2,16))},1)
    for label,layers in [('L2',[2]),('late',list(range(3,16))),('all',list(range(2,16)))]:
     mask.layers=layers;prior=None
     for state in ('clean','deleted'):
      z,_,_=r.evaluate(x,state,baseline=bl)
      if prior is not None and label=='all':assert all(torch.equal(z[k][1:],prior[k][1:]) for k in ('loss','pred','kl'))
      prior=z;r.record(group,'remove_source_'+label+'_'+state,i,x,z,b,{'state':state,'source0_masked_at':layers},1)
    mask.layers=[];checks.append({'corpus':c,'window':i,'native_explicit_exact':True,'fixed_KV_state_invariance_exact':True,'all_layer_source_removal_state_invariance_exact':True})
    del bl,bc,dc;r.checkpoint();log(f'Quiet anchor {c} {i+1}/8; {r.count} forwards')
  savej(out/'execution_checks.json',checks)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:mask.close();r.close()
