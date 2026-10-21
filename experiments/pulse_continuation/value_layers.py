"""Localize value-vector dependence while clamping the complete first-token key interface."""
import resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 sets=torch.load(ROOT/'residual_paths_run/tokens.pt',weights_only=True,map_location='cpu');packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki']
 mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 layers=list(range(2,16));variants={'mean_values':layers,'zero_values':[]}
 for l in layers:variants[f'remove_L{l}']=[k for k in layers if k!=l];variants[f'restore_L{l}']=[l]
 for label,ss in [('early',list(range(2,6))),('middle',list(range(6,12))),('late',list(range(12,16)))]:
  variants['restore_'+label]=ss;variants['remove_'+label]=[l for l in layers if l not in ss]
 for end in (3,5,8,11):variants[f'restore_prefix_through{end}']=list(range(2,end+1))
 for first in (3,6,9,12):variants[f'restore_suffix_from{first}']=list(range(first,16))
 randoms={}
 for seed in (20261021,20261022):
  gen=torch.Generator().manual_seed(seed);randoms[seed]={}
  for l in layers:
   v=mean[l]['v'].float().reshape(16,128);z=torch.randn(v.shape,generator=gen);z*=v.norm(dim=-1,keepdim=True)/z.norm(dim=-1,keepdim=True)
   randoms[seed][l]=z.reshape(1,1,2048).half();assert float(z.abs().max())<8
 torch.save({'mean':mean,'random_values':randoms},out/'frozen_interfaces.pt');torch.save({c:d['evaluation'] for c,d in sets.items()},out/'tokens.pt')
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'tokens_sha256':sha(ROOT/'residual_paths_run/tokens.pt'),'calibration_sha256':sha(ROOT/'cache_run/calibration.pt'),
  'evaluation':'same8 heldout512-token windows/corpus as quiet-anchor; all layer sets prespecified, no selection or fitting on outcomes',
  'worlds':'all K0 fixed to Wiki means at layers2..15; all V0 fixed to mean or zero per selected layer; future source K/V cannot drift due earlier V changes',
  'variants':variants,'random_controls':'two fixed seeds; each layer/head V vector norm matched to mean, random direction; mean keys retained',
  'score_start':1,'seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 r=Run(model,out);r.local.close();r.local=Paths(model)
 try:
  for c,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    b,_,bl=r.evaluate(x,keep=True);group='values_'+c;r.record(group,'clean',i,x,b,b,{'reference':'native model'},1)
    memo={}
    for name,retained in variants.items():
     key=tuple(retained)
     if key not in memo:
      donor={l:{'k':mean[l]['k'],'v':mean[l]['v'] if l in retained else torch.zeros_like(mean[l]['v'])} for l in layers}
      z,_,_=r.evaluate(x,patch={'type':'cut','layers':layers,'donor':donor},baseline=bl);memo[key]=z
     z=memo[key];r.record(group,name,i,x,z,b,{'mean_value_layers':retained,'mean_keys_layers':layers},1)
    for seed,vs in randoms.items():
     donor={l:{'k':mean[l]['k'],'v':vs[l]} for l in layers}
     z,_,_=r.evaluate(x,patch={'type':'cut','layers':layers,'donor':donor},baseline=bl)
     r.record(group,f'random_values_{seed}',i,x,z,b,{'random_seed':seed,'mean_keys_layers':layers,'norm_match':'per layer/head'},1)
    del bl;r.checkpoint();log(f'Value layers {c} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
