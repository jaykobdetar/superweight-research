"""Fresh-window tests of analytic source-normalization templates and learned residuals."""
import json,math,resource,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from normalization import Norms

def prepare(ctx,out):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text());fm=json.loads((ROOT/'factorial_run/data_manifest.json').read_text())
 sets={};meta={}
 for c in ('wiki','hume'):
  starts=[s+1024 for s in old['starts'][c]['evaluation'][:8]];excluded=[]
  for ss in old['starts'][c].values():excluded.extend((s,s+1024) for s in ss)
  excluded.extend((s-512,s+512) for s in fm[c]['starts']);excluded.extend((s,s+1024) for s in gm[c]['evaluation_starts'])
  for s in starts:assert all(s+512<=a or s>=b for a,b in excluded),(c,s)
  sets[c]=[old['corpora'][c][s:s+512].clone() for s in starts];meta[c]={'starts':starts,'excluded_spans':excluded}
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);ids=torch.tensor(tok.encode(Path(gm['plato']['path']).read_text(),add_special_tokens=False))
 starts=[s+1024 for s in gm['plato']['evaluation_starts']];sets['plato']=[ids[s:s+512].clone() for s in starts];meta['plato']={'starts':starts,'source_sha256':gm['plato']['sha256'],'excluded_spans':[(s,s+1024) for s in gm['plato']['evaluation_starts']]}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',meta);return sets

def templates(out):
 cal=torch.load(ROOT/'normalization_run/calibration.pt',weights_only=True,map_location='cpu');m=torch.stack([cal['wiki'][l]['norm'][0,0].float() for l in range(2,16)])
 ideal=torch.ones(2048)/math.sqrt(2047);ideal[1764]=-math.sqrt(2047)
 base=m[:,1764,None]/ideal[1764]*ideal[None];res=m-base
 u,s,v=torch.linalg.svd(res,full_matrices=False);center=res-res.mean(0);uc,sc,vc=torch.linalg.svd(center,full_matrices=False)
 allv={'ideal':ideal.expand(14,-1).clone(),'coordinate_matched_axis':base,'residual_only':res,'wiki_full_mean':m,
       'hume_full_mean':torch.stack([cal['hume'][l]['norm'][0,0].float() for l in range(2,16)]),'shared_residual':base+res.mean(0)}
 for rank in (1,2,4,8):allv[f'residual_rank{rank}']=base+(u[:,:rank]*s[:rank])@v[:rank]
 for seed in (20261021,20261022):
  gen=torch.Generator().manual_seed(seed);z=torch.randn((14,2048),generator=gen);z[:,1764]=0
  mask=torch.ones(2048,dtype=torch.bool);mask[1764]=False;z[:,mask]-=z[:,mask].mean(1,keepdim=True)
  z*=res.norm(dim=1,keepdim=True)/z.norm(dim=1,keepdim=True);allv[f'random_residual_{seed}']=base+z
 wrong=torch.ones(2048)/math.sqrt(2047);wrong[1763]=-math.sqrt(2047);allv['wrong_axis']=wrong.expand(14,-1).clone()
 ans={name:{l:{'norm':x[l-2][None,None].half()} for l in range(2,16)} for name,x in allv.items()}
 torch.save({'templates':ans,'mean_vectors':m,'analytic_axis':ideal,'coordinate_matched_axis':base,'residual':res,'uncentered_svd':(u,s,v),'centered_across_layers_svd':(uc,sc,vc)},out/'templates.pt')
 savej(out/'template_fit.json',{'calibration':'WikiText8 development windows from normalization_run; Hume full mean control separately fitted',
  'calibration_sha256':sha(ROOT/'normalization_run/calibration.pt'),'analytic_axis':'LN of a dominating negative coordinate1764: coord=-sqrt2047, other=1/sqrt2047',
  'axis_match':'per-layer calibration mean coordinate1764; residual subtracts that zero-sum axis',
  'svd':'uncentered14x2048 matrix of learned residuals across layers2..15; not downstream token covariance',
  'uncentered_energy':[float(s[:r].square().sum()/s.square().sum()) for r in (1,2,4,8)],
  'centered_across_layer_energy':[float(sc[:r].square().sum()/sc.square().sum()) for r in (1,2,4,8)],
  'random_controls':'two fixed seeds; coordinate1764 zero, other-coordinate mean zero, layerwise norm matched to learned residual',
  'templates':list(ans),'calibration_before_evaluation':True})
 return ans

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=prepare(ctx,out);ts=templates(out)
 r=Run(ctx['model'],out);r.local.close();r.local=Norms(ctx['model']);start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'evaluation':'8 unused512-token windows each WikiText,Hume,Plato; rank choices1,2,4,8 frozen before outcomes',
  'source':'L1/r1764/c1710 at0 only','replacement':'source0 attention-input normalized vectors at layers2..15',
  'score_start':1,'seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 try:
  for corpus,windows in sets.items():
   for i,x in enumerate(windows):
    b,_,bl=r.evaluate(x,keep=True);d,_,_=r.evaluate(x,'deleted',baseline=bl);group='natural_'+corpus
    for name,z in [('clean',b),('deleted',d)]:r.record(group,name,i,x,z,b,{'state':name},1)
    for name,donor in ts.items():
     prev=None
     for state in ('clean','deleted'):
      z,_,_=r.evaluate(x,state,{'layers':list(range(2,16)),'donor':donor,'mode':'full'},baseline=bl)
      if prev is not None:assert torch.equal(z['loss'][1:],prev['loss'][1:]) and torch.equal(z['pred'][1:],prev['pred'][1:])
      prev=z;r.record(group,name+'_'+state,i,x,z,b,{'template':name,'state':state,'layers':list(range(2,16))},1)
    del bl;r.checkpoint();log(f'Template {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
