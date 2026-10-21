"""Matched suffixes and fixed source identities separate corpus from source-cache effects."""
import resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from normalization import Norms

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=torch.load(ROOT/'template_run/tokens.pt',weights_only=True,map_location='cpu')
 template=torch.load(ROOT/'template_run/templates.pt',weights_only=True,map_location='cpu')['templates'];old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
 ids={'natural':None,'The':510,'eos':old['start_tokens']['eos'],'period':15};r=Run(ctx['model'],out);r.local.close();r.local=Paths(ctx['model']);norm=Norms(ctx['model'])
 reference={};checks=[];start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'tokens_sha256':sha(ROOT/'template_run/tokens.pt'),'templates_sha256':sha(ROOT/'template_run/templates.pt'),
  'design':'natural/The/EOS/period first-token ID crossed with three corpora; suffix IDs identical within window;512tokens',
  'source':'L1/r1764/c1710 position0 only','score_start':1,'evaluation':'template confirmation windows reused for an exploratory source-identity interaction followup; no fitting',
  'identities':ids,'seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 try:
  for corpus,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity,tid in ids.items():
     x=xx.clone()
     if tid is not None:x[0]=tid
     b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl);group=f'{corpus}_{identity}'
     if tid is not None:
      for state,c in [('clean',bc),('deleted',dc)]:
       simple={l:{k:c[l][k] for k in ('k','v')} for l in range(16)};key=(identity,state)
       if key not in reference:reference[key]=simple
       exact=all(torch.equal(reference[key][l][k],simple[l][k]) for l in range(16) for k in ('k','v'))
       assert exact,'Same source identity cache changed with suffix';checks.append({'corpus':corpus,'window':i,'identity':identity,'state':state,'all_source_kv_exact':exact})
     for name,z in [('clean',b),('deleted',d)]:r.record(group,name,i,x,z,b,{'state':name,'source_identity':identity},1)
     for state,donor,direction,ref in [('clean',dc,'damage',d),('deleted',bc,'rescue',b)]:
      for name,layers in [('L2_all',[2]),('early_2_5',list(range(2,6))),('late_3_15',list(range(3,16))),('all_2_15',list(range(2,16)))]:
       z,_,_=r.evaluate(x,state,{'type':'cut','layers':layers,'donor':donor},baseline=bl)
       r.record(group,name+'_'+direction,i,x,z,b,{'layers':layers,'direction':direction,'source_identity':identity},1)
       if name=='all_2_15':assert torch.equal(z['loss'][1:],ref['loss'][1:])
      for name in ('ideal','residual_rank2','wiki_full_mean'):
       norm.reset({'layers':list(range(2,16)),'donor':template[name],'mode':'full'})
       try:z,_,_=r.evaluate(x,state,baseline=bl)
       finally:norm.reset()
       r.record(group,name+'_'+state,i,x,z,b,{'template':name,'state':state,'source_identity':identity},1)
     del bl,bc,dc
    r.checkpoint();log(f'Identity {corpus} {i+1}/8; {r.count} forwards')
  torch.save(reference,out/'fixed_identity_source_caches.pt');savej(out/'source_identity_checks.json',checks)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:norm.close();r.close()
