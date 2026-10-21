"""Actual first-token bypass using cached K/V; exact-token and mean-cache controls."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,DynamicCache
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths

def packed(c):
 return tuple(tuple(c[l][k].clamp(-8,8).reshape(1,1,16,128).transpose(1,2).contiguous() for k in ('k','v')) for l in range(16))
def clonecache(c):return DynamicCache(tuple(tuple(t.cuda().clone() for t in pair) for pair in c))
def mean_cache(cs):return tuple(tuple(torch.stack([c[l][k].float() for c in cs]).mean(0).half() for k in range(2)) for l in range(16))
def mix(early,late):return early[:2]+late[2:]
def score(logits,x,baseline):
 losses=[];kls=[]
 for s in range(0,len(x)-1,128):
  e=min(s+128,len(x)-1);lp=logits[s:e].float().log_softmax(-1);lb=baseline[s:e].float().log_softmax(-1)
  losses.append(-lp.gather(-1,x[s+1:e+1,None].cuda()).squeeze(-1).cpu());kls.append((lb.exp()*(lb-lp)).sum(-1).cpu())
 return {'loss':torch.cat(losses),'pred':logits.argmax(-1).cpu(),'kl':torch.cat(kls)}

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 data=torch.load(ROOT/'gate_run/tokens.pt',weights_only=True,map_location='cpu');gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text())
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');texts={k:v for k,v in old['corpora'].items()}
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True)
 texts['plato']=torch.tensor(tok.encode(Path(gm['plato']['path']).read_text(),add_special_tokens=False))
 sets={c:[texts[c][s:s+1024].clone() for s in gm[c]['evaluation_starts']] for c in data}
 for c in sets:
  for i,x in enumerate(sets[c]):assert torch.equal(x[:512],data[c]['evaluation'][i]) and len(x)==1024
 torch.save(sets,out/'tokens.pt')
 r=Run(model,out);r.local.close();r.local=Paths(model);dictionary={};cal={};checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'calibration':'8 WikiText and 8 Hume development windows from prior gate batch; means of clipped native first-token cache, fixed before evaluation',
  'evaluation':'8 windows/corpus; natural and fixed The source identity; matched cropped128/512/1024 token inputs',
  'actual_bypass':'model receives tokens1: only, explicit cache_position1:, stored single-token KV states; no source hook deletion',
  'first_prediction':'cached exact-identity single-token logits reused in every bypass; later-scored primary panel excludes it, full panel includes it',
  'dictionary':'compute once per distinct first-token ID, independently of its suffix; this is ordinary deterministic prefix memoization, not fitted evaluation data',
  'cache_storage_bytes':16*2*16*128*2,'seed':20261021,'dtype':'float16','cpu_threads':4}
 savej(out/'manifest.json',meta)
 def exact(token):
  if token not in dictionary:
   r.source.state='clean';r.local.reset()
   with torch.inference_mode():
    a=model(input_ids=torch.tensor([[token]],device='cuda'),use_cache=True)
    dictionary[token]={'kv':tuple(tuple(t.detach().cpu().clone() for t in pair) for pair in a.past_key_values.to_legacy_cache()),'logit':a.logits[0,0].detach().cpu().clone()}
   r.count+=1
  return dictionary[token]
 try:
  for corpus in ('wiki','hume'):
   cs=[]
   for x in data[corpus]['calibration']:
    _,c,_=r.evaluate(x,capture=True);cs.append(packed(c))
   cal[corpus]=mean_cache(cs)
  torch.save(cal,out/'calibration.pt');exact(510);log('Actual-cache calibration frozen')
  for corpus,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity in ('natural','The'):
     for n in (128,512,1024):
      x=xx[:n].clone()
      if identity=='The':x[0]=510
      b,bc,bl=r.evaluate(x,capture=True,keep=True);group=f'cache_{corpus}_{identity}_{n}'
      token=int(x[0]);ec=exact(token);oracle=packed(bc)
      variants={'exact_full_donor':oracle,'exact_single_token':ec['kv'],'wiki_mean_all':cal['wiki'],'hume_mean_all':cal['hume'],
       'wiki_mean_late':mix(ec['kv'],cal['wiki']),'hume_mean_late':mix(ec['kv'],cal['hume']),
       'fixed_The':exact(510)['kv'],'zero_cache':tuple(tuple(torch.zeros_like(t) for t in pair) for pair in ec['kv'])}
      for full,ss in [(False,1),(True,0)]:r.record(group+('_full' if full else ''),'clean',i,x,b,b,{'identity':identity,'length':n,'full_score':full},ss)
      for name,c in variants.items():
       r.source.state='clean';r.local.reset()
       with torch.inference_mode():
        cache=clonecache(c);cp=torch.arange(1,n,device='cuda')
        result=model(input_ids=x[None,1:].cuda(),past_key_values=cache,cache_position=cp,use_cache=True)
        assert result.logits.shape[1]==n-1 and cache.get_seq_length()==n
        logits=torch.cat([ec['logit'][None].cuda(),result.logits[0,:-1]],0);z=score(logits,x,bl)
       r.count+=1
       for full,ss in [(False,1),(True,0)]:r.record(group+('_full' if full else ''),name,i,x,z,b,{'cache':name,'identity':identity,'length':n,'full_score':full},ss)
       checks.append({'corpus':corpus,'window':i,'identity':identity,'length':n,'cache':name,'actual_input_tokens':n-1,'final_cache_length':cache.get_seq_length()})
       del result,logits,cache
      del bl,bc
    r.checkpoint();log(f'Cache bypass {corpus} {i+1}/8; {r.count} forwards')
  torch.save(dictionary,out/'exact_token_dictionary.pt');savej(out/'execution_checks.json',checks)
  # End-to-end wall-clock forward timing; no losses/KL, but include cache copying/construction.
  # The one-time dictionary and calibration work is excluded and explicitly accounted above.
  x=sets['wiki'][0][:512].cuda()[None];r.source.state='clean';r.local.reset();basecache=exact(int(x[0,0]))['kv']
  modes=['full_no_cache','full_cache','bypass_exact','bypass_mean'];times=[];rng=np.random.default_rng(20261021)
  for iteration in range(27):
   for mode in rng.permutation(modes):
    torch.cuda.synchronize();t=time.perf_counter()
    with torch.inference_mode():
     if mode.startswith('full'):ans=model(input_ids=x,use_cache=mode=='full_cache')
     else:ans=model(input_ids=x[:,1:],past_key_values=clonecache(basecache if mode=='bypass_exact' else cal['wiki']),cache_position=torch.arange(1,512,device='cuda'),use_cache=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t;del ans;r.count+=1
    times.append({'iteration':iteration,'warmup':iteration<3,'mode':str(mode),'seconds':elapsed})
  savej(out/'timings.json',times)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   distinct_cached_token_identities=len(dictionary),model_hash_after=hash_model(model),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
