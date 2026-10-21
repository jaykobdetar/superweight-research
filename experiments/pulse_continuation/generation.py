"""Frozen first-token cache substitutions in greedy free-running and matched-history decoding."""
import json,resource,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import sha,savej,log,hash_model,tokenhash
from cache_bypass import clonecache,mix

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'results.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 sets=torch.load(ROOT/'extended_context_run/tokens.pt',weights_only=True,map_location='cpu')
 dictionary=torch.load(ROOT/'extended_context_run/exact_and_rank2_dictionary.pt',weights_only=True,map_location='cpu')
 mean=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki']
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);prompts={c:[x[:128].clone() for x in ws] for c,ws in sets.items()};torch.save(prompts,out/'tokens.pt')
 count=0;raw={};checks=[];length=64
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'prompt_source_sha256':sha(ROOT/'extended_context_run/tokens.pt'),'dictionary_sha256':sha(ROOT/'extended_context_run/exact_and_rank2_dictionary.pt'),
  'calibration_sha256':sha(ROOT/'cache_run/calibration.pt'),'prompt_count_per_corpus':4,'prompt_length':128,'generated_tokens':length,
  'decoding':'deterministic greedy, exactly64 tokens even if EOS occurs; EOS reported; no sampling or repetition penalties',
  'baseline':'ordinary cached full-prefix prefill and sequential decoding; same native cache backend for every condition',
  'free_running':'each condition consumes its own greedy predictions',
  'matched_history':'each condition consumes the clean greedy trajectory; KL and prediction agreement separate distribution error from divergence',
  'reference_nll':'negative log probability of clean greedy next token, not ground-truth corpus NLL or quality',
  'intervention':'only the initial first-token cache; all subsequently generated tokens computed normally, no source/norm hooks',
  'evaluation':'same fixed four contexts/corpus as extended_context; no calibration, choice or fitting on generated outcomes',
  'dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def forward(ids,cache=None,cp=None):
  nonlocal count
  kw={'input_ids':ids,'use_cache':True}
  if cache is not None:kw.update(past_key_values=cache,cache_position=cp)
  z=model(**kw);count+=1
  return z.past_key_values,z.logits[0,-1].clone()
 def decode(cache,logits,forced=None,reference_logits=None):
  ids=[];ls=[];kl=[];nll=[];margins=[]
  for step in range(length):
   pred=logits.argmax(-1);ids.append(pred.cpu());top=logits.float().topk(2).values;margins.append((top[0]-top[1]).cpu())
   if reference_logits is None:ls.append(logits.cpu())
   else:
    lp=logits.float().log_softmax(-1);lb=reference_logits[step].cuda().float().log_softmax(-1)
    kl.append((lb.exp()*(lb-lp)).sum().cpu());nll.append((-lp[forced[step]]).cpu())
   if step+1<length:
    take=pred if forced is None else forced[step].cuda()
    cache,logits=forward(take.reshape(1,1),cache,torch.tensor([128+step],device='cuda'))
  assert cache.get_seq_length()==128+length-1
  return {'pred':torch.stack(ids),'margin':torch.stack(margins),**({'logits':torch.stack(ls)} if reference_logits is None else {'kl':torch.stack(kl),'reference_nll':torch.stack(nll)})}
 try:
  with torch.inference_mode():
   for c,ws in prompts.items():
    for i,x in enumerate(ws):
     cache,logits=forward(x[None].cuda());full_first=tuple(tuple(t[:,:,:1].detach().cpu().clone() for t in p) for p in cache.to_legacy_cache())
     clean=decode(cache,logits);ec=dictionary[int(x[0])]
     variants={'exact_full_donor':full_first,'exact_single_token':ec['exact'],'wiki_mean_all':mean,'wiki_mean_late':mix(ec['exact'],mean),'rank2_late':ec['rank2'],
      'zero_cache':tuple(tuple(torch.zeros_like(t) for t in p) for p in ec['exact'])}
     raw[c,i]={'prompt':x,'clean':{'pred':clean['pred'],'margin':clean['margin']},'variants':{}}
     base_lp=clean['logits'].float().log_softmax(-1);base_nll=-base_lp.gather(1,clean['pred'][:,None]).squeeze(1)
     raw[c,i]['clean']['reference_nll']=base_nll
     for name,initial in variants.items():
      cc,first_logits=forward(x[None,1:].cuda(),clonecache(initial),torch.arange(1,128,device='cuda'));assert cc.get_seq_length()==128
      # Separate copies keep the two trajectories independent; both begin with the exact same condition prefix state.
      forced_cache=clonecache(tuple(tuple(t.detach().clone() for t in p) for p in cc.to_legacy_cache()))
      free=decode(cc,first_logits);forced=decode(forced_cache,first_logits,clean['pred'],clean['logits'])
      del free['logits']
      different=free['pred'].ne(clean['pred']);where=different.nonzero().flatten();first=int(where[0]) if len(where) else length
      forced_different=forced['pred'].ne(clean['pred']);fw=forced_different.nonzero().flatten();forced_first=int(fw[0]) if len(fw) else length
      assert first==forced_first,'Free and matched-history first divergence must coincide before histories differ'
      rec={'corpus':c,'window':i,'variant':name,'prompt_sha256':tokenhash(x),'generated_tokens':length,'free_agreement':float((~different).double().mean()),
       'identical_sequence':not bool(different.any()),'first_divergence_zero_based':first if first<length else None,'matching_prefix_length':first,
       'matched_history_agreement':float((~forced_different).double().mean()),'matched_history_kl':float(forced['kl'].double().mean()),
       'reference_delta_nll':float((forced['reference_nll']-base_nll).double().mean()),
       'clean_eos_count':int(clean['pred'].eq(tok.eos_token_id).sum()),'condition_eos_count':int(free['pred'].eq(tok.eos_token_id).sum())}
      with (out/'results.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
      raw[c,i]['variants'][name]={'free':free,'matched_history':forced}
      checks.append({'corpus':c,'window':i,'variant':name,'first_divergence_coincides':True,'final_cache_length':128+length-1})
     with (out/'continuations.jsonl').open('a') as f:
      f.write(json.dumps({'corpus':c,'window':i,'prompt':tok.decode(x),'clean':tok.decode(clean['pred']),
       'variants':{name:tok.decode(z['free']['pred']) for name,z in raw[c,i]['variants'].items()}},ensure_ascii=False)+'\n')
     torch.save(raw,out/'per_step.pt');savej(out/'execution_checks.json',checks);del clean,cache,logits,base_lp,forced_cache,cc
     log(f'Greedy generation {c} {i+1}/4; {count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta)
 finally:pass
