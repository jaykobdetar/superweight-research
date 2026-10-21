"""Targeted deterministic replay of the exact-control and hybrid-cache first divergences."""
import json,resource,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import sha,savej,log,hash_model
from cache_bypass import clonecache,mix

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'logits.pt').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 raw=torch.load(ROOT/'generation_run/per_step.pt',weights_only=True,map_location='cpu');dictionary=torch.load(ROOT/'extended_context_run/exact_and_rank2_dictionary.pt',weights_only=True,map_location='cpu')
 mean=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki'];tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True)
 cases=[]
 for line in (ROOT/'generation_run/results.jsonl').read_text().splitlines():
  z=json.loads(line)
  if z['variant'] in ('exact_full_donor','wiki_mean_late') and not z['identical_sequence']:cases.append((z['corpus'],z['window'],z['first_divergence_zero_based'],z['variant']))
 count=0;logits={};rows=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'generation_sha256':sha(ROOT/'generation_run/per_step.pt'),'selection':'all observed first divergences in exact-full-donor or hybrid mean cache; exploratory targeted diagnostic, not an independent evaluation',
  'cases':cases,'repeats':2,'history':'identical clean greedy tokens before the selected step','dtype':'float16','cpu_threads':4}
 savej(out/'manifest.json',meta)
 def fw(ids,cache=None,cp=None):
  nonlocal count
  kw={'input_ids':ids,'use_cache':True}
  if cache is not None:kw.update(past_key_values=cache,cache_position=cp)
  z=model(**kw);count+=1;return z.past_key_values,z.logits[0,-1].clone()
 with torch.inference_mode():
  for c,i,step,trigger in cases:
   x=raw[c,i]['prompt'];ec=dictionary[int(x[0])];clean_ids=raw[c,i]['clean']['pred'];cache,lg=fw(x[None].cuda())
   full_first=tuple(tuple(t[:,:,:1].detach().cpu().clone() for t in p) for p in cache.to_legacy_cache())
   variants={'clean':None,'exact_full_donor':full_first,'exact_single_token':ec['exact'],'wiki_mean_late':mix(ec['exact'],mean),'wiki_mean_all':mean,'rank2_late':ec['rank2']}
   for name,initial in variants.items():
    for rep in range(2):
     if name=='clean':cache,lg=fw(x[None].cuda())
     else:cache,lg=fw(x[None,1:].cuda(),clonecache(initial),torch.arange(1,128,device='cuda'))
     for k in range(step):cache,lg=fw(clean_ids[k].reshape(1,1).cuda(),cache,torch.tensor([128+k],device='cuda'))
     logits[c,i,step,name,rep]=lg.cpu()
     expected=raw[c,i]['clean' if name=='clean' else 'variants']
     expected_pred=expected['pred'][step] if name=='clean' else expected[name]['matched_history']['pred'][step]
     assert int(lg.argmax())==int(expected_pred)
    assert torch.equal(logits[c,i,step,name,0],logits[c,i,step,name,1]),'Repeated fixed-history logits are not deterministic'
   b=logits[c,i,step,'clean',0].float();lb=b.log_softmax(-1);top=b.topk(2);basepred=int(b.argmax())
   for name in variants:
    z=logits[c,i,step,name,0].float();lp=z.log_softmax(-1);pred=int(z.argmax());topz=z.topk(2)
    rows.append({'corpus':c,'window':i,'step':step,'selection_trigger':trigger,'variant':name,'repeats_bitwise_equal':True,'saved_prediction_reproduced':True,
     'clean_token_id':basepred,'condition_token_id':pred,'clean_token_text':tok.decode([basepred]),'condition_token_text':tok.decode([pred]),
     'clean_top2_margin':float(top.values[0]-top.values[1]),'condition_top2_margin':float(topz.values[0]-topz.values[1]),
     'clean_logit_at_clean_choice':float(b[basepred]),'clean_logit_at_condition_choice':float(b[pred]),'condition_logit_at_clean_choice':float(z[basepred]),'condition_logit_at_condition_choice':float(z[pred]),
     'kl':float((lb.exp()*(lb-lp)).sum()),'max_abs_logit_difference':float((b-z).abs().max())})
   log(f'Tie diagnostic {c} window{i} step{step}; {count} forwards')
 torch.save(logits,out/'logits.pt');savej(out/'results.json',rows)
 meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
  parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 savej(out/'manifest.json',meta)
