"""Frozen source-cache replacements through the configured 4096-token context."""
import json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from normalization import Norms
from cache_bypass import packed,clonecache,mix,score

def texts(ctx):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
 ans=dict(old['corpora']);tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True)
 gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text());pm=json.loads((ROOT/'residual_paths_run/data_manifest.json').read_text())
 ans['plato']=torch.tensor(tok.encode(Path(gm['plato']['path']).read_text(),add_special_tokens=False))
 ans['python']=torch.tensor(tok.encode('\n\n'.join(Path(p['path']).read_text() for p in pm['python']['sources']),add_special_tokens=False))
 return ans

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats()
 assert model.config.max_position_embeddings==4096
 corpora=texts(ctx);sets={};dm={}
 for c,t in corpora.items():
  starts=np.linspace(8192 if c=='python' else 32768,len(t)-4096,4,dtype=int).tolist()
  assert len(set(starts))==4 and all(b-a>=4096 for a,b in zip(starts,starts[1:])),(c,starts)
  sets[c]=[t[s:s+4096].clone() for s in starts];dm[c]={'starts':starts,'length':4096}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',dm)
 cal=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki']
 template=torch.load(ROOT/'template_run/templates.pt',weights_only=True,map_location='cpu')['templates']['residual_rank2']
 r=Run(model,out);r.local.close();r.local=Paths(model);norm=Norms(model);dictionary={};checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'evaluation':'4 fixed windows/corpus, matched crops512/1024/2048/4096; some text was previously evaluated; no fitting or selection on these outcomes',
  'calibration':'frozen prior Wiki cache means and source-normalization rank2 residual plus axis; no new fitting',
  'cache_calibration_sha256':sha(ROOT/'cache_run/calibration.pt'),'template_sha256':sha(ROOT/'template_run/templates.pt'),
  'score_start':1,'tail_score':'last512 predictor positions, or all510 available at length512',
  'configured_max_context':4096,'dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta)
 def exact(token):
  if token not in dictionary:
   pairs={};r.source.state='clean';r.local.reset()
   for name,t in [('exact',None),('rank2',template)]:
    norm.reset({'layers':list(range(2,16)),'mode':'full','donor':t} if t else None)
    with torch.inference_mode():
     z=model(input_ids=torch.tensor([[token]],device='cuda'),use_cache=True)
     pairs[name]=tuple(tuple(v.detach().cpu().clone() for v in p) for p in z.past_key_values.to_legacy_cache())
     if name=='exact':pairs['logit']=z.logits[0,0].detach().cpu().clone()
    r.count+=1
   norm.reset();dictionary[token]=pairs
  return dictionary[token]
 try:
  for c,ws in sets.items():
   for i,xx in enumerate(ws):
    for n in (512,1024,2048,4096):
     x=xx[:n].clone();norm.reset();b,bc,bl=r.evaluate(x,capture=True,keep=True);d,_,_=r.evaluate(x,'deleted',baseline=bl)
     def record(name,z):
      for suffix,ss in [('',1),('_tail512',max(1,n-513))]:
       r.record(f'extended_{c}_{n}'+suffix,name,i,x,z,b,{'length':n,'cache':name,'score_start':ss},ss)
     record('clean',b);record('deleted',d);ec=exact(int(x[0]))
     variants={'exact_full_donor':packed(bc),'exact_single_token':ec['exact'],'wiki_mean_all':cal,'wiki_mean_late':mix(ec['exact'],cal),'rank2_late':ec['rank2']}
     r.source.state='clean';r.local.reset()
     for name,cache in variants.items():
      with torch.inference_mode():
       cc=clonecache(cache);z=model(input_ids=x[None,1:].cuda(),past_key_values=cc,cache_position=torch.arange(1,n,device='cuda'),use_cache=True)
       logits=torch.cat([ec['logit'][None].cuda(),z.logits[0,:-1]],0);metrics=score(logits,x,bl)
       checks.append({'corpus':c,'window':i,'length':n,'variant':name,'final_cache_length':cc.get_seq_length(),'loss_equal_to_clean':bool(torch.equal(metrics['loss'][1:],b['loss'][1:]))})
      r.count+=1;record(name,metrics);del cc,z,logits
     del bl,bc
    r.checkpoint();log(f'Extended context {c} {i+1}/4; {r.count} forwards')
  torch.save(dictionary,out/'exact_and_rank2_dictionary.pt');savej(out/'execution_checks.json',checks)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:norm.close();r.close()
