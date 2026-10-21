"""Prespecified fresh-text confirmation with fresh, separately held-out gate calibration."""
import importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from quiet_anchor import SourceMask
from gate_subspaces import ProjectedGates

def prepare(ctx,out):
 bookroot=Path('/EXTERNAL/home/Downloads/philosophy-course-main/collection/philosophy-study-collection/texts/gutenberg')
 wiki=Path('/EXTERNAL/hf_cache/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow')
 optim=Path(importlib.util.find_spec('torch').origin).parent/'optim';codepaths=[optim/(n+'.py') for n in ('sgd','adam','adamw')]
 paths={'wiki_validation':[wiki],'dubois':[bookroot/'pg408-the-souls-of-black-folk.txt'],'hobbes':[bookroot/'pg3207-leviathan.txt'],'optimizers':codepaths}
 texts={'wiki_validation':'\n\n'.join(Dataset.from_file(str(wiki))['text'])}
 for c,ps in paths.items():
  if c!='wiki_validation':texts[c]='\n\n'.join(p.read_text() for p in ps)
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);sets={};dm={}
 for c,text in texts.items():
  ids=torch.tensor(tok.encode(text,add_special_tokens=False));code=c=='optimizers';calstarts=list(range(0,4096,512)) if code else list(range(8192,16384,1024))
  first=8192 if code else 32768;assert len(ids)>first+8*512,(c,len(ids));starts=np.linspace(first,len(ids)-512,8,dtype=int).tolist()
  assert min(b-a for a,b in zip(starts,starts[1:]))>=512
  assert all(e+512<=s or s+512<=e for s in calstarts for e in starts)
  sets[c]={'calibration':[ids[s:s+512].clone() for s in calstarts],'evaluation':[ids[s:s+512].clone() for s in starts]}
  dm[c]={'sources':[{'path':str(p),'sha256':sha(p)} for p in paths[c]],'token_count':len(ids),'calibration_starts':calstarts,'evaluation_starts':starts,'separator':'two newlines between Wiki rows or source files; no text edits within book/source','special_tokens':False}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',dm);return sets

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats();sets=prepare(ctx,out)
 packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki'];oldcal=torch.load(ROOT/'global_gates_run/calibration.pt',weights_only=True,map_location='cpu');fit=torch.load(ROOT/'gate_subspaces_run/fitted_subspaces.pt',weights_only=True,map_location='cpu')
 mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 zero={l:{'k':d['k'],'v':torch.zeros_like(d['v'])} for l,d in mean.items()};layers=list(range(2,16));patch={'type':'cut','layers':layers,'donor':mean}
 r=Run(model,out);r.local.close();r.local=Paths(model);mask=SourceMask(model);gate=ProjectedGates(model,mean);cal={};masses={};checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],'design_sha256':sha(ROOT/'CONFIRMATION.md'),
  'mean_cache_sha256':sha(ROOT/'cache_run/calibration.pt'),'old_gate_calibration_sha256':sha(ROOT/'global_gates_run/calibration.pt'),'subspace_fit_sha256':sha(ROOT/'gate_subspaces_run/fitted_subspaces.pt'),
  'fitting':'only arithmetic gate means on8 fresh calibration contexts/corpus; no changes to old source means or PCA bases; no heldout fitting',
  'identities':'natural or The510 at source0; all suffix IDs unchanged, no retokenization','score_start':1,
  'scope':'new experimental text; checkpoint training overlap unknown; pointwise paired-window intervals conditional on constants','dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta)
 try:
  for c,data in sets.items():
   masses[c]=[];mask.layers=[];gate.active_projection=False
   for x in data['calibration']:
    gate.reset('calibrate');r.evaluate(x,patch=patch);masses[c].append(gate.cache)
   cal[c]={l:{'global':torch.stack([m[l][:,1:].mean(1) for m in masses[c]]).mean(0)} for l in layers}
  torch.save(cal,out/'fresh_gate_calibration.pt');torch.save(masses,out/'calibration_source_mass.pt');log('Fresh confirmation calibration frozen')
  for c,data in sets.items():
   for i,xx in enumerate(data['evaluation']):
    for identity in ('natural','The'):
     x=xx.clone()
     if identity=='The':x[0]=510
     gate.active_projection=False;gate.reset();mask.layers=[];b,_,bl=r.evaluate(x,keep=True);group='confirm_'+c+'_'+identity
     r.record(group,'clean',i,x,b,b,{'identity':identity,'reference':'native model'},1)
     d,_,_=r.evaluate(x,'deleted',baseline=bl);r.record(group,'deleted',i,x,d,b,{'identity':identity,'source_state':'deleted'},1)
     bm,_,_=r.evaluate(x,patch=patch,baseline=bl);r.record(group,'mean_KV',i,x,bm,b,{'identity':identity,'source_cache':'frozen old Wiki means2..15'},1)
     z,_,_=r.evaluate(x,patch={'type':'cut','layers':layers,'donor':zero},baseline=bl);r.record(group,'meanK_zeroV',i,x,z,b,{'identity':identity,'source_values':'zero','source_keys':'old means'},1)
     mask.layers=layers;z,_,_=r.evaluate(x,patch=patch,baseline=bl);r.record(group,'remove_source',i,x,z,b,{'identity':identity,'source_masked_at':layers},1)
     gate.reset('oracle');z,_,_=r.evaluate(x,patch=patch,baseline=bl);r.record(group,'oracle',i,x,z,b,{'identity':identity,'control':'current gate reconstruction'},1)
     checks.append({'corpus':c,'window':i,'identity':identity,'oracle_delta_nll_to_mean':float((z['loss'][1:]-bm['loss'][1:]).double().mean()),'oracle_agreement_to_mean':float(z['pred'][1:].eq(bm['pred'][1:]).double().mean())})
     for name,fitcal in [('wiki',oldcal['wiki']),('hume',oldcal['hume']),('fresh',cal[c])]:
      gate.reset('global',fitcal);z,_,_=r.evaluate(x,patch=patch,baseline=bl)
      r.record(group,'global_'+name,i,x,z,b,{'identity':identity,'calibration':name,'gate':'arithmetic mean perhead/layer'},1)
     gate.active_projection=True
     for rank in (4,12):
      gate.reset_projection(fit,'pca',rank,True);z,_,_=r.evaluate(x,patch=patch,baseline=bl)
      r.record(group,'pca'+str(rank),i,x,z,b,{'identity':identity,'rank':rank,'basis':'old pooled dev','bounds':True},1)
     del bl
    r.checkpoint();log(f'Fresh confirmation {c} {i+1}/8; {r.count} forwards')
  savej(out/'reconstruction_checks.json',checks)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:gate.close();mask.close();r.close()
