"""Development-selected head channels for the learned static source residual."""
import collections,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from normalization import Norms

class Edges(Paths):
 def proj(self,l,k,o):
  p=self.patch
  if p.get('type')=='edges' and k in p.get('kinds','kv'):
   heads=[h for ll,h in p['edges'] if ll==l]
   if heads:
    o=o.clone()
    for h in heads:o[:,0,h*128:(h+1)*128]=p['donor'][l][k][:,0,h*128:(h+1)*128].to(o.device)
  return super().proj(l,k,o)

def prepare(ctx,out):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
 fm=json.loads((ROOT/'factorial_run/data_manifest.json').read_text());gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text());tm=json.loads((ROOT/'template_run/data_manifest.json').read_text())
 sets={};meta={}
 for c in ('wiki','hume'):
  excluded=[]
  for ss in old['starts'][c].values():excluded.extend((s,s+1024) for s in ss)
  excluded.extend((s-512,s+512) for s in fm[c]['starts']);excluded.extend((s,s+1024) for s in gm[c]['evaluation_starts']);excluded.extend((s,s+512) for s in tm[c]['starts'])
  candidates=[s for s in range(26624,len(old['corpora'][c])-512,512) if all(s+512<=a or s>=b for a,b in excluded)]
  chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,8,dtype=int)];assert len(set(chosen))==8
  assert all(b-a>=1024 for a,b in zip(chosen,chosen[1:])),chosen
  sets[c]={'evaluation':[old['corpora'][c][s:s+512].clone() for s in chosen]}
  if c=='wiki':sets[c]['development']=[old['corpora'][c][s:s+512].clone() for s in old['starts'][c]['development']]
  meta[c]={'evaluation_starts':chosen,'excluded_spans':excluded,'new_evaluation':True}
 tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);p=Path(gm['plato']['path']);plato=torch.tensor(tok.encode(p.read_text(),add_special_tokens=False));ps=[s+2048 for s in gm['plato']['evaluation_starts']]
 sets['plato']={'evaluation':[plato[s:s+512].clone() for s in ps]};meta['plato']={'evaluation_starts':ps,'excluded_spans':[(s,s+1536) for s in gm['plato']['evaluation_starts']],'source_sha256':sha(p),'new_evaluation':True}
 sources=[Path('/EXTERNAL/venv/lib/python3.12/site-packages/transformers/models/olmo/modeling_olmo.py'),Path('/EXTERNAL/venv/lib/python3.12/site-packages/transformers/cache_utils.py')]
 code=torch.tensor(tok.encode('\n\n'.join(p.read_text() for p in sources),add_special_tokens=False));assert len(code)>15000
 cs=np.linspace(8192,len(code)-1024,8,dtype=int).tolist();sets['python']={'calibration':[code[s:s+512].clone() for s in range(0,4096,512)],'evaluation':[code[s:s+512].clone() for s in cs]}
 meta['python']={'sources':[{'path':str(p),'sha256':sha(p)} for p in sources],'token_count':len(code),'calibration_starts':list(range(0,4096,512)),'evaluation_starts':cs,'separator':'two newlines','new_evaluation':True}
 for c,data in sets.items():
  for x in data['evaluation']:assert len(x)==512
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',meta);return sets

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=prepare(ctx,out)
 ts=torch.load(ROOT/'template_run/templates.pt',weights_only=True,map_location='cpu')['templates'];axis=ts['coordinate_matched_axis'];full=ts['wiki_full_mean']
 r=Run(ctx['model'],out);r.local.close();r.local=Edges(ctx['model']);norm=Norms(ctx['model']);start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'template_sha256':sha(ROOT/'template_run/templates.pt'),'screen':'224 individual layer2..15/head0..15 K0V0 repairs from coordinate-matched axis to full Wiki mean; Wiki8 development windows only',
  'selection':'rank average paired development NLL improvement; test frozen top1,2,4,8,16,32 sets and per-layer-count matched random controls',
  'heldout':'8 unused512-token windows each WikiText,Hume,Plato,local public Transformers Python source; no selection on heldout outcomes',
  'score_start':1,'seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 def ev(x,state='clean',template=None,patch=None,capture=False,baseline=None,keep=False):
  norm.reset({'layers':list(range(2,16)),'donor':template,'mode':'full'} if template else None)
  return r.evaluate(x,state,patch,capture,baseline=baseline,keep=keep)
 try:
  # Fresh code-only calibration is a transfer control; it does not select heads.
  ns=[]
  for x in sets['python']['calibration']:
   norm.reset(capture=True);r.source.state='clean';r.local.reset()
   with torch.inference_mode():ctx['model'](input_ids=x[None].cuda(),use_cache=False)
   r.count+=1;ns.append(norm.cache)
  code_mean={l:{'norm':torch.stack([c[l]['norm'].float() for c in ns]).mean(0).half()} for l in range(2,16)};torch.save(code_mean,out/'python_calibration.pt')
  effects=collections.defaultdict(list)
  for i,x in enumerate(sets['wiki']['development']):
   b,_,bl=ev(x,keep=True);a,ac,_=ev(x,template=axis,capture=True,baseline=bl);f,fc,_=ev(x,template=full,capture=True,baseline=bl)
   for name,z in [('clean',b),('axis',a),('full_mean',f)]:r.record('development_wiki',name,i,x,z,b,{'screen_anchor':name},1)
   for l in range(2,16):
    for h in range(16):
     z,_,_=ev(x,template=axis,patch={'type':'edges','edges':[(l,h)],'donor':fc},baseline=bl)
     effects[l,h].append(float((a['loss'][1:]-z['loss'][1:]).double().mean()))
     r.record('development_wiki',f'L{l}H{h}_rescue',i,x,z,b,{'layer':l,'head':h,'kinds':'kv','world':'axis','donor':'full_mean','screen':True},1)
   del bl,ac,fc;r.checkpoint();log(f'Residual screen {i+1}/8; {r.count} forwards')
  ordered=sorted(effects,key=lambda e:(-np.mean(effects[e]),e));selection={};rng=np.random.default_rng(20261021)
  for k in (1,2,4,8,16,32):
   selected=ordered[:k];counts=collections.Counter(l for l,h in selected);control=[]
   for l,n in sorted(counts.items()):
    unused=[h for h in range(16) if (l,h) not in selected];pool=unused if len(unused)>=n else list(range(16))
    control.extend((l,int(h)) for h in rng.choice(pool,n,replace=False))
   selection[k]={'selected':selected,'control':control,'control_overlap':len(set(selected)&set(control))}
  savej(out/'selection.json',{'ranking':[{'layer':l,'head':h,'mean_dev_improvement':float(np.mean(effects[l,h])),'window_improvements':effects[l,h]} for l,h in ordered],'sets':selection,'selected_before_heldout':True})
  log('Head selection frozen before unused evaluation')
  all_edges=[(l,h) for l in range(2,16) for h in range(16)]
  for corpus,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    b,_,bl=ev(x,keep=True);d,_,_=ev(x,'deleted',baseline=bl);a,ac,_=ev(x,template=axis,capture=True,baseline=bl);f,fc,_=ev(x,template=full,capture=True,baseline=bl);group='heldout_'+corpus
    for name,z in [('clean',b),('deleted',d),('axis',a),('full_mean',f)]:r.record(group,name,i,x,z,b,{'anchor':name},1)
    for name,t in [('python_full_mean',code_mean),('residual_rank2',ts['residual_rank2']),('ideal',ts['ideal'])]:
     z,_,_=ev(x,template=t,baseline=bl);r.record(group,name,i,x,z,b,{'template':name},1)
    for direction,t,donor,ref in [('rescue',axis,fc,f),('damage',full,ac,a)]:
     for k,s in selection.items():
      for label in ('selected','control'):
       z,_,_=ev(x,template=t,patch={'type':'edges','edges':s[label],'donor':donor},baseline=bl)
       r.record(group,f'top{k}_{label}_{direction}',i,x,z,b,{'edges':s[label],'direction':direction,'kinds':'kv','selection':'development_only'},1)
     for kinds in ('k','v'):
      z,_,_=ev(x,template=t,patch={'type':'edges','edges':selection[8]['selected'],'donor':donor,'kinds':kinds},baseline=bl)
      r.record(group,'top8_'+kinds+'_'+direction,i,x,z,b,{'edges':selection[8]['selected'],'direction':direction,'kinds':kinds},1)
     z,_,_=ev(x,template=t,patch={'type':'edges','edges':all_edges,'donor':donor},baseline=bl)
     assert torch.equal(z['loss'][1:],ref['loss'][1:]) and torch.equal(z['pred'][1:],ref['pred'][1:])
     r.record(group,'all224_'+direction,i,x,z,b,{'edges':'all','direction':direction,'exact_reference':'full_mean' if direction=='rescue' else 'axis'},1)
    del bl,ac,fc;r.checkpoint();log(f'Residual heldout {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:norm.close();r.close()
