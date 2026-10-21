"""Validate the selected newline route on unedited book/code windows."""
import json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from newline_heads import PathProbe

class NaturalProbe(PathProbe):
 def __init__(self,model):super().__init__(model);self.h12_patch=None
 def context(self,m,a):
  super().context(m,a);o=a[0];patch=self.h12_patch
  if patch:
   o=o.clone()
   if patch['kind']=='donor':v=patch['donor']['H12_contexts'][2]
   else:v=self.cache['H12_values'][1 if patch['kind']=='value_p' else 2]
   o[0,self.q,1536:1664]=v.to(o.device)
  self.cache['H12_applied_contexts']=o[0,[0,self.p,self.q],1536:1664].detach().cpu().clone()
  return (o,)+a[1:] if patch else None

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();sets=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');r=Run(ctx['model'],out);p=NaturalProbe(ctx['model']);raw={};obs=[];checks=[];eligibility=[];sel=json.loads((ROOT/'newline_heads_run/selection.json').read_text());h=sel['selected'];control=sel['control'];assert h==14 and control==0
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'NEWLINE_NATURAL.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'newline_heads_run/tokens.pt'),'selection_sha256':sha(ROOT/'newline_heads_run/selection.json'),'score_start':'q+1; matched withinwindow; q=second187','inputs':'unedited original token IDs','reuse':'same heldout text; unedited-input validation, no fits','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,edge=False,edges=None,hp=None,mode='clean',pos=None,bl=None,keep=False):
  p.head_edges=edges if edges is not None else ([(h,p.q,p.p)] if edge else []);p.h12_patch=hp;p.mode=mode;p.positions=pos or [];p.cache={};z,_,ll=r.evaluate(x,baseline=bl,keep=keep);cap={k:v.clone() for k,v in p.cache.items()};cap['potential_pulse']=cap['activation'].float()*p.scalar;cap['applied_pulse']=cap['potential_pulse'].clone()
  if mode!='clean':cap['applied_pulse'][pos]=0
  return z,ll,cap
 def rec(g,t,i,x,z,b,cap,spec):
  r.record(g,t,i,x,z,b,{**spec,'p':p.p,'q':p.q},p.q+1);raw[f'{g}__{t}__{i}']=cap;obs.append({'group':g,'trial':t,'window':i,'p':p.p,'q':p.q,'p_pulse':float(cap['potential_pulse'][p.p]),'q_pulse':float(cap['potential_pulse'][p.q]),'source_pulse':float(cap['potential_pulse'][0]),'token_sha256':tokenhash(x)})
 try:
  for c,windows in sets.items():
   for i,x in enumerate(windows):
    ps=(x==187).nonzero().flatten().tolist();eligible=len(ps)>=2 and ps[1]<=255;eligibility.append({'corpus':c,'window':i,'positions187':ps,'eligible':eligible})
    if not eligible:continue
    p.p,p.q=ps[:2];sham=p.p-1 if p.p else p.q-1;assert int(x[sham])!=187;g='natural_'+c;b,bl,bc=ev(x,keep=True);e,_,ec=ev(x,edge=True,bl=bl);rec(g,'clean',i,x,b,b,bc,{});rec(g,'edge',i,x,e,b,ec,{'head':h,'edge':[p.q,p.p]})
    for t,edges in [('control',[(control,p.q,p.p)]),('sham',[(h,p.q,sham)])]:
     z,_,cap=ev(x,edges=edges,bl=bl);rec(g,t,i,x,z,b,cap,{'edges':edges})
    for world,edge,donor in [('clean',False,ec),('edge',True,bc)]:
     z,_,cap=ev(x,edge,hp={'kind':'donor','donor':donor},bl=bl);rec(g,world+'_H12_donor',i,x,z,b,cap,{'world':world,'context_donor':'opposite world','head':[1,12],'query':p.q})
     for kind in ('value_p','value_q'):
      z,_,cap=ev(x,edge,hp={'kind':kind},bl=bl);rec(g,world+'_H12_'+kind,i,x,z,b,cap,{'world':world,'context':kind,'head':[1,12],'query':p.q})
     for name,pos in [('p',[p.p]),('q',[p.q]),('both',[p.p,p.q])]:
      z,_,cap=ev(x,edge,mode='row',pos=pos,bl=bl);check={'corpus':c,'window':i,'trial':world+'_row_'+name,'activation_unchanged':torch.equal(cap['activation'],(ec if edge else bc)['activation'])};checks.append(check);assert check['activation_unchanged'];rec(g,world+'_row_'+name,i,x,z,b,cap,{'world':world,'row_delete_positions':pos})
    for world,edge,name,pos in [('clean',False,'p',[p.p]),('clean',False,'q',[p.q]),('edge',True,'q',[p.q])]:
     z,_,cap=ev(x,edge,mode='column',pos=pos,bl=bl);check={'corpus':c,'window':i,'trial':world+'_column_'+name,'activation_unchanged':torch.equal(cap['activation'],(ec if edge else bc)['activation'])};checks.append(check);assert check['activation_unchanged'];rec(g,world+'_column_'+name,i,x,z,b,cap,{'world':world,'column_delete_positions':pos})
    del bl;r.checkpoint();torch.save(raw,out/'probe_traces.pt');savej(out/'pulse_observations.json',obs);savej(out/'checks.json',checks);savej(out/'eligibility.json',eligibility);log(f'Natural newline {c} {i+1}/4; {r.count} forwards')
  savej(out/'eligibility.json',eligibility);meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:p.close();r.close()
