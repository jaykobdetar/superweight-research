"""Bad-token broadcast and source-key/value factorial after newline pulse removal."""
import json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from newline_history import Probe
from newline_heads import edited

class Broadcast(Probe):
 def __init__(self,model):
  super().__init__(model);self.block_layers=[];self.block_position=64;self.patch={};self.path_cache={}
  for l,layer in enumerate(model.model.layers):
   if l>=2:
    for k in 'kv':self.handles.append(getattr(layer.self_attn,k+'_proj').register_forward_hook(lambda m,a,o,l=l,k=k:self.project(l,k,o)))
  self.handles.append(model.model.layers[2].input_layernorm.register_forward_hook(self.norm))
 def attn(self,l,args,k):
  args,k=super().attn(l,args,k)
  if l in self.block_layers:
   mask=k['attention_mask'].clone();pos=self.block_position;mask[:,:,pos+1:,pos]=-torch.inf;k['attention_mask']=mask
  return args,k
 def norm(self,m,a,o):self.cache['L2_input_p']=a[0][0,self.p].detach().cpu().clone();self.cache['L2_normalized_p']=o[0,self.p].detach().cpu().clone();return o
 def project(self,l,k,o):
  if self.patch and k in self.patch['kinds']:
   o=o.clone()
   if self.patch.get('zero') or k in self.patch.get('zero_kinds',''):o[:,self.p]=0
   else:o[:,self.p]=self.patch['donor'][l][k].to(o.device)
  self.path_cache.setdefault(l,{})[k]=o[:,self.p].detach().cpu().clone();return o

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();sets=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');r=Run(ctx['model'],out);p=Broadcast(ctx['model']);raw={};checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'NEWLINE_BROADCAST.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'newline_heads_run/tokens.pt'),'score_start':129,'scored_targets':'IDs130..511;382targets','reuse':'same heldout contexts; new exploratory mechanism panel, no fits','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,mode='clean',layers=None,pos=64,patch=None,bl=None,keep=False):
  p.mode=mode;p.positions=[64];p.block_layers=layers or [];p.block_position=pos;p.patch=patch or {};p.cache={};p.path_cache={}
  z,_,ll=r.evaluate(x,baseline=bl,keep=keep);cap={k:v.clone() for k,v in p.cache.items()};cap['paths']={l:{k:v.clone() for k,v in d.items()} for l,d in p.path_cache.items()};return z,ll,cap
 def rec(g,t,i,x,z,b,cap,spec):r.record(g,t,i,x,z,b,spec,129);raw[f'{g}__{t}__{i}']=cap
 def exact(a,b,label,c,i):
  check={'corpus':c,'window':i,'closure':label,'loss_exact':torch.equal(a['loss'][129:],b['loss'][129:]),'pred_exact':torch.equal(a['pred'][129:],b['pred'][129:])};checks.append(check);assert check['loss_exact'] and check['pred_exact'],check
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    x=edited(xx);g='broadcast_'+c;b,bl,bc=ev(x,keep=True);zs={'clean':b};cs={'clean':bc};rec(g,'clean',i,x,b,b,bc,{})
    for mode in ('row','column'):
     z,_,cap=ev(x,mode,bl=bl);zs[mode]=z;cs[mode]=cap;rec(g,mode,i,x,z,b,cap,{'mode':mode,'position':64})
    masked={}
    for mode in ('clean','row','column'):
     z,_,cap=ev(x,mode,list(range(2,16)),bl=bl);masked[mode]=z;rec(g,mode+'_block_p_all',i,x,z,b,cap,{'mode':mode,'block_key_position':64,'layers':list(range(2,16)),'queries':'strictly later'})
     z,_,cap=ev(x,mode,patch={'kinds':'v','zero':True},bl=bl);rec(g,mode+'_zero_p_values',i,x,z,b,cap,{'mode':mode,'zero_value_position':64,'layers':list(range(2,16)),'keys':'current'})
    z,_,cap=ev(x,patch={'kinds':'kv','zero_kinds':'v','donor':cs['clean']['paths']},bl=bl);rec(g,'cleanK_zero_p_values',i,x,z,b,cap,{'mode':'clean','keys':'frozen clean donor','values':'zero','position':64,'layers':list(range(2,16))})
    for mode in ('row','column'):exact(masked[mode],masked['clean'],mode+'_block_p_all_to_clean_block_p_all',c,i)
    for target,donor in [('row','clean'),('row','column'),('clean','row')]:
     for kinds in ('k','v','kv'):
      z,_,cap=ev(x,target,patch={'kinds':kinds,'donor':cs[donor]['paths']},bl=bl);rec(g,target+'_from_'+donor+'_'+kinds,i,x,z,b,cap,{'state':target,'donor':donor,'kinds':kinds,'position':64,'layers':list(range(2,16))})
      if kinds=='kv':exact(z,zs[donor],target+'_from_'+donor+'_kv_to_'+donor,c,i)
    for name,layers in [('early',list(range(2,6))),('middle',list(range(6,12))),('late',list(range(12,16)))]:
     for mode in ('clean','row'):
      z,_,cap=ev(x,mode,layers,bl=bl);rec(g,mode+'_block_p_'+name,i,x,z,b,cap,{'mode':mode,'block_key_position':64,'layers':layers,'queries':'strictly later'})
    for mode in ('clean','row'):
     z,_,cap=ev(x,mode,list(range(2,16)),pos=128,bl=bl);rec(g,mode+'_block_q_all',i,x,z,b,cap,{'mode':mode,'block_key_position':128,'layers':list(range(2,16)),'queries':'strictly later'})
    del bl;r.checkpoint();torch.save(raw,out/'probe_traces.pt');savej(out/'checks.json',checks);log(f'Newline broadcast {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:p.close();r.close()
