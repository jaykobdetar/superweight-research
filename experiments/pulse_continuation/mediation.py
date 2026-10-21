"""Layerwise source-cache causal cuts and later candidate query dependence."""
import json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,tokenhash,log,hash_model

class Paths:
 def __init__(self,model):
  self.handles=[];self.reset()
  for l,layer in enumerate(model.model.layers):
   for k in ('q','k','v'):self.handles.append(getattr(layer.self_attn,k+'_proj').register_forward_hook(lambda m,i,o,l=l,k=k:self.proj(l,k,o)))
 def reset(self,patch=None,capture=False):self.patch=patch or {};self.capture=capture;self.cache={}
 def proj(self,l,k,o):
  p=self.patch
  if p.get('type')=='cut' and l in p['layers'] and k in p.get('kinds','kv'):
   o=o.clone();donor=p['donor'][l][k].to(o.device)
   if p.get('heads') is None:o[:,0]=donor[:,0]
   else:
    for h in p['heads']:o[:,0,h*128:(h+1)*128]=donor[:,0,h*128:(h+1)*128]
  if p.get('type')=='candidate' and l==p['layer'] and k in p['kinds']:
   o=o.clone();h=p['head'];o[:,:,h*128:(h+1)*128]=p['donor']['candidate'][l][h][k].to(o.device)
  if self.capture:
   if k in 'kv':self.cache.setdefault(l,{})[k]=o[:,:1].detach().cpu().clone()
   if l in (12,15):
    for h in (0,5 if l==12 else 11):self.cache.setdefault('candidate',{}).setdefault(l,{}).setdefault(h,{})[k]=o[:,:,h*128:(h+1)*128].detach().cpu().clone()
  return o
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists()
 sets=torch.load(ROOT/'gate_run/tokens.pt',weights_only=True,map_location='cpu');r=Run(ctx['model'],out);r.local.close();r.local=Paths(ctx['model'])
 start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'factorial_sha256':sha(ROOT/'factorial.py'),
  'model_hash_before':ctx['original_hash'],'tokens_sha256':sha(ROOT/'gate_run/tokens.pt'),'source':'L1/r1764/c1710 position0 only',
  'evaluation':'same fresh gate-batch evaluation windows; exploratory mechanism panel, no fitting to evaluation',
  'score_start':1,'score_reason':'all-source-cache cut protects predictions at positions1+; predictor0 still has damaged residual; all comparisons omit it equally',
  'candidate_patch':'all Q/K/V positions for named later head; tests downstream dependence, not direct source transmission',
  'seed':20261021,'dtype':'float16','cpu_threads':4}
 savej(out/'manifest.json',meta)
 cuts={'L2_H15':([2],[15],'kv'),'L2_other_heads':([2],list(range(15)),'kv'),'L2_all':([2],None,'kv'),
       'L2_keys':([2],None,'k'),'L2_values':([2],None,'v'),'early_2_3':(list(range(2,4)),None,'kv'),
       'early_2_5':(list(range(2,6)),None,'kv'),'early_2_8':(list(range(2,9)),None,'kv'),
       'early_2_11':(list(range(2,12)),None,'kv'),'all_2_15':(list(range(2,16)),None,'kv'),
       'late_3_15':(list(range(3,16)),None,'kv'),'late_6_15':(list(range(6,16)),None,'kv'),'late_12_15':(list(range(12,16)),None,'kv')}
 try:
  cals={}
  for corpus in ('wiki','hume'):
   cc=[]
   for x in sets[corpus]['calibration']:
    _,c,_=r.evaluate(x,capture=True);cc.append(c)
   cals[corpus]={l:{k:torch.stack([c[l][k].float() for c in cc]).mean(0).half() for k in ('k','v')} for l in range(16)}
  torch.save(cals,out/'calibration.pt');log('Cache-cut calibration frozen')
  for corpus,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl);group='natural_'+corpus
    torch.save({'clean':bc,'deleted':dc},out/f'paths_{corpus}_{i}.pt')
    for trial,z in [('clean',b),('deleted',d)]:r.record(group,trial,i,x,z,b,{'state':trial},1)
    for state,donor,direction,ref in [('clean',dc,'damage',d),('deleted',bc,'rescue',b)]:
     for name,(layers,heads,kinds) in cuts.items():
      z,_,_=r.evaluate(x,state,{'type':'cut','layers':layers,'heads':heads,'kinds':kinds,'donor':donor},baseline=bl)
      r.record(group,name+'_'+direction,i,x,z,b,{'layers':layers,'heads':heads,'kinds':kinds,'direction':direction},1)
      if name=='all_2_15':assert torch.equal(z['loss'][1:],ref['loss'][1:]) and torch.equal(z['pred'][1:],ref['pred'][1:]),'Full causal cut not exact'
     for cal in ('wiki','hume'):
      for name,layers in [('all_0_15',list(range(16))),('all_2_15',list(range(2,16))),('L2_all',[2])]:
       z,_,_=r.evaluate(x,state,{'type':'cut','layers':layers,'donor':cals[cal]},baseline=bl)
       r.record(group,f'constant_{cal}_{name}_{state}',i,x,z,b,{'calibration':cal,'layers':layers,'state':state},1)
     for l,h in [(12,5),(15,11)]:
      for head in (h,0):
       for kinds in (('q','k','v','qk','qv','kv','qkv') if head==h else ('q','kv','qkv')):
        z,_,_=r.evaluate(x,state,{'type':'candidate','layer':l,'head':head,'kinds':kinds,'donor':donor},baseline=bl)
        r.record(group,f'L{l}H{head}_{kinds}_{direction}',i,x,z,b,{'layer':l,'head':head,'kinds':kinds,'positions':'all','direction':direction},1)
    del bl,bc,dc;r.checkpoint();log(f'Mediation {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
