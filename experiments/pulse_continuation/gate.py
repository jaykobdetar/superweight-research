"""Frozen gate/value decompositions and donor-free replacements, sequential GPU job."""
import json,math,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from factorial import Run,Local,sha,savej,tokenhash,log,hash_model,rotate_half
from transformers import AutoTokenizer

def parts(cache,h):
 sl=slice(h*128,(h+1)*128);q=cache['q'][0,:,sl].clamp(-8,8);k=cache['k'][0,:,sl].clamp(-8,8);v=cache['v'][0,:,sl].clamp(-8,8).float()
 co=cache['cos'][0].to(q.device);si=cache['sin'][0].to(q.device)
 q=(q*co+rotate_half(q)*si).to(q.dtype).float();k=(k*co+rotate_half(k)*si).to(k.dtype).float()
 scores=q@k.T/math.sqrt(128);scores.masked_fill_(torch.ones_like(scores,dtype=torch.bool).triu(1),-torch.inf)
 a=scores.softmax(-1)[:,0];u=torch.zeros_like(v);u[1:]=scores[1:,1:].softmax(-1)@v[1:]
 y=a[:,None]*v[0]+(1-a[:,None])*u
 return {'a':a,'v':v[0],'u':u,'y':y}

class Gate(Local):
 def reset(self,patch=None,capture=False):super().reset(patch,capture);self.gpu={}
 def proj(self,k,o):
  p=self.patch
  if p.get('type')=='constant_kv' and k in 'kv':
   o=o.clone();h=p['head'];sl=slice(h*128,(h+1)*128)
   o[:,0,sl]=p['cal'][k][0,0,sl].to(o.device)
   if k=='v' and p.get('zero_value'):o[:,0,sl]=0
  o=super().proj(k,o)
  if p.get('type')=='gate':self.gpu[k]=o.detach().clone()
  return o
 def context(self,m,inputs):
  super().context(m,inputs);p=self.patch
  if p.get('type')!='gate':return None
  h=p['head'];sl=slice(h*128,(h+1)*128);self.gpu.update(cos=self.rope[0],sin=self.rope[1]);z=parts(self.gpu,h)
  a=z['a'].clone();v=z['v'];g=p['gate'];vm=p['value'];dev=inputs[0].device
  if g=='donor':a=p['donor']['a'].to(dev)
  elif g=='constant':a=torch.full_like(a,p['cal']['gate'][h])
  elif g=='position':a=p['cal']['gate_position'][h].to(dev).clone()
  elif g=='zero':a=torch.zeros_like(a)
  elif g=='one':a=torch.ones_like(a)
  elif g!='native':raise ValueError(g)
  a[0]=1 # The sole available first key always has mass one.
  if vm=='donor':v=p['donor']['v'].to(dev)
  elif vm=='zero':v=torch.zeros_like(v)
  elif vm=='constant':v=p['cal']['v'][0,0,sl].to(dev).clamp(-8,8).float()
  elif vm!='native':raise ValueError(vm)
  proposed=a[:,None]*v+(1-a[:,None])*z['u']
  o=inputs[0].clone()
  if p.get('reconstruct'):o[0,:,sl]=proposed.to(o.dtype)
  else:o[0,:,sl]=(o[0,:,sl].float()+(proposed-z['y'])).to(o.dtype)
  return (o,)

def prepare(ctx,out):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');sets={};meta={}
 prior=json.loads((ROOT/'factorial_run/data_manifest.json').read_text())
 for corpus in ('wiki','hume'):
  starts=[s+3072 for s in old['starts'][corpus]['evaluation'][:8]];excluded=[]
  for split in old['starts'][corpus].values():excluded.extend((s,s+1024) for s in split)
  excluded.extend((s-512,s+512) for s in prior[corpus]['starts'])
  for s in starts:assert all(s+512<=a or s>=b for a,b in excluded),(corpus,s)
  sets[corpus]={'calibration':[old['corpora'][corpus][s:s+512].clone() for s in old['starts'][corpus]['development']],
                'evaluation':[old['corpora'][corpus][s:s+512].clone() for s in starts]}
  meta[corpus]={'calibration_starts':old['starts'][corpus]['development'],'evaluation_starts':starts,'evaluation_disjoint_from_prior_spans':True}
 path=Path('/EXTERNAL/home/Downloads/philosophy-course-main/collection/philosophy-study-collection/texts/gutenberg/pg1497-the-republic.txt')
 text=path.read_text();tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True)
 ids=torch.tensor(tok.encode(text,add_special_tokens=False));starts=[32768+4096*i for i in range(8)]
 assert len(ids)>starts[-1]+512
 sets['plato']={'evaluation':[ids[s:s+512].clone() for s in starts]}
 meta['plato']={'path':str(path),'sha256':sha(path),'total_tokens':len(ids),'evaluation_starts':starts,'tokenizer_special_tokens':False,'text_transform':'none; interior evaluation avoids file header'}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',meta);return sets

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=prepare(ctx,out)
 r=Run(ctx['model'],out);r.local.close();r.local=Gate(ctx['model']);start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'factorial_sha256':sha(ROOT/'factorial.py'),
  'model_hash_before':ctx['original_hash'],'source':'L1/r1764/c1710 position0 only','evaluation':'8 fresh windows each WikiText/Hume/Plato, length512',
  'calibration':'8 old development windows per WikiText/Hume; arithmetic means, no optimization or outcome-based selection',
  'replacement':'native context plus proposed-minus-reconstructed context; separate direct-reconstruction controls',
  'dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta)
 try:
  cals={}
  for corpus in ('wiki','hume'):
   caches=[];metrics=[]
   for i,x in enumerate(sets[corpus]['calibration']):
    _,c,_=r.evaluate(x,capture=True);caches.append({k:c[k][:,:1].clone() for k in ('k','v')});metrics.append({h:parts(c,h) for h in (0,15)})
   cals[corpus]={k:torch.stack([c[k].float() for c in caches]).mean(0).half() for k in ('k','v')}
   cals[corpus]['gate']={h:float(torch.cat([m[h]['a'][1:] for m in metrics]).mean()) for h in (0,15)}
   cals[corpus]['gate_position']={h:torch.stack([m[h]['a'] for m in metrics]).mean(0) for h in (0,15)}
  torch.save(cals,out/'calibration.pt');log('Gate calibration frozen')
  for corpus,sets0 in sets.items():
   for i,x in enumerate(sets0['evaluation']):
    b,bc,bl=r.evaluate(x,capture=True,keep=True);d,dc,_=r.evaluate(x,'deleted',capture=True,baseline=bl);group='natural_'+corpus
    for trial,z in [('clean',b),('deleted',d)]:r.record(group,trial,i,x,z,b,{'state':trial})
    assert all(torch.equal(bc[k][:,1:],dc[k][:,1:]) for k in ('q','k','v'))
    decom={state:{h:parts(c,h) for h in (0,15)} for state,c in [('clean',bc),('deleted',dc)]}
    torch.save(decom,out/f'decomposition_{corpus}_{i}.pt')
    for head in (15,0):
     for state,donor,ref,direction in [('clean',dc,b,'damage'),('deleted',bc,d,'rescue')]:
      other='deleted' if state=='clean' else 'clean';pdon=decom[other][head]
      variants=[('identity','native','native',False,'wiki'),('reconstruction','native','native',True,'wiki'),
                ('zero_value','native','zero',False,'wiki'),('remove_source','zero','zero',False,'wiki'),
                ('constant_gate_zero_value','constant','zero',False,'wiki'),('constant_kv','native','constant',False,'wiki')]
      if head==15:variants += [('donor_gate','donor','native',False,'wiki'),('donor_value','native','donor',False,'wiki'),
                ('donor_both','donor','donor',False,'wiki'),('donor_gate_zero_value','donor','zero',False,'wiki'),
                ('zero_head','one','zero',False,'wiki'),('constant_value','native','constant',False,'wiki'),
                ('constant_gate','constant','native',False,'wiki'),('position_gate_zero_value','position','zero',False,'wiki'),
                ('hume_gate_zero_value','constant','zero',False,'hume'),('hume_position_gate_zero_value','position','zero',False,'hume'),
                ('constant_key_zero_value','native','zero',False,'wiki'),('hume_constant_kv','native','constant',False,'hume')]
      for name,g,v,reconstruct,cal in variants:
       if name in ('constant_kv','constant_key_zero_value','hume_constant_kv'):
        patch={'type':'constant_kv','head':head,'cal':cals[cal],'zero_value':v=='zero'}
       else:patch={'type':'gate','head':head,'gate':g,'value':v,'donor':pdon,'cal':cals[cal],'reconstruct':reconstruct}
       z,_,_=r.evaluate(x,state,patch,baseline=bl)
       if name=='identity':assert torch.equal(z['loss'],ref['loss']) and torch.equal(z['pred'],ref['pred'])
       r.record(group,f'H{head}_{name}_{state}',i,x,z,b,{'head':head,'operation':name,'state':state,'calibration_corpus':cal,'gate':g,'value':v,'reconstruct':reconstruct})
      if head==15:
       for kinds in ('k','v','kv'):
        z,_,_=r.evaluate(x,state,{'type':'qkv','head':head,'kinds':kinds,'donor':donor},baseline=bl)
        r.record(group,f'H15_native_{kinds}_{direction}',i,x,z,b,{'head':head,'kinds':kinds,'direction':direction,'native_reference':True})
    del bl,bc,dc;r.checkpoint();log(f'Gate {corpus} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:r.close()
