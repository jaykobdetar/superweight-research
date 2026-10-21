"""Fresh-window source/later factorial with full tokenwise neuron traces."""
import json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash

class Neuron:
 def __init__(self,model):
  self.mode='clean';self.scope='source';self.cache={};self.handles=[];m=model.model.layers[1].mlp.down_proj;self.weight=m.weight.detach().clone();self.weight[:,1710]=0;self.scalar=float(m.weight[1764,1710]);self.second_scalar=float(model.model.layers[2].mlp.down_proj.weight[1764,8041]);self.handles.append(m.register_forward_hook(self.hook));self.handles.append(model.model.layers[2].mlp.down_proj.register_forward_pre_hook(self.second))
  for name in ('gate','up'):self.handles.append(getattr(model.model.layers[1].mlp,name+'_proj').register_forward_hook(lambda m,a,o,name=name:self.factor(name,o)))
 def factor(self,name,o):self.cache[name]=o[0,:,1710].detach().cpu().clone();return o
 def second(self,m,a):self.cache['second_activation']=a[0][0,:,8041].detach().cpu().clone()
 def hook(self,m,args,o):
  self.cache['activation']=args[0][0,:,1710].detach().cpu().clone();z=o
  if self.mode!='clean':
   alt=F.linear(args[0],self.weight,m.bias);z=o.clone();p={'source':0,'later':slice(1,None),'all':slice(None)}[self.scope]
   if self.mode=='row':z[:,p,1764]=alt[:,p,1764]
   elif self.mode=='column':z[:,p]=alt[:,p]
   else:raise ValueError(self.mode)
  return z
 def close(self):
  for h in self.handles:h.remove()

def prepare(ctx,out):
 prior=json.loads((ROOT/'confirmation_run/data_manifest.json').read_text());tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);sets={};dm={};seen=set()
 for c,d in prior.items():
  ps=[Path(s['path']) for s in d['sources']]
  for p,s in zip(ps,d['sources']):assert sha(p)==s['sha256'],p
  text='\n\n'.join(Dataset.from_file(str(ps[0]))['text']) if c=='wiki_validation' else '\n\n'.join(p.read_text() for p in ps)
  ids=torch.tensor(tok.encode(text,add_special_tokens=False));assert len(ids)==d['token_count'];starts=[s+768 for s in d['evaluation_starts'][:4]];excluded=d['calibration_starts']+d['evaluation_starts']
  assert all(s+512<=e or e+512<=s for s in starts for e in excluded);assert all(s+512<=len(ids) for s in starts)
  sets[c]=[ids[s:s+512].clone() for s in starts]
  for x in sets[c]:h=tokenhash(x);assert h not in seen;seen.add(h)
  dm[c]={**d,'evaluation_starts':starts,'original_excluded_starts':excluded,'parent_manifest_sha256':sha(ROOT/'confirmation_run/data_manifest.json'),'fitting':'none; fresh disjoint windows'}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',dm);return sets,tok

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();sets,tok=prepare(ctx,out);r=Run(ctx['model'],out);neuron=Neuron(ctx['model']);raw={};observations=[];checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'LATER_USES.md'),'model_hash_before':ctx['original_hash'],'target':[1,1764,1710],'contexts':'16 fresh disjoint512-token windows,4/text','scopes':['source0','later1+','all'],'score_start':1,'identities':['natural','The510','period15'],'amplitude_thresholds':[1,10,100],'fitting':'none','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def capture():
  a={k:v.clone() for k,v in neuron.cache.items()};a['potential_first']=a['activation'].float()*neuron.scalar;a['potential_second']=a['second_activation'].float()*neuron.second_scalar;a['applied_first']=a['potential_first'].clone()
  if neuron.mode!='clean':a['applied_first'][{'source':0,'later':slice(1,None),'all':slice(None)}[neuron.scope]]=0
  a.update(mode=neuron.mode,scope=neuron.scope);return a
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity,tid in [('natural',None),('The',510),('period',15)]:
     x=xx.clone()
     if tid is not None:x[0]=tid
     g=f'later_{c}_{identity}';neuron.mode='clean';neuron.scope='source';b,_,bl=r.evaluate(x,keep=True);base=capture();r.record(g,'clean',i,x,b,b,{'identity':identity},1);raw[f'{g}__clean__{i}']=base
     amp=base['potential_first'];top=(amp[1:].abs().topk(8).indices+1).tolist();observations.append({'group':g,'window':i,'source_pulse':float(amp[0]),'later_abs_max':float(amp[1:].abs().max()),'later_abs_mean':float(amp[1:].abs().mean()),'counts_above':{str(t):int((amp[1:].abs()>t).sum()) for t in (1,10,100)},'top_positions':[{'position':p,'token_id':int(x[p]),'potential_pulse':float(amp[p]),'nearby_tokens':x[max(0,p-4):p+5].tolist(),'nearby_text':tok.decode(x[max(0,p-4):p+5])} for p in top]})
     for mode in ('row','column'):
      for scope in ('source','later','all'):
       neuron.mode=mode;neuron.scope=scope;z,_,_=r.evaluate(x,baseline=bl);cap=capture();check={'group':g,'window':i,'trial':mode+'_'+scope,'first_gate_exact':torch.equal(cap['gate'],base['gate']),'first_up_exact':torch.equal(cap['up'],base['up']),'first_activation_exact':torch.equal(cap['activation'],base['activation'])};checks.append(check);assert all(check[k] for k in ('first_gate_exact','first_up_exact','first_activation_exact'));r.record(g,mode+'_'+scope,i,x,z,b,{'mode':mode,'scope':scope,'identity':identity},1);raw[f'{g}__{mode}_{scope}__{i}']=cap
     del bl
    r.checkpoint();torch.save(raw,out/'neuron_traces.pt');savej(out/'pulse_observations.json',observations);savej(out/'checks.json',checks);log(f'Later uses {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:neuron.close();r.close()
