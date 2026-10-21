"""Controlled prior-newline token and early-attention-edge interventions."""
import json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash

class Probe:
 def __init__(self,model):
  self.p=64;self.q=128;self.mode='clean';self.positions=[];self.edges=[];self.cache={};self.masks={};self.handles=[]
  m=model.model.layers[1].mlp.down_proj;self.weight=m.weight.detach().clone();self.weight[:,1710]=0;self.scalar=float(m.weight[1764,1710]);self.handles.append(m.register_forward_hook(self.output))
  for l,layer in enumerate(model.model.layers):self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k,l=l:self.attn(l,a,k),with_kwargs=True))
  a=model.model.layers[1].self_attn;self.handles.append(a.v_proj.register_forward_hook(self.value));self.handles.append(a.o_proj.register_forward_pre_hook(self.context))
 def attn(self,l,args,k):
  x=k.get('hidden_states',args[0] if args else None);n=x.shape[1]
  if n not in self.masks:
   m=torch.zeros((n,n),device=x.device,dtype=x.dtype);m.masked_fill_(torch.ones((n,n),device=x.device,dtype=torch.bool).triu(1),-torch.inf);self.masks[n]=m[None,None]
  m=self.masks[n]
  if any(ll==l for ll,dest,source in self.edges):
   m=m.clone()
   for ll,dest,source in self.edges:
    if ll==l:m[:,:,dest,source]=-torch.inf
  k=dict(k);k['attention_mask']=m;return args,k
 def value(self,m,a,o):self.cache['H12_values']=o[0,[0,self.p,self.q],12*128:13*128].clamp(-8,8).detach().cpu().clone();return o
 def context(self,m,a):self.cache['H12_contexts']=a[0][0,[0,self.p,self.q],12*128:13*128].detach().cpu().clone()
 def output(self,m,a,o):
  self.cache['activation']=a[0][0,:,1710].detach().cpu().clone();z=o
  if self.mode!='clean':
   alt=F.linear(a[0],self.weight,m.bias);z=o.clone()
   if self.mode=='row':z[:,self.positions,1764]=alt[:,self.positions,1764]
   elif self.mode=='column':z[:,self.positions]=alt[:,self.positions]
   else:raise ValueError(self.mode)
  return z
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);probe=Probe(ctx['model']);sets=torch.load(ROOT/'later_uses_run/tokens.pt',weights_only=True,map_location='cpu');torch.save(sets,out/'tokens.pt');raw={};obs=[];checks=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'NEWLINE_HISTORY.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'later_uses_run/tokens.pt'),'prefix_edits':'replace187/535 through128 with209; source0=510; p64 andq128 overwritten byspecifiedIDs; suffix129+ unchanged','p_ids':[187,535,186,209,510,15,64],'q_ids':[187,535],'score_start':129,'scored_targets':'originalIDs130..511 (382targets)','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,mode='clean',pos=None,edges=None,bl=None,keep=False):
  probe.mode=mode;probe.positions=pos or [];probe.edges=edges or [];z,_,ll=r.evaluate(x,baseline=bl,keep=keep);cap={k:a.clone() for k,a in probe.cache.items()};cap['potential_pulse']=cap['activation'].float()*probe.scalar;cap['applied_pulse']=cap['potential_pulse'].clone()
  if mode!='clean':cap['applied_pulse'][pos]=0
  return z,ll,cap
 def record(g,t,i,x,z,b,cap,spec):
  r.record(g,t,i,x,z,b,spec,129);raw[f'{g}__{t}__{i}']=cap;obs.append({'group':g,'trial':t,'window':i,'p_pulse':float(cap['potential_pulse'][64]),'q_pulse':float(cap['potential_pulse'][128]),'source_pulse':float(cap['potential_pulse'][0]),'token_sha256':tokenhash(x)})
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    base=xx.clone();prefix=base[:129];prefix[(prefix==187)|(prefix==535)]=209;base[0]=510
    for p_id in (187,535,186,209,510,15,64):
     for q_id in (187,535):
      x=base.clone();x[64]=p_id;x[128]=q_id;assert torch.equal(x[129:],xx[129:]);g=f'newline_{c}_p{p_id}_q{q_id}';b,bl,bc=ev(x,keep=True);record(g,'clean',i,x,b,b,bc,{'p_id':p_id,'q_id':q_id})
      if p_id in (209,187,535):
       for t,mode,pos in [('delete_p','row',[64]),('delete_q','row',[128]),('delete_both','row',[64,128]),('column_q','column',[128])]:
        z,_,cap=ev(x,mode,pos,bl=bl);assert torch.equal(cap['activation'],bc['activation']);record(g,t,i,x,z,b,cap,{'mode':mode,'positions':pos,'p_id':p_id,'q_id':q_id})
      if p_id==q_id==187:
       for t,edges in [('exclude_L0',[(0,128,64)]),('exclude_L1',[(1,128,64)]),('exclude_both',[(0,128,64),(1,128,64)]),('sham_L0',[(0,128,63)])]:
        z,_,cap=ev(x,edges=edges,bl=bl);record(g,t,i,x,z,b,cap,{'edges':edges,'p_id':p_id,'q_id':q_id})
      del bl
    r.checkpoint();torch.save(raw,out/'probe_traces.pt');savej(out/'pulse_observations.json',obs);log(f'Newline history {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:probe.close();r.close()
