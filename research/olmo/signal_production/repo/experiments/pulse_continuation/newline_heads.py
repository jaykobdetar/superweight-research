"""Select a newline transmission head on development; test K/V/Q held out."""
import json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer
from transformers.models.olmo.modeling_olmo import rotate_half
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from newline_history import Probe

def edited(xx,pid=187,qid=187):
 x=xx.clone();a=x[:129];a[(a==187)|(a==535)]=209;x[0]=510;x[64]=pid;x[128]=qid
 assert torch.equal(x[129:],xx[129:]);return x

class PathProbe(Probe):
 def __init__(self,model):
  super().__init__(model);self.head_edges=[];self.context_patch=None;self.current={}
  a=model.model.layers[0].self_attn
  for k in 'qkv':self.handles.append(getattr(a,k+'_proj').register_forward_hook(lambda m,a,o,k=k:self.projection(k,o)))
  self.handles.append(a.o_proj.register_forward_pre_hook(self.context0))
  for k in ('gate','up'):self.handles.append(getattr(model.model.layers[1].mlp,k+'_proj').register_forward_hook(lambda m,a,o,k=k:self.factor(k,o)))
 def factor(self,k,o):self.cache[k]=o[0,[0,self.p,self.q],1710].detach().cpu().clone();return o
 def attn(self,l,args,k):
  if l==0:self.rope=k['position_embeddings']
  args,k=super().attn(l,args,k)
  if l==0 and self.head_edges:
   m=k['attention_mask'].expand(-1,16,-1,-1).clone()
   for h,dest,source in self.head_edges:m[:,h,dest,source]=-torch.inf
   k['attention_mask']=m
  return args,k
 def projection(self,k,o):
  self.current[k]=o;self.cache['L0_'+k]=o[0,[0,self.p,self.q]].detach().cpu().clone();return o
 def context0(self,m,a):
  o=a[0];self.cache['L0_context']=o[0,[0,self.p,self.q]].detach().cpu().clone();n=o.shape[1]
  raw={k:v.clone() if self.context_patch else v for k,v in self.current.items()}
  patch=self.context_patch
  if patch:
   for h in patch['heads']:
    sl=slice(h*128,(h+1)*128)
    for k in patch['kinds']:
     pos=self.q if k=='q' else self.p;idx=2 if k=='q' else 1
     raw[k][:,pos,sl]=patch['donor']['L0_'+k][idx,sl].to(o.device)
  q,k,v=[raw[t].clamp(-8,8).view(1,n,16,128).transpose(1,2) for t in 'qkv'];co,si=self.rope;co=co.unsqueeze(1);si=si.unsqueeze(1);q=(q*co+rotate_half(q)*si).to(q.dtype);k=(k*co+rotate_half(k)*si).to(k.dtype)
  mask=self.masks[n][:,:,self.q:self.q+1]
  if self.head_edges:
   mask=mask.expand(-1,16,-1,-1).clone()
   for h,dest,source in self.head_edges:
    if dest==self.q:mask[:,h,0,source]=-torch.inf
  score=(q[:,:,self.q:self.q+1].float()@k.float().transpose(-1,-2))/(128**.5)+mask.float();prob=score.softmax(-1)
  self.cache['L0_mass_0_p_q']=prob[0,:,0,[0,self.p,self.q]].detach().cpu().clone()
  if patch:
   z=F.scaled_dot_product_attention(q[:,:,self.q:self.q+1],k,v,attn_mask=mask,dropout_p=0.0,is_causal=False).transpose(1,2).reshape(1,1,2048)
   y=o.clone()
   for h in patch['heads']:
    sl=slice(h*128,(h+1)*128);y[:,self.q,sl]=z[:,0,sl]
   self.cache['L0_patched_context_q']=y[0,self.q].detach().cpu().clone()
   return (y,)+a[1:]

def prepare(ctx,out):
 old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');dev=[(c,i,old['corpora'][c][s:s+512].clone()) for c in ('wiki','hume') for i,s in enumerate(old['starts'][c]['development'])]
 prior=json.loads((ROOT/'confirmation_run/data_manifest.json').read_text());later=json.loads((ROOT/'later_uses_run/data_manifest.json').read_text());tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True);sets={};dm={};seen=set()
 for c,d in prior.items():
  ps=[Path(s['path']) for s in d['sources']]
  for p,s in zip(ps,d['sources']):assert sha(p)==s['sha256'],p
  text='\n\n'.join(Dataset.from_file(str(ps[0]))['text']) if c=='wiki_validation' else '\n\n'.join(p.read_text() for p in ps)
  ids=torch.tensor(tok.encode(text,add_special_tokens=False));assert len(ids)==d['token_count'];starts=[4096,5120,6144,7168] if c=='optimizers' else [16384,18432,20480,22528];excluded=d['calibration_starts']+d['evaluation_starts']+later[c]['evaluation_starts']
  assert all(s+512<=e or e+512<=s for s in starts for e in excluded);assert all(s+512<=len(ids) for s in starts)
  sets[c]=[ids[s:s+512].clone() for s in starts]
  for x in sets[c]:h=tokenhash(x);assert h not in seen;seen.add(h)
  dm[c]={**d,'evaluation_starts':starts,'excluded_starts':excluded,'selection':'separate old Wiki/Hume development only'}
 torch.save({'development':dev,'evaluation':sets},out/'tokens.pt');savej(out/'data_manifest.json',dm);return dev,sets

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();dev,sets=prepare(ctx,out);r=Run(ctx['model'],out);p=PathProbe(ctx['model']);raw={};obs=[];checks=[];screen=[]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'NEWLINE_HEADS.md'),'model_hash_before':ctx['original_hash'],'score_start':129,'scored_targets':'original IDs130..511;382targets','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,heads=None,patch=None,delete=None,bl=None,keep=False):
  p.mode='row' if delete else 'clean';p.positions=delete or [];p.head_edges=heads or [];p.context_patch=patch;p.cache={}
  z,_,ll=r.evaluate(x,baseline=bl,keep=keep);cap={k:v.clone() for k,v in p.cache.items()};cap['potential_pulse']=cap['activation'].float()*p.scalar;cap['applied_pulse']=cap['potential_pulse'].clone()
  if delete:cap['applied_pulse'][delete]=0
  return z,ll,cap
 def record(g,t,i,x,z,b,cap,spec):
  r.record(g,t,i,x,z,b,spec,129);raw[f'{g}__{t}__{i}']=cap;obs.append({'group':g,'trial':t,'window':i,'p_pulse':float(cap['potential_pulse'][64]),'q_pulse':float(cap['potential_pulse'][128]),'source_pulse':float(cap['potential_pulse'][0]),'token_sha256':tokenhash(x)})
 try:
  for c,i,xx in dev:
   x=edited(xx);g='dev_'+c;b,bl,bc=ev(x,keep=True);record(g,'clean',i,x,b,b,bc,{});z,_,ac=ev(x,[(h,128,64) for h in range(16)],bl=bl);record(g,'edge_all',i,x,z,b,ac,{'heads':'all'})
   gap=float(ac['potential_pulse'][128]-bc['potential_pulse'][128]);assert abs(gap)>1
   for h in range(16):
    z,_,hc=ev(x,[(h,128,64)],bl=bl);record(g,'edge_H'+str(h),i,x,z,b,hc,{'head':h});screen.append({'corpus':c,'window':i,'head':h,'fraction':float(hc['potential_pulse'][128]-bc['potential_pulse'][128])/gap})
   del bl
  ranking=sorted([{'head':h,'mean_fraction':sum(a['fraction'] for a in screen if a['head']==h)/len(dev)} for h in range(16)],key=lambda a:-a['mean_fraction']);selected=ranking[0]['head'];control=0 if selected!=0 else 1;savej(out/'selection.json',{'selected':selected,'control':control,'ranking':ranking,'screen':screen,'frozen_before_heldout':True});log(f'Newline head selection H{selected}, control H{control}')
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    xs={'U':edited(xx,209,187),'R':edited(xx,187,187),'Q535':edited(xx,187,535)};bases={};caps={};logs={}
    for world,x in xs.items():
     b,bl,bc=ev(x,keep=world!='Q535');bases[world]=b;caps[world]=bc;logs[world]=bl;record(f'heads_{c}_{world}','clean',i,x,b,b,bc,{'world':world})
    assert torch.equal(caps['U']['L0_q'][2],caps['R']['L0_q'][2])
    for k in 'qkv':assert torch.equal(caps['U']['L0_'+k][[0,2]],caps['R']['L0_'+k][[0,2]])
    checks.append({'corpus':c,'window':i,'same_q_query':True,'same_source_and_q_projections':True})
    for world in ('U','R'):
     x=xs[world];b=bases[world];bl=logs[world];g=f'heads_{c}_{world}';donor=caps['R' if world=='U' else 'U']
     trials=[('edge_selected',[(selected,128,64)]),('edge_control',[(control,128,64)]),('edge_all',[(h,128,64) for h in range(16)]),('sham_selected',[(selected,128,63)])]
     for t,edges in trials:
      z,_,cap=ev(x,edges,bl=bl);record(g,t,i,x,z,b,cap,{'head_edges':edges})
     for label,h in [('selected',selected),('control',control)]:
      for kinds in ('','k','v','kv','q'):
       d=caps['Q535'] if kinds=='q' else donor;patch={'heads':[h],'kinds':kinds,'donor':d};z,_,cap=ev(x,patch=patch,bl=bl);t=label+'_'+(kinds or 'oracle');record(g,t,i,x,z,b,cap,{'head':h,'kinds':kinds,'donor':'Q535' if kinds=='q' else ('R' if world=='U' else 'U')})
     if world=='R':
      for edge in (False,True):
       for name,pos in [('p',[64]),('q',[128]),('both',[64,128])]:
        edges=[(selected,128,64)] if edge else [];z,_,cap=ev(x,heads=edges,delete=pos,bl=bl);record(g,('edge_' if edge else '')+'delete_'+name,i,x,z,b,cap,{'head_edges':edges,'row_delete_positions':pos})
    del logs,bases,caps
    r.checkpoint();torch.save(raw,out/'probe_traces.pt');savej(out/'pulse_observations.json',obs);savej(out/'checks.json',checks);log(f'Newline heads {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:p.close();r.close()
