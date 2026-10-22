"""Fresh source-set K/V factorial for Llama; see LLAMA_SOURCE_KV.md."""
import argparse,itertools,json,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,th,modelhash,Probe,Runner
from fresh_inputs import prepare_fresh

class KV:
 def __init__(self,m,start):
  self.start=start;self.patch={};self.mask='native';self.cache={};self.queries=None;self.errors=[];self.handles=[];self.masks={}
  for i,layer in enumerate(m.model.layers):
   if i<start:continue
   att=layer.self_attn
   for k in ('k','v'):self.handles.append(getattr(att,k+'_proj').register_forward_hook(lambda m,a,o,i=i,k=k:self.proj(i,k,o)))
   self.handles.append(att.register_forward_pre_hook(self.attn,with_kwargs=True))
  self.handles.append(m.model.layers[start].self_attn.q_proj.register_forward_hook(self.query))
 def proj(self,i,k,o):
  if (i,k) in self.patch:
   old=o;o=o.clone();donor=self.patch[i,k].to(o.device);o[0,0]=donor;assert torch.equal(o[:,1:],old[:,1:]);self.errors.append(float((o[0,0].float()-donor.float()).abs().max()))
  self.cache.setdefault(i,{})[k]=o[0,0].detach().cpu().clone();return o
 def query(self,m,a,o):self.queries=o.detach().cpu().clone()
 def attn(self,m,args,kwargs):
  if self.mask=='native':return None
  x=kwargs.get('hidden_states',args[0] if args else None);assert x is not None;n=x.shape[1];old=kwargs.get('attention_mask');assert old is None or old.shape[-2:]==(n,n)
  key=(n,self.mask,x.dtype,x.device)
  if key not in self.masks:
   mask=torch.zeros((1,1,n,n),dtype=x.dtype,device=x.device);mask.masked_fill_(torch.ones((n,n),device=x.device,dtype=torch.bool).triu(1),float('-inf'))
   if self.mask=='blocked':mask[0,0,1:,0]=float('-inf')
   self.masks[key]=mask
  kwargs=dict(kwargs);mask=self.masks[key]
  if old is not None:mask=mask.masked_fill(~old,float('-inf')) if old.dtype==torch.bool else old+mask
  kwargs['attention_mask']=mask;return args,kwargs
 def reset(self,patch=None,mask='native'):self.patch=patch or {};self.mask=mask;self.cache={};self.queries=None;self.errors=[]
 def close(self):
  for h in self.handles:h.remove()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists();parent=ROOT/'llama_run';old=json.loads((parent/'manifest.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256'];s=json.loads((parent/'selection.json').read_text())['selected'];dump(out/'selection.json',s);start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['llama']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True);items,dm=prepare_fresh(cfg,tok,parent,out)
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'common_sha256':sha(ROOT/'run_transfer.py'),'fresh_inputs_sha256':sha(ROOT/'fresh_inputs.py'),'protocol_sha256':sha(ROOT/'LLAMA_SOURCE_KV.md'),'selection_sha256':sha(out/'selection.json'),'parent_manifest_sha256':sha(parent/'manifest.json'),'model_id':old['model_id'],'revision':old['revision'],'dtype':'bfloat16','attention':'sdpa','seed':20261021};dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats();m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True);m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()};p=Probe(m,'llama');p.selected=s;p.capture_all=False;kv=KV(m,s['layer']+1);run=Runner(m,p,out);donors={};patch_checks=[]
 def ev(x,world,patch=None,mask='native',bl=None,keep=False):
  p.setpatch({'C':None,'R':'row','N':'column'}[world]);kv.reset(patch,mask);z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);d={'kv':{l:{k:v.clone() for k,v in zz.items()} for l,zz in kv.cache.items()},'queries':kv.queries.clone()};return z,ll,cap,norms,d
 def flat(d,keys='kv'):return {(l,k):v for l,dd in d['kv'].items() for k,v in dd.items() if k in keys}
 try:
  for item in items:
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'panel':'kv','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity};key=f"kv__{item['corpus']}__{item['window']}__512__{identity}";refs={};ds={};b,bl,bc,bn,bd=ev(x,'C',keep=True)
    for world in 'CRN':
     if world=='C':z,cap,norms,d=b,bc,bn,bd
     else:z,_,cap,norms,d=ev(x,world,bl=bl)
     refs[world]=z;ds[world]=d;run.rec(key+'__'+world,{**base,'trial':world},x,z,b,cap,norms)
    for world in 'RN':
     ch={**base,'trial':'first_queries_'+world,'query_later_exact':torch.equal(ds[world]['queries'][:,1:],ds['C']['queries'][:,1:])};run.checks.append(ch);assert ch['query_later_exact']
    donors[key]=ds
    for recipient in 'RN':
     for kd,vd in itertools.product('CRN',repeat=2):
      patch={**flat(ds[kd],'k'),**flat(ds[vd],'v')};z,_,cap,norms,d=ev(x,recipient,patch,bl=bl);trial=f'{recipient}_K{kd}_V{vd}';run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);patch_checks.append({**base,'trial':trial,'patches':len(kv.errors),'max_error':max(kv.errors)})
      if kd==vd:
       ch={**base,'trial':trial,'loss_exact':torch.equal(z['loss'][1:],refs[kd]['loss'][1:]),'pred_exact':torch.equal(z['pred'][1:],refs[kd]['pred'][1:])};run.checks.append(ch);assert ch['loss_exact'] and ch['pred_exact'],ch
    blocked_refs={}
    for world in 'CRN':
     for fixed in (False,True):
      patch={k:torch.zeros_like(v) for k,v in flat(ds['C'],'v').items()}
      if fixed:patch.update(flat(ds['C'],'k'))
      z,_,cap,norms,d=ev(x,world,patch,bl=bl);trial=world+'_zeroV_'+('cleanK' if fixed else 'currentK');run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);patch_checks.append({**base,'trial':trial,'patches':len(kv.errors),'max_error':max(kv.errors)})
     maskrefs={}
     for mask in ('causal','blocked'):
      z,_,cap,norms,d=ev(x,world,mask=mask,bl=bl);trial=world+'_mask_'+mask;run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);maskrefs[mask]=z
     blocked_refs[world]=maskrefs['blocked']
     if world!='C':
      ch={**base,'trial':world+'_blocked_source_closure','loss_exact':torch.equal(maskrefs['blocked']['loss'][1:],blocked_refs['C']['loss'][1:]),'pred_exact':torch.equal(maskrefs['blocked']['pred'][1:],blocked_refs['C']['pred'][1:])};run.checks.append(ch);assert ch['loss_exact'] and ch['pred_exact'],ch
     run.checks.append({**base,'trial':world+'_mask_reference','max_native_loss_drift':float((maskrefs['causal']['loss'][1:]-refs[world]['loss'][1:]).abs().max()),'native_pred_agreement':float(maskrefs['causal']['pred'][1:].eq(refs[world]['pred'][1:]).double().mean())})
    del bl
   run.checkpoint();torch.save(donors,out/'donors.pt');dump(out/'patch_checks.json',patch_checks);print('KV_DONE',item['corpus'],item['window'],run.count,flush=True)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'];dump(out/'manifest.json',meta);print('COMPLETE',meta['forward_passes'],meta['wall_seconds'],flush=True)
 finally:kv.close();p.close()
if __name__=='__main__':main()
