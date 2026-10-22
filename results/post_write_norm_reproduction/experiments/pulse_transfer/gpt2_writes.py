"""Causal factorial for source-coordinate restoration; see GPT2_WRITES.md."""
import argparse,itertools,json,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Probe,Runner
from fresh_inputs_v2 import prepare_fresh

class Writes(Probe):
 def __init__(self,m,s,coords):
  super().__init__(m,'gpt2');self.selected=s;self.capture_all=False;self.coords=coords;self.active_coords=coords;self.E=self.A=self.M=False;self.special=None;self.vectors={};self.locality=0;self.zero_checks=0
  for i,layer in enumerate(self.layers):
   self.handles.append(layer.register_forward_hook(lambda m,a,o,i=i:self.block(i,o)))
   if i>s['layer']:self.handles.append(layer.attn.c_proj.register_forward_hook(lambda m,a,o,i=i:self.attn(i,o)))
 def reset(self,E=False,A=False,M=False,coords=None,special=None):self.E,self.A,self.M=E,A,M;self.active_coords=coords if coords is not None else self.coords;self.special=special;self.vectors={};self.locality=self.zero_checks=0
 def local(self,b,a):assert torch.equal(b[:,1:],a[:,1:]);self.locality+=1
 def save(self,i,name,b,a):self.vectors.setdefault(i,{})[name]={'before':b[0,0].detach().cpu().clone(),'after':a[0,0].detach().cpu().clone()}
 def down(self,i,m,a,o):
  o=super().down(i,m,a,o)
  if self.selected is None or i<=self.selected['layer']:return o
  before=o
  if self.M:o=o.clone();o[0,0,self.active_coords]=0
  self.local(before,o);self.save(i,'mlp',before,o);return o
 def attn(self,i,o):
  before=o
  if self.A:o=o.clone();o[0,0,self.active_coords]=0
  self.local(before,o);self.save(i,'attention',before,o);return o
 def block(self,i,o):
  before=o[0];after=before
  if i==self.selected['layer']:
   if self.special=='zero_all':after=before.clone();after[0,0]=0
   elif self.E:
    after=before.clone();after[0,0,self.active_coords]=0
    if self.special=='rescale':
     r=after[0,0].norm();assert r>0;after[0,0]*=before[0,0].norm()/r
   elif self.special=='identity':after=before.clone()
  self.local(before,after);self.save(i,'residual',before,after)
  if self.E and self.A and self.M and i>=self.selected['layer']:assert torch.count_nonzero(after[0,0,self.active_coords])==0;self.zero_checks+=1
  return (after,)+o[1:]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists();parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text());sel=json.loads((parent/'selection.json').read_text());group=json.loads((ROOT/'gpt2_groups_run/group.json').read_text());s=sel['selected'];coords=[s['row'],group['other_row']];random=[r['row'] for r in sel['controls'][:2]];assert len(set(coords+random))==4;start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True);items,dm=prepare_fresh(cfg,tok,parent,out,additional_parents=(ROOT/'gpt2_groups_run',));meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'common_sha256':sha(ROOT/'run_transfer.py'),'fresh_inputs_sha256':sha(ROOT/'fresh_inputs_v2.py'),'protocol_sha256':sha(ROOT/'GPT2_WRITES.md'),'parent_manifest_sha256':sha(parent/'manifest.json'),'group_sha256':sha(ROOT/'gpt2_groups_run/group.json'),'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa','seed':20261021,'coordinates':coords,'random_coordinates':random};dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats();m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True);m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()};p=Writes(m,s,coords);run=Runner(m,p,out);ops=[]
 conditions=[(f'E{int(e)}A{int(aa)}M{int(mm)}',e,aa,mm,None,None) for e,aa,mm in itertools.product((False,True),repeat=3)]+[('random_E1A1M1',True,True,True,random,None),('rescale_E1',True,False,False,None,'rescale'),('zero_all',False,False,False,None,'zero_all')];dump(out/'conditions.json',conditions)
 def ev(x,E=False,A=False,M=False,coords=None,special=None,bl=None,keep=False):
  p.reset(E,A,M,coords,special);z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);cap['selected']['writers']=p.vectors;return z,ll,cap,norms
 try:
  for item in items:
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'panel':'writes','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity};key=f"writes__{item['corpus']}__{item['window']}__512__{identity}";b,bl,bc,bn=ev(x,keep=True)
    for trial,E,A,M,cc,sp in conditions:
     if trial=='E0A0M0':z,cap,norms=b,bc,bn
     else:z,_,cap,norms=ev(x,E,A,M,cc,sp,bl=bl)
     run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);ops.append({**base,'trial':trial,'locality_checks':p.locality,'zero_checks':p.zero_checks})
    if identity=='natural' and item['window']==0:
     for trial,sp in [('repeat',None),('identity','identity')]:
      z,_,cap,norms=ev(x,special=sp,bl=bl);check={**base,'trial':trial,'loss_exact':torch.equal(z['loss'],b['loss']),'pred_exact':torch.equal(z['pred'],b['pred'])};run.checks.append(check);assert check['loss_exact'] and check['pred_exact'];run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
    del bl
   run.checkpoint();dump(out/'operation_checks.json',ops);print('WRITES_DONE',item['corpus'],item['window'],run.count,flush=True)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'];dump(out/'manifest.json',meta);print('COMPLETE',meta['forward_passes'],meta['wall_seconds'],flush=True)
 finally:p.close()
if __name__=='__main__':main()
