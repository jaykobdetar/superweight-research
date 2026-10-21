"""Joint contribution factorial with fresh held-out passages; see GPT2_GROUPS.md."""
import argparse,itertools,json,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Probe,Runner
from fresh_inputs import prepare_fresh

class Groups(Probe):
 def __init__(self,m):
  self.kindpatch='clean';self.alt=None;self.cols=[];self.coords=[];super().__init__(m,'gpt2');self.capture_all=False
 def setup(self,selected):
  self.selected=selected;self.handles.append(self.layers[selected['layer']].register_forward_hook(self.block))
 def set(self,kind,cols=None,coords=None):
  self.kindpatch=kind;self.cols=cols or [];self.coords=coords or [];self.alt=None;self.patch=None
  if kind in ('row','column','oracle'):
   w=self.downs[self.selected['layer']].weight.detach().clone()
   for c in self.cols:
    if kind=='row':w[c,self.selected['row']]=0
    elif kind=='column':w[c,:]=0
   self.alt=w
 def down(self,i,m,a,o):
  old=o;o=super().down(i,m,a,o)
  if self.selected is None or i!=self.selected['layer']:return o
  if self.alt is not None:
   zz=torch.addmm(m.bias,a[0].reshape(-1,a[0].shape[-1]),self.alt).view_as(o);o=o.clone()
   if self.kindpatch=='row':o[0,0,self.selected['row']]=zz[0,0,self.selected['row']]
   else:o[0,0]=zz[0,0]
  elif self.kindpatch=='mlp_coordinate':o=o.clone();o[0,0,self.selected['row']]=0
  assert torch.equal(o[:,1:],old[:,1:]);self.cache['selected']['output_source_after']=o[0,0].detach().cpu().clone();self.cache['selected']['group_activations']=a[0][0,0,self.group_cols].detach().cpu().clone();return o
 def block(self,m,a,o):
  before=o[0];after=before
  if self.kindpatch=='residual_coordinates':after=before.clone();after[0,0,self.coords]=0
  elif self.kindpatch=='residual_all':after=before.clone();after[0,0]=0
  assert torch.equal(before[:,1:],after[:,1:]);self.cache['selected'].update(block_before=before[0,0].detach().cpu().clone(),block_after=after[0,0].detach().cpu().clone());return (after,)+o[1:]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists();parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256'] and sha(parent/'selection_raw.pt')==old['selection_raw_sha256'];selection=json.loads((parent/'selection.json').read_text());s=selection['selected'];raw=torch.load(parent/'selection_raw.pt',weights_only=True);z=raw['layers'][s['layer']];cols=sorted([c for c in range(len(z['scores'])) if int(z['weight_max_rows'][c])==s['row']],key=lambda c:(-float(z['scores'][c]),c))[:3];other=next(v['row'] for v in selection['top_ten'] if v['row']!=s['row']);group={'selected':s,'columns':cols,'other_row':other,'random_row':selection['controls'][0]['row'],'selection_parent_sha256':sha(parent/'selection.json')};dump(out/'group.json',group)
 start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True);items,dm=prepare_fresh(cfg,tok,parent,out);meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'common_sha256':sha(ROOT/'run_transfer.py'),'fresh_inputs_sha256':sha(ROOT/'fresh_inputs.py'),'protocol_sha256':sha(ROOT/'GPT2_GROUPS.md'),'group_sha256':sha(out/'group.json'),'parent_manifest_sha256':sha(parent/'manifest.json'),'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa','seed':20261021};dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats();m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True);m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()};p=Groups(m);p.setup(s);p.group_cols=cols;run=Runner(m,p,out)
 conditions=[('clean','clean',[],[])]+[(f"{kind}_{''.join(str(i) for i in range(3) if bits[i])}",kind,[cols[i] for i in range(3) if bits[i]],[]) for kind in ('row','column') for bits in itertools.product((0,1),repeat=3) if any(bits)]
 conditions += [('mlp_coordinate','mlp_coordinate',[],[]),('residual_selected','residual_coordinates',[],[s['row']]),('residual_other','residual_coordinates',[],[other]),('residual_both','residual_coordinates',[],[s['row'],other]),('residual_random','residual_coordinates',[],[group['random_row']]),('residual_all','residual_all',[],[])];dump(out/'conditions.json',conditions)
 try:
  for item in items:
   for identity in ('natural','bos'):
    x=item['tokens'].clone()
    if identity=='bos':x[0]=dm['literals']['start_special']['id']
    base={'panel':'groups','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity};key=f"groups__{item['corpus']}__{item['window']}__512__{identity}";p.set('clean');b,bl,bc,bn=run.ev(x,keep=True)
    for trial,kind,cc,rr in conditions:
     if trial=='clean':z,cap,norms=b,bc,bn
     else:p.set(kind,cc,rr);z,_,cap,norms=run.ev(x,baseline=bl)
     assert torch.equal(cap['selected']['group_activations'],bc['selected']['group_activations']);run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
    if identity=='natural' and item['window']==0:
     for trial,kind in [('repeat','clean'),('oracle','oracle')]:
      p.set(kind);z,_,cap,norms=run.ev(x,baseline=bl);check={**base,'trial':trial,'loss_exact':torch.equal(z['loss'],b['loss']),'pred_exact':torch.equal(z['pred'],b['pred'])};run.checks.append(check);assert check['loss_exact'] and check['pred_exact'];run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
    del bl
   run.checkpoint();print('GROUP_DONE',item['corpus'],item['window'],run.count,flush=True)
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'];dump(out/'manifest.json',meta);print('COMPLETE',meta['forward_passes'],meta['wall_seconds'],flush=True)
 finally:p.close()
if __name__=='__main__':main()
