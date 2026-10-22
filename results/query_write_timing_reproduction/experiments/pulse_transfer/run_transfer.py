"""Offline sequential causal-signature transfer. See PROTOCOL.md before use."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,hashlib,json,math,platform,resource,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def modelhash(m):
 h=hashlib.sha256()
 for name,p in m.named_parameters():
  h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def strip_book(s):
 a=s.find('*** START OF');b=s.rfind('*** END OF')
 return s[s.find('\n',a)+1:b if b>=0 else None] if a>=0 else s

def prepare(cfg,tok,out):
 sources={k:Path(v) for k,v in cfg['sources'].items()};texts={};meta={'sources':{},'passages':[],'literals':{}}
 for k,p in sources.items():
  texts[k]='\n\n'.join(Dataset.from_file(str(p))['text']) if k.startswith('wiki_') else p.read_text()
  if k in ('hume','dubois','hobbes'):texts[k]=strip_book(texts[k])
  meta['sources'][k]={'name':p.name,'sha256':sha(p),'processed_text_sha256':hashlib.sha256(texts[k].encode()).hexdigest(),'characters':len(texts[k])}
 texts['optimizers']='\n\n'.join(texts[k] for k in ('sgd','adam','adamw'))
 items=[]
 for split,corpora in [('development',('wiki_test','hume')),('evaluation',('wiki_validation','dubois','hobbes','optimizers'))]:
  for c in corpora:
   s=texts[c];assert len(s)>8192
   for i,f in enumerate((.1,.3,.5,.7)):
    a=math.floor(f*(len(s)-8192));passage=s[a:a+8192];ids=torch.tensor(tok.encode(passage,add_special_tokens=False),dtype=torch.long);assert len(ids)>=512,(c,i,len(ids))
    x=ids[:512].clone();items.append({'split':split,'corpus':c,'window':i,'tokens':x});meta['passages'].append({'split':split,'corpus':c,'window':i,'char_start':a,'char_end':a+8192,'passage_sha256':hashlib.sha256(passage.encode()).hexdigest(),'passage_tokens':len(ids),'window_token_sha256':th(x)})
 for name,lit in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]:meta['literals'][name]={'literal':lit,'ids':tok.encode(lit,add_special_tokens=False)}
 meta['literals']['start_special']={'label':'bos' if tok.bos_token_id is not None else 'eos','id':tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id}
 assert len(meta['literals']['the']['ids'])==1
 torch.save(items,out/'tokens.pt');dump(out/'data_manifest.json',meta);return items,meta

class Probe:
 def __init__(self,m,kind):
  self.model=m;self.kind=kind;self.layers=m.transformer.h if kind=='gpt2' else m.model.layers
  self.downs=[x.mlp.c_proj if kind=='gpt2' else x.mlp.down_proj for x in self.layers]
  self.capture_all=True;self.selected=None;self.patch=None;self.cache={};self.norms={};self.source=0;self.handles=[]
  for i,d in enumerate(self.downs):self.handles.append(d.register_forward_hook(lambda m,a,o,i=i:self.down(i,m,a,o)))
  for i,layer in enumerate(self.layers):
   n=layer.ln_1 if kind=='gpt2' else layer.input_layernorm
   self.handles.append(n.register_forward_hook(lambda m,a,o,i=i:self.norm(i,a,o)))
 def matrix(self,i):
  w=self.downs[i].weight
  return w.T if self.kind=='gpt2' else w
 def down(self,i,m,a,o):
  if self.capture_all:self.cache[i]={'a0':a[0][0,0].detach().cpu().clone(),'out0':o[0,0].detach().cpu().clone()}
  if self.selected is not None and i==self.selected['layer']:
   r,c=self.selected['row'],self.selected['column'];self.cache['selected']={'a0':float(a[0][0,0,c]),'aq':float(a[0][0,128,c]) if a[0].shape[1]>128 else None,'output_source_before':o[0,self.source].detach().cpu().clone()}
   if self.patch is not None:
    state=self.patch;w=state['weight']
    if self.kind=='gpt2':z=torch.addmm(m.bias,a[0].reshape(-1,a[0].shape[-1]),w).view_as(o)
    else:z=F.linear(a[0],w,m.bias)
    o=o.clone()
    if state['mode']=='row':o[0,self.source,state['row']]=z[0,self.source,state['row']]
    else:o[0,self.source]=z[0,self.source]
   self.cache['selected']['output_source_after']=o[0,self.source].detach().cpu().clone()
  return o
 def norm(self,i,a,o):
  if self.selected is not None and i>self.selected['layer']:
   z=o[0,self.source].float();r=self.selected['row'];x=a[0][0,self.source].float()
   self.norms[i]={'input_rms':float(x.square().mean().sqrt()),'output_rms':float(z.square().mean().sqrt()),'output_max_abs':float(z.abs().max()),'target_coordinate':float(z[r]),'target_energy_fraction':float(z[r].square()/z.square().sum().clamp_min(1e-30))}
  return o
 def setpatch(self,mode,row=None,col=None):
  self.patch=None
  if mode is None:return
  i=self.selected['layer'];r=self.selected['row'] if row is None else row;c=self.selected['column'] if col is None else col;w=self.downs[i].weight.detach().clone()
  if mode=='row':
   if self.kind=='gpt2':w[c,r]=0
   else:w[r,c]=0
  elif mode=='column':
   if self.kind=='gpt2':w[c,:]=0
   else:w[:,c]=0
  elif mode!='oracle':raise ValueError(mode)
  self.patch={'weight':w,'mode':mode,'row':r,'column':c}
 def close(self):
  for h in self.handles:h.remove()

class Runner:
 def __init__(self,m,p,out):self.m=m;self.p=p;self.out=out;self.count=0;self.raw={};self.traces={};self.checks=[]
 def ev(self,x,baseline=None,keep=False):
  self.p.cache={};self.p.norms={}
  with torch.inference_mode():
   logits=self.m(x[None].cuda(),use_cache=False).logits[0,:-1];loss=[];kl=[]
   for a in range(0,len(logits),32):
    b=min(a+32,len(logits));lp=logits[a:b].float().log_softmax(-1);loss.append(-lp.gather(-1,x[a+1:b+1,None].cuda()).squeeze(-1).cpu())
    if baseline is None:kl.append(torch.zeros(b-a))
    else:
     lb=baseline[a:b].float().log_softmax(-1);kl.append((lb.exp()*(lb-lp)).sum(-1).cpu())
   z={'loss':torch.cat(loss),'kl':torch.cat(kl),'pred':logits.argmax(-1).cpu()}
  assert all(torch.isfinite(z[k]).all() for k in ('loss','kl'));self.count+=1
  return z,logits if keep else None,dict(self.p.cache),dict(self.p.norms)
 def rec(self,key,meta,x,z,b,cap,norms,score_index=1):
  assert key not in self.raw;self.raw[key]={'token_sha256':th(x),'score_index':score_index,**z};self.traces[key]={'capture':cap.get('selected',{}),'norms':norms}
  zz={k:v[score_index:] for k,v in z.items()};bb={k:v[score_index:] for k,v in b.items()};d=cap.get('selected',{});w=self.p.selected['weight']
  row={**meta,'key':key,'token_sha256':th(x),'input_tokens':len(x),'score_index':score_index,'target_start':score_index+1,'scored_tokens':len(zz['loss']),'nll':float(zz['loss'].double().mean()),'baseline_nll':float(bb['loss'].double().mean()),'agreement':float(zz['pred'].eq(bb['pred']).double().mean()),'kl':float(zz['kl'].double().mean()),'source_contribution':d.get('a0',0)*w,'q_contribution':d.get('aq',0)*w}
  with (self.out/'windows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 def checkpoint(self):
  torch.save(self.raw,self.out/'per_token.pt');torch.save(self.traces,self.out/'traces.pt');dump(self.out/'checks.json',self.checks);dump(self.out/'progress.json',{'forward_passes':self.count,'records':len(self.raw),'utc':time.strftime('%FT%TZ',time.gmtime())})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--model',choices=['gpt2','llama'],required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists();start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
 modelcfg=cfg['models'][args.model];path=Path(modelcfg['path']);meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model':args.model,'model_id':modelcfg['model_id'],'revision':modelcfg['revision'],'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'seed':20261021,'dtype':'float32' if args.model=='gpt2' else 'bfloat16','attention':'sdpa','threads':4,'versions':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__},'checkpoint_files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(path.iterdir()) if p.is_file() and (p.suffix in ('.safetensors','.json') or p.name=='merges.txt')}}
 if args.model=='gpt2':assert meta['checkpoint_files']['model.safetensors']['sha256']=='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
 dump(out/'manifest.json',meta);tok=AutoTokenizer.from_pretrained(path,local_files_only=True);items,dm=prepare(cfg,tok,out)
 torch.cuda.reset_peak_memory_stats();m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32 if args.model=='gpt2' else torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True);m.eval();m.requires_grad_(False);versions={n:p._version for n,p in m.named_parameters()};meta['model_hash_before']=modelhash(m);dump(out/'model_config.json',m.config.to_dict());dump(out/'manifest.json',meta);p=Probe(m,args.model);run=Runner(m,p,out)
 print('MODEL_READY',args.model,flush=True)
 try:
  activations={i:[] for i in range(len(p.layers)-1)};dev=[]
  for item in items:
   if item['split']!='development':continue
   z,_,cap,_=run.ev(item['tokens'][:512]);dev.append({'corpus':item['corpus'],'window':item['window'],'loss':z['loss'],'pred':z['pred'],'token_sha256':th(item['tokens'][:512])})
   for i in activations:activations[i].append(cap[i]['a0'])
  selection={};rank=[]
  for i,aa in activations.items():
   a=torch.stack(aa).float();w=p.matrix(i).detach();mx,rows=w.abs().max(0);scores=a.abs().mean(0)*mx.cpu().float();assert torch.isfinite(a).all() and torch.isfinite(mx).all() and torch.isfinite(scores).all()
   selection[i]={'activations':a,'weight_max_abs':mx.cpu(),'weight_max_rows':rows.cpu(),'scores':scores}
   score_list=scores.tolist();row_list=rows.cpu().tolist();ranked_columns=sorted(range(len(scores)),key=lambda c:(-score_list[c],row_list[c],c))
   for c in ranked_columns[:10]:rank.append({'layer':i,'row':int(rows[c]),'column':c,'score':float(scores[c]),'weight':float(w[int(rows[c]),c])})
  rank.sort(key=lambda z:(-z['score'],z['layer'],z['row'],z['column']));s=rank[0];p.selected=s;p.capture_all=False;w=p.matrix(s['layer']).detach().float();s['column_energy_fraction']=float(w[s['row'],s['column']].square()/w[:,s['column']].square().sum());s['column_norm']=float(w[:,s['column']].norm());controls=[]
  for seed in (20261021,20261022,20261023):
   rng=np.random.default_rng(seed);r=int(rng.integers(w.shape[0]));c=int(rng.integers(w.shape[1]));assert (r,c)!=(s['row'],s['column']);controls.append({'seed':seed,'row':r,'column':c,'weight':float(w[r,c])})
  del w;torch.save({'layers':selection,'development':dev},out/'selection_raw.pt');dump(out/'selection.json',{'selected':s,'top_ten':rank[:10],'controls':controls,'rule':'mean_abs_source0_activation times column_max_abs_weight; all layers except final'});meta['selection_sha256']=sha(out/'selection.json');meta['selection_raw_sha256']=sha(out/'selection_raw.pt');dump(out/'manifest.json',meta);print('SELECTED',json.dumps(s),flush=True)
  evalitems=[x for x in items if x['split']=='evaluation'];the=dm['literals']['the']['ids'][0];special=dm['literals']['start_special'];assert special['id'] is not None
  for item in evalitems:
   for n in (256,512):
    for identity in ('natural','the',special['label']):
     x=item['tokens'][:n].clone()
     if identity=='the':x[0]=the
     elif identity==special['label']:x[0]=special['id']
     base={'panel':'source0','corpus':item['corpus'],'window':item['window'],'length':n,'identity':identity};key=f"source0__{item['corpus']}__{item['window']}__{n}__{identity}";p.source=0;p.setpatch(None);b,bl,bc,bn=run.ev(x,keep=True);run.rec(key+'__clean',{**base,'trial':'clean'},x,b,b,bc,bn)
     if identity=='natural' and item['window']==0:
      for trial in ('repeat','oracle'):
       p.setpatch(None if trial=='repeat' else 'oracle');z,_,cap,norms=run.ev(x,baseline=bl);check={**base,'trial':trial,'loss_exact':torch.equal(z['loss'],b['loss']),'pred_exact':torch.equal(z['pred'],b['pred'])};run.checks.append(check);assert check['loss_exact'] and check['pred_exact'],check;run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
     for control in [None]+controls:
      for mode in ('row','column'):
       trial=mode if control is None else f"random{control['seed']}__{mode}";p.setpatch(mode,None if control is None else control['row'],None if control is None else control['column']);z,_,cap,norms=run.ev(x,baseline=bl);run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
     p.setpatch(None);del bl
   run.checkpoint();print('SOURCE_DONE',item['corpus'],item['window'],run.count,flush=True)
  eligible=all(len(dm['literals'][k]['ids'])==1 for k in ('space','newline','double_newline'));dump(out/'history_eligibility.json',{'eligible':eligible,'reason':'all four literals encode as one token' if eligible else 'newline/space literal encodes to multiple tokens; no substitute'})
  if eligible:
   nl=dm['literals']['newline']['ids'][0];dbl=dm['literals']['double_newline']['ids'][0];sp=dm['literals']['space']['ids'][0]
   for item in evalitems:
    for n in (256,512):
     for variant in ('U','R','R_double'):
      x=item['tokens'][:n].clone();pre=x[:129];pre[(pre==nl)|(pre==dbl)]=sp;x[0]=the;x[64]=sp if variant=='U' else nl;x[128]=dbl if variant=='R_double' else nl;p.source=128;p.setpatch(None);z,_,cap,norms=run.ev(x);base={'panel':'history','corpus':item['corpus'],'window':item['window'],'length':n,'identity':'the','trial':variant};key=f"history__{item['corpus']}__{item['window']}__{n}__{variant}";run.rec(key,base,x,z,z,cap,norms,129)
    run.checkpoint();print('HISTORY_DONE',item['corpus'],item['window'],run.count,flush=True)
  run.checkpoint();meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),wall_seconds=time.time()-start,forward_passes=run.count,records=len(run.raw),model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'];dump(out/'manifest.json',meta);print('COMPLETE',args.model,meta['forward_passes'],meta['wall_seconds'],flush=True)
 finally:p.close()
if __name__=='__main__':main()
