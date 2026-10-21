#!/usr/bin/env python3
"""Bounded, offline causal follow-up. No model weights are mutated."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse, contextlib, csv, hashlib, json, math, resource, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from pulse_interventions import PulseIntervention
from single_projection_core import hash_model

ROOT=Path(__file__).resolve().parent
OLD=Path('/EXTERNAL/original_experiment')
MODEL=Path('/EXTERNAL/hf_cache/models--allenai--OLMo-1B-0724-hf/snapshots/d7cbab742d80589e714b1a2d7f838dcd21cbe143')
WIKI=Path('/EXTERNAL/hf_cache/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow')
HUME=Path('/EXTERNAL/home/Downloads/philosophy-course-main/collection/philosophy-study-collection/texts/gutenberg/pg9662-an-enquiry-concerning-human-understanding.txt')
CAND=[(2,15),(12,5),(15,11)]
CTRL=[(2,0),(12,0),(15,0)]
LAYERS=[2,12,15]
SEED=20261021
ORIGINAL_PULSE=-250.0545883178711

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def savej(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def tokenhash(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def log(s):print(time.strftime('%H:%M:%S',time.gmtime()),s,flush=True)

def prepare(out):
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 texts={'wiki':'\n\n'.join(Dataset.from_file(str(WIKI))['text']), 'hume':HUME.read_text()}
 # Remove Gutenberg legal/front/back wrappers using explicit markers, retain actual work.
 h=texts['hume']; start=h.find('*** START OF'); stop=h.rfind('*** END OF')
 if start>=0:h=h[h.find('\n',start)+1:stop if stop>=0 else None]
 texts['hume']=h
 ids={k:torch.tensor(tok.encode(v,add_special_tokens=False),dtype=torch.long) for k,v in texts.items()}
 begins={}
 for k,x in ids.items():
  assert len(x)>60000,(k,len(x))
  begins[k]={'calibration':list(range(0,16384,1024)), 'development':list(range(18432,26624,1024)),
             'evaluation':np.linspace(32768,len(x)-2048,12,dtype=int).tolist()}
  assert min(np.diff(begins[k]['evaluation']))>=2048
 special={'natural':None,'eos':tok.eos_token_id}
 for text,name in [('The','the'),('.','period')]:
  val=tok.encode(text,add_special_tokens=False);assert len(val)==1,(text,val);special[name]=val[0]
 torch.save({'corpora':ids,'starts':begins,'start_tokens':special},out/'tokens.pt')
 meta={'source_paths':{'wiki':str(WIKI),'hume':str(HUME)},'source_sha256':{'wiki':sha(WIKI),'hume':sha(HUME)},
       'text_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in texts.items()},
       'token_count':{k:len(v) for k,v in ids.items()},'starts':begins,'start_tokens':special,
       'tokenization':'pinned tokenizer; add_special_tokens=False; ID0 replacement after tokenization',
       'gutenberg_wrappers_removed':start>=0}
 savej(out/'data_manifest.json',meta)
 return ids,begins,special

class Hooks:
 """Capture preclip K/V, pre-o_proj contexts, and block residuals, then patch."""
 def __init__(self,model):
  self.model=model;self.capture=False;self.cache={};self.patch={};self.handles=[]
  for l in LAYERS:
   a=model.model.layers[l].self_attn
   for kind in ('k','v'):
    self.handles.append(getattr(a,kind+'_proj').register_forward_hook(lambda m,i,o,l=l,k=kind:self.proj(l,k,o)))
   self.handles.append(a.o_proj.register_forward_pre_hook(lambda m,i,l=l:(self.context(l,i[0]),)))
   self.handles.append(model.model.layers[l].register_forward_hook(lambda m,i,o,l=l:self.residual(l,o)))
 def reset(self,capture=False,patch=None):self.capture=capture;self.cache={};self.patch=patch or {}
 def proj(self,l,k,o):
  p=self.patch
  if p.get('type') in ('kv','constant_kv') and k in p['kinds']:
   targets=[h for ll,h in p['heads'] if ll==l]
   if targets:
    o=o.clone();pos=slice(0,1) if p.get('positions','zero')=='zero' else slice(None)
    for h in targets:
     sl=slice(h*128,(h+1)*128)
     source=p['donor'][(l,k)][...,pos,sl].to(o.device) if p['type']=='kv' else p['constants'][(l,k,h)].to(o.device).view(1,1,128)
     o[...,pos,sl]=source
  if self.capture:self.cache[(l,k)]=o.detach().cpu().clone()
  return o
 def context(self,l,o):
  p=self.patch
  if p.get('type') in ('context','constant_context'):
   targets=[h for ll,h in p['heads'] if ll==l]
   if targets:
    o=o.clone()
    for h in targets:
     sl=slice(h*128,(h+1)*128)
     if p['type']=='context':o[...,sl]=p['donor'][(l,'context')][...,sl].to(o.device)
     else:
      c=p['constants'][(l,'context',h)].to(o.device)
      o[:,0,sl]=c[0];o[:,1:,sl]=c[1]
  if self.capture:self.cache[(l,'context')]=o.detach().cpu().clone()
  return o
 def residual(self,l,o):
  p=self.patch
  if p.get('type')=='direction' and p['layer']==l:
   o=o.clone();d=p['delta'][0,9:].to(o.device).float();mu=p['mean'].to(o.device)
   mode=p['mode'];v=p['vectors'].to(o.device)
   if mode=='full':a=d
   elif mode=='mean':a=mu.expand_as(d)
   elif mode=='mean_pc8':a=mu+((d-mu)@v[:,:8])@v[:,:8].T
   else:
    n=int(mode.replace('pc','').replace('random',''));u=v[:,:n]
    a=(d@u)@u.T
    if mode.startswith('random'):
     reference=p['reference'].to(o.device);ref=(d@reference)@reference.T
     a=a*(ref.norm(dim=-1,keepdim=True)/a.norm(dim=-1,keepdim=True).clamp_min(1e-12))
   o[:,9:]=(o[:,9:].float()+p['sign']*a).to(o.dtype)
  if self.capture:self.cache[(l,'residual')]=o.detach().cpu().clone()
  return o
 def close(self):
  for h in self.handles:h.remove()

class Runner:
 def __init__(self,out,model,pulse,hooks):
  self.out=out;self.model=model;self.pulse=pulse;self.hooks=hooks;self.count=0;self.specs={};self.raw={}
 def run(self,x,state='clean',capture=False,patch=None,amplitude=ORIGINAL_PULSE,mask='first',offset=0):
  self.pulse.configure({'mode':'baseline' if state=='clean' else 'ablate','selected':[(1,1764,1710)],
    'ablation_positions':mask,'pulse_values':{1:amplitude} if state=='pulse' else {}})
  self.hooks.reset(capture,patch)
  ids=x[None].cuda();pos=torch.arange(len(x),device='cuda')[None]+offset
  with torch.inference_mode():
   logits=self.model(input_ids=ids,position_ids=pos,use_cache=False).logits
   # Chunked log-softmax limits transient FP32 vocabulary memory for 1024 tokens.
   nll=[];kl=[];pred=logits[0,:-1].argmax(-1).cpu()
   for s in range(0,len(x)-1,128):
    e=min(s+128,len(x)-1)
    nll.append(F.cross_entropy(logits[0,s:e].float(),ids[0,s+1:e+1],reduction='none').cpu())
   loss=torch.cat(nll)
  assert torch.isfinite(loss).all()
  self.count+=1
  return {'loss':loss,'pred':pred},self.hooks.cache
 def record(self,group,trial,index,x,result,baseline,spec):
  key=group+'__'+trial
  if key not in self.specs:self.specs[key]=spec
  self.raw.setdefault(key,[]).append({'window':index,'token_sha256':tokenhash(x),**result})
  rec={'group':group,'trial':trial,'window':index,'input_tokens':len(x),'token_sha256':tokenhash(x),
    'nll':result['loss'].double().mean().item(),'baseline_nll':baseline['loss'].double().mean().item(),
    'agreement':result['pred'].eq(baseline['pred']).double().mean().item()}
  for name,sl in [('first127',slice(0,127)),('later9',slice(9,None)),('suffix512',slice(512,None))]:
   a=result['loss'][sl];b=baseline['loss'][sl]
   if len(a):rec[name+'_delta_nll']=(a.double()-b.double()).mean().item()
  with (self.out/'windows.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
  return rec
 def checkpoint(self):
  torch.save(self.raw,self.out/'per_token.pt');savej(self.out/'conditions.json',self.specs)
  savej(self.out/'progress.json',{'forward_passes':self.count,'groups_completed':len(self.raw),'utc':time.strftime('%FT%TZ',time.gmtime())})

def window(data,starts,corpus,split,i,length=512,identity=None):
 s=starts[corpus][split][i];x=data[corpus][s:s+length].clone()
 if identity is not None:x[0]=identity
 return x

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'run'));args=p.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 if (out/'windows.jsonl').exists():raise FileExistsError('Refuse to overwrite existing results')
 torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED)
 started=time.time();data,starts,identities=prepare(out)
 manifest={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model':str(MODEL),'revision':MODEL.name,
  'seed':SEED,'dtype':'float16','attention':'native SDPA','cpu_threads':4,'candidates':CAND,'controls':CTRL,
  'code_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'design_sha256':sha(ROOT/'DESIGN.md'),
  'model_files_sha256':{p.name:sha(p) for p in MODEL.glob('*.safetensors')},
  'source_reports':{p.name:sha(p) for p in (ROOT/'provenance').glob('*report.html')},
  'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'resource_policy':'single GPU model, four CPU threads; offline'}
 savej(out/'manifest.json',manifest)
 log('Loading one pinned model')
 model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
 model.eval();model.requires_grad_(False)
 before=hash_model(model);manifest['model_hash_before']=before;savej(out/'manifest.json',manifest)
 versions={n:p._version for n,p in model.named_parameters()}
 hooks=Hooks(model)
 with PulseIntervention({1:model.model.layers[1].mlp.down_proj},[(1,1764,1710)]) as pulse:
  r=Runner(out,model,pulse,hooks)
  # Stage 1: exact old anchors, no new sweeps.
  old=torch.load(OLD/'main_run/tokens.pt',weights_only=True)
  expected={}
  for line in (OLD/'pulse_run/window_results.jsonl').read_text().splitlines():
   z=json.loads(line);expected[(z['trial'],z['window'])]=z['nll']
  errors={t:[] for t in ['baseline','first:ablate:all','first:pulse:0:1']}
  for i,x in enumerate(old['evaluation']):
   b,_=r.run(x)
   for trial,state in [('baseline','clean'),('first:ablate:all','deleted'),('first:pulse:0:1','pulse')]:
    z=b if state=='clean' else r.run(x,state,mask='all')[0]
    rec=r.record('replication',trial,i,x,z,b,{'state':state,'mask':'all'})
    errors[trial].append(rec['nll']-expected[(trial,i)])
  savej(out/'replication.json',{k:{'max_abs_window_nll_error':max(map(abs,v)),'mean_error':float(np.mean(v))} for k,v in errors.items()})
  assert max(abs(e) for v in errors.values() for e in v)<0.002,'Anchor replication failed'
  r.checkpoint();log('Replication passed for all 32 windows and three anchor conditions')
  # Stage 2 development: head-context screening, covariance fitting, constants.
  deltas={l:[] for l in LAYERS};constants={};screen={h:{'damage':[],'rescue':[]} for h in CAND}
  for i in range(8):
   x=window(data,starts,'wiki','development',i)
   b,bc=r.run(x,capture=True);d,dc=r.run(x,'deleted',capture=True)
   for l in LAYERS:deltas[l].append((dc[(l,'residual')].float()-bc[(l,'residual')].float()))
   for l,h in CAND+CTRL:
    for k in ('k','v'):constants.setdefault((l,k,h),[]).append(bc[(l,k)][0,0,h*128:(h+1)*128].float())
    a=bc[(l,'context')][0,:,h*128:(h+1)*128].float()
    constants.setdefault((l,'context',h),[]).append(torch.stack([a[0],a[1:].mean(0)]))
   for h in CAND:
    for direction,state,donor,ref in [('damage','clean',dc,b),('rescue','deleted',bc,d)]:
     z,_=r.run(x,state,patch={'type':'context','heads':[h],'donor':donor})
     value=float((z['loss']-ref['loss']).mean());screen[h][direction].append(value)
     r.record('development',f'L{h[0]}H{h[1]}_{direction}',i,x,z,b,{'type':'context','heads':[h],'direction':direction})
  constants={k:torch.stack(v).mean(0) for k,v in constants.items()}
  implicated=[h for h,v in screen.items() if np.mean(v['damage'])>0.02 or -np.mean(v['rescue'])>0.02]
  savej(out/'selection.json',{'rule':'damage > .02 or rescue improvement > .02 mean nats on development only','selected':implicated,
       'screen':{str(k):v for k,v in screen.items()}})
  torch.save(constants,out/'replacement_constants.pt')
  bases={};spectrum=[]
  for l,ds in deltas.items():
   raw=torch.cat([a[0,9:] for a in ds]);mu=raw.mean(0);centered=raw-mu
   cov=centered.T@centered/(len(raw)-1)
   vals,vec=torch.linalg.eigh(cov);vals=vals.flip(0).clamp_min(0);vec=vec.flip(1)
   random,_=torch.linalg.qr(torch.randn(2048,32))
   bases[l]={'mean':mu,'vectors':vec[:,:32].clone(),'random':random}
   fr=vals/vals.sum();energy=raw.square().sum()
   spectrum.append({'layer':l,'tokens':len(raw),'mean_energy_fraction':float(len(raw)*mu.square().sum()/energy),
      'pc1_centered_fraction':float(fr[0]),'pc8_centered_fraction':float(fr[:8].sum()),'pc32_centered_fraction':float(fr[:32].sum()),
      'effective_rank':float(vals.sum().square()/vals.square().sum()),'rank90':int((fr.cumsum(0)<.9).sum()+1)})
   torch.save({'window_deltas':ds,'mean':mu,'covariance':cov,'eigenvalues':vals,'singular_values':(vals*(len(raw)-1)).sqrt(),
      'top_vectors':vec[:,:32].clone(),'random_vectors':random},out/f'covariance_L{l}.pt')
  torch.save(bases,out/'bases.pt');savej(out/'spectrum.json',spectrum);r.checkpoint()
  log(f'Development frozen: selected {implicated}; centered covariance saved')
  # Stage 3 held-out causal pathways, directions, then frozen simple replacements.
  for i in range(12):
   x=window(data,starts,'wiki','evaluation',i)
   b,bc=r.run(x,capture=True);d,dc=r.run(x,'deleted',capture=True);p0,pc=r.run(x,'pulse',capture=True)
   for trial,z in [('clean',b),('deleted',d),('pulse',p0)]:r.record('causal',trial,i,x,z,b,{'state':trial})
   # Identity hooks are exact controls on both recipient states.
   for state,donor,ref in [('clean',bc,b),('deleted',dc,d)]:
    z,_=r.run(x,state,patch={'type':'kv','heads':CAND+CTRL,'kinds':'kv','positions':'all','donor':donor})
    assert torch.equal(z['pred'],ref['pred']) and torch.equal(z['loss'],ref['loss'])
    r.record('causal','identity_'+state,i,x,z,b,{'patch':'same-state kv all','state':state})
   for head in CAND+CTRL:
    l,h=head
    options=[('k','zero'),('v','zero'),('kv','zero'),('context','all')]
    if head in CAND:options += [('k','all'),('v','all'),('kv','all')]
    for kind,positions in options:
     for direction,state,donor in [('rescue','deleted',bc),('damage','clean',dc)]:
      patch={'type':'context' if kind=='context' else 'kv','heads':[head],'kinds':kind,'positions':positions,'donor':donor}
      z,_=r.run(x,state,patch=patch)
      name=f'L{l}H{h}_{kind}_{positions}_{direction}'
      r.record('causal',name,i,x,z,b,{k:v for k,v in patch.items() if k!='donor'}|{'state':state,'direction':direction})
    if head in CAND:
     for kind in ('k','v'):
      z,_=r.run(x,'pulse',patch={'type':'kv','heads':[head],'kinds':kind,'positions':'zero','donor':dc})
      r.record('causal',f'L{l}H{h}_{kind}_zero_sabotage_pulse',i,x,z,b,{'state':'pulse','deleted_donor':True,'kind':kind,'head':head})
   for group,heads in [('candidate',CAND),('control',CTRL)]:
    for direction,state,donor in [('rescue','deleted',bc),('damage','clean',dc)]:
     for kind in ('kv','context'):
      z,_=r.run(x,state,patch={'type':kind,'heads':heads,'kinds':'kv','positions':'zero','donor':donor})
      r.record('causal',f'{group}_set_{kind}_{direction}',i,x,z,b,{'heads':heads,'patch':kind,'positions':'zero' if kind=='kv' else 'all','direction':direction})
   for l in LAYERS:
    delta=(dc[(l,'residual')].float()-bc[(l,'residual')].float());basis=bases[l]
    # Projection retention on heldout, explicitly separate centered covariance and mean.
    rr=delta[0,9:];cc=rr-basis['mean'];vv=basis['vectors']
    retention={'window':i,'layer':l,'centered_energy':float(cc.square().sum()),
      'pc1_energy':float((cc@vv[:,:1]).square().sum()),'pc8_energy':float((cc@vv[:,:8]).square().sum()),
      'pc32_energy':float((cc@vv).square().sum())}
    with (out/'heldout_retention.jsonl').open('a') as f:f.write(json.dumps(retention)+'\n')
    for mode in ('pc1','pc8','pc32','random8','mean','mean_pc8','full'):
     for direction,state,sign in [('rescue','deleted',-1),('damage','clean',1)]:
      patch={'type':'direction','layer':l,'delta':delta,'mean':basis['mean'],'vectors':basis['random'] if mode.startswith('random') else vv,
       'reference':vv[:,:8],'mode':mode,'sign':sign}
      z,_=r.run(x,state,patch=patch)
      r.record('directions',f'L{l}_{mode}_{direction}',i,x,z,b,{'layer':l,'mode':mode,'direction':direction,'positions':'9+','oracle_donor':mode!='mean'})
   selected_ctrl=[(l,0) for l,h in implicated]
   for group,heads in [('selected',implicated),('control',selected_ctrl)]:
    if not heads:continue
    for kind in ('constant_kv','constant_context'):
     for state in ('clean','pulse'):
      patch={'type':kind,'heads':heads,'kinds':'kv','positions':'zero','constants':constants}
      z,_=r.run(x,state,patch=patch)
      r.record('replacements',f'{group}_{kind}_{state}',i,x,z,b,{'state':state,'heads':heads,'replacement':kind,'fitted':'development'})
   if i%2==1:r.checkpoint();log(f'Causal evaluation {i+1}/12; {r.count} forwards')
  r.checkpoint();log('Held-out causal pathway and directional interventions complete')
  # Stage 4 fresh pulse calibration; no evaluation activations used to set amplitude.
  cal={};cal_values=[]
  def capture_input(m,inputs,output):cal_values.append(float(inputs[0][0,0,1710])*float(m.weight[1764,1710]))
  handle=model.model.layers[1].mlp.down_proj.register_forward_hook(capture_input)
  for corpus in ('wiki','hume'):
   for identity,tid in identities.items():
    cal_values.clear()
    for i in range(16):r.run(window(data,starts,corpus,'calibration',i,128,tid))
    a=np.array(cal_values);cal[(corpus,identity)]=float(a.mean())
    with (out/'fresh_calibration.jsonl').open('a') as f:f.write(json.dumps({'corpus':corpus,'identity':identity,'values':a.tolist(),'mean':float(a.mean()),'std':float(a.std(ddof=1))})+'\n')
  handle.remove()
  # Calibration context length is irrelevant to position0 under causal attention;
  # later length tests retain the exact same start IDs and prefix tokens.
  for corpus in ('wiki','hume'):
   for identity,tid in identities.items():
    for length in (128,512,1024):
     group=f'generalize_{corpus}_{identity}_{length}'
     for i in range(8):
      x=window(data,starts,corpus,'evaluation',i,length,tid);b,_=r.run(x)
      for trial,state,amp in [('clean','clean',0),('deleted','deleted',0),('old_pulse','pulse',ORIGINAL_PULSE),('fresh_pulse','pulse',cal[(corpus,identity)])]:
       z=b if state=='clean' else r.run(x,state,amplitude=amp)
       if isinstance(z,tuple):z=z[0]
       r.record(group,trial,i,x,z,b,{'corpus':corpus,'identity':identity,'start_id':tid,'length':length,'state':state,'pulse':amp,'source_mask':'first'})
     r.checkpoint();log(f'Generalization {corpus}/{identity}/{length}')
  # Reset/continuation: identical 512-token suffix under 0 versus 512 prefix tokens.
  for corpus in ('wiki','hume'):
   for i in range(8):
    whole=window(data,starts,corpus,'evaluation',i,1024)
    for setting,x in [('continued',whole),('reset',whole[512:].clone())]:
     b,_=r.run(x)
     for trial,state in [('clean','clean'),('deleted','deleted'),('pulse','pulse')]:
      z=b if state=='clean' else r.run(x,state)[0]
      if setting=='continued':
       zz={k:v[512:] for k,v in z.items()};bb={k:v[512:] for k,v in b.items()}
      else:zz,bb=z,b
      r.record(f'boundary_{corpus}_{setting}',trial,i,x,zz,bb,{'source_position':0,'scoring':'same suffix target IDs (whole positions513:1024)'})
  for i in range(8):
   x=window(data,starts,'wiki','evaluation',i);b,_=r.run(x)
   for trial,state in [('clean','clean'),('deleted','deleted'),('pulse','pulse')]:
    z,_=r.run(x,state,offset=128)
    r.record('rope_offset128',trial,i,x,z,b,{'state':state,'offset':128,'same_tokens':True})
  r.checkpoint()
  after=hash_model(model);manifest.update(model_hash_after=after,model_unchanged=after==before,
    parameter_versions_unchanged=versions=={n:p._version for n,p in model.named_parameters()},forward_passes=r.count,
    completed_utc=time.strftime('%FT%TZ',time.gmtime()),wall_seconds=time.time()-started,
    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',manifest);assert before==after and manifest['parameter_versions_unchanged']
 hooks.close();log('COMPLETE; weights unchanged')

if __name__=='__main__':main()
