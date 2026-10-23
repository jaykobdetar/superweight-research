"""Does a later token generate the pulse when its early attention history is removed?"""
import json,resource,sys,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,tokenhash,log,hash_model

class Masks:
 def __init__(self,model):
  self.handles=[];self.mode='native';self.p=0;self.cache={};self.trace=None
  for l,layer in enumerate(model.model.layers):self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m,a,k,l=l:self.hook(l,a,k),with_kwargs=True))
  self.handles.append(model.model.layers[1].mlp.down_proj.register_forward_pre_hook(self.source))
  self.weight=float(model.model.layers[1].mlp.down_proj.weight[1764,1710])
 def hook(self,l,args,kwargs):
  if self.mode=='native':return None
  x=kwargs.get('hidden_states',args[0] if args else None);n=x.shape[1]
  if n not in self.cache:
   m=torch.zeros((n,n),dtype=x.dtype,device=x.device);m.masked_fill_(torch.ones((n,n),dtype=torch.bool,device=x.device).triu(1),-torch.inf);self.cache[n]=m[None,None]
  mask=self.cache[n];layers={'explicit':[],'L0':[0],'L1':[1],'L0_L1':[0,1],'all':list(range(16))}[self.mode]
  if l in layers:mask=mask.clone();mask[:,:,self.p,:self.p]=-torch.inf
  kwargs=dict(kwargs);kwargs['attention_mask']=mask;return args,kwargs
 def source(self,m,args):self.trace=args[0][0,[0,self.p]].detach().cpu().clone()
 def close(self):
  for h in self.handles:h.remove()

def prepare(ctx,out):
 rm=json.loads((ROOT/'residual_paths_run/data_manifest.json').read_text());old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu');tok=AutoTokenizer.from_pretrained(ctx['model_path'],local_files_only=True)
 gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text());ids={k:v for k,v in old['corpora'].items()}
 ids['plato']=torch.tensor(tok.encode(Path(gm['plato']['path']).read_text(),add_special_tokens=False))
 ids['python']=torch.tensor(tok.encode('\n\n'.join(Path(p['path']).read_text() for p in rm['python']['sources']),add_special_tokens=False))
 sets={c:[{'prefix':ids[c][s-512:s].clone(),'suffix':ids[c][s:s+512].clone(),'start':s} for s in rm[c]['evaluation_starts'][:4]] for c in rm}
 torch.save(sets,out/'tokens.pt');savej(out/'data_manifest.json',{'source_manifest_sha256':sha(ROOT/'residual_paths_run/data_manifest.json'),'windows':'first4 heldout windows/corpus from residual screen, real512-token prefixes added','fitting':'none','prefix_overlap':'prefixes may include earlier evaluated/calibration text; no constants fitted in this intervention'});return sets

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();sets=prepare(ctx,out);r=Run(ctx['model'],out);mask=Masks(ctx['model']);traces={};start=time.time();torch.cuda.reset_peak_memory_stats()
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'design':'same suffix, real prefix128/512, source0 and suffix-boundary token both The510 or period15; alter only boundary-token attention row in specified layers',
  'mask_modes':['explicit','native','L0','L1','L0_L1','all'],'deletions':['none','source0','boundary','both'],
  'score_start':'prefix p for suffix511-target panel; p+1 for later510-target panel; all comparisons within same mask use matched mask-clean logits',
  'explicit_mask_control':'standard causal4D FP16 mask in all layers, same backend mask shape as isolation conditions; native automatic mask separately compared',
  'pulse_measurement':'FP32 product of original scalar weight and captured pre-down-projection activation1710; also retain full8192D intermediate at0 andp',
  'source':'L1/r1764/c1710 at specified positions only','seed':20261021,'dtype':'float16','cpu_threads':4};savej(out/'manifest.json',meta)
 try:
  for corpus,windows in sets.items():
   for i,w in enumerate(windows):
    for identity,tid in [('The',510),('period',15)]:
     for p in (128,512):
      x=torch.cat([w['prefix'][-p:],w['suffix']]);x[0]=tid;x[p]=tid;mask.p=p;mask.mode='explicit'
      eb,_,ebl=r.evaluate(x,keep=True);eref=mask.trace.clone();cross=f'mask_effect_{corpus}_{identity}_{p}'
      r.record(cross,'clean',i,x,eb,eb,{'mask':'explicit','identity':identity,'prefix':p},p)
      for mode in ('explicit','native','L0','L1','L0_L1','all'):
       mask.mode=mode
       if mode=='explicit':b=eb;bl=ebl;trace=eref
       else:
        crossz,_,bl=r.evaluate(x,baseline=ebl,keep=True);trace=mask.trace.clone()
        r.record(cross,mode,i,x,crossz,eb,{'mask':mode,'identity':identity,'prefix':p},p)
        b={**crossz,'kl':torch.zeros_like(crossz['kl'])}
       key=f'{corpus}_{identity}_{p}_{mode}_{i}';traces[key]=trace
       values=trace[:,1710].float()*mask.weight
       obs={'corpus':corpus,'window':i,'identity':identity,'prefix':p,'mode':mode,'source0_pulse':float(values[0]),'boundary_pulse':float(values[1]),
            'intermediate_max_abs_difference':float((trace[0].float()-trace[1].float()).abs().max()),'intermediate_exact':torch.equal(trace[0],trace[1]),'token_sha256':tokenhash(x)}
       with (out/'pulse_observations.jsonl').open('a') as f:f.write(json.dumps(obs)+'\n')
       group=f'{corpus}_{identity}_{p}_{mode}'
       for later,ss in [(False,p),(True,p+1)]:r.record(group+('_later' if later else ''),'clean',i,x,b,b,{'mask':mode,'prefix':p,'identity':identity,'later_only':later},ss)
       for name,position in [('delete0',0),('delete_boundary',p),('delete_both',[0,p])]:
        z,_,_=r.evaluate(x,'deleted',source_position=position,baseline=bl)
        assert torch.equal(mask.trace,trace),'Deletion unexpectedly changed its own MLP input at the same layer'
        for later,ss in [(False,p),(True,p+1)]:r.record(group+('_later' if later else ''),name,i,x,z,b,{'mask':mode,'prefix':p,'identity':identity,'source_positions':position,'later_only':later},ss)
       if mode!='explicit':del bl
      del ebl
    r.checkpoint();torch.save(traces,out/'source_intermediates.pt');log(f'Empty history {corpus} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,
   model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},
   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:mask.close();r.close()
