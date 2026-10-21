"""Factor H14's downstream effect into early attention/MLP residual components."""
import itertools,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model,tokenhash
from newline_heads import PathProbe,edited

class Components(PathProbe):
 def __init__(self,model):
  super().__init__(model);self.swaps={};self.components={}
  modules={'A0':model.model.layers[0].self_attn.o_proj,'M0':model.model.layers[0].mlp,'A1':model.model.layers[1].self_attn.o_proj}
  for name,m in modules.items():self.handles.append(m.register_forward_hook(lambda m,a,o,name=name:self.component(name,o)))
  self.handles.append(model.model.layers[1].post_attention_layernorm.register_forward_hook(self.norm1))
 def component(self,name,o):
  before=o[0,self.q].detach().cpu().clone()
  if name in self.swaps:o=o.clone();o[0,self.q]=self.swaps[name].to(o.device)
  self.components[name]={'native':before,'applied':o[0,self.q].detach().cpu().clone()};return o
 def norm1(self,m,a,o):self.cache['before_mlp1']=a[0][0,self.q].detach().cpu().clone();self.cache['normalized_mlp1']=o[0,self.q].detach().cpu().clone();return o

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();sets=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');eligibility=json.loads((ROOT/'newline_natural_run/eligibility.json').read_text());eligible={(a['corpus'],a['window']):a for a in eligibility if a['eligible']};r=Run(ctx['model'],out);p=Components(ctx['model']);raw={};obs=[];checks=[];h=json.loads((ROOT/'newline_heads_run/selection.json').read_text())['selected'];assert h==14
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'NEWLINE_COMPONENTS.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'newline_heads_run/tokens.pt'),'eligibility_sha256':sha(ROOT/'newline_natural_run/eligibility.json'),'source':'frozen C/E donors for three full2048D block outputs atq','score_start':'q+1; parent-matched targets','reuse':'16constructed+12natural exploratory contexts; no fitting','dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 def ev(x,edge=False,swaps=None,bl=None,keep=False):
  p.head_edges=[(h,p.q,p.p)] if edge else [];p.swaps=swaps or {};p.cache={};p.components={};z,_,ll=r.evaluate(x,baseline=bl,keep=keep);cap={k:v.clone() for k,v in p.cache.items()};cap['components']={n:{k:v.clone() for k,v in d.items()} for n,d in p.components.items()};return z,ll,cap
 def rec(g,t,i,x,z,b,cap,spec):
  r.record(g,t,i,x,z,b,{**spec,'p':p.p,'q':p.q},p.q+1);raw[f'{g}__{t}__{i}']=cap;obs.append({'group':g,'trial':t,'window':i,'p':p.p,'q':p.q,'q_pulse':float(cap['activation'][p.q])*p.scalar,'q_gate':float(cap['gate'][2]),'q_up':float(cap['up'][2]),'token_sha256':tokenhash(x)})
 def exact(g,i,t,z,ref,cap,rc):
  check={'group':g,'window':i,'trial':t,'full_loss_exact':torch.equal(z['loss'],ref['loss']),'full_pred_exact':torch.equal(z['pred'],ref['pred']),'q_mlp_input_exact':torch.equal(cap['before_mlp1'],rc['before_mlp1']),'q_intermediate_exact':torch.equal(cap['activation'][p.q],rc['activation'][p.q])};checks.append(check);assert all(check[k] for k in ('full_loss_exact','full_pred_exact','q_mlp_input_exact','q_intermediate_exact')),check
 try:
  for kind in ('constructed','natural'):
   for c,windows in sets.items():
    for i,xx in enumerate(windows):
     if kind=='natural':
      if (c,i) not in eligible:continue
      p.p,p.q=eligible[c,i]['positions187'][:2];x=xx.clone();assert int(x[p.p])==int(x[p.q])==187
     else:p.p,p.q=64,128;x=edited(xx)
     g=kind+'_'+c;b,bl,bc=ev(x,keep=True);e,_,ec=ev(x,edge=True,bl=bl);rec(g,'clean',i,x,b,b,bc,{});rec(g,'edge',i,x,e,b,ec,{'head':h});donors={'C':bc,'E':ec}
     for code in itertools.product('CE',repeat=3):
      swaps={name:donors[world]['components'][name]['applied'] for name,world in zip(('A0','M0','A1'),code)};z,_,cap=ev(x,swaps=swaps,bl=bl);t='mix_'+''.join(code);rec(g,t,i,x,z,b,cap,{'base':'C','frozen_components':dict(zip(('A0','M0','A1'),code))})
      if len(set(code))==1:exact(g,i,t,z,b if code[0]=='C' else e,cap,donors[code[0]])
     for base,edge,donor in [('C',False,'E'),('E',True,'C')]:
      for name in ('A0','M0','A1'):
       swaps={name:donors[donor]['components'][name]['applied']};z,_,cap=ev(x,edge,swaps,bl=bl);t=base+'_from_'+donor+'_'+name;rec(g,t,i,x,z,b,cap,{'base':base,'donor':donor,'component':name,'other_components':'endogenous'})
       if name=='A0':exact(g,i,t,z,e if donor=='E' else b,cap,donors[donor])
     del bl;r.checkpoint();torch.save(raw,out/'probe_traces.pt');savej(out/'factor_observations.json',obs);savej(out/'checks.json',checks);log(f'Newline components {kind} {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:p.close();r.close()
