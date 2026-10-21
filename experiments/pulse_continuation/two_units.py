"""Two known super-weight units: row versus entire-column removal factorial."""
import itertools,json,resource,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model

class Units:
 def __init__(self,model):
  self.states={1:'clean',2:'clean'};self.scope='source';self.cache={};self.handles=[];self.weights={};self.scalars={};self.cols={1:1710,2:8041}
  for l,c in self.cols.items():
   m=model.model.layers[l].mlp.down_proj;self.scalars[l]=float(m.weight[1764,c]);self.weights[l]=m.weight.detach().clone();self.weights[l][:,c]=0;self.handles.append(m.register_forward_hook(lambda m,a,o,l=l:self.hook(l,m,a,o)))
 def hook(self,l,m,args,o):
  state=self.states[l];z=o
  if state!='clean':
   alt=F.linear(args[0],self.weights[l],m.bias);z=o.clone();p=0 if self.scope=='source' else slice(None)
   if state=='row':z[:,p,1764]=alt[:,p,1764]
   elif state=='column':z[:,p]=alt[:,p]
   else:raise ValueError(state)
  a=args[0][0,0,self.cols[l]].detach().cpu();potential=float(a.float()*self.scalars[l]);self.cache[l]={'activation':a,'weight':self.scalars[l],'potential_unremoved_contribution':potential,'applied_target_contribution':potential if state=='clean' else 0.,'state':state,'scope':self.scope,'down_before':o[0,0].detach().cpu().clone(),'down_after':z[0,0].detach().cpu().clone()};return z
 def close(self):
  for h in self.handles:h.remove()

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);unit=Units(ctx['model']);sets=torch.load(ROOT/'upstream_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];torch.save(sets,out/'tokens.pt');saved={};checks=[]
 prior=torch.load(ROOT/'column_factorial_run/per_token.pt',weights_only=True,map_location='cpu');prior={k:{z['window']:z for z in v} for k,v in prior.items()}
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'TWO_UNITS.md'),'model_hash_before':ctx['original_hash'],'tokens_parent_sha256':sha(ROOT/'upstream_run/tokens.pt'),'prior_metrics_sha256':sha(ROOT/'column_factorial_run/per_token.pt'),'targets':{'first':[1,1764,1710],'second':[2,1764,8041]},'states':['clean','row','column'],'scopes':['source','all'],'score_start':1,'dtype':'float16','cpu_threads':4,'seed':20261021};savej(out/'manifest.json',meta)
 try:
  for c,windows in sets.items():
   for i,xx in enumerate(windows):
    for identity,tid in [('natural',None),('The',510),('period',15)]:
     x=xx.clone()
     if tid is not None:x[0]=tid
     unit.states={1:'clean',2:'clean'};unit.scope='source';b,_,bl=r.evaluate(x,keep=True);bc={l:d.copy() for l,d in unit.cache.items()}
     for scope in ('source','all'):
      unit.scope=scope;g=f'units_{c}_{identity}_{scope}'
      for a,zstate in itertools.product(('clean','row','column'),repeat=2):
       unit.states={1:a,2:zstate};trial='clean' if a==zstate=='clean' else f'first_{a}_second_{zstate}'
       if a==zstate=='clean':z=b;cap={l:{**d,'scope':scope} for l,d in bc.items()}
       else:z,_,_=r.evaluate(x,baseline=bl);cap={l:d.copy() for l,d in unit.cache.items()}
       r.record(g,trial,i,x,z,b,{'first':a,'second':zstate,'scope':scope,'identity':identity},1);saved[f'{g}__{trial}__{i}']=cap
       if scope=='source' and zstate=='clean':
        key=f'first_{c}_{identity}__'+{'clean':'clean','row':'delete_row','column':'delete_column'}[a];ref=prior[key][i];check={'group':g,'trial':trial,'window':i,'loss_exact':torch.equal(z['loss'][1:],ref['loss']),'pred_exact':torch.equal(z['pred'][1:],ref['pred'])};checks.append(check);assert check['loss_exact'] and check['pred_exact'],check
     del bl
    r.checkpoint();torch.save(saved,out/'unit_traces.pt');savej(out/'checks.json',checks);log(f'Two units {c} {i+1}/4; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:unit.close();r.close()
