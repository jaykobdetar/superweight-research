"""Causal angle/direction decomposition of downstream normalized source states."""
import itertools,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from newline_history import Probe
class Geometry:
 def __init__(self,model,probe):
  self.probe=probe;self.patch={};self.cache={};self.handles=[]
  for l in range(2,16):self.handles.append(model.model.layers[l].input_layernorm.register_forward_hook(lambda m,a,o,l=l:self.norm(l,a,o)))
 def norm(self,l,a,o):
  p=self.probe.p;self.cache[l]={'raw':a[0][0,p].detach().cpu().clone(),'norm':o[0,p].detach().cpu().clone()}
  if l in self.patch:o=o.clone();o[0,p]=self.patch[l].to(o.device)
  return o
 def close(self):
  for h in self.handles:h.remove()
def decompose(z,u):
 x=z.float();mean=x.mean();y=x-mean;r=y.norm();assert torch.isfinite(r) and r>0;a=(y@u)/r;b=y/r-a*u;s=b.norm();assert torch.isfinite(s) and s>1e-8;v=b/s
 return {'mean':mean,'radius':r,'a':a,'s':s,'v':v}
def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not(out/'windows.jsonl').exists();start=time.time();torch.cuda.reset_peak_memory_stats();r=Run(ctx['model'],out);p=Probe(ctx['model']);geo=Geometry(ctx['model'],p);raw={};obs=[];checks=[]
 first=torch.load(ROOT/'upstream_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];natural=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];eligible={(a['corpus'],a['window']):a for a in json.loads((ROOT/'newline_natural_run/eligibility.json').read_text()) if a['eligible']};torch.save({'first':first,'natural':natural},out/'tokens.pt')
 u=torch.full((2048,),-1/2048,dtype=torch.float32);u[1764]+=1;u/=u.norm();random={}
 for seed in range(20261021,20261024):
  gen=torch.Generator().manual_seed(seed);random[seed]={}
  for l in range(2,16):
   v=torch.randn(2048,generator=gen);v-=v.mean();v-=u*(v@u);random[seed][l]=v/v.norm()
 torch.save(random,out/'random_controls.pt')
 meta={'script_sha256':sha(__file__),'design_sha256':sha(ROOT/'SOURCE_GEOMETRY.md'),'model_hash_before':ctx['original_hash'],'input_hashes':{s:sha(ROOT/s) for s in ['upstream_run/tokens.pt','newline_heads_run/tokens.pt','newline_natural_run/eligibility.json']},'seed':20261021,'dtype':'float16','source_norm_layers':list(range(2,16)),'started_utc':time.strftime('%FT%TZ',time.gmtime())};savej(out/'manifest.json',meta)
 def ev(x,mode='clean',patch=None,bl=None,keep=False):
  p.mode=mode;p.positions=[p.p];geo.patch=patch or {};geo.cache={};z,_,ll=r.evaluate(x,baseline=bl,keep=keep);return z,ll,{l:{k:v.clone() for k,v in d.items()} for l,d in geo.cache.items()}
 try:
  for kind,sets in [('first',first),('newline',natural)]:
   for c,ws in sets.items():
    for i,x in enumerate(ws):
     if kind=='newline' and (c,i) not in eligible:continue
     p.p=0 if kind=='first' else eligible[c,i]['positions187'][0];p.q=128 if kind=='first' else eligible[c,i]['positions187'][1];g=kind+'_'+c;ss=p.p+1;donors={};refs={};ds={}
     b,bl,bc=ev(x,keep=True)
     for world,mode in [('C','clean'),('R','row'),('N','column')]:
      if world=='C':z,cap=b,bc
      else:z,_,cap=ev(x,mode,bl=bl)
      donors[world]=cap;refs[world]=z;ds[world]={l:decompose(d['norm'],u) for l,d in cap.items()};r.record(g,{'C':'clean','R':'row','N':'column'}[world],i,x,z,b,{'source':p.p,'mode':mode},ss)
      for l,d in ds[world].items():obs.append({'group':g,'window':i,'source':p.p,'world':world,'layer':l,**{k:float(v) for k,v in d.items() if k!='v'}})
     raw[f'{g}__{i}']={'source':p.p,'donors':donors}
     for a,d in itertools.product('CRN',repeat=2):
      patch={l:(ds[a][l]['mean']+ds[a][l]['radius']*(ds[a][l]['a']*u+ds[a][l]['s']*ds[d][l]['v'])).half() for l in range(2,16)};z,_,_=ev(x,'row',patch,bl);r.record(g,'angle_'+a+'_direction_'+d,i,x,z,b,{'source':p.p,'recipient':'row','axis_radius_donor':a,'direction_donor':d},ss)
      if a==d:checks.append({'group':g,'window':i,'kind':'reconstruction','world':a,'max_norm_error':max(float((patch[l].float()-donors[a][l]['norm'].float()).abs().max()) for l in patch),'nll_drift':float((z['loss'][ss:]-refs[a]['loss'][ss:]).double().mean()),'agreement':float(z['pred'][ss:].eq(refs[a]['pred'][ss:]).float().mean())})
     for a in 'CRN':
      patch={l:donors[a][l]['norm'] for l in range(2,16)};z,_,_=ev(x,'row',patch,bl);r.record(g,'full_'+a,i,x,z,b,{'source':p.p,'recipient':'row','full_norm_donor':a},ss);ch={'group':g,'window':i,'kind':'closure','world':a,'loss_exact':torch.equal(z['loss'][ss:],refs[a]['loss'][ss:]),'pred_exact':torch.equal(z['pred'][ss:],refs[a]['pred'][ss:])};checks.append(ch);assert ch['loss_exact'] and ch['pred_exact'],ch
     for a in 'CN':
      for seed,vs in random.items():
       patch={l:(ds[a][l]['mean']+ds[a][l]['radius']*(ds[a][l]['a']*u+ds[a][l]['s']*vs[l])).half() for l in range(2,16)};z,_,_=ev(x,'row',patch,bl);r.record(g,f'angle_{a}_random_{seed}',i,x,z,b,{'source':p.p,'recipient':'row','axis_radius_donor':a,'random_seed':seed},ss)
     patch={l:(ds['C'][l]['mean']-ds['C'][l]['radius']*u).half() for l in range(2,16)};z,_,_=ev(x,'row',patch,bl);r.record(g,'negative_axis_only',i,x,z,b,{'source':p.p,'recipient':'row','axis':1764,'radius':'clean'},ss)
     del bl;r.checkpoint();torch.save(raw,out/'donors.pt');savej(out/'geometry_observations.json',obs);savej(out/'checks.json',checks);log(f'Geometry {kind} {c} {i+1}/4;{r.count}forwards')
  meta.update(forward_passes=r.count,wall_seconds=time.time()-start,completed_utc=time.strftime('%FT%TZ',time.gmtime()),model_hash_after=hash_model(ctx['model']),parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in ctx['model'].named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);savej(out/'manifest.json',meta)
 finally:geo.close();p.close();r.close()
