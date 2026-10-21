"""Centered gate spectra plus current-recipient directional interventions."""
import json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from factorial import Run,sha,savej,log,hash_model
from mediation import Paths
from quiet_anchor import SourceMask
from global_gates import Gates

class ProjectedGates(Gates):
 def reset_projection(self,fit,kind,rank,clamp=True):
  self.reset('oracle');self.fit=fit;self.kind=kind;self.rank=rank;self.clamp=clamp;self.bounds={}
 def context(self,l,inputs):
  if not getattr(self,'active_projection',False):return super().context(l,inputs)
  a=self.probabilities(l);del self.gpu[l];o=inputs[0];n=o.shape[1];mu=self.fit[l]['mean'].to(o.device);x=a[:,1:].T
  if self.rank==16:y=x
  elif self.rank==0:y=mu.expand_as(x)
  else:
   basis=self.fit[l][self.kind][:,:self.rank].to(o.device);y=mu+(x-mu)@basis@basis.T
  self.bounds[l]={'below_zero':int((y<0).sum()),'above_one':int((y>1).sum()),'count':y.numel(),'min':float(y.min()),'max':float(y.max())}
  if self.clamp:y=y.clamp(0,1)
  aa=y[None,:,:,None];u=o[:,1:].reshape(1,n-1,16,128).float();vv=self.mean[l]['v'].to(o.device).reshape(1,1,16,128).float()
  result=o.clone();result[:,1:]=((1-aa)*u+aa*vv).reshape(1,n-1,2048).to(o.dtype);return (result,)

def fit_subspaces(out):
 cal=torch.load(ROOT/'global_gates_run/calibration_source_mass.pt',weights_only=True,map_location='cpu');fits={};spectrum={}
 gens={seed:torch.Generator().manual_seed(seed) for seed in (20261021,20261022)}
 for l in range(2,16):
  x=torch.cat([w[l][:,1:].T.double() for c in ('wiki','hume') for w in cal[c]],0);mu=x.mean(0);z=x-mu;cov=z.T@z/(len(z)-1)
  eigen,basis=torch.linalg.eigh(cov);eigen=eigen.flip(0);basis=basis.flip(1);fits[l]={'mean':mu.float(),'pca':basis.float(),'covariance':cov,'eigenvalues':eigen}
  for seed,gen in gens.items():
   q,_=torch.linalg.qr(torch.randn((16,16),generator=gen,dtype=torch.float64));fits[l][f'random_{seed}']=q.float()
  spectrum[l]={'mean_gate':mu.tolist(),'eigenvalues':eigen.tolist(),'centered_energy':{str(k):float(eigen[:k].sum()/eigen.sum()) for k in (1,2,4,8,12,16)}}
 torch.save(fits,out/'fitted_subspaces.pt');savej(out/'calibration_spectra.json',spectrum);return fits

def run(ctx,out,params):
 out.mkdir(parents=True,exist_ok=True);assert not (out/'windows.jsonl').exists();model=ctx['model'];start=time.time();torch.cuda.reset_peak_memory_stats();fit=fit_subspaces(out)
 sets=torch.load(ROOT/'residual_paths_run/tokens.pt',weights_only=True,map_location='cpu');packed=torch.load(ROOT/'cache_run/calibration.pt',weights_only=True,map_location='cpu')['wiki']
 mean={l:{k:packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous() for j,k in enumerate('kv')} for l in range(16)}
 r=Run(model,out);r.local.close();r.local=Paths(model);mask=SourceMask(model);gate=ProjectedGates(model,mean);layers=list(range(2,16));patch={'type':'cut','layers':layers,'donor':mean}
 variants=[(f'pca{k}','pca',k,True) for k in (0,1,2,4,8,12,16)]
 variants += [(f'random{seed}_rank{k}',f'random_{seed}',k,True) for seed in (20261021,20261022) for k in (4,8)]
 variants += [(f'pca{k}_unclamped','pca',k,False) for k in (2,4,8)]
 meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'script_sha256':sha(__file__),'model_hash_before':ctx['original_hash'],
  'calibration_mass_sha256':sha(ROOT/'global_gates_run/calibration_source_mass.pt'),'evaluation_tokens_sha256':sha(ROOT/'residual_paths_run/tokens.pt'),
  'fit':'centered sample covariance per layer over16-head gate vector; pool8 Wiki+8 Hume development windows,511 queries/window; no heldout fitting',
  'projection':'current recipient causal full gate a projected as mu+P(a-mu) at each layer; fixed mean values and native conditional nonzero-key mixture',
  'primary_bounds':'projected gates clamped to[0,1]; bounding is nonlinear, so post-bound gates need not retain the nominal linear rank',
  'unclamped_controls':'ranks2/4/8 retain affine subspace exactly before FP rounding, but can extrapolate outside probability bounds',
  'rank16_control':'uses unmodified current gate, exact fast-path to previous oracle recombination',
  'random':'two fixed Gaussian QR bases per layer, rank-matched nested4/8; same fitted means as PCA',
  'variants':variants,'score_start':1,'dtype':'float16','cpu_threads':4,'seed':20261021}
 savej(out/'manifest.json',meta);torch.save({c:d['evaluation'] for c,d in sets.items()},out/'tokens.pt');heldout={};bounds=[]
 try:
  for c,data in sets.items():
   for i,x in enumerate(data['evaluation']):
    gate.active_projection=False;gate.reset();mask.layers=[];b,_,bl=r.evaluate(x,keep=True);group='subspace_'+c;r.record(group,'clean',i,x,b,b,{'reference':'native model'},1)
    gate.reset('calibrate');bm,_,_=r.evaluate(x,patch=patch,baseline=bl);heldout[c,i]=gate.cache
    r.record(group,'mean_KV',i,x,bm,b,{'reference':'native source mean cache'},1);mask.layers=layers;gate.active_projection=True
    for name,kind,rank,clamp in variants:
     gate.reset_projection(fit,kind,rank,clamp);z,_,_=r.evaluate(x,patch=patch,baseline=bl)
     r.record(group,name,i,x,z,b,{'basis':kind,'rank':rank,'bounded':clamp},1)
     bounds.append({'corpus':c,'window':i,'variant':name,'layers':gate.bounds})
     if name=='pca4':
      gate.reset_projection(fit,kind,rank,clamp);zd,_,_=r.evaluate(x,'deleted',patch,baseline=bl)
      assert all(torch.equal(z[k][1:],zd[k][1:]) for k in ('loss','pred','kl'))
      r.record(group,'pca4_deleted',i,x,zd,b,{'basis':kind,'rank':rank,'bounded':clamp,'source_state':'deleted'},1)
    del bl;r.checkpoint();torch.save(heldout,out/'heldout_native_gates.pt');savej(out/'projection_bounds.json',bounds);log(f'Gate subspaces {c} {i+1}/8; {r.count} forwards')
  meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=r.count,wall_seconds=time.time()-start,model_hash_after=hash_model(model),
   parameter_versions_unchanged=ctx['versions']=={n:p._version for n,p in model.named_parameters()},peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  savej(out/'manifest.json',meta);r.checkpoint()
 finally:gate.close();mask.close();r.close()
