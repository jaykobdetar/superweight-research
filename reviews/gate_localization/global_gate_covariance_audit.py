"""Independent CPU eigendecomposition of pooled same-query calibration gates."""
from pathlib import Path
import json,hashlib,resource
import torch
P=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
rawfile=P/'gpt2_gate_subspace_run/calibration_raw.pt';raw=torch.load(rawfile,map_location='cpu',weights_only=True);record=json.loads((P/'review/global_gate_covariance_exploration.json').read_text())
assert hashlib.sha256(rawfile.read_bytes()).hexdigest()==record['raw_sha256'];assert hashlib.sha256((P/'analyze_global_gate_covariance.py').read_bytes()).hexdigest()==record['source_sha256']
# Explicit coordinate fill makes the layer/head/query alignment test independent.
x=torch.empty((8,510,96),dtype=torch.float64)
for n,item in enumerate(raw.values()):
 for l in range(4,12):
  for h in range(12):x[n,:,12*(l-4)+h]=item[l]['b'][h,:510].double()
errors={};reported={}
def close(a,b,label,tol=1e-10):
 e=float((torch.as_tensor(a,dtype=torch.float64)-torch.as_tensor(b,dtype=torch.float64)).abs().max());errors[label]=e;assert e<tol,(label,e)
def stats(z):
 z=z.reshape(-1,96);mean=z.mean(0);c=z-mean;cov=c.T@c/(len(z)-1);ev,u=torch.linalg.eigh(cov);ev=ev.flip(0);u=u.flip(1);prob=ev/ev.sum()
 return mean,cov,ev,u,prob
for name,z in [('raw_b',x),('position_detrended_r',x-torch.arange(1,511,dtype=torch.float64).log()[None,:,None])]:
 row=record['spaces'][name];mean,cov,ev,u,prob=stats(z);ranks=[1,3,6,12,24,48,96]
 vals=[float(prob[:k].sum())for k in ranks];pr=float(1/(prob*prob).sum());er=float((-(prob[prob>0]*prob[prob>0].log()).sum()).exp())
 close(mean,row['mean'],name+'/mean');close(cov,row['covariance'],name+'/cov');close(ev,row['eigenvalues'],name+'/ev');close(vals,row['cumulative_variance'],name+'/cumulative');close(pr,row['participation_ratio'],name+'/participation');close(er,row['entropy_effective_rank'],name+'/entropy')
 for drop in range(8):
  _,_,e,v,p=stats(z[[i for i in range(8)if i!=drop]]);rr=row['leave_one_passage_out'][drop]
  close(p[0],rr['pc1_variance'],name+'/loo'+str(drop)+'/pc1')
  for k in [1,3,6,12]:close(torch.linalg.svdvals(u[:,:k].T@v[:,:k]),rr['subspace_cosines'][str(k)],name+'/loo'+str(drop)+'/cos'+str(k))
 for label,sl in [('early_queries1_126',slice(0,126)),('later_queries127_510',slice(126,510))]:
  mean,cov,e,v,p=stats(z[:,sl]);rr=row[label];close(cov,rr['covariance'],name+'/'+label+'/cov');close(e,rr['eigenvalues'],name+'/'+label+'/ev');close([float(p[:k].sum())for k in ranks],rr['cumulative_variance'],name+'/'+label+'/cumulative')
 reported[name]={'cumulative_variance':vals,'participation_ratio':pr,'entropy_effective_rank':er}
result={'status':'passed','scope':'Descriptive calibration covariance only; no intervention or heldout fit. Rawb is centered but not position-detrended, r subtracts logq then centers. Coordinates are not standardized. Same-query concatenation L4..11 and heads0..11.','raw_sha256':record['raw_sha256'],'source_sha256':record['source_sha256'],'shape':[8,510,96],'max_recomputation_error':max(errors.values()),'comparisons':len(errors),'statistics':reported,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_loaded':False,'gpu_used':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
