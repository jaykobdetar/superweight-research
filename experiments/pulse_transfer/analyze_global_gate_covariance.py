"""Descriptive calibration-only covariance across layers; no model or causal claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
def stats(x):
    center=x-x.mean(0);_,s,vh=np.linalg.svd(center,full_matrices=False)
    eigen=s*s/(len(x)-1);p=eigen/eigen.sum();cov=center.T@center/(len(x)-1)
    ev=np.linalg.eigvalsh(cov)[::-1]
    return {'vectors':len(x),'coordinates':x.shape[1],
        'cumulative_variance':[float(p[:k].sum())for k in (1,3,6,12,24,48,96)],
        'reported_ranks':[1,3,6,12,24,48,96],
        'participation_ratio':float(1/(p*p).sum()),
        'entropy_effective_rank':float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),
        'svd_cov_eigenvalue_max_error':float(np.max(np.abs(ev-eigen))),
        'mean':x.mean(0).tolist(),'covariance':cov.tolist(),'eigenvalues':eigen.tolist()},vh.T
def main():
    torch.set_num_threads(1);p=ROOT/'gpt2_gate_subspace_run/calibration_raw.pt'
    raw=torch.load(p,weights_only=True,map_location='cpu');assert len(raw)==8
    b=np.stack([np.concatenate([d[l]['b'][:,:510].T.double().numpy()for l in range(4,12)],axis=1)for d in raw.values()])
    assert b.shape==(8,510,96)
    r=b-np.log(np.arange(1,511))[None,:,None];out={}
    for label,z in [('raw_b',b),('position_detrended_r',r)]:
        summary,u=stats(z.reshape(-1,96));summary['leave_one_passage_out']=[]
        for drop in range(8):
            subset,v=stats(np.delete(z,drop,axis=0).reshape(-1,96))
            summary['leave_one_passage_out'].append({'omitted_calibration_index':drop,
                'pc1_variance':subset['cumulative_variance'][0],
                'subspace_cosines':{str(k):np.linalg.svd(u[:,:k].T@v[:,:k],compute_uv=False).tolist()for k in (1,3,6,12)}})
        for segment,sl in [('early_queries1_126',slice(0,126)),('later_queries127_510',slice(126,510))]:
            summary[segment],_=stats(z[:,sl,:].reshape(-1,96))
        out[label]=summary
    result={'status':'descriptive_calibration_only','model_forwards':0,'model_loaded':False,'gpu_used':False,
        'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'layout':'Concatenate L4..11, twelve ordered heads each, at the same calibration query;8passages×510queries×96coordinates.',
        'limits':'These correlated calibration vectors are not held-out samples or causal interventions. Layer/head axes are distinct computations; dimensionality does not identify one global mechanism. No evaluation data, no fitting for an experiment, no change to frozen localization directions/design. Segment views share passages.',
        'spaces':out,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (ROOT/'review/global_gate_covariance_exploration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{f:v[f]for f in ('cumulative_variance','participation_ratio','entropy_effective_rank')}for k,v in out.items()},indent=2))
if __name__=='__main__':main()
