"""Fixed-map and synthetic direction arithmetic only; no model/GPU load."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,json,resource,sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from gate_localization_inputs import directions,sha,MAP_SHA
torch.set_num_threads(1)
p=root/'gpt2_gate_fixed512_run/calibration_maps.pt';assert sha(p)==MAP_SHA
maps=torch.load(p,weights_only=True,map_location='cpu');ds=directions(maps)
g=torch.Generator().manual_seed(20261025);normerrors={}
for l,d in ds.items():
    x=torch.randn(511,12,generator=g);u=d['u'].float();v=d['v'].float();s=x@u
    a=s[:,None]*u;b=s[:,None]*v
    aa=a.norm(dim=1);bb=b.norm(dim=1)
    error=float(((aa-bb).abs()/torch.maximum(torch.ones_like(aa),torch.maximum(aa,bb))).max())
    assert error<=1e-6;normerrors[l]=error
    assert not torch.count_nonzero(torch.zeros(12)@u*u)
synthetic={l:{'basis_r':torch.eye(12,dtype=torch.float64),'basis_qr':torch.eye(12,dtype=torch.float64)}for l in range(4,12)}
sd=directions(synthetic);assert all(d['qr_column']==1 and d['sign_multiplier']==1 for d in sd.values())
neg=copy.deepcopy(synthetic)
for l in neg:neg[l]['basis_qr'][:,1]*=-1
nd=directions(neg);assert all(torch.equal(nd[l]['v'],sd[l]['v'])for l in nd)
bad=copy.deepcopy(synthetic);bad[4]['basis_r'][0,0]=float('nan')
try:directions(bad)
except AssertionError:pass
else:raise AssertionError('NaN must fail')
destination=root/'gpt2_gate_localization_directions';destination.mkdir(exist_ok=True)
assert not any(destination.iterdir())
torch.save(ds,destination/'directions.pt')
result={'status':'passed','model_loaded':False,'model_forwards':0,'gpu_used':False,'maps_sha256':MAP_SHA,
    'helper_sha256':sha(root/'gate_localization_inputs.py'),'proposal_sha256':sha(root/'GPT2_GATE_LOCALIZATION_PROPOSAL.md'),
    'script_sha256':sha(__file__),'directions_sha256':sha(destination/'directions.pt'),
    'checks':['eight original frozen directions','first degenerate QR column skipped','sign canonicalization invariant','nonfinite direction rejected','zero coefficient exact','511 per-layer FP32 coefficient norms'],
    'max_per_layer_relative_norm_error':normerrors,
    'layers':{l:{k:(v.tolist()if isinstance(v,torch.Tensor)else v)for k,v in d.items()}for l,d in ds.items()},
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
assert result['peak_rss_kib']<3*1024**2
(destination/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='layers'},indent=2))
