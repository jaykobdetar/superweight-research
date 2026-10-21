"""Small independent-readout formula check against the frozen helper; no weights/model/GPU."""
from pathlib import Path
import sys,json,torch,resource
p=Path(__file__).resolve().parents[1];sys.path[:0]=[str(p/'review'),str(p)]
from gate_additive_readout_audit import family_metrics
from gate_additive_readout import AdditiveReadout
torch.set_num_threads(1);g=torch.Generator().manual_seed(117);zs=[torch.randn(37,173,generator=g)for _ in range(11)];t=torch.randint(173,(37,),generator=g)
a=AdditiveReadout(zs[0],t)
for l,z in zip(range(4,12),zs[2:10]):a.add_singleton(l,z)
d=a.finish(zs[10],zs[1]);ind,add,pe=family_metrics(zs[0].double(),zs[1].double(),[z.double()for z in zs[2:10]],zs[10].double(),t)
errors={k:float((v.double()-ind[k].double()).abs().max())for k,v in d.items()};assert max(errors.values())<1e-12 and torch.equal(add,a.additive)
out={'status':'passed','random_logit_fixture_only':True,'model_forwards':0,'gpu_used':False,'max_errors':errors,'partition_identity_error':pe,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
