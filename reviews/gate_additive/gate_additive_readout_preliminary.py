"""Independent algebra and dimension-scaled CPU readout checks; no model."""
from pathlib import Path
import sys,json,hashlib,resource
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from gate_additive_readout import AdditiveReadout

def run():
 torch.set_num_threads(1);g=torch.Generator().manual_seed(971123)
 base=torch.randn(511,83,generator=g);target=torch.randint(83,(511,),generator=g)
 xs=[base+torch.randn(base.shape,generator=g)*.2 for _ in range(8)]
 j1=base+torch.randn(base.shape,generator=g)*.3;j2=j1+torch.randn(base.shape,generator=g)*.1
 def calc(j,c):
  obj=AdditiveReadout(base,target)
  for l,x in zip(range(4,12),xs):obj.add_singleton(l,x)
  return obj.finish(j,c),obj.additive
 a,add=calc(j1,base);b,_=calc(j2,base);c,_=calc(j1,j2)
 dense=base.double()+sum((x.double()-base.double()for x in xs),torch.zeros_like(base,dtype=torch.float64))
 ce=lambda x:-x.double().log_softmax(-1)[torch.arange(len(target)),target]
 expected_R=ce(dense)-ce(base)-sum((ce(x)-ce(base)for x in xs),torch.zeros(len(target),dtype=torch.float64))
 expected_J=ce(j1)-ce(dense)
 checks={'additive_max_error':float((add-dense).abs().max()),'J_max_error':float((a['J']-expected_J).abs().max()),'R_max_error':float((a['R']-expected_R).abs().max()),'identity_max_error':float((a['J']+a['R']-a['joint_minus_sum']).abs().max())}
 r_partition=dense.logsumexp(-1)+7*base.double().logsumexp(-1)-sum((x.double().logsumexp(-1)for x in xs),torch.zeros(len(target),dtype=torch.float64))
 j_partition=j1.double().logsumexp(-1)-dense.logsumexp(-1)-(j1.double()-dense)[torch.arange(len(target)),target]
 checks['R_log_partition_identity_max']=float((a['R']-r_partition).abs().max())
 checks['J_partition_target_identity_max']=float((a['J']-j_partition).abs().max())
 target2=(target+1)%base.shape[1];other=AdditiveReadout(base,target2)
 for l,x in zip(range(4,12),xs):other.add_singleton(l,x)
 other_result=other.finish(j1,base);checks['R_target_change_max']=float((a['R']-other_result['R']).abs().max())
 assert max(checks.values())<1e-12
 assert torch.equal(a['R'],b['R'])and not torch.equal(a['J'],b['J'])
 assert all(torch.equal(a[k],c[k])for k in ('J','R','joint_minus_sum'))
 # Check a prospective theoretical forward-error policy on actual FP32 CPU dot products.
 bounds=[];d=768;u=2**-24;n=2*d+2;gamma=n*u/(1-n*u)
 for scale in (1e-4,1.,1e4):
  h=torch.randn(8,d,generator=g)*scale;w=torch.randn(137,d,generator=g);bias=torch.randn(137,generator=g)
  native=h@w.T+bias;ref=h.double()@w.double().T+bias.double()
  budget=gamma*(h.double().abs()@w.double().abs().T+bias.double().abs())+1e-12
  err=(native.double()-ref).abs();assert bool((err<=budget).all())
  p=native.double().log_softmax(-1);q=ref.log_softmax(-1);delta=err.max(-1).values
  nllerr=(p-q).abs().max(-1).values;kl=(p.exp()*(p-q)).sum(-1)
  assert bool((nllerr<=2*delta+1e-10).all()) and bool((kl<=delta.square()/2+1e-10).all())
  bounds.append({'scale':scale,'maximum_absolute_error':float(err.max()),'maximum_error_to_bound':float((err/budget).max()),'maximum_KL':float(kl.max())})
 out={'status':'passed_preliminary_helper_only','helper_sha256':hashlib.sha256((P/'gate_additive_readout.py').read_bytes()).hexdigest(),'algebra':checks,'R_independent_of_joint_and_native_reference':True,'J_independent_of_native_reference':True,'FP32_readout_bound_cpu_fixtures':bounds,'bound_caveat':'Conservative sampled readout sanity guard; not empirical fidelity or a guarantee for untested GPU algorithms. Production review still pending.','peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_loaded':False,'gpu_used':False}
 Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':run()
