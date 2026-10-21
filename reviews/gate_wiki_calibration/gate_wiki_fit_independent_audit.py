"""Independent bounded CPU audit of retained n4 fit and n8 regression; no model."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from pathlib import Path
import sys,json,hashlib,time,resource
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fit_gate_calibration import fit_maps_n
from gpt2_gate_subspace import fit_maps,maps_digest
from gate_localization_inputs import directions

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def read(p):return json.loads(p.read_text())
def equal(a,b):
 if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b,(a,b)
def plain(x):return json.loads(json.dumps(x))

def main():
 start=time.time();torch.set_num_threads(1);ref=ROOT/'gpt2_gate_fixed512_run';out=ROOT/'wiki_calibration_preparation';fitter=ROOT/'fit_gate_calibration.py'
 assert sha(fitter)=='9cc6ea695f6c3753f8878a5d5146fa9a371be264685c2e26d8f831cd2b147c7e'
 raw=load(ref/'calibration_raw.pt');original=load(ref/'calibration_maps.pt');original_spectra=read(ref/'calibration_spectra.json')
 assert sha(ref/'calibration_raw.pt')=='d41d5a91c699d30d5414a2916f7350a9093eda4b0004d8bb7b2fb458ebb31935'
 m8,s8=fit_maps_n(raw,8);legacy,ls=fit_maps(raw);equal(m8,original);equal(m8,legacy);assert plain(s8)==original_spectra==plain(ls)
 assert maps_digest(m8)=='fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81'
 meta=read(out/'old_wiki_fit.json');names=['wiki_test__'+str(i)for i in range(4)]
 assert [k for k in raw if k.startswith('wiki_test__')]==meta['subset_keys']==names
 assert meta['source_raw_sha256']==sha(ref/'calibration_raw.pt')and meta['fitter_sha256']==sha(fitter)
 for file,key in [('old_wiki_maps.pt','maps_sha256'),('old_wiki_directions.pt','directions_sha256'),('old_wiki_spectra.json','spectra_sha256')]:assert sha(out/file)==meta[key]
 subset={k:raw[k]for k in names};m4,s4=fit_maps_n(subset,4);saved=load(out/'old_wiki_maps.pt');ss=read(out/'old_wiki_spectra.json');equal(m4,saved);assert plain(s4)==ss;equal(directions(m4),load(out/'old_wiki_directions.pt'))
 assert maps_digest(m4)==meta['maps_tensor_digest']=='e59fab6ed96e92d34a3286680b989f8ca12608edee7923baaf088f5d1b40573d'
 full_checks=loo_checks=cosine_checks=0;errs={'mean':0.,'covariance':0.,'eigenvalues':0.,'LOO_covariance':0.,'LOO_eigenvalues':0.,'LOO_cosines':0.};gaps={}
 def check_space(x,record,prefix=''):
  nonlocal full_checks,loo_checks
  flat=x.reshape(-1,12).double();mu=flat.mean(0);c=flat-mu;cov=c.T@c/(flat.shape[0]-1)
  _,sing,vh=torch.linalg.svd(c,full_matrices=False);basis=vh.T;eig=sing.square()/(flat.shape[0]-1)
  assert record['vectors']==len(flat)and record['covariance_denominator']==len(flat)-1
  me=float((mu-torch.tensor(record['mean'],dtype=torch.float64)).abs().max());ce=float((cov-torch.tensor(record['covariance'],dtype=torch.float64)).abs().max());ee=float((eig-torch.tensor(record['eigenvalues'],dtype=torch.float64)).abs().max())
  assert me<=1e-12 and ce<=1e-12 and ee<=1e-10
  if prefix:loo_checks+=1;errs['LOO_covariance']=max(errs['LOO_covariance'],ce);errs['LOO_eigenvalues']=max(errs['LOO_eigenvalues'],ee)
  else:full_checks+=1;errs['mean']=max(errs['mean'],me);errs['covariance']=max(errs['covariance'],ce);errs['eigenvalues']=max(errs['eigenvalues'],ee)
  # Independent eigensolver consistency, without requiring signed basis identity.
  assert float((torch.linalg.eigvalsh(cov).flip(0)-eig).abs().max())<=1e-9
  return basis
 for layer in range(4,12):
  b=torch.stack([raw[k][layer]['b'][:,:510].T for k in names]).double();a=torch.stack([raw[k][layer]['mass'][:,:510].T for k in names]).double();logq=torch.arange(1,511,dtype=torch.float64).log()[None,:,None]
  spaces={'b':b,'mass':a,'r':b-logq,'position_centered_b':b-b.mean(0,keepdim=True)};bases={k:check_space(v,ss[str(layer)][k])for k,v in spaces.items()}
  assert b.shape==(4,510,12)
  for key,space in [('alpha','r'),('mean_b','b'),('mean_mass','mass')]:assert torch.equal(saved[layer][key],spaces[space].reshape(-1,12).mean(0))
  for key in ['basis_r','basis_b']:
   u=saved[layer][key];assert bool((u[u.abs().argmax(0),torch.arange(12)]>=0).all())
  equal(saved[layer]['basis_qr'],original[layer]['basis_qr'])
  lo=ss[str(layer)]['leave_one_passage_out'];assert len(lo)==4 and [x['omitted_passage']for x in lo]==names
  for omit,record in enumerate(lo):
   keep=[i for i in range(4)if i!=omit]
   for space,v in spaces.items():
    value=b[keep]-b[keep].mean(0,keepdim=True)if space=='position_centered_b'else v[keep]
    rep=record['spaces'][space];ub=check_space(value,rep,'LOO');assert rep['vectors']==1530 and rep['covariance_denominator']==1529
    for rank,reported in rep['principal_cosines_to_full'].items():
     k=int(rank);actual=torch.linalg.svdvals(bases[space][:,:k].T@ub[:,:k]);error=float((actual-torch.tensor(reported,dtype=torch.float64)).abs().max())if k else 0.
     assert error<=1e-10;errs['LOO_cosines']=max(errs['LOO_cosines'],error);cosine_checks+=1
  eig=ss[str(layer)]['r']['eigenvalues'];gap=(eig[0]-eig[1])/eig[0];assert eig[0]>0 and gap>1e-12;gaps[layer]=gap
 failures=[]
 for n in [3,8]:
  try:fit_maps_n(subset,n)
  except AssertionError:failures.append(n)
  else:raise AssertionError('wrong n accepted')
 result={'status':'passed','fitter_sha256':sha(fitter),'legacy_source_sha256':sha(ROOT/'gpt2_gate_subspace.py'),'reviewed_revision_sha256':sha(ROOT/'GPT2_GATE_WIKI_CALIBRATION_REVISION.md'),'old_wiki_metadata_sha256':sha(out/'old_wiki_fit.json'),'old_wiki_artifact_sha256':{p.name:sha(p)for p in out.glob('old_wiki_*')},'raw_calibration_sha256':sha(ref/'calibration_raw.pt'),'exact_n8_legacy_and_saved_maps_reports':True,'n8_digest':maps_digest(m8),'n4_digest':maps_digest(m4),'exact_n4_refit_and_directions':True,'subset_keys':names,'pooled_vectors':2040,'covariance_denominator':2039,'LOO_vectors':1530,'LOO_denominator':1529,'direct_full_space_checks':full_checks,'direct_LOO_space_checks':loo_checks,'LOO_principal_cosine_checks':cosine_checks,'errors':errs,'leading_relative_gaps':gaps,'rejected_sample_counts':failures,'wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_loads':0,'tokenizer_loads':0,'model_forwards':0,'gpu_used':False,'limits':'Fitter and existing old-Wiki preparation only. Fresh calibration, context hooks, storage budget and final protocol/source freeze remain separate.'}
 assert result['peak_rss_kib']<3*1024**2
 return result
if __name__=='__main__':
 result=main();Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
