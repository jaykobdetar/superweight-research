"""CPU regression of count generalization using frozen calibration raw arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fit_gate_calibration import fit_maps_n
from gpt2_gate_subspace import fit_maps,maps_digest
from gate_localization_inputs import directions
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 start=time.time();torch.set_num_threads(4)
 ref=ROOT/'gpt2_gate_fixed512_run';out=ROOT/'wiki_calibration_preparation'
 assert not out.exists(),'Preserve prior fitting evidence'
 out.mkdir();assert sha(ref/'calibration_raw.pt')=='d41d5a91c699d30d5414a2916f7350a9093eda4b0004d8bb7b2fb458ebb31935'
 raw=torch.load(ref/'calibration_raw.pt',weights_only=True,map_location='cpu')
 m8=torch.load(ref/'calibration_maps.pt',weights_only=True,map_location='cpu');s8=json.loads((ref/'calibration_spectra.json').read_text())
 generalized,reports=fit_maps_n(raw,8);legacy,legacy_report=fit_maps(raw)
 for l in m8:
  for n in m8[l]:assert torch.equal(generalized[l][n],m8[l][n])and torch.equal(generalized[l][n],legacy[l][n]),(l,n)
 assert json.loads(json.dumps(reports))==s8==json.loads(json.dumps(legacy_report))
 assert maps_digest(generalized)=='fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81'
 names=[n for n in raw if n.startswith('wiki_test__')];assert names==['wiki_test__'+str(i)for i in range(4)]
 subset={n:raw[n]for n in names};m4,s4=fit_maps_n(subset,4);ds=directions(m4);checks=[]
 for l in range(4,12):
  b=torch.stack([r[l]['b'][:,:510].T for r in subset.values()]).double()
  a=torch.stack([r[l]['mass'][:,:510].T for r in subset.values()]).double()
  logq=torch.arange(1,511,dtype=torch.float64).log()[None,:,None]
  spaces={'b':b,'mass':a,'r':b-logq,'position_centered_b':b-b.mean(0,keepdim=True)}
  for space,x in spaces.items():
   flat=x.reshape(-1,12);centered=flat-flat.mean(0);cov=centered.T@centered/(len(flat)-1)
   rep=s4[l][space];assert len(flat)==2040 and rep['covariance_denominator']==2039 and rep['vectors']==2040
   error=float((cov-torch.tensor(rep['covariance'],dtype=torch.float64)).abs().max());assert error<=1e-12
   checks.append({'layer':l,'space':space,'covariance_max_abs_error':error})
  for record in s4[l]['leave_one_passage_out']:
   for sp in record['spaces'].values():assert sp['vectors']==1530 and sp['covariance_denominator']==1529
  assert len(s4[l]['leave_one_passage_out'])==4
  eig=s4[l]['r']['eigenvalues'];assert eig[0]>0 and (eig[0]-eig[1])/eig[0]>1e-12
 try:fit_maps_n(subset,8)
 except AssertionError:pass
 else:raise AssertionError('Four passages silently accepted as eight')
 torch.save(m4,out/'old_wiki_maps.pt');torch.save(ds,out/'old_wiki_directions.pt')
 (out/'old_wiki_spectra.json').write_text(json.dumps(s4,indent=2)+'\n')
 metadata={'fit_name':'W_old4','calibration_world':'P_source_V0_zero_L4_to_L11','source_raw_sha256':sha(ref/'calibration_raw.pt'),
  'subset_keys':names,'passages':4,'queries_per_passage':510,'pooled_vectors':2040,'covariance_denominator':2039,
  'maps_tensor_digest':maps_digest(m4),'maps_sha256':sha(out/'old_wiki_maps.pt'),'directions_sha256':sha(out/'old_wiki_directions.pt'),
  'spectra_sha256':sha(out/'old_wiki_spectra.json'),'fitter_sha256':sha(ROOT/'fit_gate_calibration.py'),'new_model_forwards':0,
  'scope':'New four-Wiki fit from original raw calibration; not the original mixed-eight map and not new model inference.'}
 (out/'old_wiki_fit.json').write_text(json.dumps(metadata,indent=2)+'\n')
 result={'status':'passed','original_eight_maps_and_reports_exact':True,'original_eight_digest':maps_digest(generalized),
  'direct_four_passage_covariances':checks,'four_passage_LOO_checked':32,'wrong_sample_count_rejected':True,
  'fitter_sha256':sha(ROOT/'fit_gate_calibration.py'),'old_wiki_fit':metadata,'wall_seconds':time.time()-start,
  'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_forwards':0,'model_loaded':False,'gpu_used':False}
 Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in('direct_four_passage_covariances','old_wiki_fit')},indent=2))
if __name__=='__main__':main()
