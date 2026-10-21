"""CPU geometry, frozen control, retention and causal metric audit for gate PCA."""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'gate_subspaces_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
for field,p in [('script_sha256',ROOT/'gate_subspaces.py'),('calibration_mass_sha256',ROOT/'global_gates_run/calibration_source_mass.pt'),('evaluation_tokens_sha256',ROOT/'residual_paths_run/tokens.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']and meta['forward_passes']==544
raw,groups,result=load_audit(OUT,require_tables=False);assert result['rows']==544 and result['groups']==68
result['summary_tables_available']=(OUT/'summary.csv').exists()and(OUT/'contrasts.csv').exists()
mass=torch.load(ROOT/'global_gates_run/calibration_source_mass.pt',map_location='cpu',weights_only=True)
fit=torch.load(OUT/'fitted_subspaces.pt',map_location='cpu',weights_only=True);spectra=json.loads((OUT/'calibration_spectra.json').read_text())
gens={seed:torch.Generator().manual_seed(seed)for seed in [20261021,20261022]};cov_errors=[];orth_errors=[];eigen_errors=[];fullrank_errors=[]
for l in range(2,16):
 x=torch.cat([w[l][:,1:].T.double()for c in ['wiki','hume']for w in mass[c]],0);assert x.shape==(8176,16)
 mu=x.mean(0);z=x-mu;cov=z.T@z/(len(z)-1);d=fit[l];basis=d['pca'].double()
 cov_errors.append(float((cov-d['covariance']).abs().max()));assert torch.allclose(mu.float(),d['mean'],atol=1e-7,rtol=0)
 eig=torch.linalg.eigvalsh(cov).flip(0);assert torch.allclose(eig,d['eigenvalues'],atol=1e-12,rtol=1e-10)
 eigen_errors.append(float((cov@basis-basis*d['eigenvalues'][None]).abs().max()));orth_errors.append(float((basis.T@basis-torch.eye(16)).abs().max()))
 fullrank_errors.append(float(((z.float()@d['pca']@d['pca'].T)-z.float()).abs().max()))
 assert max(abs(a-b)for a,b in zip(spectra[str(l)]['mean_gate'],mu.tolist()))<1e-12
 for k in [1,2,4,8,12,16]:assert abs(spectra[str(l)]['centered_energy'][str(k)]-float(eig[:k].sum()/eig.sum()))<1e-12
 for seed,gen in gens.items():
  q,_=torch.linalg.qr(torch.randn(16,16,generator=gen,dtype=torch.float64));assert torch.equal(q.float(),d[f'random_{seed}'])
assert max(cov_errors)<1e-12 and max(orth_errors)<2e-7 and max(eigen_errors)<1e-7 and max(fullrank_errors)<1e-6
old=torch.load(ROOT/'global_gates_run/per_token.pt',map_location='cpu',weights_only=True);old={k:{z['window']:z for z in v}for k,v in old.items()}
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);prior=torch.load(ROOT/'global_gates_run/tokens.pt',map_location='cpu',weights_only=True)
heldout=torch.load(OUT/'heldout_native_gates.pt',map_location='cpu',weights_only=True);assert len(heldout)==32
comparisons={};vs_oracle={};retention={}
for c,ws in sets.items():
 g='subspace_'+c
 for i,x in enumerate(ws):
  assert torch.equal(x,prior[c][i])
  for t,ot in [('clean','clean'),('mean_KV','mean_KV'),('pca16','oracle_clean')]:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],old['gates_'+c+'__'+ot][i][k])
  for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__pca4'][i][k],raw[g+'__pca4_deleted'][i][k])
 for key,rows in groups.items():
  if not key.startswith(g+'__'):continue
  assert all(z['score_start']==1 and z['scored_tokens']==510 for z in rows.values());t=key.split('__')[1]
  comparisons[c+'__'+t]=compare(raw,groups,g,t,'clean');comparisons[c+'__'+t].update(agreement=float(np.mean([z['agreement']for z in rows.values()])),kl=float(np.mean([z['kl_clean_to_condition']for z in rows.values()])))
  vs_oracle[c+'__'+t]=effect(groups,g,t,'pca16')
 for l in range(2,16):
  x=torch.cat([heldout[c,i][l][:,1:].T.double()for i in range(8)],0);assert x.shape==(4088,16)and torch.isfinite(x).all()and(x>=0).all()and(x<=1).all()
  mu=fit[l]['mean'].double();z=x-mu;own=x-x.mean(0);basis=fit[l]['pca'].double()
  retention[c+'__L'+str(l)]={'native_world_development_mean_centered':{str(k):float((z@basis[:,:k]).square().sum()/z.square().sum())for k in [1,2,4,8,12,16]},'native_world_heldout_own_mean_centered':{str(k):float((own@basis[:,:k]).square().sum()/own.square().sum())for k in [1,2,4,8,12,16]}}
bounds=json.loads((OUT/'projection_bounds.json').read_text());assert len(bounds)==448
reported=json.loads((OUT/'heldout_spectra.json').read_text());assert len(reported)==56
for row in reported:
 c,l=row['corpus'],row['layer'];x=torch.cat([heldout[c,i][l][:,1:].T.double()for i in range(8)],0)
 own=x-x.mean(0);eig=torch.linalg.eigvalsh(own.T@own/(len(x)-1)).flip(0)
 assert row['queries']==4088 and np.allclose(row['heldout_mean'],x.mean(0).tolist(),atol=1e-12,rtol=0)
 assert np.allclose(row['centered_eigenvalues'],eig.tolist(),atol=1e-12,rtol=0)
 assert abs(row['centered_pc1_share']-float(eig[0]/eig.sum()))<1e-12
 assert row['centered_effective_rank90']==int((eig.cumsum(0)/eig.sum()<.9).sum())+1
 for label,rlabel in [('dev_basis_retained_centered_energy','native_world_heldout_own_mean_centered'),('dev_basis_retained_deviation_from_dev_mean','native_world_development_mean_centered')]:
  assert all(abs(v-retention[c+'__L'+str(l)][rlabel][k])<1e-12 for k,v in row[label].items())
result['heldout_spectra_rows_replayed']=len(reported)
paired={}
for c in sets:
 for k in [4,8]:
  for seed in [20261021,20261022]:paired[f'{c}__random{seed}_minus_pca{k}']=effect(groups,'subspace_'+c,f'random{seed}_rank{k}',f'pca{k}')
 for k in [2,4,8]:paired[f'{c}__unclamped_minus_clamped{k}']=effect(groups,'subspace_'+c,f'pca{k}_unclamped',f'pca{k}')
result['matched_control_differences']=paired
bound_summary={}
for b in bounds:
 c,t=b['corpus'],b['variant'];a=bound_summary.setdefault(c+'__'+t,{'outside':0,'count':0,'min':1.0,'max':0.0})
 assert len(b['layers'])==14
 for l,d in b['layers'].items():
  assert d['count']==8176 and 0<=d['below_zero']+d['above_one']<=d['count']
  if t in ['pca0','pca16']:assert d['below_zero']==d['above_one']==0
  a['outside']+=d['below_zero']+d['above_one'];a['count']+=d['count'];a['min']=min(a['min'],d['min']);a['max']=max(a['max'],d['max'])
for a in bound_summary.values():a['fraction_outside_before_clamp']=a['outside']/a['count']
result.update(centered_development_covariance_replayed=True,max_covariance_error=max(cov_errors),max_orthonormality_error=max(orth_errors),max_eigen_equation_error=max(eigen_errors),max_cpu_fullrank_projection_error=max(fullrank_errors),rank16_is_oracle_fastpath_not_projection=True,seeded_nested_random_qr_exact=True,native_mean_and_oracle_anchors_exact=True,pca4_source_state_invariance_exact=True,comparisons=comparisons,vs_oracle=vs_oracle,native_gate_retention=retention,projection_bound_summary=bound_summary,bound_replay_limit='Dynamic recipient gate vectors are not retained; bounds records are structurally checked and aggregated, not independently reconstructed for each low-rank forward.',peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['comparisons','vs_oracle','native_gate_retention','projection_bound_summary']},indent=2))
