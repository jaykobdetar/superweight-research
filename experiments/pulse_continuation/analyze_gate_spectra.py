"""Held-out descriptive gate spectra, separate from causal intervention scores."""
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;out=ROOT/'gate_subspaces_run'
fit=torch.load(out/'fitted_subspaces.pt',weights_only=True,map_location='cpu');held=torch.load(out/'heldout_native_gates.pt',weights_only=True,map_location='cpu');rows=[]
for c in ('wiki','hume','plato','python'):
 for l in range(2,16):
  x=torch.cat([held[c,i][l][:,1:].T.double() for i in range(8)],0);own=x-x.mean(0);dev=x-fit[l]['mean'].double();basis=fit[l]['pca'].double();cov=own.T@own/(len(x)-1)
  eig=torch.linalg.eigvalsh(cov).flip(0);den=float(own.square().sum());dd=float(dev.square().sum())
  rows.append({'corpus':c,'layer':l,'queries':len(x),'heldout_mean':x.mean(0).tolist(),'centered_eigenvalues':eig.tolist(),
   'centered_pc1_share':float(eig[0]/eig.sum()),'centered_effective_rank90':int((eig.cumsum(0)/eig.sum()<.9).sum())+1,
   'dev_basis_retained_centered_energy':{str(k):float((own@basis[:,:k]).square().sum())/den for k in (1,2,4,8,12,16)},
   'dev_basis_retained_deviation_from_dev_mean':{str(k):float((dev@basis[:,:k]).square().sum())/dd for k in (1,2,4,8,12,16)}})
(out/'heldout_spectra.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Saved',len(rows),'corpus/layer centered spectra; separate dev-centered deviations.')
