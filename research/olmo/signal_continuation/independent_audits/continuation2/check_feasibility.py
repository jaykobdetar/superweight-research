from pathlib import Path
import json,numpy as np,hashlib
P=Path('/workspace/shared/olmo_signal_production');R=Path(__file__).resolve().parent;z=np.load(P/'results/mlp1_replay.npz');f=np.load(P/'continuation/results/feature_feasibility_inputs.npz');saved=json.loads((P/'continuation/results/feature_feasibility.json').read_text());A=f['rows'];D=2048;neurons=[1710,3950,4786];rows=[]
for j in neurons:
 for name in ['gate_proj','up_proj']:
  with open(P/'tensors'/f'{name}.f32','rb') as file:file.seek(j*D*4);v=np.fromfile(file,dtype='<f4',count=D).astype('f8')
  rows.append(v-v.mean())
assert np.array_equal(A,np.stack(rows));U,s,Vt=np.linalg.svd(A,full_matrices=False);assert min(s)>0;labels=z['labels'].tolist();C=np.load(P/'inputs/compact_down_projection.npz');ci={int(j):i for i,j in enumerate(C['neuron_ids'])};aw=np.array([(C['columns'][1764,ci[j]]-C['columns'][:,ci[j]].mean(dtype='f8'))/np.sqrt(1-1/D) for j in neurons]);out=[]
for panel in ['water_cycle','library','bread','programming']:
 ir=labels.index([panel,'newline_R']);ie=labels.index([panel,'newline_R_edge_H14']);hr=z['normalized_pre_mlp1'][ir].astype('f8');he=z['normalized_pre_mlp1'][ie].astype('f8');hr-=hr.mean();he-=he.mean();fr=A@hr;fe=A@he;rad=he@he
 for pd,bd in [('R','E'),('E','R')]:
  b=np.r_[fr[:2] if pd=='R' else fe[:2],fr[2:] if bd=='R' else fe[2:]];hmin=Vt.T@((U.T@b)/s);mn=float(hmin@hmin);sv=[c for c in saved['cases'] if c['panel']==panel and c['primary_donor']==pd and c['pair_donor']==bd][0]
  rr=np.array([v for j in neurons for v in [z['gate'][ir,j],z['up'][ir,j]]]);ee=np.array([v for j in neurons for v in [z['gate'][ie,j],z['up'][ie,j]]]);bb=np.r_[rr[:2] if pd=='R' else ee[:2],rr[2:] if bd=='R' else ee[2:]];hh=Vt.T@((U.T@bb)/s)
  r={'panel':panel,'primary':pd,'backup':bd,'min_norm2_svd':mn,'saved_delta':mn-sv['minimum_squared_norm'],'min_norm2_cached_rounded_factors':float(hh@hh),'margin_vs_ceiling':mn-D,'min_feature_l2_error_lower_bound_if_infeasible':float(s[-1]*max(np.sqrt(mn)-np.sqrt(D),0))}
  if pd=='R':
   h=f[panel+'_primaryR_pairE'];pre=h*np.sqrt(1e-5/(1-float(h@h)/D));back=(pre-pre.mean())/np.sqrt(np.mean((pre-pre.mean())**2)+1e-5);r.update(constructed_constraint_max_error=float(abs(A@h-b).max()),constructed_norm_delta=float(h@h-rad),constructed_mean=float(h.mean()),ln_preimage_max_error=float(abs(back-h).max()))
  out.append(r)
 h=f[panel+'_primaryE_nullR'];v=A@h;ww=v[::2]/(1+np.exp(-v[::2]))*v[1::2]*aw;out.append({'panel':panel,'construction':'E-primary_R-nullspace','primary_constraint_error':float(abs(A[:2]@(h-he)).max()),'norm_delta':float(h@h-rad),'mean':float(h.mean()),'primary_axis':float(ww[0]),'backup_pair_axis':float(ww[1:].sum())})
assert max(abs(r.get('saved_delta',0)) for r in out)<1e-9;assert all(r['margin_vs_ceiling']>0 for r in out if r.get('primary')=='E');assert all(r['margin_vs_ceiling']<0 for r in out if r.get('primary')=='R');assert all(r.get('backup_pair_axis',1)>0 for r in out)
result={'method':'Independent thin-SVD pseudoinverse; six existing row slices verified against saved compact rows; no full projection or new weights.','A_matches_six_existing_rows_exact':True,'gram_condition':float((s[0]/s[-1])**2),'cases':out};(R/'feature_feasibility_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
