"""Offline numerical audit of stored replay artifacts; NOT full-model inference."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path(__file__).resolve().parent;O=R/'results';z=np.load(O/'mlp1_replay.npz');d=np.load(R/'inputs/compact_down_projection.npz');D=2048;J=1710;axis=np.full(D,-1/D);axis[1764]+=1;axis/=np.linalg.norm(axis)
wrow=np.fromfile(R/'inputs/olmo_training_origins/sources/trajectory/main/down_row1764.f32',dtype='<f4');labs=z['labels'].tolist();act=z['activation'];cols={int(j):d['columns'][:,i].astype(np.float64) for i,j in enumerate(d['neuron_ids'])}
def vec(i,ids):
 c=np.stack([cols[j]-cols[j].mean() for j in ids],axis=1);return c@act[i,ids].astype(np.float64)
assert np.max(abs(z['raw_writes']-act.astype(np.float64)*wrow))<1e-10
aw=act.astype(np.float64)*(wrow.astype(np.float64)-d['column_means'])/np.sqrt(1-1/D)
assert np.max(abs(z['axis_writes']-aw))<1e-10
errors=[]
for a in json.loads((O/'source_decomposition.json').read_text()):
 i=labs.index([a['panel'],a['condition']]);v=z['column'][i]
 errors.extend([abs(a['carry_axis']-z['carry'][i]@axis),abs(a['other_mlp_axis']-z['other_mlp'][i]@axis),abs(a['remaining_axis']-v@axis),abs(a['column_alignment']-v@axis/np.linalg.norm(v))])
assert max(errors)<1e-10
errs=[]
for a in json.loads((O/'alternate_writer_interventions.json').read_text()):
 i=labs.index([a['panel'],a['condition']]);carry=z['carry'][i];base=z['column'][i];name=a['intervention']
 choices={'column_deleted':base,'column_drop7579':base-vec(i,[7579]),'only7579_plus_carry':carry+vec(i,[7579]),'column_drop_source_set':base-vec(i,[7579,7845,6947,2637]),'source_set_only_plus_carry':carry+vec(i,[7579,7845,6947,2637]),'column_drop_query_pair':base-vec(i,[3950,4786]),'query_pair_only_plus_carry':carry+vec(i,[3950,4786]),'column_drop_newline_positive_trio':base-vec(i,[2476,2637,7845]),'carry_only':carry};v=choices[name]
 errs.append(abs(a['alignment']-v@axis/np.linalg.norm(v)))
assert max(errs)<1e-6
q=np.load(R/'inputs/olmo_training_origins/combined/results/query_states/main.npz');g=q['gate'][3].astype(np.float64);g-=g.mean();u=q['up'][3].astype(np.float64);u-=u.mean();b=abs(float(wrow[8041]))*1024*(np.linalg.norm(g)*np.linalg.norm(u)+abs(g@u));assert abs(b-.16032984931200406)<1e-14
report={'scope':'Numerical consistency of compact frozen arrays and included selected weight rows/columns. Not full checkpoint or independent model inference.','source_decomposition_max_abs_error':max(errors),'alternate_intervention_max_alignment_error':max(errs),'coordinate8041_bound':b,'all_checks_pass':True}
print(json.dumps(report,indent=2))
