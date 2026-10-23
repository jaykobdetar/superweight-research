"""Exploratory surgical L1-output interventions; geometry, not downstream loss."""
from pathlib import Path
import numpy as np,json,csv,resource
R=Path(__file__).resolve().parent;O=R/'results';z=np.load(O/'mlp1_replay.npz');D=2048;J=1710;K=1764
down=np.memmap(R/'tensors/down_proj.f32',dtype='<f4',mode='r',shape=(D,8192));act=z['activation'];labs=z['labels'].tolist();axis=np.full(D,-1/D);axis[K]+=1;axis/=np.linalg.norm(axis)
def vector(i,ids):
 m=np.array(down[:,ids],dtype=np.float64);m-=m.mean(axis=0,keepdims=True);return m@act[i,ids].astype(np.float64)
def geom(v):
 n=np.linalg.norm(v);p=v@axis;return {'axis':float(p),'norm':float(n),'alignment':float(p/n),'orthogonal_norm':float(np.linalg.norm(v-p*axis))}
rows=[];checks=[];cancellation=[]
for i,lab in enumerate(labs):
 a=act[i].copy();a[J]=0
 direct=np.array(a@down.T,dtype=np.float64);direct-=direct.mean();other=z['mlp'][i]-vector(i,[J]);err=float(np.max(abs(direct-other)));checks.append({'panel':lab[0],'condition':lab[1],'direct_exclusion_vs_subtraction_max_abs':err});assert err<0.0002
 # The direct exclusion avoids subtracting the dominant scalar at the output.
 carry=z['carry'][i];base=carry+direct
 conditions={'column_deleted':base,'column_drop7579':base-vector(i,[7579]),'only7579_plus_carry':carry+vector(i,[7579]),'column_drop_source_set':base-vector(i,[7579,7845,6947,2637]),'source_set_only_plus_carry':carry+vector(i,[7579,7845,6947,2637]),'column_drop_query_pair':base-vector(i,[3950,4786]),'query_pair_only_plus_carry':carry+vector(i,[3950,4786]),'column_drop_newline_positive_trio':base-vector(i,[2476,2637,7845]),'carry_only':carry}
 for name,v in conditions.items():rows.append({'panel':lab[0],'condition':lab[1],'intervention':name,**geom(v)})
 writes=z['axis_writes'][i].copy();writes[J]=0
 cancellation.append({'panel':lab[0],'condition':lab[1],'negative_other_sum':float(writes[writes<0].sum()),'positive_other_sum':float(writes[writes>0].sum()),'net_other_sum':float(writes.sum()),'net_abs_over_total_abs':float(abs(writes.sum())/abs(writes).sum())})
mix=[]
for panel in ['water_cycle','library','bread','programming']:
 ir=labs.index([panel,'newline_R']);ie=labs.index([panel,'newline_R_edge_H14'])
 for c in 'RE':
  for m in 'RE':
   ci=ir if c=='R' else ie;mi=ir if m=='R' else ie
   other=z['other_mlp'][mi];v=z['carry'][ci]+other
   mix.append({'panel':panel,'carry_donor':c,'other_mlp_donor':m,**geom(v)})
for name,data in [('alternate_writer_interventions',rows),('cancellation',cancellation),('h14_nonprimary_component_swaps',mix)]:
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (O/(name+'.json')).write_text(json.dumps(data,indent=2))
meta={'label':'Post-hoc exploratory source-output interventions selected after inspecting decomposition','source_set':[7579,7845,6947,2637],'query_pair':[3950,4786],'newline_positive_trio':[2476,2637,7845],'scope':'Full columns removed or retained at L1 output; source geometry only. Neither necessity for loss nor sufficiency for model function is measured. No holdout claim.','checks':checks,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(O/'geometry_intervention_checks.json').write_text(json.dumps(meta,indent=2))
for i in range(5):
 print('SOURCE',labs[i]);print([a for a in rows if a['panel']==labs[i][0] and a['condition']==labs[i][1]]);print(cancellation[i])
print('H14 swaps',mix)
