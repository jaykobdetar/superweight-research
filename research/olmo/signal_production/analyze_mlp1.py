"""Replay the final L1 MLP from immutable cached normalized inputs.
Not a full-model forward or a downstream behavioral intervention.
"""
from pathlib import Path
import json,hashlib,csv,time,resource,os
import numpy as np
R=Path(__file__).resolve().parent;O=R/'results';O.mkdir(exist_ok=True)
B=Path(os.environ.get('OLMO_SOURCE_ROOT',str(R/'inputs/olmo_training_origins')))
REV='d7cbab742d80589e714b1a2d7f838dcd21cbe143';D=2048;N=8192;J=1710;K=1764
meta=json.loads((B/'phase2/results/main/float32.json').read_text());assert meta['revision']==REV
zs=np.load(B/'phase2/results/source_states/main.npz');zq=np.load(B/'combined/results/query_states/main.npz');qm=json.loads((B/'combined/results/query_states/main.json').read_text());assert qm['commit']==REV
refs=[];labels=[];hs=[]
for i,t in enumerate([510,15,187,535,209]):
 refs.append(meta['single_tokens'][str(t)]);labels.append(['source0',str(t)]);hs.append(zs['normalized_pre_mlp1'][i])
by={(a['panel'],a['condition']):a for a in meta['observations']}
for lab,h in zip(qm['labels'],zq['normalized_pre_mlp1']):refs.append(by[tuple(lab)]);labels.append(lab);hs.append(h)
h=np.stack(hs);gatew=np.memmap(R/'tensors/gate_proj.f32',dtype='<f4',mode='r',shape=(N,D));upw=np.memmap(R/'tensors/up_proj.f32',dtype='<f4',mode='r',shape=(N,D));down=np.memmap(R/'tensors/down_proj.f32',dtype='<f4',mode='r',shape=(D,N))
checks={'revision':REV,'state_count':len(h),'scope':'FP32 cached-prefix L1 MLP replay; no new prefix forward and no downstream layers'}
assert np.array_equal(down[K],np.fromfile(B/'sources/trajectory/main/down_row1764.f32',dtype='<f4'))
for jj,j in enumerate(qm['neuron_ids']):
 assert np.array_equal(gatew[j],zq['gate'][jj]) and np.array_equal(upw[j],zq['up'][jj])
 assert np.array_equal(zq['gate'][jj],zs['gate'][jj]) and np.array_equal(zq['up'][jj],zs['up'][jj])
 assert down[K,j]==zq['down'][jj]==zs['down'][jj]
checks['all_cached_rows_bitwise_match']=True
g=h@gatew.T;v=h@upw.T;act=g*np.exp(-np.logaddexp(np.float32(0),-g))*v
expected=np.array([a['contribution'][0 if i<5 else -1] for i,a in enumerate(refs)])
known=act[:,qm['neuron_ids']]*down[K,qm['neuron_ids']]
checks['known_contribution_max_scaled_error']=float(np.max(abs(known-expected)/np.maximum(abs(expected),1)))
assert checks['known_contribution_max_scaled_error']<1e-4
norms=np.array([a['pre_mlp1_centered_norm'][0 if i<5 else -1] for i,a in enumerate(refs)],dtype=np.float64)
hc=h.astype(np.float64)-h.astype(np.float64).mean(axis=1,keepdims=True)
carry=hc*(norms/np.linalg.norm(hc,axis=1))[:,None]
hl=carry/np.sqrt(np.mean(carry*carry,axis=1,keepdims=True)+1e-5)
checks['normalized_state_reconstruction_max_abs']=float(np.max(abs(hl-h)))
assert checks['normalized_state_reconstruction_max_abs']<2e-5
mlp_raw=act@down.T
without_act=act.copy();without_act[:,J]=0
without_raw=without_act@down.T
mlp=mlp_raw.astype(np.float64);mlp-=mlp.mean(axis=1,keepdims=True)
col=np.array(down[:,J],dtype=np.float64);col-=col.mean();primary=act[:,J,None].astype(np.float64)*col
other=without_raw.astype(np.float64);other-=other.mean(axis=1,keepdims=True)
checks['direct_other_vs_subtracted_primary_max_abs']=float(np.max(abs(other-(mlp-primary))))
assert checks['direct_other_vs_subtracted_primary_max_abs']<0.0002
clean=carry+mlp;column=carry+other
axis=np.full(D,-1/D);axis[K]+=1;axis/=np.linalg.norm(axis)
row_raw=mlp_raw.copy();row_raw[:,K]=without_raw[:,K]
row_mlp=row_raw.astype(np.float64);row_mlp-=row_mlp.mean(axis=1,keepdims=True)
row=carry+row_mlp
def scalar(x):return float(x@axis)
def geo(x):
 a=scalar(x);rad=float(np.linalg.norm(x));return {'axis_projection':a,'centered_norm':rad,'alignment':a/rad,'orthogonal_norm':float(np.linalg.norm(x-a*axis)),'normalized_coordinate1764':float(x[K]/np.sqrt(np.mean(x*x)+1e-5))}
column_means=np.mean(down,axis=0,dtype=np.float64)
axis_weights=(np.array(down[K],dtype=np.float64)-column_means)/np.sqrt(1-1/D)
axis_writes=act.astype(np.float64)*axis_weights
raw_writes=act.astype(np.float64)*np.array(down[K],dtype=np.float64)
rows=[];top=[];geometries=[];factor=[]
for i,lab in enumerate(labels):
 cp=scalar(carry[i]);op=scalar(other[i]);pp=scalar(primary[i]);den=cp+op
 rows.append({'panel':lab[0],'condition':lab[1],'carry_axis':cp,'other_mlp_axis':op,'primary_axis':pp,'remaining_axis':den,'other_fraction_signed':op/den if den else None,'primary_gate':float(g[i,J]),'primary_up':float(v[i,J]),'primary_scalar_write':float(raw_writes[i,J]),'clean_alignment':geo(clean[i])['alignment'],'row_alignment':geo(row[i])['alignment'],'column_alignment':geo(column[i])['alignment']})
 order=np.argsort(-np.abs(axis_writes[i]));order=[int(j) for j in order if j!=J][:20]
 for rank,j in enumerate(order,1):top.append({'panel':lab[0],'condition':lab[1],'rank':rank,'neuron':j,'gate':float(g[i,j]),'up':float(v[i,j]),'activation':float(act[i,j]),'down1764':float(down[K,j]),'raw_write':float(raw_writes[i,j]),'axis_write':float(axis_writes[i,j])})
 conditions={'clean':clean[i],'row_deleted':row[i],'column_deleted':column[i],'column_no_carry_axis':column[i]-cp*axis,'column_no_other_mlp_axis':column[i]-op*axis,'column_no_both_axis':column[i]-(cp+op)*axis,'primary_only':primary[i]}
 for name,x in conditions.items():geometries.append({'panel':lab[0],'condition':lab[1],'intervention':name,**geo(x)})
 if i>=5 and lab[1]=='newline_R':
  e=labels.index([lab[0],'newline_R_edge_H14'])
  for ga in ['R','E']:
   for ua in ['R','E']:
    gg=float(g[i if ga=='R' else e,J]);uu=float(v[i if ua=='R' else e,J]);cc=gg*np.exp(-np.logaddexp(0,-gg))*uu*float(down[K,J])
    factor.append({'panel':lab[0],'gate_donor':ga,'up_donor':ua,'gate':gg,'up':uu,'contribution':float(cc)})
for name,data in [('source_decomposition',rows),('alternate_writers_top20',top),('source_geometry_interventions',geometries),('h14_factorial',factor)]:
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (O/(name+'.json')).write_text(json.dumps(data,indent=2))
np.savez_compressed(O/'mlp1_replay.npz',labels=np.array(labels),normalized_pre_mlp1=h,gate=g,up=v,activation=act,axis_writes=axis_writes,raw_writes=raw_writes,carry=carry,mlp=mlp,other_mlp=other,clean=clean,row=row,column=column)
checks['primary_column_squared_fraction']=float(down[K,J]**2/np.sum(np.array(down[:,J],dtype=np.float64)**2))
checks['coordinate8041_bound']=float(abs(down[K,8041])*1024*(np.linalg.norm(gatew[8041].astype(np.float64)-gatew[8041].mean(dtype=np.float64))*np.linalg.norm(upw[8041].astype(np.float64)-upw[8041].mean(dtype=np.float64))+abs((gatew[8041].astype(np.float64)-gatew[8041].mean(dtype=np.float64))@(upw[8041].astype(np.float64)-upw[8041].mean(dtype=np.float64)))))
checks['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;assert checks['peak_rss_kib']<1024*1024
checks['all_checks_pass']=True;(O/'replay_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
for row in rows:print(row)
for lab in labels[:5]:print('TOP',lab,[x for x in top if x['panel']==lab[0] and x['condition']==lab[1]][:5])
