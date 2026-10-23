from pathlib import Path
import json,hashlib,numpy as np,resource
R=Path('/workspace/shared/olmo_signal_production');B=Path('/workspace/shared/olmo_recovered/olmo_training_development_complete/olmo_training_origins');D=2048;N=8192;K=1764;J=1710
out={};manifest=json.loads((B/'BUNDLE_MANIFEST.json').read_text())['files']
files=['sources/trajectory/main/header.json','sources/trajectory/main/index.json','sources/trajectory/main/down_row1764.f32','phase2/results/main/float32.json','phase2/results/source_states/main.npz','combined/results/query_states/main.npz','combined/results/query_states/main.json','phase2/checks/source_cross_time.py','combined/checks/query_cross_time.py']
out['input_manifest_checks']={}
for f in files:
 p=B/f;m=manifest[f];out['input_manifest_checks'][f]=(p.stat().st_size==m['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'])
header=json.loads((B/files[0]).read_text());idx=json.loads((B/files[1]).read_text());off=8+(B/files[0]).stat().st_size
out['tensor_receipt_checks']={};Ws={}
for typ,shape in [('gate_proj',(N,D)),('up_proj',(N,D)),('down_proj',(D,N))]:
 p=R/'tensors'/(typ+'.f32');rr=json.loads(Path(str(p)+'.receipt.json').read_text());h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 md=header[rr['key']];expected_url='https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/d7cbab742d80589e714b1a2d7f838dcd21cbe143/'+idx['weight_map'][rr['key']]
 out['tensor_receipt_checks'][typ]={'sha_match':h.hexdigest()==rr['sha256'],'range_match':(rr['start']==off+md['data_offsets'][0] and rr['end']==off+md['data_offsets'][1]-1),'shape_match':list(shape)==md['shape']==rr['shape'],'fp32':md['dtype']=='F32','size_match':p.stat().st_size==rr['size']==rr['received_bytes']==int(np.prod(shape))*4,'url_match':rr['url']==expected_url}
 Ws[typ]=np.memmap(p,dtype='<f4',mode='r',shape=shape)
gw,uw,dw=[Ws[t] for t in ['gate_proj','up_proj','down_proj']]
zs=np.load(B/'phase2/results/source_states/main.npz');zq=np.load(B/'combined/results/query_states/main.npz');qm=json.loads((B/'combined/results/query_states/main.json').read_text());h=np.concatenate([zs['normalized_pre_mlp1'],zq['normalized_pre_mlp1']],axis=0).astype(np.float64);labels=[['source0',str(t)] for t in [510,15,187,535,209]]+qm['labels'];z=np.load(R/'results/mlp1_replay.npz');out['labels_match']=labels==z['labels'].tolist();out['h_cached_exact']=np.array_equal(h,z['normalized_pre_mlp1']);out['cached_rows_exact']=all(np.array_equal(w[j],c[k]) for w,c in [(gw,zq['gate']),(uw,zq['up'])] for k,j in enumerate(qm['neuron_ids']))
# Independent high precision on the exact FP32 cache, blockwise full nonlinear MLP.
g=np.empty((len(h),N));u=np.empty_like(g)
for lo in range(0,N,256):
 hi=lo+256;g[:,lo:hi]=h@gw[lo:hi].astype(np.float64).T;u[:,lo:hi]=h@uw[lo:hi].astype(np.float64).T
act=g/(1+np.exp(-g))*u
mlp=np.empty((len(h),D));out['fp64_g_max_delta']=float(abs(g-z['gate']).max());out['fp64_u_max_delta']=float(abs(u-z['up']).max());out['fp64_act_max_delta']=float(abs(act-z['activation']).max())
for lo in range(0,D,128):mlp[:,lo:lo+128]=act@dw[lo:lo+128].astype(np.float64).T
mlp_raw=mlp.copy();mlp-=mlp.mean(axis=1,keepdims=True);out['fp64_mlp_max_delta']=float(abs(mlp-z['mlp']).max())
meta=json.loads((B/'phase2/results/main/float32.json').read_text());by={(a['panel'],a['condition']):a for a in meta['observations']};norms=np.array([meta['single_tokens'][str(t)]['pre_mlp1_centered_norm'][0] for t in [510,15,187,535,209]]+[by[tuple(l)]['pre_mlp1_centered_norm'][-1] for l in qm['labels']]);hc=h-h.mean(axis=1,keepdims=True);carry=hc/np.linalg.norm(hc,axis=1,keepdims=True)*norms[:,None]
# Scalar identity independent of vector dot implementation.
axis=np.full(D,-1/D);axis[K]+=1;axis/=np.linalg.norm(axis);col=dw[:,J].astype(np.float64);col-=col.mean();primary=act[:,J,None]*col
masked_act=act.copy();masked_act[:,J]=0;other_raw=np.empty_like(mlp)
for lo in range(0,D,128):other_raw[:,lo:lo+128]=masked_act@dw[lo:lo+128].astype(np.float64).T
other=other_raw-other_raw.mean(axis=1,keepdims=True);clean=carry+mlp;column=carry+other;row_raw=mlp_raw.copy();row_raw[:,K]=other_raw[:,K];row=carry+row_raw-row_raw.mean(axis=1,keepdims=True)
out['fp64_direct_other_max_delta']=float(abs(other-z['other_mlp']).max());out['fp64_scalar_edge_deleted_max_delta']=float(abs(row-z['row']).max());out['fp64_direct_other_vs_subtraction_max_delta']=float(abs(other-(mlp-primary)).max())
column_means=dw.mean(axis=0,dtype=np.float64);axis_weights=(dw[K].astype(np.float64)-column_means)/np.sqrt(1-1/D);axis_writes=act*axis_weights;raw_writes=act*dw[K].astype(np.float64)
geom=lambda v:{'axis':float(v@axis),'norm':float(np.linalg.norm(v)),'alignment':float(v@axis/np.linalg.norm(v))}
rows=[]
for i,l in enumerate(labels):
 aw=axis_writes[i].copy();aw[J]=0;order=[int(j) for j in np.argsort(-abs(aw))[:10]]
 rows.append({'label':l,'carry_axis':float(carry[i]@axis),'other_axis':float(aw.sum()),'remaining_axis':float(carry[i]@axis+aw.sum()),'primary_raw':float(raw_writes[i,J]),'clean':geom(clean[i]),'single_weight_deleted':geom(row[i]),'column_deleted':geom(column[i]),'negative':float(aw[aw<0].sum()),'positive':float(aw[aw>0].sum()),'top_alternates':[(j,float(aw[j])) for j in order]})
out['selected_recomputation']=rows;out['axis_sum_identity_max_error']=float(abs(axis_writes.sum(axis=1)-mlp@axis).max());out['carry_l2_reconstruction_delta']=float(abs(carry-z['carry']).max());out['h_max_abs_mean']=float(abs(h.mean(axis=1)).max());out['h_norm_sq_max']=float((h*h).sum(axis=1).max())
rowg=gw[8041].astype(np.float64);rowu=uw[8041].astype(np.float64);rowg-=rowg.mean();rowu-=rowu.mean();prod=np.linalg.norm(rowg)*np.linalg.norm(rowu);dot=rowg@rowu;base=(D/2)*(prod+abs(dot));out['8041_bound']={'activation_real_arithmetic':float(base),'raw_coordinate1764':float(base*abs(dw[K,8041])),'centered_axis':float(base*abs(axis_weights[8041])),'whole_column_l2':float(base*np.linalg.norm(dw[:,8041].astype(np.float64))),'centered_column_l2':float(base*np.linalg.norm(dw[:,8041].astype(np.float64)-column_means[8041])),'max_observed_raw':float(abs(raw_writes[:,8041]).max())}
out['primary_raw_column_squared_fraction']=float(dw[K,J]**2/np.dot(dw[:,J].astype(np.float64),dw[:,J].astype(np.float64)));out['primary_centered_axis_squared_fraction']=float((col@axis)**2/(col@col));out['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
p=Path('/workspace/shared/olmo_signal_audit/audit.json');p.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='selected_recomputation'},indent=2));print('SELECTED');print(json.dumps(rows[:5]+[r for r in rows if r['label'][1]=='newline_R_edge_H14'],indent=2))
