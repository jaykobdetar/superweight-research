"""CPU replay of source/boundary down-column component factorial."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'column_factorial_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta and meta['forward_passes']==1200
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
for field,p in [('script_sha256',ROOT/'column_factorial.py'),('design_sha256',ROOT/'COLUMN_FACTORIAL.md'),('tokens_parent_sha256',ROOT/'upstream_run/tokens.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4);assert result['rows']==960 and result['groups']==240
traces=torch.load(OUT/'source_traces.pt',map_location='cpu',weights_only=True);assert len(traces)==960
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);oldsets=torch.load(ROOT/'upstream_run/tokens.pt',map_location='cpu',weights_only=True)['evaluation']
confirm=torch.load(ROOT/'confirmation_run/per_token.pt',map_location='cpu',weights_only=True);confirm={k:{z['window']:z for z in v}for k,v in confirm.items()}
mlp=torch.load(ROOT/'upstream_mlp_run/per_token.pt',map_location='cpu',weights_only=True);mlp={k:{z['window']:z for z in v}for k,v in mlp.items()}
up=torch.load(ROOT/'upstream_run/per_token.pt',map_location='cpu',weights_only=True);up={k:{z['window']:z for z in v}for k,v in up.items()}
column=torch.load(OUT/'down_column1710.pt',map_location='cpu',weights_only=True);assert column.shape==(2048,)and column[1764]==.63232421875
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==240 and all(z[k]for z in checks for k in ['row_implementation_loss_exact','row_implementation_pred_exact','intermediate_unchanged'])
groupnames=sorted({k.split('__')[0]for k in groups});effects={};desc={}
for g in groupnames:
 for i in range(4):
  d={t:traces[f'{g}__{t}__{i}']for t in ['clean','delete_row','delete_complement','delete_column']};b=d['clean']
  for t,z in d.items():
   assert torch.equal(z['intermediate'],b['intermediate'])
   if g.startswith('boundary_'):assert all(torch.equal(z[k][0],b[k][0])for k in z)
   else:assert all(torch.equal(z[k][0],z[k][1])for k in z)
  target=1
  row=d['delete_row']['input'][target];comp=d['delete_complement']['input'][target];full=d['delete_column']['input'][target];clean=b['input'][target]
  assert torch.equal(row[:1764],clean[:1764])and torch.equal(row[1765:],clean[1765:])and comp[1764]==clean[1764]
  assert row[1764]==full[1764]and torch.equal(comp[:1764],full[:1764])and torch.equal(comp[1765:],full[1765:])
  for k in ['k','v']:assert all((z[k].abs()<=8).all()for z in d.values())
 for t in ['delete_row','delete_complement','delete_column']:
  effects[g+'__'+t]=compare(raw,groups,g,t,'clean')
  desc[g+'__'+t]={'normalized_target_coordinate_mean':float(np.mean([float(traces[f'{g}__{t}__{i}']['normalized'][1,1764])for i in range(4)])),'target_input_delta_mean':float(np.mean([float(traces[f'{g}__{t}__{i}']['input'][1,1764].float()-traces[f'{g}__clean__{i}']['input'][1,1764].float())for i in range(4)]))}
for c,ws in sets.items():
 for i,xx in enumerate(ws):
  assert torch.equal(xx,oldsets[c][i])
  for identity,tid in [('natural',None),('The',510),('period',15)]:
   x=xx.clone()
   if tid is not None:x[0]=tid
   g=f'first_{c}_{identity}';h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
   for t in ['clean','delete_row','delete_complement','delete_column']:assert groups[g+'__'+t][i]['token_sha256']==h and groups[g+'__'+t][i]['score_start']==1 and len(raw[g+'__'+t][i]['loss'])==510
   if identity!='period':
    for t,pt in [('clean','clean'),('delete_row','deleted')]:
     for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],confirm[f'confirm_{c}_{identity}__{pt}'][i][k])
  for identity,tid in [('The',510),('period',15)]:
   for p in [128,256]:
    x=xx[256-p:].clone();x[0]=x[p]=tid;h=hashlib.sha256(x.numpy().tobytes()).hexdigest();priorg=f'upstream_{c}_{identity}_{p}'
    for world,pt in [('ordinary','ordinary'),('isolated','isolated'),('induced','ordinary_H12')]:
     g=f'boundary_{c}_{identity}_{p}_{world}'
     for t in ['clean','delete_row','delete_complement','delete_column']:assert groups[g+'__'+t][i]['token_sha256']==h and groups[g+'__'+t][i]['score_start']==p+1 and len(raw[g+'__'+t][i]['loss'])==254
     for k in ['loss','pred']:assert torch.equal(raw[g+'__clean'][i][k],mlp[priorg+'__'+pt][i][k])
     reference=mlp[priorg+'__ordinary_H12_delete_boundary'][i]if world=='induced'else up[priorg+'__'+world+'_delete_boundary'][i]
     for k in ['loss','pred']:assert torch.equal(raw[g+'__delete_row'][i][k],reference[k])
interaction_rows=list(csv.DictReader((OUT/'interaction_contrasts.csv').open()));assert len(interaction_rows)==180;interactions={}
for row in interaction_rows:
 g,name=row['group'],row['contrast'];coef={'joint_minus_individuals':{'delete_column':1,'delete_row':-1,'delete_complement':-1,'clean':1},'column_vs_row':{'delete_column':1,'delete_row':-1},'complement_vs_clean':{'delete_complement':1,'clean':-1}}[name]
 # Average the paired raw token contrast within each window before bootstrap.
 vals=[sum(w*float(raw[g+'__'+t][i]['loss'].double().mean())for t,w in coef.items())for i in range(4)];q=interval(vals);interactions[g+'__'+name]=q
 for a,b in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
 assert int(row['windows'])==4 and abs(float(row['ratio'])-math.exp(q['mean']))<1e-12
row_energy=float(column[1764].float().square());total_energy=float(column.float().square().sum())
result.update(interaction_rows_replayed=180,all_world_specific_baselines_verified=True,prior_source_and_boundary_anchors_exact=True,runtime_row_control_flags=240,unchanged_pre_projection_intermediates=True,L2_input_component_partition_exact=True,boundary_source0_unchanged=True,known_column_target_weight_exact=True,column_target_fraction_of_squared_norm=row_energy/total_energy,column_l2_norm=math.sqrt(total_energy),effects=effects,interaction_effects=interactions,trace_descriptions=desc,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limits='Runtime control asserts full-sequence equality against established Source implementation; independent prior-artifact replay covers retained scores. L2 keys are clipped pre-RoPE. Source0 panels have duplicate trace rows for position0.')
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['effects','interaction_effects','trace_descriptions']},indent=2))
