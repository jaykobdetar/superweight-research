"""CPU-only scalar-factor/routing numerical audit using saved intermediates."""
import csv,hashlib,inspect,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
import audit_common as common
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'upstream_mlp_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta and meta['forward_passes']==1664
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
for field,p in [('script_sha256',ROOT/'upstream_mlp.py'),('design_sha256',ROOT/'UPSTREAM_MLP.md'),('tokens_parent_sha256',ROOT/'upstream_run/tokens.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
# Same raw audit with this experiment's ordinary baseline, without editing shared helper.
ns=dict(vars(common));exec(inspect.getsource(common.load_audit).replace("'__clean'","'__ordinary'"),ns)
raw,groups,result=ns['load_audit'](OUT,expected_windows=4,require_tables=False);assert result['rows']==1664 and result['groups']==416
prior=torch.load(ROOT/'upstream_run/per_token.pt',map_location='cpu',weights_only=True);prior={k:{z['window']:z for z in v}for k,v in prior.items()}
factors=torch.load(OUT/'factor_donors.pt',map_location='cpu',weights_only=True);obs=json.loads((OUT/'factor_observations.json').read_text());assert len(factors)==len(obs)==1664
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==192
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);oldsets=torch.load(ROOT/'upstream_run/tokens.pt',map_location='cpu',weights_only=True)['evaluation']
weight=.63232421875;cpu_errors=[];fg={};og={}
for z in obs:
 g,t,i=z['group'],z['trial'],z['window'];key=f'{g}__{t}__{i}';d=factors[key];base=factors[f'{g}__ordinary__{i}'];fg.setdefault(g,{}).setdefault(t,{})[i]=d;og.setdefault(g,{}).setdefault(t,{})[i]=z
 assert all(torch.equal(d[k][0],base[k][0])for k in ['norm_input','norm_output','gate','up','intermediate','clipped_H12_values'])
 assert z['gate']==float(d['gate'][1])and z['up']==float(d['up'][1])and z['intermediate']==float(d['intermediate'][1,1710])
 assert z['pulse']==float(d['intermediate'][1,1710].float()*weight)and z['source_pulse']==float(d['intermediate'][0,1710].float()*weight)
 calc=torch.nn.functional.silu(d['gate'][1])*d['up'][1];err=float(calc.float()-d['intermediate'][1,1710].float());cpu_errors.append(err)
 assert z['cpu_factor_reconstruction_exact']==torch.equal(calc,d['intermediate'][1,1710])and z['cpu_factor_reconstruction_difference']==err
 assert z['token_sha256']==groups[g+'__'+t][i]['token_sha256']
for c,ws in sets.items():
 for i,xx in enumerate(ws):
  assert torch.equal(xx,oldsets[c][i])
  for identity,tid in [('The',510),('period',15)]:
   for p in [128,256]:
    x=xx[256-p:].clone();x[0]=x[p]=tid;g=f'upstream_{c}_{identity}_{p}';h=hashlib.sha256(x.numpy().tobytes()).hexdigest();a=fg[g]
    for t,ds in a.items():assert groups[g+'__'+t][i]['token_sha256']==h and groups[g+'__'+t][i]['score_start']==p+1 and len(raw[g+'__'+t][i]['loss'])==254
    for t,pt in [('ordinary','ordinary'),('isolated','isolated'),('ordinary_H12','ordinary_top1'),('isolated_H12','isolated_top1')]:
     for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],prior[g+'__'+pt][i][k])
    for recipient,other in [('ordinary','isolated'),('isolated','ordinary')]:
     for kinds in [('gate',),('up',),('gate','up')]:
      d=a[recipient+'_swap_'+'_'.join(kinds)][i];b=a[recipient][i];donor=a[other][i]
      for k in ['norm_input','norm_output','head_context','clipped_H12_values']:assert torch.equal(d[k],b[k])
      for k in ['gate','up']:assert d[k][1]==(donor if k in kinds else b)[k][1]
      assert torch.equal(d['intermediate'][1,:1710],b['intermediate'][1,:1710])and torch.equal(d['intermediate'][1,1711:],b['intermediate'][1,1711:])
      if len(kinds)==2:assert d['intermediate'][1,1710]==donor['intermediate'][1,1710]
    for kinds in [('gate',),('up',),('gate','up')]:
     d=a['induced_restore_'+'_'.join(kinds)][i];b=a['ordinary_H12'][i];donor=a['ordinary'][i]
     for k in ['norm_input','norm_output','head_context','clipped_H12_values']:assert torch.equal(d[k],b[k])
     for k in ['gate','up']:assert d[k][1]==(donor if k in kinds else b)[k][1]
     if len(kinds)==2:assert d['intermediate'][1,1710]==donor['intermediate'][1,1710]
    for k,v in a['ordinary_H12'][i].items():assert torch.equal(v,a['ordinary_H12_delete_boundary'][i][k])
    for mode in ['value0','valuep','zero']:
     d=a['H12_'+mode][i];b=a['ordinary'][i];expected=b['head_context'].clone();expected[:,1536:1664]=0 if mode=='zero'else d['clipped_H12_values'][0 if mode=='value0'else 1]
     assert torch.equal(d['head_context'],expected)and torch.equal(d['clipped_H12_values'],b['clipped_H12_values'])
    for t,target in [('H12_value0','ordinary_H12'),('H12_selfmask','H12_valuep'),('isolated_complement31','H12_valuep')]:
     for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],raw[g+'__'+target][i][k])
     for k in ['norm_input','norm_output','gate','up','intermediate','head_context','clipped_H12_values']:assert torch.equal(a[t][i][k],a[target][i][k])
    assert torch.equal(a['ordinary'][i]['clipped_H12_values'][0],a['isolated'][i]['head_context'][0,1536:1664])
for z in checks:
 g,i=z['group'],z['window'];a=fg[g]
 if 'world'in z:
  world=z['world'];other='isolated'if world=='ordinary'else'ordinary';assert z['donor_intermediate_exact']and a[world+'_swap_gate_up'][i]['intermediate'][1,1710]==a[other][i]['intermediate'][1,1710]
 else:
  a,b=a['H12_valuep'][i],a['H12_selfmask'][i]
  assert z['self_context_vs_mask_intermediate_maxdiff']==float((a['intermediate'].float()-b['intermediate'].float()).abs().max())
  assert z['self_context_vs_mask_pulse_diff']==float((a['intermediate'][1,1710].float()-b['intermediate'][1,1710].float())*weight)
effects={};pulse_summary={};route_controls={}
for g in fg:
 for t in fg[g]:
  reference='ordinary_H12'if t.startswith('induced_restore')or t=='ordinary_H12_delete_boundary'else'isolated'if t.startswith('isolated_')else'ordinary'
  effects[g+'__'+t]=common.compare(raw,groups,g,t,reference)
  pulse_summary[g+'__'+t]={k:common.interval([z[k]for z in og[g][t].values()])for k in ['pulse','gate','up','intermediate']}
 route_controls[g]=common.compare(raw,groups,g,'H12_selfmask','H12_valuep')
factor_rows=list(csv.DictReader((OUT/'factor_summary.csv').open()));assert len(factor_rows)==416
for row in factor_rows:
 values=pulse_summary[row['group']+'__'+row['trial']];assert int(row['windows'])==4
 for k in ['gate','up','intermediate','pulse']:
  for column,stat in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[column])-values[k][stat])<1e-12
result.update(factor_summary_rows_replayed=len(factor_rows),prior_world_and_H12_output_anchors_exact=True,factor_patch_localization_verified=True,both_factor_scalar_closure_exact=True,source0_unchanged=True,boundary_deletion_inputs_unchanged=True,value_route_contexts_verified=True,current_value0_equals_isolated_H12_donor_condition=True,selfmask_valuep_and_isolated_complement31_equal=True,all_checks_replayed=192,cpu_half_reconstruction_mismatches=sum(e!=0 for e in cpu_errors),max_cpu_half_reconstruction_error=max(map(abs,cpu_errors)),effects=effects,pulse_summary=pulse_summary,route_controls=route_controls,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['effects','pulse_summary','route_controls']},indent=2))
