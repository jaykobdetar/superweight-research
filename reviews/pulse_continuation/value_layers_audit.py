"""CPU audit of frozen-key source-value layer sets and norm-matched controls."""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'value_layers_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
for field,p in [('script_sha256',ROOT/'value_layers.py'),('tokens_sha256',ROOT/'residual_paths_run/tokens.pt'),('calibration_sha256',ROOT/'cache_run/calibration.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
layers=list(range(2,16));variants={'mean_values':layers,'zero_values':[]}
for l in layers:variants[f'remove_L{l}']=[k for k in layers if k!=l];variants[f'restore_L{l}']=[l]
for label,ss in [('early',list(range(2,6))),('middle',list(range(6,12))),('late',list(range(12,16)))]:variants['restore_'+label]=ss;variants['remove_'+label]=[l for l in layers if l not in ss]
for end in [3,5,8,11]:variants[f'restore_prefix_through{end}']=list(range(2,end+1))
for first in [3,6,9,12]:variants[f'restore_suffix_from{first}']=list(range(first,16))
assert variants==meta['variants']and len(variants)==44
unique=len(set(tuple(v)for v in variants.values()));assert meta['forward_passes']==32*(unique+3)
raw,groups,result=load_audit(OUT,require_tables=False);assert result['rows']==1504 and result['groups']==188
result['summary_tables_available']=(OUT/'summary.csv').exists()and(OUT/'contrasts.csv').exists()
frozen=torch.load(OUT/'frozen_interfaces.pt',map_location='cpu',weights_only=True)
packed=torch.load(ROOT/'cache_run/calibration.pt',map_location='cpu',weights_only=True)['wiki']
mean=frozen['mean'];norm_errors=[];random_max=0
for l in range(16):
 for j,k in enumerate('kv'):assert torch.equal(mean[l][k],packed[l][j].transpose(1,2).reshape(1,1,2048).contiguous())
for seed in [20261021,20261022]:
 gen=torch.Generator().manual_seed(seed)
 for l in layers:
  v=mean[l]['v'].float().reshape(16,128);z=torch.randn(v.shape,generator=gen);z*=v.norm(dim=-1,keepdim=True)/z.norm(dim=-1,keepdim=True)
  saved=frozen['random_values'][seed][l];assert torch.equal(saved,z.reshape(1,1,2048).half())
  assert z.abs().max()<8 and saved.abs().max()<=8
  norm_errors.append(float(((saved.float().reshape(16,128).norm(dim=-1)-v.norm(dim=-1))/v.norm(dim=-1)).abs().max()));random_max=max(random_max,float(saved.abs().max()))
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);prior=torch.load(ROOT/'quiet_anchor_run/tokens.pt',map_location='cpu',weights_only=True)
old=torch.load(ROOT/'quiet_anchor_run/per_token.pt',map_location='cpu',weights_only=True);old={k:{z['window']:z for z in v}for k,v in old.items()}
aliases=[];comparisons={};relative_to_mean={};relative_to_zero={}
for c,ws in sets.items():
 g='values_'+c
 for i,x in enumerate(ws):
  assert torch.equal(x,prior[c][i])
  for t,ot in [('clean','clean'),('mean_values','K_mean_V_mean_clean'),('zero_values','K_mean_V_zero_clean')]:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],old['quiet_'+c+'__'+ot][i][k])
 seen={}
 for name,ss in variants.items():
  ss=tuple(ss)
  if ss in seen:
   aliases.append([c,name,seen[ss]])
   for i in range(8):
    for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+name][i][k],raw[g+'__'+seen[ss]][i][k])
  seen[ss]=name
 for key,rows in groups.items():
  if not key.startswith(g+'__'):continue
  assert all(z['score_start']==1 and z['scored_tokens']==510 for z in rows.values())
  t=key.split('__')[1]
  comparisons[c+'__'+t]={'vs_clean':effect(groups,g,t,'clean'),'agreement':float(np.mean([z['agreement']for z in rows.values()])),'kl':float(np.mean([z['kl_clean_to_condition']for z in rows.values()]))}
  relative_to_mean[c+'__'+t]=effect(groups,g,t,'mean_values');relative_to_zero[c+'__'+t]=effect(groups,g,t,'zero_values')
result.update(script_calibration_model_integrity_passed=True,all_prespecified_layer_sets_replayed=True,unique_layer_sets=unique,forward_passes=meta['forward_passes'],native_mean_zero_anchors_exact=True,seeded_random_controls_exactly_replayed=True,max_random_abs_value=random_max,max_relative_head_norm_error_after_fp16=max(norm_errors),identical_layer_set_aliases=aliases,comparisons=comparisons,relative_to_mean=relative_to_mean,relative_to_zero=relative_to_zero,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['comparisons','relative_to_mean','relative_to_zero','identical_layer_set_aliases']},indent=2))
