"""CPU-only audit of global source key/value and source-column removal interventions."""
import ast,hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'quiet_anchor_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
for field,path in [('script_sha256',ROOT/'quiet_anchor.py'),('tokens_sha256',ROOT/'residual_paths_run/tokens.pt'),('calibration_sha256',ROOT/'cache_run/calibration.pt')]:assert meta[field]==hashlib.sha256(path.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'] and meta['forward_passes']==832
raw,groups,result=load_audit(OUT);assert result['rows']==832 and result['groups']==104
old=torch.load(ROOT/'residual_paths_run/per_token.pt',map_location='cpu',weights_only=True)
old={k:{z['window']:z for z in v}for k,v in old.items()}
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);prior=torch.load(ROOT/'residual_paths_run/tokens.pt',map_location='cpu',weights_only=True)
for c,ws in sets.items():
 g='quiet_'+c
 for i,x in enumerate(ws):
  assert torch.equal(x,prior[c]['evaluation'][i])
  for t in ['clean','deleted']:
   for k in ['loss','pred','kl']:
    assert torch.equal(raw[g+'__'+t][i][k],old['heldout_'+c+'__'+t][i][k])
    assert torch.equal(raw[g+'__'+t][i][k],raw[g+'__native_mask_'+t][i][k])
  for t in ['K_mean_V_mean','K_mean_V_zero','K_zero_V_mean','K_zero_V_zero','remove_source_all']:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t+'_clean'][i][k],raw[g+'__'+t+'_deleted'][i][k])
 for key,items in groups.items():
  if key.startswith(g+'__'):assert all(z['score_start']==1 and z['scored_tokens']==510 for z in items.values())

# Pure mask hook extracted without importing model or GPU packages.
node=next(n for n in ast.parse((ROOT/'quiet_anchor.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SourceMask')
env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<SourceMask AST>','exec'),env)
m=env['SourceMask'].__new__(env['SourceMask']);m.explicit=True;m.layers=[2,3];m.masks={};x=torch.randn(1,9,2048).half()
base=torch.zeros(1,1,9,9).half().masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),-torch.inf)
for l in range(16):
 a,k=m.hook(l,(),{'hidden_states':x});want=base.clone()
 if l in [2,3]:want[:,:,1:,0]=-torch.inf
 assert torch.equal(k['attention_mask'],want) and torch.equal(m.masks[9],base)
m.explicit=False;assert m.hook(2,(x,),{})is None
packed=torch.load(ROOT/'cache_run/calibration.pt',map_location='cpu',weights_only=True)['wiki']
for pair in packed:
 for t in pair:
  u=t.transpose(1,2).reshape(1,1,2048).contiguous()
  assert torch.equal(u.clamp(-8,8).reshape(1,1,16,128).transpose(1,2).contiguous(),t)
checks=json.loads((OUT/'execution_checks.json').read_text());assert len(checks)==32 and all(z['native_explicit_exact']and z['fixed_KV_state_invariance_exact']and z['all_layer_source_removal_state_invariance_exact']for z in checks)
comparisons={};zero_vs_removal={}
for c in sets:
 g='quiet_'+c
 for key,rows in groups.items():
  if not key.startswith(g+'__'):continue
  t=key.split('__')[1]
  comparisons[c+'__'+t]={'vs_clean':effect(groups,g,t,'clean'),'agreement':float(np.mean([z['agreement']for z in rows.values()])),'kl':float(np.mean([z['kl_clean_to_condition']for z in rows.values()]))}
 for state in ['clean','deleted']:
  for km in ['native','mean','zero']:zero_vs_removal[c+'__K_'+km+'_V_zero_'+state]=effect(groups,g,'K_'+km+'_V_zero_'+state,'remove_source_all_'+state)
result.update(model_hashes_script_calibration_passed=True,prior_anchors_exact=True,native_explicit_exact=True,fixed_key_value_and_full_removal_state_invariance_exact=True,source_mask_localization_and_cache_immutability_passed=True,mean_cache_unpack_roundtrip_exact=True,comparisons=comparisons,zero_vs_removal=zero_vs_removal,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['comparisons','zero_vs_removal']},indent=2))
