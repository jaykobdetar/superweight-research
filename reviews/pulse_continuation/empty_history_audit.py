"""CPU-only mask localization, paired scores, and saved pulse trace audit."""
import ast,hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,interval,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'empty_history_run'

# Extract only the hook class; do not import an experiment or model package.
node=next(n for n in ast.parse((ROOT/'empty_history.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='Masks')
env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<Masks AST>','exec'),env)
m=env['Masks'].__new__(env['Masks']);m.cache={};m.p=3;x=torch.randn(1,9,2048).half()
standard=torch.zeros(1,1,9,9).half().masked_fill(torch.ones(9,9,dtype=torch.bool).triu(1),-torch.inf)
for mode,layers in [('explicit',[]),('L0',[0]),('L1',[1]),('L0_L1',[0,1]),('all',list(range(16)))]:
 m.mode=mode
 for l in range(16):
  args,kwargs=m.hook(l,(),{'hidden_states':x,'attention_mask':None,'sentinel':7})
  expected=standard.clone()
  if l in layers:expected[:,:,m.p,:m.p]=-torch.inf
  assert torch.equal(kwargs['attention_mask'],expected) and kwargs['sentinel']==7 and kwargs['hidden_states'] is x
  assert torch.equal(m.cache[9],standard),'Isolation mutated cached causal mask'
m.mode='native';assert m.hook(0,(x,),{}) is None

meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['script_sha256']==hashlib.sha256((ROOT/'empty_history.py').read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
raw,groups,result=load_audit(OUT,expected_windows=4,require_tables=False)
result['summary_tables_available']=(OUT/'summary.csv').exists() and (OUT/'contrasts.csv').exists()
assert result['rows']==3456 and result['groups']==864 and meta['forward_passes']==1536
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True)
prior=torch.load(ROOT/'residual_paths_run/tokens.pt',map_location='cpu',weights_only=True)
traces=torch.load(OUT/'source_intermediates.pt',map_location='cpu',weights_only=True)
obs=[json.loads(s) for s in (OUT/'pulse_observations.jsonl').read_text().splitlines()]
assert len(obs)==len(traces)==384
data=json.loads((OUT/'data_manifest.json').read_text());assert data['source_manifest_sha256']==hashlib.sha256((ROOT/'residual_paths_run/data_manifest.json').read_bytes()).hexdigest()
pulse_groups={};source_weight_estimates=[];max_explicit_native={};effects={};mask_effects={}
for z in obs:
 c,identity,p,mode,i=(z[k] for k in ['corpus','identity','prefix','mode','window']);t=traces[f'{c}_{identity}_{p}_{mode}_{i}']
 assert t.shape==(2,8192) and torch.isfinite(t).all()
 assert z['intermediate_exact']==torch.equal(t[0],t[1])
 assert z['intermediate_max_abs_difference']==float((t[0].float()-t[1].float()).abs().max())
 for j,key in enumerate(['source0_pulse','boundary_pulse']):
  if float(t[j,1710])!=0:source_weight_estimates.append(z[key]/float(t[j,1710]))
 pulse_groups.setdefault(f'{c}_{identity}_{p}_{mode}',[]).append(z)
 # Same source token at index zero remains unaffected by boundary mask edits.
 assert torch.equal(t[0],traces[f'{c}_{identity}_{p}_explicit_{i}'][0])
 if mode=='native':assert torch.equal(t,traces[f'{c}_{identity}_{p}_explicit_{i}'])
 assert torch.equal(t,traces[f'{c}_{identity}_{p}_L0_L1_{i}']) if mode=='all' else True
 # Suffix and targets retained; only the two specified input tokens replaced.
 w=sets[c][i];assert torch.equal(w['suffix'],prior[c]['evaluation'][i])
 x=torch.cat([w['prefix'][-p:],w['suffix']]);x[0]=x[p]=510 if identity=='The' else 15
 assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==z['token_sha256']
 for later,n in [('',511),('_later',510)]:
  group=f'{c}_{identity}_{p}_{mode}'+later
  for trial in ['clean','delete0','delete_boundary','delete_both']:
   assert groups[group+'__'+trial][i]['scored_tokens']==n
   assert groups[group+'__'+trial][i]['score_start']==p+(later!='')
   assert raw[group+'__'+trial][i]['token_sha256']==z['token_sha256']
   if later:
    for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__'+trial][i][k],raw[group[:-6]+'__'+trial][i][k][1:])
assert max(source_weight_estimates)-min(source_weight_estimates)<1e-5
for c in sets:
 for identity in ['The','period']:
  for p in [128,512]:
   cross=f'mask_effect_{c}_{identity}_{p}'
   max_explicit_native[f'{c}_{identity}_{p}']=compare(raw,groups,cross,'native','clean')
   assert all(torch.count_nonzero(raw[cross+'__native'][i]['kl'])==0 for i in range(4))
   for mode in ['explicit','native','L0','L1','L0_L1','all']:
    group=f'{c}_{identity}_{p}_{mode}'
    for i in range(4):
     for k in ['loss','pred']:assert torch.equal(raw[cross+'__'+('clean' if mode=='explicit' else mode)][i][k],raw[group+'__clean'][i][k])
    if mode!='explicit':mask_effects[group]=effect(groups,cross,mode,'clean')
    for later in ['','_later']:
     g=group+later
     for trial in ['delete0','delete_boundary','delete_both']:effects[g+'__'+trial]=effect(groups,g,trial,'clean')
     # Additive NLL interaction, not a mediated fraction.
     effects[g+'__interaction']=interval([groups[g+'__delete_both'][i]['nll']-groups[g+'__delete0'][i]['nll']-groups[g+'__delete_boundary'][i]['nll']+groups[g+'__clean'][i]['nll'] for i in range(4)])
result.update(mask_localization_and_cache_immutability=True,all_scoring_and_token_alignment_passed=True,source_weight_reconstructed_range=[min(source_weight_estimates),max(source_weight_estimates)],pulse_summary={k:{'source0_range':[min(z['source0_pulse'] for z in v),max(z['source0_pulse'] for z in v)],'boundary_range':[min(z['boundary_pulse'] for z in v),max(z['boundary_pulse'] for z in v)],'exact_intermediate_count':sum(z['intermediate_exact'] for z in v),'max_intermediate_difference':max(z['intermediate_max_abs_difference'] for z in v)} for k,v in pulse_groups.items()},native_control=max_explicit_native,mask_effects=mask_effects,deletion_effects=effects,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['pulse_summary','native_control','mask_effects','deletion_effects']},indent=2))
