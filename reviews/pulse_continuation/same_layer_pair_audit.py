"""CPU-only corrected same-layer pair raw result and localization replay."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'same_layer_pair_run'
m=json.loads((OUT/'manifest.json').read_text());assert m['completed_utc'] and m['forward_passes']==144 and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] and m['targets']==[[1,1764,1710],[1,1764,8041]]
for f,p in [('script_sha256',ROOT/'same_layer_pair.py'),('design_sha256',ROOT/'SAME_LAYER_PAIR.md'),('tokens_parent_sha256',ROOT/'upstream_run/tokens.pt')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,4);assert result['rows']==144 and result['groups']==36
tr=torch.load(OUT/'traces.pt',weights_only=True,map_location='cpu');checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==48 and all(z['loss_exact']and z['pred_exact']for z in checks)
old=torch.load(ROOT/'column_factorial_run/per_token.pt',weights_only=True,map_location='cpu');old={k:{a['window']:a for a in v}for k,v in old.items()};anchors=0;ratios={};observations=[]
for key,rs in groups.items():
 g,t=key.split('__');e=effect(groups,g,t,'clean');ratios[key]={'ppl_ratio':math.exp(e['mean']),'delta_nll':e,'agreement':interval([r['agreement']for r in rs.values()])}
 states=('clean','clean')if t=='clean'else(t.split('_')[1],t.split('_')[3])
 for i,r in rs.items():
  z=tr[f'{key}__{i}'];b=tr[f'{g}__clean__{i}'];assert torch.equal(z['activations'],b['activations']) and torch.equal(z['weights'],b['weights']) and torch.equal(z['before'],b['before']);assert r['score_start']==1 and r['scored_tokens']==510
  if 'column' not in states:
   mask=torch.arange(2048)!=1764;assert torch.equal(z['before'][mask],z['after'][mask])
  if states[1]=='clean':
   ref=old['first_'+g[5:]+'_natural__'+{'clean':'clean','row':'delete_row','column':'delete_column'}[states[0]]][i];a=raw[key][i];assert torch.equal(a['loss'],ref['loss']) and torch.equal(a['pred'],ref['pred']);anchors+=1
  if t=='clean':observations.append({'group':g,'window':i,'first_intermediate':float(z['activations'][0]),'second_intermediate':float(z['activations'][1]),'first_potential':float(z['activations'][0])*float(z['weights'][0]),'second_potential':float(z['activations'][1])*float(z['weights'][1])})
result.update(prior_checks_replayed=anchors,all_144_intermediates_unchanged=True,row_only_output_localization_verified=True,ratios=ratios,source_intermediates=observations,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','source_intermediates']},indent=2))
for k,v in ratios.items():
 if k.endswith('__first_clean_second_row')or k.endswith('__first_clean_second_column')or k.endswith('__first_column_second_column'):print(k,v)
print('second potential range',min(z['second_potential']for z in observations),max(z['second_potential']for z in observations))
