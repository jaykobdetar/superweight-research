"""Independent CPU-only component donor/closure and summary replay."""
import csv,hashlib,itertools,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'newline_components_run'
m=json.loads((OUT/'manifest.json').read_text());assert m['completed_utc'] and m['forward_passes']==448 and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'newline_components.py'),('design_sha256',ROOT/'NEWLINE_COMPONENTS.md'),('tokens_parent_sha256',ROOT/'newline_heads_run/tokens.pt'),('eligibility_sha256',ROOT/'newline_natural_run/eligibility.json')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,4);assert result['rows']==448 and result['groups']==112
tr=torch.load(OUT/'probe_traces.pt',weights_only=True,map_location='cpu');obs={(a['group'],a['trial'],a['window']):a for a in json.loads((OUT/'factor_observations.json').read_text())};assert len(obs)==len(tr)==448
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==112 and all(all(z[k] for k in ['full_loss_exact','full_pred_exact','q_mlp_input_exact','q_intermediate_exact'])for z in checks)
closures=0;applied=0;windows=[];metrics={}
for key,items in groups.items():
 g,t=key.split('__');metrics[key]={'pulse':interval([obs[g,t,i]['q_pulse']for i in range(4)]),'delta_nll':effect(groups,g,t,'clean')}
 for i,row in items.items():
  z=tr[f'{key}__{i}'];o=obs[g,t,i];q=o['q'];bc=tr[f'{g}__clean__{i}'];ec=tr[f'{g}__edge__{i}'];donors={'C':bc,'E':ec}
  assert o['token_sha256']==row['token_sha256'] and row['score_start']==q+1 and row['scored_tokens']==510-q
  assert o['q_pulse']==float(z['activation'][q])*.63232421875 and o['q_gate']==float(z['gate'][2]) and o['q_up']==float(z['up'][2])
  assert torch.equal(z['activation'][:q],bc['activation'][:q])
  if t.startswith('mix_'):swaps=dict(zip(['A0','M0','A1'],t[4:]))
  elif '_from_'in t:base,_,donor,name=t.split('_');swaps={name:donor}
  else:swaps={}
  for name,d in z['components'].items():
   assert torch.equal(d['applied'],donors[swaps[name]]['components'][name]['applied']if name in swaps else d['native']);applied+=1
  ref={'mix_CCC':'clean','mix_EEE':'edge','C_from_E_A0':'edge','E_from_C_A0':'clean'}.get(t)
  if ref:
   a,b=raw[key][i],raw[g+'__'+ref][i];zz=tr[f'{g}__{ref}__{i}'];assert torch.equal(a['loss'],b['loss']) and torch.equal(a['pred'],b['pred']) and torch.equal(a['kl'],b['kl'])
   assert torch.equal(z['before_mlp1'],zz['before_mlp1']) and torch.equal(z['activation'],zz['activation']);closures+=1
  windows.append({'group':g,'trial':t,'window':i,'q_pulse':o['q_pulse'],'q_gate':o['q_gate'],'q_up':o['q_up'],'delta_nll':row['nll']-row['baseline_nll'],'ppl_ratio':math.exp(row['nll']-row['baseline_nll']),'agreement':row['agreement']})
fs=list(csv.DictReader((OUT/'factor_summary.csv').open()));assert len(fs)==112
for a in fs:
 for k in ['q_pulse','q_gate','q_up']:
  z=interval([obs[a['group'],a['trial'],i][k]for i in range(4)])
  for f,s in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(a[f])-z[s])<1e-12
cs=list(csv.DictReader((OUT/'component_contrasts.csv').open()));assert len(cs)==98
for a in cs:
 g=a['group'];terms=json.loads(a['terms'])
 for k in ['nll','q_pulse','q_gate','q_up']:
  z=interval([sum(v*(groups[g+'__'+t][i]['nll']if k=='nll'else obs[g,t,i][k])for t,v in terms)for i in range(4)])
  for f,s in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(a[f])-z[s])<1e-12
anchors=0
for kind,file in [('constructed','newline_heads_run'),('natural','newline_natural_run')]:
 old=torch.load(ROOT/file/'per_token.pt',weights_only=True,map_location='cpu');old={k:{a['window']:a for a in v}for k,v in old.items()}
 for g in sorted({k.split('__')[0]for k in raw if k.startswith(kind+'_')}):
  c=g[len(kind)+1:]
  for t in ['clean','edge']:
   oldkey=('heads_'+c+'_R__'+('clean'if t=='clean'else'edge_selected'))if kind=='constructed'else g+'__'+t
   if oldkey not in old:raise AssertionError((oldkey,list(old)[:8]))
   for i,a in raw[g+'__'+t].items():
    b=old[oldkey][i];assert torch.equal(a['loss'],b['loss']) and torch.equal(a['pred'],b['pred']) and a['token_sha256']==b['token_sha256'];anchors+=1
result.update(closure_checks_replayed=closures,component_application_checks=applied,prior_anchors=anchors,factor_summary_rows_replayed=len(fs),component_contrasts_replayed=len(cs),metrics=metrics,per_window=windows,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
with Path(__file__).with_name('newline_components_per_window.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(windows[0]));w.writeheader();w.writerows(windows)
print(json.dumps({k:v for k,v in result.items()if k not in ['metrics','per_window']},indent=2))
for g in sorted({k.split('__')[0]for k in raw}):
 print(g)
 for t in ['clean','edge','mix_ECC','mix_CEC','mix_CCE','mix_EEC','mix_ECE','mix_CEE','C_from_E_M0','C_from_E_A1','E_from_C_M0','E_from_C_A1']:
  a=metrics[g+'__'+t];print(t,round(a['pulse']['mean'],3),round(math.exp(a['delta_nll']['mean']),5))
