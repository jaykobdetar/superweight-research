"""CPU-only unedited-input eligibility, H12 donor, pulse and output audit."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'newline_natural_run'
m=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in m and m['forward_passes']==228
assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'newline_natural.py'),('design_sha256',ROOT/'NEWLINE_NATURAL.md'),('tokens_parent_sha256',ROOT/'newline_heads_run/tokens.pt'),('selection_sha256',ROOT/'newline_heads_run/selection.json')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4);assert result['rows']==228 and result['groups']==57
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');old=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];traces=torch.load(OUT/'probe_traces.pt',weights_only=True,map_location='cpu');obs=json.loads((OUT/'pulse_observations.json').read_text());assert len(traces)==len(obs)==228;obs={(z['group'],z['trial'],z['window']):z for z in obs}
elig=json.loads((OUT/'eligibility.json').read_text());checks=json.loads((OUT/'checks.json').read_text());assert len(elig)==16 and len(checks)==108 and all(z['activation_unchanged']for z in checks)
ratios={};pulses={};paired={};windows=[];strata=[]
for c,ws in sets.items():
 for i,x in enumerate(ws):
  assert torch.equal(x,old[c][i]);ps=(x==187).nonzero().flatten().tolist();ee=next(z for z in elig if z['corpus']==c and z['window']==i);assert ee['positions187']==ps and ee['eligible']==(len(ps)>=2 and ps[1]<=255)
  if not ee['eligible']:continue
  p,q=ps[:2];sham=p-1 if p else q-1;assert int(x[sham])!=187;g='natural_'+c;digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();bc=traces[f'{g}__clean__{i}'];ec=traces[f'{g}__edge__{i}']
  for key,items in groups.items():
   if not key.startswith(g+'__'):continue
   t=key.split('__')[1];z=traces[f'{key}__{i}'];o=obs[g,t,i];row=items[i];assert row['token_sha256']==digest==o['token_sha256']and row['score_start']==q+1 and row['scored_tokens']==510-q and o['p']==p and o['q']==q
   assert torch.equal(z['potential_pulse'],z['activation'].float()*.63232421875);applied=z['potential_pulse'].clone();world=ec if t.startswith('edge')else bc
   if '_row_'in t or '_column_'in t:
    pos={'p':[p],'q':[q],'both':[p,q]}[t.rsplit('_',1)[1]];applied[pos]=0
    for k in ['activation','gate','up','H12_contexts','H12_values']:assert torch.equal(z[k],world[k])
   assert torch.equal(z['applied_pulse'],applied)
   for k,j in [('p_pulse',p),('q_pulse',q),('source_pulse',0)]:assert o[k]==float(z['potential_pulse'][j])
   for k in 'qkv':assert torch.equal(z['L0_'+k],bc['L0_'+k])
   if '_H12_'in t:
    assert torch.equal(z['H12_contexts'],world['H12_contexts'])and torch.equal(z['H12_values'],world['H12_values'])
    expected=world['H12_contexts'].clone();kind=t.split('_H12_')[1]
    expected[2]=(bc if t.startswith('edge')else ec)['H12_contexts'][2]if kind=='donor'else world['H12_values'][1 if kind=='value_p'else 2]
    assert torch.equal(z['H12_applied_contexts'],expected)
   else:assert torch.equal(z['H12_applied_contexts'],z['H12_contexts'])
   assert torch.equal(z['activation'][:q],bc['activation'][:q])
   windows.append({'group':g,'trial':t,'window':i,'p':p,'q':q,'targets':510-q,'q_pulse':o['q_pulse'],'p_pulse':o['p_pulse'],'delta_nll':row['nll']-row['baseline_nll'],'ppl_ratio':math.exp(row['nll']-row['baseline_nll']),'agreement':row['agreement'],'kl':row['kl_clean_to_condition']})
for key,items in groups.items():
 g,t=key.split('__');qq=effect(groups,g,t,'clean');ratios[key]={'ppl_ratio':math.exp(qq['mean']),'delta_nll':qq,'agreement':interval([z['agreement']for z in items.values()]),'kl':interval([z['kl_clean_to_condition']for z in items.values()])};pulses[key]={k:interval([obs[g,t,i][k]for i in range(4)])for k in ['p_pulse','q_pulse','source_pulse']}
 for name,ids in [('p0',[i for i in range(4)if obs[g,t,i]['p']==0]),('p_positive',[i for i in range(4)if obs[g,t,i]['p']>0])]:
  if not ids:continue
  rs=[items[i]for i in ids];d=interval([z['nll']-z['baseline_nll']for z in rs]);a=interval([z['agreement']for z in rs]);k=interval([z['kl_clean_to_condition']for z in rs]);strata.append({'group':g,'stratum':name,'trial':t,'windows':len(ids),'nll':sum(z['nll']for z in rs)/len(ids),'ppl_ratio':math.exp(d['mean']),'ratio_low':math.exp(d['low']),'ratio_high':math.exp(d['high']),'delta_nll':d['mean'],'delta_low':d['low'],'delta_high':d['high'],'agreement':a['mean'],'agreement_low':a['low'],'agreement_high':a['high'],'kl':k['mean'],'kl_low':k['low'],'kl_high':k['high']})
ss=list(csv.DictReader((OUT/'pulse_summary.csv').open()));assert len(ss)==57
for row in ss:
 for k in ['p_pulse','q_pulse','source_pulse']:
  z=pulses[row['group']+'__'+row['trial']][k]
  for a,b in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[a])-z[b])<1e-12
cs=list(csv.DictReader((OUT/'natural_contrasts.csv').open()));assert len(cs)==27
for row in cs:
 g=row['group'];ts=json.loads(row['terms']);dd={}
 for k in ['nll','q_pulse']:
  vals=[sum(v*(groups[g+'__'+t][i]['nll']if k=='nll'else obs[g,t,i][k])for t,v in ts)for i in range(4)];z=interval(vals);dd[k]=z
  for a,b in [(k+'_difference','mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[a])-z[b])<1e-12
 paired[g+'__'+row['contrast']]=dd
for file,rows in [('newline_natural_per_window.csv',windows),('newline_natural_p0_stratified_summary.csv',strata)]:
 with (ROOT/'review'/file).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result.update(unmodified_parent_token_hashes_verified=True,eligibility_replayed=16,eligible_cases=12,p0_cases=[z for z in windows if z['trial']=='clean'and z['p']==0],activation_checks_replayed=108,all_H12_donor_value_application_captures_exact=True,matched_variable_target_ids_verified=True,pulse_summary_rows_replayed=57,natural_contrasts_replayed=27,ratios=ratios,pulses=pulses,paired_effects=paired,per_window=windows,stratified_summary_rows=len(strata),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','pulses','paired_effects','per_window']},indent=2))
