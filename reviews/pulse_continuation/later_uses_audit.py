"""Independent CPU replay of fresh source/later factorial metrics and traces."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
import torch.nn.functional as F
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'later_uses_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta and meta['forward_passes']==336
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
for field,p in [('script_sha256',ROOT/'later_uses.py'),('design_sha256',ROOT/'LATER_USES.md')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4,require_tables=False);assert result['rows']==336 and result['groups']==84
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');traces=torch.load(OUT/'neuron_traces.pt',weights_only=True,map_location='cpu');assert len(traces)==336
prior=json.loads((ROOT/'confirmation_run/data_manifest.json').read_text());dm=json.loads((OUT/'data_manifest.json').read_text());old=torch.load(ROOT/'confirmation_run/tokens.pt',weights_only=True,map_location='cpu')
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==288 and all(z[k]for z in checks for k in ['first_gate_exact','first_up_exact','first_activation_exact'])
obs=json.loads((OUT/'pulse_observations.json').read_text());assert len(obs)==48;obs={(z['group'],z['window']):z for z in obs}
seen=set();ratios={};effects={};amplitudes={};cpu_factor_err=[]
for c,ws in sets.items():
 d=dm[c];starts=[s+768 for s in prior[c]['evaluation_starts'][:4]];excluded=prior[c]['calibration_starts']+prior[c]['evaluation_starts'];assert d['evaluation_starts']==starts and d['original_excluded_starts']==excluded and d['parent_manifest_sha256']==hashlib.sha256((ROOT/'confirmation_run/data_manifest.json').read_bytes()).hexdigest()
 assert all(abs(s-e)>=512 for s in starts for e in excluded)and all(abs(s-e)>=512 for j,s in enumerate(starts)for e in starts[j+1:])
 for source in d['sources']:assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
 for i,xx in enumerate(ws):
  h=hashlib.sha256(xx.numpy().tobytes()).hexdigest();assert h not in seen;seen.add(h);assert not any(torch.equal(xx,q) for part in old[c].values()for q in part)
  for identity,tid in [('natural',None),('The',510),('period',15)]:
   x=xx.clone()
   if tid is not None:x[0]=tid
   g=f'later_{c}_{identity}';digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();base=traces[f'{g}__clean__{i}'];amp=base['potential_first'];o=obs[g,i]
   assert o['source_pulse']==float(amp[0])and o['later_abs_max']==float(amp[1:].abs().max())and o['later_abs_mean']==float(amp[1:].abs().mean())
   assert o['counts_above']=={str(t):int((amp[1:].abs()>t).sum())for t in [1,10,100]}
   assert [z['position']for z in o['top_positions']]==(amp[1:].abs().topk(8).indices+1).tolist()
   for z in o['top_positions']:
    p=z['position'];assert z['token_id']==int(x[p])and z['nearby_tokens']==x[max(0,p-4):p+5].tolist()and z['potential_pulse']==float(amp[p])
   for trial in ['clean','row_source','row_later','row_all','column_source','column_later','column_all']:
    a=traces[f'{g}__{trial}__{i}'];row=groups[g+'__'+trial][i];assert row['token_sha256']==digest and row['score_start']==1 and row['scored_tokens']==510
    for k in ['gate','up','activation']:assert torch.equal(a[k],base[k])
    assert torch.equal(a['potential_first'],a['activation'].float()*.63232421875)and torch.equal(a['potential_second'],a['second_activation'].float()*(-.59228515625))
    applied=a['potential_first'].clone()
    if trial!='clean':applied[{'source':0,'later':slice(1,None),'all':slice(None)}[a['scope']]]=0
    assert torch.equal(a['applied_first'],applied)
    cpu_factor_err.append(float((F.silu(a['gate'])*a['up']-a['activation']).abs().max()))
   for kind in ['row','column']:
    # Causality: later-position interventions do not change source0's next-layer activation.
    assert traces[f'{g}__{kind}_later__{i}']['second_activation'][0]==base['second_activation'][0]
    assert traces[f'{g}__{kind}_all__{i}']['second_activation'][0]==traces[f'{g}__{kind}_source__{i}']['second_activation'][0]
 for identity in ['natural','The','period']:
  g=f'later_{c}_{identity}'
  for t in ['clean','row_source','row_later','row_all','column_source','column_later','column_all']:
   q=effect(groups,g,t,'clean');ratios[g+'__'+t]={'ppl_ratio':math.exp(q['mean']),'delta_nll':q,'agreement':sum(z['agreement']for z in groups[g+'__'+t].values())/4}
  for kind in ['row','column']:
   effects[g+'__'+kind+'__all_minus_source']=effect(groups,g,kind+'_all',kind+'_source')
   effects[g+'__'+kind+'__source_later_interaction']=interval([groups[g+'__'+kind+'_all'][i]['nll']-groups[g+'__'+kind+'_source'][i]['nll']-groups[g+'__'+kind+'_later'][i]['nll']+groups[g+'__clean'][i]['nll']for i in range(4)])
  for scope in ['source','later','all']:effects[g+'__column_vs_row__'+scope]=effect(groups,g,'column_'+scope,'row_'+scope)
  amps=torch.stack([traces[f'{g}__clean__{i}']['potential_first']for i in range(4)])
  amplitudes[g]={'source_pulse':interval(amps[:,0].tolist()),'later_abs_max':float(amps[:,1:].abs().max()),'later_counts_above':{str(t):int((amps[:,1:].abs()>t).sum())for t in [1,10,100]},'later_abs_max_per_window':amps[:,1:].abs().max(1).values.tolist()}
large_pulse_tokens={};large_pulse_examples=[]
for (g,i),o in obs.items():
 c=g.removeprefix('later_').rsplit('_',1)[0];xx=sets[c][i];amp=traces[f'{g}__clean__{i}']['potential_first'];ps=(amp[1:].abs()>1).nonzero().flatten()+1
 for p in ps.tolist():
  tid=int(xx[p]);large_pulse_tokens[str(tid)]=large_pulse_tokens.get(str(tid),0)+1;large_pulse_examples.append({'group':g,'window':i,'position':p,'token_id':tid,'pulse':float(amp[p])})
rows=list(csv.DictReader((OUT/'scope_interactions.csv').open()))if (OUT/'scope_interactions.csv').exists()else []
for row in rows:
 q=effects[row['group']+'__'+row['component']+'__'+row['contrast']]
 for a,b in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
 assert abs(float(row['ratio'])-math.exp(q['mean']))<1e-12
result.update(scope_contrasts_replayed=len(rows),unique_natural_windows=len(seen),source_hashes_and_frozen_spans_verified=True,matched_targets_verified=True,all_288_first_neuron_trace_checks_replayed=True,potential_and_applied_pulse_exact=True,source0_second_activation_causality_checks=True,cpu_half_factor_max_error=max(cpu_factor_err),observation_rows_replayed=48,later_large_pulse_token_counts=large_pulse_tokens,later_large_pulse_examples=large_pulse_examples,ratios=ratios,paired_effects=effects,native_amplitudes=amplitudes,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','paired_effects','native_amplitudes','later_large_pulse_examples']},indent=2))
