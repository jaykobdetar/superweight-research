"""CPU-only two-unit factorial, trace and paired interaction replay."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'two_units_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta and meta['forward_passes']==816
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
for field,p in [('script_sha256',ROOT/'two_units.py'),('design_sha256',ROOT/'TWO_UNITS.md'),('tokens_parent_sha256',ROOT/'upstream_run/tokens.pt'),('prior_metrics_sha256',ROOT/'column_factorial_run/per_token.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4);assert result['rows']==864 and result['groups']==216
prior=torch.load(ROOT/'column_factorial_run/per_token.pt',map_location='cpu',weights_only=True);prior={k:{z['window']:z for z in v}for k,v in prior.items()}
traces=torch.load(OUT/'unit_traces.pt',map_location='cpu',weights_only=True);assert len(traces)==864
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);old=torch.load(ROOT/'upstream_run/tokens.pt',map_location='cpu',weights_only=True)['evaluation']
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==144 and all(z['loss_exact']and z['pred_exact']for z in checks)
states=['clean','row','column']
def trial(a,b):return 'clean'if a==b=='clean'else f'first_{a}_second_{b}'
weights={1:set(),2:set()};rounding=[];potentials={}
for c,ws in sets.items():
 for i,xx in enumerate(ws):
  assert torch.equal(xx,old[c][i])
  for identity,tid in [('natural',None),('The',510),('period',15)]:
   x=xx.clone()
   if tid is not None:x[0]=tid
   h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
   for scope in ['source','all']:
    g=f'units_{c}_{identity}_{scope}'
    for a in states:
     for b in states:
      t=trial(a,b);z=traces[f'{g}__{t}__{i}'];row=groups[g+'__'+t][i]
      assert row['token_sha256']==h and row['score_start']==1 and row['scored_tokens']==510
      native=traces[f'{g}__clean__{i}'];first=traces[f'{g}__{trial(a,"clean")}__{i}']
      for l,state in [(1,a),(2,b)]:
       d=z[l];weights[l].add(d['weight']);assert d['scope']==scope and d['state']==state
       assert d['potential_unremoved_contribution']==float(d['activation'].float()*d['weight'])
       assert d['applied_target_contribution']==(d['potential_unremoved_contribution']if state=='clean'else 0.)
       baseline=native[l]if l==1 else first[l]
       assert torch.equal(d['activation'],baseline['activation'])and torch.equal(d['down_before'],baseline['down_before'])
       if state=='clean':assert torch.equal(d['down_after'],d['down_before'])
       if state=='row':assert torch.equal(d['down_after'][:1764],d['down_before'][:1764])and torch.equal(d['down_after'][1765:],d['down_before'][1765:])
       if state!='clean':rounding.append(float(d['down_after'][1764].float()-d['down_before'][1764].float())+d['potential_unremoved_contribution'])
       source=traces[f'units_{c}_{identity}_source__{t}__{i}'][l]
       for k in ['activation','down_before','down_after']:assert torch.equal(d[k],source[k])
       for k in ['weight','potential_unremoved_contribution','applied_target_contribution','state']:assert d[k]==source[k]
      if scope=='source'and b=='clean':
       pt={'clean':'clean','row':'delete_row','column':'delete_column'}[a]
       for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+t][i][k],prior[f'first_{c}_{identity}__{pt}'][i][k])
    for a in states:
     rr=traces[f'{g}__{trial(a,"row")}__{i}'][2];cc=traces[f'{g}__{trial(a,"column")}__{i}'][2];assert rr['down_after'][1764]==cc['down_after'][1764]
   gs=f'units_{c}_{identity}_source';ga=f'units_{c}_{identity}_all'
   for k in ['loss','pred','kl']:assert torch.equal(raw[gs+'__clean'][i][k],raw[ga+'__clean'][i][k])
assert all(len(w)==1 for w in weights.values())and next(iter(weights[1]))==.63232421875
conditional=list(csv.DictReader((OUT/'conditional_contrasts.csv').open()));assert len(conditional)==192;ce={}
for row in conditional:
 g,name=row['group'],row['contrast'];kind=name.split('_')[1]
 if '_given_first_'in name:
  a=name.split('_given_first_')[1];coef={trial(a,kind):1,trial(a,'clean'):-1}
 else:coef={trial('column',kind):1,trial('column','clean'):-1,trial('clean',kind):-1,'clean':1}
 vals=[sum(w*float(raw[g+'__'+t][i]['loss'].double().mean())for t,w in coef.items())for i in range(4)];q=interval(vals);ce[g+'__'+name]=q
 for a,b in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
 assert abs(float(row['ppl_ratio'])-math.exp(q['mean']))<1e-12
scope_rows=list(csv.DictReader((OUT/'scope_contrasts.csv').open()));assert len(scope_rows)==108;se={}
for row in scope_rows:
 g,t=row['group'],row['trial'];a=groups[g+'_all__'+t];b=groups[g+'_source__'+t];assert all(a[i]['token_sha256']==b[i]['token_sha256']for i in range(4))
 q=interval([float(raw[g+'_all__'+t][i]['loss'].double().mean()-raw[g+'_source__'+t][i]['loss'].double().mean())for i in range(4)]);se[g+'__'+t]=q
 for a,b in [('all_minus_source_delta','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
for g in sorted({k.split('__')[0]for k in groups}):
 for a in states:
  for b in states:
   t=trial(a,b);potentials[g+'__'+t]={str(l):{'potential':interval([traces[f'{g}__{t}__{i}'][l]['potential_unremoved_contribution']for i in range(4)]),'applied':interval([traces[f'{g}__{t}__{i}'][l]['applied_target_contribution']for i in range(4)])}for l in [1,2]}
result.update(conditional_rows_replayed=192,scope_rows_replayed=108,prior_source_only_replays_exact=144,potential_and_applied_labels_correct=True,row_deletion_localization_verified=True,source0_traces_identical_across_scopes=True,first_unit_input_unaffected_by_interventions=True,second_unit_input_fixed_given_first_state=True,scalar_weights={str(l):next(iter(w))for l,w in weights.items()},max_target_delta_vs_potential_rounding_difference=max(map(abs,rounding)),conditional_effects=ce,scope_effects=se,potential_and_applied_summaries=potentials,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['conditional_effects','scope_effects','potential_and_applied_summaries']},indent=2))
