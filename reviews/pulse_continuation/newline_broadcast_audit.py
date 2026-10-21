"""CPU-only broadcast mask, donor, closure, and paired outcome replay."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'newline_broadcast_run'
m=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in m and m['forward_passes']==432
assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'newline_broadcast.py'),('design_sha256',ROOT/'NEWLINE_BROADCAST.md'),('tokens_parent_sha256',ROOT/'newline_heads_run/tokens.pt')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4);assert result['rows']==432 and result['groups']==108
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');old=torch.load(ROOT/'newline_heads_run/tokens.pt',weights_only=True,map_location='cpu')['evaluation'];traces=torch.load(OUT/'probe_traces.pt',weights_only=True,map_location='cpu');prior=torch.load(ROOT/'newline_heads_run/per_token.pt',weights_only=True,map_location='cpu');prior={k:{z['window']:z for z in v}for k,v in prior.items()};assert len(traces)==432
checks=json.loads((OUT/'checks.json').read_text());assert len(checks)==80 and all(z['loss_exact']and z['pred_exact']for z in checks)
closure_pairs=[('row_block_p_all','clean_block_p_all'),('column_block_p_all','clean_block_p_all'),('row_from_clean_kv','clean'),('row_from_column_kv','column'),('clean_from_row_kv','row')]
ratios={};effects={};closures={};key_drift={};norms={}
for c,ws in sets.items():
 g='broadcast_'+c
 for i,xx in enumerate(ws):
  assert torch.equal(xx,old[c][i]);x=xx.clone();a=x[:129];a[(a==187)|(a==535)]=209;x[0]=510;x[64]=x[128]=187;digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();clean=traces[f'{g}__clean__{i}']
  for key,items in groups.items():
   if not key.startswith(g+'__'):continue
   t=key.split('__')[1];z=traces[f'{key}__{i}'];assert items[i]['token_sha256']==digest and items[i]['score_start']==129 and items[i]['scored_tokens']==382
   for k in ['activation','H12_values','H12_contexts']:assert torch.equal(z[k],clean[k])
   mode='row'if t=='row'or t.startswith('row_')else('column'if t=='column'or t.startswith('column_')else'clean');state=traces[f'{g}__{mode}__{i}']
   for k in ['L2_input_p','L2_normalized_p']:assert torch.equal(z[k],state[k])
   for l in range(2,16):
    assert set(z['paths'][l])=={'k','v'}
    for k in 'kv':
     assert z['paths'][l][k].shape==(1,2048)
     if '_block_'in t:assert torch.equal(z['paths'][l][k],state['paths'][l][k])
     if '_from_'in t:
      _,donor,kinds=t.split('_from_')[0],t.split('_from_')[1].split('_')[0],t.split('_')[-1]
      if k in kinds:assert torch.equal(z['paths'][l][k],traces[f'{g}__{donor}__{i}']['paths'][l][k])
     if t.endswith('_zero_p_values'):
      if k=='v':assert(z['paths'][l][k]==0).all()
      elif t=='cleanK_zero_p_values':assert torch.equal(z['paths'][l][k],clean['paths'][l][k])
   if t in ['clean','row']:
    pt='clean'if t=='clean'else'delete_p'
    for k in ['loss','pred','kl']:assert torch.equal(raw[key][i][k],prior[f'heads_{c}_R__{pt}'][i][k])
  row=traces[f'{g}__row__{i}'];col=traces[f'{g}__column__{i}'];assert row['L2_input_p'][1764]==col['L2_input_p'][1764]
  assert torch.equal(row['L2_input_p'][:1764],clean['L2_input_p'][:1764])and torch.equal(row['L2_input_p'][1765:],clean['L2_input_p'][1765:])
  for a,b in closure_pairs:
   for k in ['loss','pred','kl']:assert torch.equal(raw[g+'__'+a][i][k],raw[g+'__'+b][i][k])
 for a,b in closure_pairs:closures[g+'__'+a+'_to_'+b]=compare(raw,groups,g,a,b)
 for key,items in groups.items():
  if not key.startswith(g+'__'):continue
  t=key.split('__')[1];q=effect(groups,g,t,'clean');ratios[key]={'ppl_ratio':math.exp(q['mean']),'delta_nll':q,'agreement':interval([z['agreement']for z in items.values()]),'kl':interval([z['kl_clean_to_condition']for z in items.values()])}
 pairs=[('cleanK_zero_p_values','clean_zero_p_values'),('cleanK_zero_p_values','clean_block_p_all'),('clean_zero_p_values','clean_block_p_all'),('row_zero_p_values','row'),('column_zero_p_values','column'),('row_from_clean_k','row'),('row_from_clean_v','row'),('row_from_column_k','row'),('row_from_column_v','row'),('clean_from_row_k','clean'),('clean_from_row_v','clean')]
 for band in ['early','middle','late','all']:
  pairs += [('row_block_p_'+band,'row'),('row_block_p_'+band,'clean_block_p_'+band)]
 pairs += [('row_block_q_all','row'),('clean_block_q_all','clean')]
 for a,b in pairs:effects[g+'__'+a+'_minus_'+b]=effect(groups,g,a,b)
 for mode in ['clean','row','column']:
  key_drift[g+'__'+mode]={str(l):interval([float((traces[f'{g}__{mode}_zero_p_values__{i}']['paths'][l]['k'].float()-traces[f'{g}__{mode}__{i}']['paths'][l]['k'].float()).norm())for i in range(4)])for l in range(2,16)}
  norms[g+'__'+mode]={name:interval([float(traces[f'{g}__{mode}__{i}'][name][1764])for i in range(4)])for name in ['L2_input_p','L2_normalized_p']}
bcrows=list(csv.DictReader((OUT/'broadcast_contrasts.csv').open()));assert len(bcrows)==48
for row in bcrows:
 g=row['group'];terms=json.loads(row['terms']);q=interval([sum(v*groups[g+'__'+t][i]['nll']for t,v in terms)for i in range(4)])
 for a,b in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
 effects[g+'__'+row['contrast']]=q
result.update(broadcast_contrasts_replayed=48,architectural_closures_replayed=80,prior_clean_and_row_anchors_exact=32,all_first_neuron_and_H12_traces_unchanged=True,matched_382_target_ids_verified=True,postpatch_preclip_donor_and_zero_captures_verified=True,all_blocking_preserves_p_KV_states=True,row_column_raw_L2_target_equal=True,ratios=ratios,paired_effects=effects,closure_details=closures,current_key_zeroV_key_drift=key_drift,L2_target_states=norms,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','paired_effects','closure_details','current_key_zeroV_key_drift','L2_target_states']},indent=2))
