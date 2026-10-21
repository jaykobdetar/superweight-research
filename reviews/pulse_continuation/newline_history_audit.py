"""CPU-only matched token-history/edge experiment and analysis audit."""
import csv,hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
from audit_common import load_audit,interval,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'newline_history_run'
m=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in m and m['forward_passes']==672
assert m['model_hash_before']==m['model_hash_after']and m['parameter_versions_unchanged']
for f,p in [('script_sha256',ROOT/'newline_history.py'),('design_sha256',ROOT/'NEWLINE_HISTORY.md'),('tokens_parent_sha256',ROOT/'later_uses_run/tokens.pt')]:assert m[f]==hashlib.sha256(p.read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT,expected_windows=4);assert result['rows']==672 and result['groups']==168
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');prior=torch.load(ROOT/'later_uses_run/tokens.pt',weights_only=True,map_location='cpu');traces=torch.load(OUT/'probe_traces.pt',weights_only=True,map_location='cpu');obs=json.loads((OUT/'pulse_observations.json').read_text());assert len(traces)==len(obs)==672;obs={(r['group'],r['trial'],r['window']):r for r in obs}
pids=[187,535,186,209,510,15,64];qids=[187,535];ratios={};pulses={};interactions={};edges={};cross={}
for c,ws in sets.items():
 for i,xx in enumerate(ws):
  assert torch.equal(xx,prior[c][i]);base=xx.clone();prefix=base[:129];prefix[(prefix==187)|(prefix==535)]=209;base[0]=510;firsttrace=None
  for pid in pids:
   for qid in qids:
    x=base.clone();x[64]=pid;x[128]=qid;assert torch.equal(x[130:],xx[130:]);digest=hashlib.sha256(x.numpy().tobytes()).hexdigest();g=f'newline_{c}_p{pid}_q{qid}';bc=traces[f'{g}__clean__{i}'];trials=['clean']
    if pid in [209,187,535]:trials+=['delete_p','delete_q','delete_both','column_q']
    if pid==qid==187:trials+=['exclude_L0','exclude_L1','exclude_both','sham_L0']
    for t in trials:
     cap=traces[f'{g}__{t}__{i}'];o=obs[g,t,i];r=groups[g+'__'+t][i];assert r['token_sha256']==digest==o['token_sha256']and r['score_start']==129 and r['scored_tokens']==382
     assert cap['activation'].shape==(512,)and cap['H12_values'].shape==cap['H12_contexts'].shape==(3,128)
     assert torch.equal(cap['potential_pulse'],cap['activation'].float()*.63232421875)
     applied=cap['potential_pulse'].clone();pos={'delete_p':[64],'delete_q':[128],'delete_both':[64,128],'column_q':[128]}.get(t,[])
     if pos:applied[pos]=0
     assert torch.equal(cap['applied_pulse'],applied)
     for k,p in [('p_pulse',64),('q_pulse',128),('source_pulse',0)]:assert o[k]==float(cap['potential_pulse'][p])
     if pos:
      for k in ['activation','H12_values','H12_contexts']:assert torch.equal(cap[k],bc[k])
     if t.startswith('exclude_')or t=='sham_L0':
      assert torch.equal(cap['activation'][:128],bc['activation'][:128]);assert torch.equal(cap['H12_values'][:2],bc['H12_values'][:2]);assert torch.equal(cap['H12_contexts'][:2],bc['H12_contexts'][:2])
      if t=='exclude_L1':assert torch.equal(cap['H12_values'],bc['H12_values'])
     if firsttrace is None:firsttrace=cap
     for k in ['H12_values','H12_contexts']:assert torch.equal(cap[k][0],firsttrace[k][0])
     assert cap['activation'][0]==firsttrace['activation'][0]
  # q ID cannot alter earlier p state.
  for pid in pids:
   a=traces[f'newline_{c}_p{pid}_q187__clean__{i}'];b=traces[f'newline_{c}_p{pid}_q535__clean__{i}']
   for k in ['activation']:assert torch.equal(a[k][:128],b[k][:128])
 for pid in pids:
  for qid in qids:
   g=f'newline_{c}_p{pid}_q{qid}';ts=[k.split('__')[1]for k in groups if k.startswith(g+'__')]
   for t in ts:
    q=effect(groups,g,t,'clean');ratios[g+'__'+t]={'ppl_ratio':math.exp(q['mean']),'delta_nll':q,'agreement':sum(z['agreement']for z in groups[g+'__'+t].values())/4}
    pulses[g+'__'+t]={k:interval([obs[g,t,i][k]for i in range(4)])for k in ['p_pulse','q_pulse','source_pulse']}
   if pid in [209,187,535]:interactions[g]=interval([groups[g+'__delete_both'][i]['nll']-groups[g+'__delete_p'][i]['nll']-groups[g+'__delete_q'][i]['nll']+groups[g+'__clean'][i]['nll']for i in range(4)])
   if pid!=209:
    ref=f'newline_{c}_p209_q{qid}';cross[g]={'q_pulse':interval([obs[g,'clean',i]['q_pulse']-obs[ref,'clean',i]['q_pulse']for i in range(4)]),'nll_on_matched_targets':interval([groups[g+'__clean'][i]['nll']-groups[ref+'__clean'][i]['nll']for i in range(4)])}
 g=f'newline_{c}_p187_q187'
 for t in ['exclude_L0','exclude_L1','exclude_both']:edges[g+'__'+t+'_minus_sham']={'nll':effect(groups,g,t,'sham_L0'),'q_pulse':interval([obs[g,t,i]['q_pulse']-obs[g,'sham_L0',i]['q_pulse']for i in range(4)])}
ss=list(csv.DictReader((OUT/'pulse_summary.csv').open()));assert len(ss)==168
for row in ss:
 d=pulses[row['group']+'__'+row['trial']]
 for k in ['p_pulse','q_pulse','source_pulse']:
  for a,b in [(k,'mean'),(k+'_low','low'),(k+'_high','high')]:assert abs(float(row[a])-d[k][b])<1e-12
cs=list(csv.DictReader((OUT/'pulse_contrasts.csv').open()));assert len(cs)==160
for row in cs:
 g,t,ref=row['group'],row['trial'],row['reference'];vals=[obs[g,t,i]['q_pulse']-obs[g if ref=='clean'else ref,'clean',i]['q_pulse']for i in range(4)];q=interval(vals)
 for a,b in [('q_pulse_difference','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
result.update(pulse_summary_rows_replayed=len(ss),pulse_contrast_rows_replayed=len(cs),matched_382_target_ids_verified=True,all_672_observations_replayed=True,projection_patch_inputs_identical=True,causal_source_and_prequery_trace_checks=True,potential_and_applied_labels_exact=True,ratios=ratios,pulses=pulses,deletion_interactions=interactions,edge_vs_sham=edges,cross_prefix_effects=cross,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['ratios','pulses','deletion_interactions','edge_vs_sham','cross_prefix_effects']},indent=2))
