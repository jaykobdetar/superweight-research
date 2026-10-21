"""Independent upstream raw-score, selection, pulse and closure replay; CPU only."""
import csv,hashlib,json,math,os,resource
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import interval,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'upstream_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta and meta['forward_passes']==2384
assert meta['model_hash_before']==meta['model_hash_after']and meta['parameter_versions_unchanged']
for field,p in [('script_sha256',ROOT/'upstream.py'),('design_sha256',ROOT/'UPSTREAM.md'),('development_parent_sha256',ROOT.parent/'pulse_causal_tranche/run/tokens.pt'),('heldout_parent_sha256',ROOT/'confirmation_run/tokens.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
raw=torch.load(OUT/'per_token.pt',map_location='cpu',weights_only=True);raw={k:{z['window']:z for z in v}for k,v in raw.items()}
rows=[json.loads(s)for s in (OUT/'windows.jsonl').read_text().splitlines()];groups=defaultdict(dict)
for z in rows:
 key=z['group']+'__'+z['trial'];assert z['window']not in groups[key];groups[key][z['window']]=z
assert len(rows)==2112 and len(groups)==528 and set(raw)==set(groups)
for key,items in raw.items():
 assert len(items)==4;ordinary=raw[key.split('__')[0]+'__ordinary']
 for i,z in items.items():
  r=groups[key][i];assert len(z['loss'])==254==r['scored_tokens'];assert z['token_sha256']==r['token_sha256']==ordinary[i]['token_sha256']
  assert abs(float(z['loss'].double().mean())-r['nll'])<1e-12 and abs(float(z['kl'].double().mean())-r['kl_clean_to_condition'])<1e-12
  assert abs(float(ordinary[i]['loss'].double().mean())-r['baseline_nll'])<1e-12 and float(z['pred'].eq(ordinary[i]['pred']).double().mean())==r['agreement']
summaries=list(csv.DictReader((OUT/'summary.csv').open()));assert len(summaries)==528
for row in summaries:
 rs=list(groups[row['group']+'__'+row['trial']].values());assert int(row['windows'])==4 and abs(float(row['nll'])-np.mean([r['nll']for r in rs]))<1e-12
 for vals,fields in [([r['nll']-r['baseline_nll']for r in rs],['delta_nll','delta_low','delta_high']),([r['agreement']for r in rs],['agreement','agreement_low','agreement_high']),([r['kl_clean_to_condition']for r in rs],['kl','kl_low','kl_high'])]:
  q=interval(vals)
  for field,k in zip(fields,['mean','low','high']):assert abs(float(row[field])-q[k])<1e-12
 for a,b in [('ppl_ratio','delta_nll'),('ratio_low','delta_low'),('ratio_high','delta_high')]:assert abs(float(row[a])-math.exp(float(row[b])))<1e-12
contrasts=list(csv.DictReader((OUT/'contrasts.csv').open()))
for row in contrasts:
 q=effect(groups,row['group'],row['trial'],row['reference'])
 for a,b in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[a])-q[b])<1e-12
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);prior=torch.load(ROOT/'confirmation_run/tokens.pt',map_location='cpu',weights_only=True)
traces=torch.load(OUT/'source_intermediates.pt',map_location='cpu',weights_only=True);donors=torch.load(OUT/'head_donors.pt',map_location='cpu',weights_only=True)
obs=json.loads((OUT/'pulse_observations.json').read_text());assert len(obs)==len(traces)==2112 and len(donors)==72
weight=.63232421875;og=defaultdict(dict)
for z in obs:
 g,t,i=z['group'],z['trial'],z['window'];trace=traces[f'{g}__{t}__{i}'];ref=donors[f'{g}_{i}']['intermediates'];og[g+'__'+t][i]=z
 assert trace.shape==(2,8192)and torch.equal(trace[0],ref['ordinary'][0])
 assert z['pulse']==float(trace[1,1710].float()*weight)and z['source_pulse']==float(trace[0,1710].float()*weight)
 assert z['ordinary_pulse']==float(ref['ordinary'][1,1710].float()*weight)and z['isolated_pulse']==float(ref['isolated'][1,1710].float()*weight)
 assert z['intermediate_source_boundary_equal']==torch.equal(trace[0],trace[1])
 assert abs(z['pulse_gap_fraction']-(z['pulse']-z['ordinary_pulse'])/(z['isolated_pulse']-z['ordinary_pulse']))<1e-12
 assert z['token_sha256']==groups[g+'__'+t][i]['token_sha256']
 if t.endswith('_delete_boundary'):assert torch.equal(trace,traces[f'{g}__{t.removesuffix("_delete_boundary")}__{i}'])
ps=list(csv.DictReader((OUT/'pulse_summary.csv').open()));assert len(ps)==528
for row in ps:
 rs=list(og[row['group']+'__'+row['trial']].values())
 for field,cols in [('pulse_gap_fraction',['gap_fraction','gap_low','gap_high']),('pulse',['pulse','pulse_low','pulse_high'])]:
  q=interval([z[field]for z in rs])
  for a,b in zip(cols,['mean','low','high']):assert abs(float(row[a])-q[b])<1e-12
frozen=json.loads((OUT/'frozen_sets.json').read_text());screen=json.loads((OUT/'screen.json').read_text());assert len(screen)==256
for z in screen:
 ref=donors['dev_'+str(z['window'])]['intermediates'];t0=ref['ordinary'][1,1710].float();t1=ref['isolated'][1,1710].float();activation=torch.tensor(z['pulse']/weight).half().float()
 expected=float(activation-t0)/float(t1-t0);assert abs(z['gap_fraction']-expected)<1e-12
scores={(l,h):float(np.mean([z['gap_fraction']for z in screen if z['layer']==l and z['head']==h]))for l in [0,1]for h in range(16)}
ranked=sorted(scores,key=lambda a:(-scores[a],a));assert [(z['layer'],z['head'])for z in frozen['ranking']]==ranked
assert all(z['mean_gap_fraction']==scores[z['layer'],z['head']]for z in frozen['ranking'])
rng=np.random.default_rng(20261021);overlaps={}
for k in [1,2,4,8,16,32]:
 chosen=ranked[:k];assert frozen['sets']['top'+str(k)]==[list(z)for z in chosen]
 if k<32:
  control=[(l,int(h))for l in [0,1]for h in rng.choice(16,sum(ll==l for ll,h in chosen),replace=False)];assert frozen['sets']['control'+str(k)]==[list(z)for z in control]
  overlaps[str(k)]={'overlap_count':len(set(chosen)&set(control)),'contains_L1H12':(1,12)in control,'identical_to_selected':control==chosen}
effects={};pulse_effects={};closure=json.loads((OUT/'closure_checks.json').read_text());assert len(closure)==128 and all(z[k]for z in closure for k in ['intermediate_exact','loss_exact','pred_exact'])
for c,ws in sets['evaluation'].items():
 for i,xx in enumerate(ws):
  assert torch.equal(xx,prior[c]['evaluation'][i])
  for identity,tid in [('The',510),('period',15)]:
   for p in [128,256]:
    x=xx[256-p:].clone();x[0]=x[p]=tid;g=f'upstream_{c}_{identity}_{p}';h=hashlib.sha256(x.numpy().tobytes()).hexdigest()
    assert torch.equal(x[p+2:],xx[258:])
    for key,items in groups.items():
     if key.startswith(g+'__'):assert items[i]['token_sha256']==h and items[i]['score_start']==p+1
    ref=donors[f'{g}_{i}']['intermediates'];assert torch.equal(ref['isolated'][0],ref['isolated'][1])
    for world,target in [('ordinary','isolated'),('isolated','ordinary')]:
     assert torch.equal(traces[f'{g}__{world}_top32__{i}'],ref[target])
     for k in ['loss','pred','kl']:
      assert torch.equal(raw[g+'__'+world+'_top32'][i][k],raw[g+'__'+target][i][k])
      assert torch.equal(raw[g+'__'+world+'_top1'][i][k],raw[g+'__'+world+'_control1'][i][k])
for g in sorted(set(z['group']for z in rows)):
 for t,ref in [('isolated','ordinary'),('ordinary_top1','ordinary'),('isolated_top1','isolated'),('ordinary_L0','ordinary'),('ordinary_L1','ordinary')]+[(t,t.removesuffix('_delete_boundary'))for t in ['ordinary_delete_boundary','isolated_delete_boundary','ordinary_top2_delete_boundary','ordinary_top8_delete_boundary','ordinary_top32_delete_boundary']]:effects[g+'__'+t]=compare(raw,groups,g,t,ref)
 for t in ['ordinary','isolated','ordinary_top1','isolated_top1','ordinary_top2','ordinary_top8','ordinary_L0','ordinary_L1']:
  pulse_effects[g+'__'+t]={'gap_fraction':interval([z['pulse_gap_fraction']for z in og[g+'__'+t].values()]),'pulse':interval([z['pulse']for z in og[g+'__'+t].values()])}
result={'raw_rows_replayed':len(rows),'summary_rows_replayed':len(summaries),'contrasts_replayed':len(contrasts),'pulse_summary_rows_replayed':len(ps),'all_retained_closures_exact':True,'runtime_full_sequence_closure_flags':128,'scored_target_alignment_pass':True,'screen_fraction_reconstruction_from_saved_pulse_pass':True,'ranking_and_seeded_control_replay_pass':True,'all_source0_traces_unchanged':True,'conditional_deletion_input_traces_unchanged':True,'control_overlaps':overlaps,'top_screen':frozen['ranking'][:8],'effects':effects,'pulse_effects':pulse_effects,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Full sequence closure was asserted at runtime; independent raw-output replay covers retained254 targets. Single-screen intermediates were not saved; their fractions are reconstructed from saved pulse amplitude and donor trace using known scalar weight.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['effects','pulse_effects']},indent=2))
