"""CPU replay of greedy trajectory, matched-history, and prompt summary metrics."""
import collections,csv,hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'generation_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
for field,p in [('script_sha256',ROOT/'generation.py'),('prompt_source_sha256',ROOT/'extended_context_run/tokens.pt'),('dictionary_sha256',ROOT/'extended_context_run/exact_and_rank2_dictionary.pt'),('calibration_sha256',ROOT/'cache_run/calibration.pt')]:assert meta[field]==hashlib.sha256(p.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged'] and meta['forward_passes']==13216
raw=torch.load(OUT/'per_step.pt',weights_only=True,map_location='cpu');assert len(raw)==16
prompts=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');extended=torch.load(ROOT/'extended_context_run/tokens.pt',weights_only=True,map_location='cpu')
eos=json.loads(Path('/EXTERNAL/hf_cache/models--allenai--OLMo-1B-0724-hf/snapshots/d7cbab742d80589e714b1a2d7f838dcd21cbe143/config.json').read_text())['eos_token_id']
records=[json.loads(s) for s in (OUT/'results.jsonl').read_text().splitlines()];assert len(records)==96
groups=collections.defaultdict(list);divergences=[]
for r in records:
 c,i,t=r['corpus'],r['window'],r['variant'];z=raw[c,i];clean=z['clean'];f=z['variants'][t]['free'];h=z['variants'][t]['matched_history'];x=z['prompt']
 assert torch.equal(x,prompts[c][i]) and torch.equal(x,extended[c][i][:128])
 assert r['prompt_sha256']==hashlib.sha256(x.numpy().tobytes()).hexdigest()
 for item in [clean,f,h]:
  assert item['pred'].shape==item['margin'].shape==(64,) and torch.isfinite(item['margin']).all() and (item['margin']>=0).all()
 assert torch.isfinite(h['kl']).all() and torch.isfinite(h['reference_nll']).all()
 diff=f['pred'].ne(clean['pred']);hd=h['pred'].ne(clean['pred'])
 first=int(diff.nonzero()[0]) if diff.any() else 64;hf=int(hd.nonzero()[0]) if hd.any() else 64
 assert first==hf==r['matching_prefix_length']
 assert r['first_divergence_zero_based']==(first if first<64 else None)
 assert r['identical_sequence']==(first==64)
 assert torch.equal(f['pred'][:min(64,first+1)],h['pred'][:min(64,first+1)])
 assert torch.equal(f['margin'][:min(64,first+1)],h['margin'][:min(64,first+1)])
 fields={'free_agreement':float((~diff).double().mean()),'matched_history_agreement':float((~hd).double().mean()),'matched_history_kl':float(h['kl'].double().mean()),'reference_delta_nll':float((h['reference_nll']-clean['reference_nll']).double().mean())}
 for k,v in fields.items():assert abs(v-r[k])<1e-12
 assert r['clean_eos_count']==int(clean['pred'].eq(eos).sum()) and r['condition_eos_count']==int(f['pred'].eq(eos).sum())
 for corpus in [c,'pooled']:groups[corpus,t].append(r)
 if first<64:divergences.append({'corpus':c,'window':i,'variant':t,'first':first,'clean_margin_at_first':float(clean['margin'][first]),'condition_margin_at_first':float(f['margin'][first])})
checks=json.loads((OUT/'execution_checks.json').read_text());assert len(checks)==96
assert all(z['first_divergence_coincides'] and z['final_cache_length']==191 for z in checks)
summary_path=OUT/'summary.csv';summary=list(csv.DictReader(summary_path.open())) if summary_path.exists() else []
for r in summary:
 rs=groups[r['corpus'],r['variant']];assert int(r['prompts'])==len(rs) and int(r['identical_sequences'])==sum(z['identical_sequence'] for z in rs)
 for k in ['free_agreement','matching_prefix_length','matched_history_agreement','matched_history_kl','reference_delta_nll']:
  d=interval([z[k] for z in rs])
  for suffix,stat in [('', 'mean'),('_low','low'),('_high','high')]:assert abs(float(r[k+suffix])-d[stat])<1e-12
if summary:assert len(summary)==30
out={'records':96,'prompts':16,'forward_passes':13216,'summary_rows_replayed':len(summary),'hashes_model_integrity_passed':True,'metrics_and_eos_counts_replayed':True,'free_and_forced_paths_identical_through_first_divergence':True,'final_cache_lengths_valid':True,'distribution_level_kl_replay_available':False,'distribution_level_kl_replay_limit':'Logits were discarded; aggregates are verified from per-step KL and log-probability metrics, with formula reviewed statically.','summary':{c+'__'+t:{'identical_sequences':sum(z['identical_sequence']for z in rs),'prompts':len(rs),**{k:interval([z[k]for z in rs])for k in ['free_agreement','matching_prefix_length','matched_history_agreement','matched_history_kl','reference_delta_nll']}}for (c,t),rs in groups.items()},'divergences':divergences,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['summary','divergences']},indent=2))
