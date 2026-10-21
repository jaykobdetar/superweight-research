"""CPU-only audit of frozen-cache replacements over nested context crops."""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'extended_context_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
for field,path in [('script_sha256',ROOT/'extended_context.py'),('cache_calibration_sha256',ROOT/'cache_run/calibration.pt'),('template_sha256',ROOT/'template_run/templates.pt')]:assert meta[field]==hashlib.sha256(path.read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
raw,groups,result=load_audit(OUT,expected_windows=4,require_tables=False)
assert result['rows']==896 and result['groups']==224
result['summary_tables_available']=(OUT/'summary.csv').exists() and (OUT/'contrasts.csv').exists()
sets=torch.load(OUT/'tokens.pt',map_location='cpu',weights_only=True);dm=json.loads((OUT/'data_manifest.json').read_text())
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',map_location='cpu',weights_only=True)
dictionary=torch.load(OUT/'exact_and_rank2_dictionary.pt',map_location='cpu',weights_only=True)
identities={int(x[0])for ws in sets.values() for x in ws};assert identities==set(dictionary)
assert meta['forward_passes']==448+2*len(dictionary)
for token,d in dictionary.items():
 for name in ['exact','rank2']:
  assert len(d[name])==16
  for pair in d[name]:
   assert len(pair)==2
   for t in pair:assert t.shape==(1,16,1,128) and torch.isfinite(t).all()
 for l in [0,1]:
  for k in [0,1]:assert torch.equal(d['exact'][l][k],d['rank2'][l][k])
 assert torch.isfinite(d['logit']).all()
checks=json.loads((OUT/'execution_checks.json').read_text());assert len(checks)==320
for z in checks:
 assert z['final_cache_length']==z['length']
 g=f"extended_{z['corpus']}_{z['length']}";i=z['window'];t=z['variant']
 assert z['loss_equal_to_clean']==torch.equal(raw[g+'__'+t][i]['loss'],raw[g+'__clean'][i]['loss'])
variants=['clean','deleted','exact_full_donor','exact_single_token','wiki_mean_all','wiki_mean_late','rank2_late']
comparisons={};effects={};crop_numerics={}
for c,ws in sets.items():
 starts=dm[c]['starts'];assert len(starts)==len(ws)==4 and all(b-a>=4096 for a,b in zip(starts,starts[1:]))
 for i,(x,s) in enumerate(zip(ws,starts)):
  assert len(x)==4096
  if c in ['wiki','hume']:assert torch.equal(x,old['corpora'][c][s:s+4096])
  for n in [512,1024,2048,4096]:
   g=f'extended_{c}_{n}';tokenhash=hashlib.sha256(x[:n].numpy().tobytes()).hexdigest()
   for trial in variants:
    a,b=raw[g+'__'+trial][i],raw[g+'_tail512__'+trial][i]
    assert a['token_sha256']==b['token_sha256']==tokenhash
    assert len(a['loss'])==n-2 and len(b['loss'])==min(512,n-2)
    assert groups[g+'__'+trial][i]['score_start']==1
    assert groups[g+'_tail512__'+trial][i]['score_start']==max(1,n-513)
    for k in ['loss','pred','kl']:assert torch.equal(a[k][-min(512,n-2):],b[k])
    if trial=='exact_full_donor':
     for k in ['loss','pred','kl']:assert torch.equal(a[k],raw[g+'__clean'][i][k])
 for n in [512,1024,2048,4096]:
  g=f'extended_{c}_{n}'
  for tail in ['', '_tail512']:
   group=g+tail
   for trial in variants[1:]:
    comparisons[group+'__'+trial]=compare(raw,groups,group,trial,'clean')
    rows=list(groups[group+'__'+trial].values())
    comparisons[group+'__'+trial].update(mean_agreement=float(np.mean([r['agreement']for r in rows])),mean_kl=float(np.mean([r['kl_clean_to_condition']for r in rows])))
   for trial in ['wiki_mean_all','wiki_mean_late','rank2_late']:effects[group+'__'+trial]=effect(groups,group,trial,'exact_single_token')
  if n!=512:
   for trial in variants:
    a=raw[f'extended_{c}_512__'+trial];b=raw[g+'__'+trial]
    crop_numerics[g+'__'+trial]={'max_first510_token_loss_delta':max(float((a[i]['loss']-b[i]['loss'][:510]).abs().max())for i in range(4)), 'first510_prediction_mismatches':sum(int(a[i]['pred'].ne(b[i]['pred'][:510]).sum())for i in range(4))}
result.update(hashes_model_integrity_passed=True,nested_crops_matched_scoring_passed=True,all_cache_lengths_valid=True,all_full_donor_loss_pred_kl_exact=True,rank2_dictionary_early_layers_exact=True,distinct_identities=len(dictionary),forward_passes=meta['forward_passes'],comparisons=comparisons,vs_exact_single_token=effects,common_crop_numerics=crop_numerics,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','vs_exact_single_token','common_crop_numerics']},indent=2))
