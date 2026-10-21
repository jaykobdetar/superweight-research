"""CPU distribution-level replay of saved targeted greedy divergence logits."""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'generation_ties_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc'in meta
assert meta['script_sha256']==hashlib.sha256((ROOT/'generation_ties.py').read_bytes()).hexdigest()
assert meta['generation_sha256']==hashlib.sha256((ROOT/'generation_run/per_step.pt').read_bytes()).hexdigest()
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
prior=[json.loads(s)for s in (ROOT/'generation_run/results.jsonl').read_text().splitlines()]
cases=[[z['corpus'],z['window'],z['first_divergence_zero_based'],z['variant']]for z in prior if z['variant']in ['exact_full_donor','wiki_mean_late']and not z['identical_sequence']]
assert cases==meta['cases']and len(cases)==4
assert meta['forward_passes']==sum(1+12*(step+1)for c,i,step,trigger in cases)==892
raw=torch.load(ROOT/'generation_run/per_step.pt',map_location='cpu',weights_only=True)
logits=torch.load(OUT/'logits.pt',map_location='cpu',weights_only=True);rows=json.loads((OUT/'results.json').read_text());assert len(logits)==48 and len(rows)==24
errors=[];triggers=[]
for r in rows:
 c,i,step,name=r['corpus'],r['window'],r['step'],r['variant'];z=logits[c,i,step,name,0]
 assert torch.equal(z,logits[c,i,step,name,1])and torch.isfinite(z).all()
 b=logits[c,i,step,'clean',0].float();z=z.float();bp=int(b.argmax());zp=int(z.argmax())
 target=raw[c,i]['clean']['pred'][step]if name=='clean'else raw[c,i]['variants'][name]['matched_history']['pred'][step]
 assert zp==int(target)and r['condition_token_id']==zp and r['clean_token_id']==bp
 bt=b.topk(2).values;zt=z.topk(2).values
 vals={'clean_top2_margin':float(bt[0]-bt[1]),'condition_top2_margin':float(zt[0]-zt[1]),'clean_logit_at_clean_choice':float(b[bp]),'clean_logit_at_condition_choice':float(b[zp]),'condition_logit_at_clean_choice':float(z[bp]),'condition_logit_at_condition_choice':float(z[zp]),'max_abs_logit_difference':float((b-z).abs().max())}
 for k,v in vals.items():assert r[k]==v
 lb=b.log_softmax(-1);lp=z.log_softmax(-1);kl=float((lb.exp()*(lb-lp)).sum());errors.append(abs(kl-r['kl']))
 assert abs(kl-r['kl'])<2e-6
 if name==r['selection_trigger']:triggers.append({**r,'cpu_recomputed_kl':kl})
result={'selected_cases_exactly_replayed':True,'rows':24,'saved_logit_vectors':48,'all_repeated_logits_bitwise_equal':True,'all_original_step_predictions_reproduced':True,'all_margins_selected_logits_max_deltas_exact':True,'max_cpu_gpu_kl_absolute_difference':max(errors),'trigger_cases':triggers,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
