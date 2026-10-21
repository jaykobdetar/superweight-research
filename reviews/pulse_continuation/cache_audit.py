"""CPU-only actual-prefix-cache correctness, numerical-control and timing audit."""
import hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,compare,interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'cache_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'cache_bypass.py').read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==2592 and result['groups']==324
tokens=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');gate=torch.load(ROOT/'gate_run/tokens.pt',weights_only=True,map_location='cpu')
checks=json.loads((OUT/'execution_checks.json').read_text());assert len(checks)==1152
for r in checks:assert r['actual_input_tokens']==r['length']-1 and r['final_cache_length']==r['length']
dictionary=torch.load(OUT/'exact_token_dictionary.pt',weights_only=True,map_location='cpu')
assert len(dictionary)==meta['distinct_cached_token_identities']==21
cal=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu')
for corpus in ['wiki','hume','plato']:
    for i,x in enumerate(tokens[corpus]):
        assert len(x)==1024 and torch.equal(x[:512],gate[corpus]['evaluation'][i])
        for identity in ['natural','The']:
            for length in [128,512,1024]:
                xx=x[:length].clone()
                if identity=='The':xx[0]=510
                group=f'cache_{corpus}_{identity}_{length}';expected_hash=hashlib.sha256(xx.numpy().tobytes()).hexdigest()
                assert raw[group+'__clean'][i]['token_sha256']==expected_hash
                for trial in ['clean','exact_full_donor','exact_single_token','wiki_mean_all','hume_mean_all','wiki_mean_late','hume_mean_late','fixed_The','zero_cache']:
                    later=raw[group+'__'+trial][i];full=raw[group+'_full__'+trial][i]
                    for k in ['loss','pred','kl']:assert torch.equal(later[k],full[k][1:])
                refs=[raw[group+'_full__'+t][i] for t in ['exact_full_donor','exact_single_token','wiki_mean_all','hume_mean_all','wiki_mean_late','hume_mean_late','fixed_The','zero_cache']]
                for r in refs[1:]:
                    for k in ['loss','pred','kl']:assert torch.equal(r[k][:1],refs[0][k][:1])
                if identity=='The':
                    for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__fixed_The'][i][k],raw[group+'__exact_single_token'][i][k])
for cached in [v['kv'] for v in dictionary.values()]+list(cal.values()):
    assert len(cached)==16
    for pair in cached:
        for t in pair:assert t.shape==(1,16,1,128) and t.dtype==torch.float16 and torch.isfinite(t).all() and t.abs().max()<=8
assert sum(t.numel()*t.element_size() for pair in next(iter(dictionary.values()))['kv'] for t in pair)==meta['cache_storage_bytes']==131072
result.update(execution_checks_passed=True,nested_crops_and_token_hashes_passed=True,full_primary_scoring_alignment_passed=True,first_prediction_identical_across_bypass_variants=True,fixed_The_matches_exact_for_The_inputs=True,cache_shape_dtype_clip_passed=True,controls={},replacements={})
for corpus in ['wiki','hume','plato']:
    for identity in ['natural','The']:
        for length in [128,512,1024]:
            group=f'cache_{corpus}_{identity}_{length}'
            result['controls'][group]={t:compare(raw,groups,group,t,'clean') for t in ['exact_full_donor','exact_single_token']}
            result['controls'][group]['single_vs_full_donor']=compare(raw,groups,group,'exact_single_token','exact_full_donor')
            for trial in ['wiki_mean_all','hume_mean_all','wiki_mean_late','hume_mean_late','fixed_The','zero_cache']:
                rs=list(groups[group+'__'+trial].values())
                result['replacements'][group+'__'+trial]={'vs_clean':effect(groups,group,trial,'clean'),'vs_exact_single_token':effect(groups,group,trial,'exact_single_token'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rs])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rs]))}
timing=json.loads((OUT/'timings.json').read_text());assert len(timing)==108
modes=['full_no_cache','full_cache','bypass_exact','bypass_mean'];timeby={m:[] for m in modes}
for iteration in range(27):
    rs=[r for r in timing if r['iteration']==iteration];assert {r['mode'] for r in rs}==set(modes) and len(rs)==4
    assert all(r['warmup']==(iteration<3) and r['seconds']>0 for r in rs)
    if iteration>=3:
        for r in rs:timeby[r['mode']].append(r['seconds'])
result['timing']={m:{'n':len(v),'mean_ms':float(np.mean(v)*1000),'median_ms':float(np.median(v)*1000)} for m,v in timeby.items()}
result['timing_paired_difference_ms']={m:interval((np.asarray(timeby[m])-np.asarray(timeby['full_cache']))*1000) for m in ['bypass_exact','bypass_mean']}
result['timing_mean_ratio_to_full_cache']={m:float(np.mean(timeby[m])/np.mean(timeby['full_cache'])) for m in ['bypass_exact','bypass_mean']}
assert meta['forward_passes']==144+1152+16+len(dictionary)+108==1441
result['forward_count_reconciles']=True;result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['controls','replacements']},indent=2))
