"""CPU-only source-identity audit; validates persisted evidence and matched targets."""
import hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'identity_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'identity.py').read_bytes()).hexdigest()
assert meta['tokens_sha256']==hashlib.sha256((ROOT/'template_run/tokens.pt').read_bytes()).hexdigest()
assert meta['templates_sha256']==hashlib.sha256((ROOT/'template_run/templates.pt').read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==1536 and result['groups']==192 and meta['forward_passes']==1536
tokens=torch.load(ROOT/'template_run/tokens.pt',weights_only=True,map_location='cpu')
prior=torch.load(ROOT/'template_run/per_token.pt',weights_only=True,map_location='cpu');prior={k:{r['window']:r for r in v} for k,v in prior.items()}
checks=json.loads((OUT/'source_identity_checks.json').read_text());assert len(checks)==144
tuples={(r['corpus'],r['window'],r['identity'],r['state']) for r in checks}
assert len(tuples)==144 and all(r['all_source_kv_exact'] for r in checks)
assert tuples=={(c,i,identity,state) for c in ['wiki','hume','plato'] for i in range(8) for identity in ['The','eos','period'] for state in ['clean','deleted']}
reference=torch.load(OUT/'fixed_identity_source_caches.pt',weights_only=True,map_location='cpu')
assert set(reference)=={(identity,state) for identity in ['The','eos','period'] for state in ['clean','deleted']}
for identity in ['The','eos','period']:
    for state in ['clean','deleted']:
        assert set(reference[identity,state])==set(range(16))
        for l in range(16):
            for k in ['k','v']:
                v=reference[identity,state][l][k];assert v.shape==(1,1,2048) and torch.isfinite(v).all()
                if l<2:assert torch.equal(v,reference[identity,'clean'][l][k])
result.update(fixed_identity_runtime_checks_complete=True,cache_reference_shapes_and_pre_source_equality_passed=True,cache_equality_replay_limit='Only one reference per fixed identity/state and runtime assertion results are saved; all144 original tensor comparisons cannot be independently rerun from artifacts.',results={},matched_identity_effect_changes={})
for corpus,windows in tokens.items():
    for identity,tid in meta['identities'].items():
        group=f'{corpus}_{identity}'
        for i,xx in enumerate(windows):
            x=xx.clone()
            if tid is not None:x[0]=tid
            assert torch.equal(x[1:],xx[1:])
            expected=hashlib.sha256(x.numpy().tobytes()).hexdigest()
            assert raw[group+'__clean'][i]['token_sha256']==expected
            for direction,state in [('rescue','clean'),('damage','deleted')]:
                for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__all_2_15_'+direction][i][k],raw[group+'__'+state][i][k])
            for name in ['ideal','residual_rank2','wiki_full_mean']:
                for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__'+name+'_clean'][i][k],raw[group+'__'+name+'_deleted'][i][k])
            if identity=='natural':
                for trial in ['clean','deleted','ideal_clean','ideal_deleted','residual_rank2_clean','residual_rank2_deleted','wiki_full_mean_clean','wiki_full_mean_deleted']:
                    for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__'+trial][i][k],prior['natural_'+corpus+'__'+trial][i][k])
        for trial in ['deleted','L2_all_rescue','early_2_5_rescue','late_3_15_rescue','all_2_15_rescue','L2_all_damage','early_2_5_damage','late_3_15_damage','all_2_15_damage','ideal_clean','residual_rank2_clean','wiki_full_mean_clean']:
            rows=list(groups[group+'__'+trial].values())
            result['results'][group+'__'+trial]={'vs_clean':effect(groups,group,trial,'clean'),'vs_deleted':effect(groups,group,trial,'deleted'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rows])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rows]))}
        if identity!='natural':
            for trial in ['L2_all_rescue','ideal_clean','residual_rank2_clean']:
                gnat=groups[corpus+'_natural__'+trial];bnat=groups[corpus+'_natural__clean']
                gfix=groups[group+'__'+trial];bfix=groups[group+'__clean']
                # Difference of intervention effects on identical suffix target IDs.
                vals=[(gfix[i]['nll']-bfix[i]['nll'])-(gnat[i]['nll']-bnat[i]['nll']) for i in range(8)]
                result['matched_identity_effect_changes'][group+'__'+trial]=interval(vals)
        for key,rows in groups.items():
            if key.startswith(group+'__'):assert all(r['score_start']==1 and r['scored_tokens']==510 for r in rows.values())
result.update(suffix_target_and_input_hash_checks_passed=True,all_layer_cut_exact_loss_prediction_kl=True,template_clean_deleted_identity_exact=True,natural_conditions_exactly_replicate_prior_template=True,score_range='predictors1..510; target token indices2..511',peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['results','matched_identity_effect_changes']},indent=2))
