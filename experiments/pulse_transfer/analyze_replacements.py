"""Reaggregate private records and prespecified paired replacement contrasts."""
import argparse, json, math, resource
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_transfer import dump
from analyze_transfer import aggregate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();p=args.run
    rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()]
    raw=torch.load(p/'per_token.pt',weights_only=True,map_location='cpu')
    manifest=json.loads((p/'manifest.json').read_text())
    assert manifest['forward_passes']==528 and manifest['records']==len(rows)==len(raw)==520
    assert manifest['model_hash_before']==manifest['model_hash_after'] and manifest['parameter_versions_unchanged']
    groups=defaultdict(list);by_input=defaultdict(dict);error=0.
    for r in rows:
        key=r['key'];prefix='__'.join(key.split('__')[:5]);z=raw[key];b=raw[prefix+'__C'];j=r['score_index']
        assert j==1 and r['scored_tokens']==510 and z['token_sha256']==b['token_sha256']==r['token_sha256']
        vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),
              'kl':float(z['kl'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean())}
        for k,v in vals.items():
            assert math.isfinite(v);error=max(error,abs(v-r[k]));assert abs(v-r[k])<1e-12
        groups[r['corpus'],r['identity'],r['trial']].append(r)
        by_input[r['corpus'],r['identity'],r['window']][r['trial']]=r
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','oracle')]
    dump(p/'summary.json',summary)
    contrasts=[]
    for (c,i,w),r in by_input.items():
        values={'complement_interaction':r['N']['nll']-r['R']['nll']-r['D']['nll']+r['C']['nll'],
                'column_minus_complement':r['N']['nll']-r['D']['nll'],
                'meanKV_minus_native':r['C_mean_kv']['nll']-r['C']['nll'],
                'randomKV_minus_meanKV':r['C_random_mean_kv']['nll']-r['C_mean_kv']['nll']}
        for t,v in values.items():contrasts.append({'corpus':c,'identity':i,'window':w,'trial':t,'delta_nll':v})
    dump(p/'paired_contrasts.json',contrasts);gg=defaultdict(list)
    for r in contrasts:gg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    ps=[]
    for (c,i,t),values in gg.items():
        a=np.array(values);rng=np.random.default_rng(20261021);bs=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
        ps.append({'corpus':c,'identity':i,'trial':t,'n':len(a),'mean_delta_nll':float(a.mean()),
                   'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',ps)
    checks=json.loads((p/'checks.json').read_text());assert len(checks)==72 and all(c['loss_exact'] and c['pred_exact'] for c in checks)
    patches=json.loads((p/'patch_checks.json').read_text());assert len(patches)==320 and all(c['max_error']==0 and c['patches'] in (26,52) for c in patches)
    pulse=json.loads((p/'pulse_checks.json').read_text());assert len(pulse)==64 and all(c['locality'] for c in pulse)
    cal=torch.load(p/'calibration.pt',weights_only=True,map_location='cpu')
    detail={'calibration_inputs':8,'distinct_calibration_first_tokens':len({r['first_token_id'] for r in cal['calibration']}),
            'applied_signed_mean_activation':float(cal['activation_mean']),
            'generic_source_contribution_semantics':'Native potential activation times selected weight; use applied_signed_mean_activation for mean_pulse interventions.'}
    coverage=json.loads((p/'first_token_coverage.json').read_text())
    for identity in ('natural','bos'):
        rr=[r for r in coverage if r['identity']==identity];detail[identity+'_coverage']={'inputs':len(rr),'seen':sum(r['seen_in_calibration'] for r in rr),'distinct_first_tokens':len({r['first_token_id'] for r in rr})}
    dump(p/'calibration_summary.json',detail)
    dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':error,'closures':len(checks),'projection_records':len(patches),'pulse_records':len(pulse),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['identity']=='natural':print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5),round(r['mean_kl'],5))

if __name__=='__main__':main()
