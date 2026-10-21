"""Reaggregate preserved records and predeclared source K/V factorial contrasts."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_transfer import dump
from analyze_transfer import aggregate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();p=a.run
    rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()]
    raw=torch.load(p/'per_token.pt',weights_only=True,map_location='cpu')
    m=json.loads((p/'manifest.json').read_text());assert len(rows)==len(raw)==m['records']==m['forward_passes']==616
    assert m.get('status')!='failed' and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
    groups=defaultdict(list);byinput=defaultdict(dict);err=0.
    for r in rows:
        key=r['key'];pre='__'.join(key.split('__')[:5]);z=raw[key];b=raw[pre+'__C'];j=r['score_index']
        assert j==z['score_index']==1 and r['scored_tokens']==510 and z['token_sha256']==r['token_sha256']==b['token_sha256']
        vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),
              'kl':float(z['kl'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean())}
        for k,v in vals.items():err=max(err,abs(v-r[k]));assert math.isfinite(v) and abs(v-r[k])<1e-12
        groups[r['corpus'],r['identity'],r['trial']].append(r);byinput[r['corpus'],r['identity'],r['window']][r['trial']]=r
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','identity')]
    dump(p/'summary.json',summary);contrasts=[]
    for (c,i,w),r in byinput.items():
        n={t:x['nll'] for t,x in r.items()}
        ee,eb,be,bb=(n[t] for t in ('E_KE_VE','E_KE_VB','E_KB_VE','E_KB_VB'))
        v={'B_minus_E':n['B']-n['E'],'factorial_interaction':ee-eb-be+bb,
           'B_key_minus_E_key_given_EV':be-ee,'B_key_minus_E_key_given_BV':bb-eb,
           'B_value_minus_E_value_given_EK':eb-ee,'B_value_minus_E_value_given_BK':bb-be,
           'E_key_repair_minus_B':eb-n['B'],'E_value_repair_minus_B':be-n['B']}
        for world in 'EB':
            for fixed in ('currentK','EK'):
                zero=n[f'{world}_zeroV_{fixed}'];blocked=n[f'{world}_mask_blocked'];mask=n[f'{world}_mask_causal']
                v[f'{world}_blocked_minus_zeroV_{fixed}_raw']=blocked-zero
                v[f'{world}_blocked_minus_zeroV_{fixed}_adjusted']=(blocked-mask)-(zero-n[world])
            v[world+'_mask_reference_drift']=n[world+'_mask_causal']-n[world]
        for t,d in v.items():contrasts.append({'corpus':c,'identity':i,'window':w,'trial':t,'delta_nll':d})
    dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
    for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    paired=[]
    for (c,i,t),ds in cg.items():
        x=np.array(ds);assert len(x)==4;rng=np.random.default_rng(20261021);bs=x[rng.integers(0,4,(2000,4))].mean(1)
        paired.append({'corpus':c,'identity':i,'trial':t,'n':4,'mean_delta_nll':float(x.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',paired)
    checks=json.loads((p/'checks.json').read_text());ops=json.loads((p/'operation_checks.json').read_text())
    assert len(checks)==392 and len(ops)==616
    for ch in checks:
        for key,v in ch.items():
            if key.endswith('_exact'):assert v
    for op in ops:
        assert op['projection_checks']==8 and op['writer_locality_checks']==30 and op['max_patch_error']==0.
        assert op.get('first_queries_exact',True)
    dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':err,'check_records':len(checks),
                        'projection_locality_checks':sum(o['projection_checks'] for o in ops),'exact_patch_copies':sum(o['patches'] for o in ops),
                        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['identity']=='natural' and r['trial'] in ('E','B','E_KE_VB','E_KB_VE','E_zeroV_currentK','E_mask_blocked'):
            print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5))

if __name__=='__main__':main()
