"""Headwise value factorial: exact saved-metric audit and paired contrasts."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_transfer import dump
from analyze_transfer import aggregate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);p=ap.parse_args().run
    rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()];raw=torch.load(p/'per_token.pt',weights_only=True,map_location='cpu')
    m=json.loads((p/'manifest.json').read_text());assert len(rows)==len(raw)==m['records']==m['forward_passes']==520
    assert m.get('status')!='failed' and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
    groups=defaultdict(list);byinput=defaultdict(dict);err=0.
    for r in rows:
        z=raw[r['key']];pre='__'.join(r['key'].split('__')[:5]);b=raw[pre+'__C'];j=r['score_index']
        assert j==z['score_index']==1 and r['scored_tokens']==510 and z['token_sha256']==r['token_sha256']==b['token_sha256']
        vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),'kl':float(z['kl'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean())}
        for k,v in vals.items():err=max(err,abs(v-r[k]));assert math.isfinite(v) and abs(v-r[k])<1e-12
        groups[r['corpus'],r['identity'],r['trial']].append(r);byinput[r['corpus'],r['identity'],r['window']][r['trial']]=r
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','identity')];dump(p/'summary.json',summary)
    contrasts=[]
    for (c,i,w),r in byinput.items():
        n={t:x['nll'] for t,x in r.items()};ee,eb,be,bb=(n[t] for t in ('B_DE_NE','B_DE_NB','B_DB_NE','B_DB_NB'))
        v={'normB_minus_normE_given_directionE':eb-ee,'normB_minus_normE_given_directionB':bb-be,
           'directionB_minus_directionE_given_normE':be-ee,'directionB_minus_directionE_given_normB':bb-eb,
           'factorial_interaction':ee-eb-be+bb,'randomE_minus_E':n['B_randomE']-ee,'randomB_minus_B_diagonal':n['B_randomB']-bb,
           'C_zeroV_minus_C':n['C_zeroV']-n['C'],'E_zeroV_minus_E':n['E_zeroV']-n['E'],'B_zeroV_minus_B':n['B_zeroV']-n['B'],
           'B_EK_zeroV_minus_E':n['B_EK_zeroV']-n['E']}
        for t,d in v.items():contrasts.append({'corpus':c,'identity':i,'window':w,'trial':t,'delta_nll':d})
    dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
    for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    paired=[]
    for (c,i,t),ds in cg.items():
        x=np.array(ds);assert len(x)==4;rng=np.random.default_rng(20261021);bs=x[rng.integers(0,4,(2000,4))].mean(1)
        paired.append({'corpus':c,'identity':i,'trial':t,'n':4,'mean_delta_nll':float(x.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',paired)
    checks=json.loads((p/'checks.json').read_text());ops=json.loads((p/'operation_checks.json').read_text());vc=json.loads((p/'value_checks.json').read_text())
    assert len(checks)==200 and len(ops)==520 and len(vc)==2048
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    assert all(o['projection_checks']==8 and o['writer_locality_checks']==30 and o['max_patch_error']==0 and o.get('first_queries_exact',True) for o in ops)
    assert all(v['max_relative_norm_error']<2e-6 for v in vc)
    dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':err,'check_records':len(checks),'value_geometry_checks':len(vc),
                        'projection_locality_checks':sum(o['projection_checks'] for o in ops),'exact_patch_copies':sum(o['patches'] for o in ops),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['identity']=='natural' and r['trial'] in ('E','B','B_DE_NB','B_DB_NE','B_randomE','B_randomB','C_zeroV'):
            print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5))

if __name__=='__main__':main()
