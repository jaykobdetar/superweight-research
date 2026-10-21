"""Paired query-bias/source-access effects and aligned attention diagnostics."""
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
    diag=torch.load(p/'attention_diagnostics.pt',weights_only=True,map_location='cpu');m=json.loads((p/'manifest.json').read_text())
    assert len(rows)==len(raw)==len(diag)==m['records']==m['forward_passes']==776
    assert m.get('status')!='failed' and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
    groups=defaultdict(list);inputs=defaultdict(dict);error=0.;attrows=[]
    for r in rows:
        key=r['key'];z=raw[key];pre='__'.join(key.split('__')[:5]);b=raw[pre+'__C'];j=r['score_index']
        assert j==z['score_index']==1 and r['scored_tokens']==510 and z['token_sha256']==r['token_sha256']==b['token_sha256']
        values={'nll':float(z['loss'][1:].double().mean()),'baseline_nll':float(b['loss'][1:].double().mean()),'kl':float(z['kl'][1:].double().mean()),'agreement':float(z['pred'][1:].eq(b['pred'][1:]).double().mean())}
        for k,v in values.items():error=max(error,abs(v-r[k]));assert math.isfinite(v) and abs(v-r[k])<1e-12
        groups[r['corpus'],r['identity'],r['trial']].append(r);inputs[r['corpus'],r['identity'],r['window']][r['trial']]=r
        assert set(diag[key])==set(range(4,12))
        for l,d in diag[key].items():
            assert 0<=d['context_max_abs_error']<=1e-4
            for head in range(12):
                masses=d['source_mass'][head,1:-1];assert len(masses)==510 and torch.isfinite(masses).all() and masses.min()>=0 and masses.max()<=1
                if '_zero_' in r['trial']:assert torch.count_nonzero(d['source_vector_norm'])==0
                if r['trial'].endswith('_blocked'):assert torch.count_nonzero(d['source_mass'][:,1:])==0
                attrows.append({k:r[k] for k in ('corpus','identity','window','trial')}|{'layer':l,'head':head,'scored_queries':510,
                    'mean_source_mass':float(masses.double().mean()),'min_source_mass':float(masses.min()),'max_source_mass':float(masses.max()),
                    'mean_source_vector_norm':float(d['source_vector_norm'][head,1:-1].double().mean()),
                    'mean_nonsource_vector_norm':float(d['nonsource_vector_norm'][head,1:-1].double().mean())})
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','identity')];dump(p/'summary.json',summary)
    dump(p/'attention_summary.json',attrows);contrasts=[]
    for (c,i,w),r in inputs.items():
        n={t:x['nll'] for t,x in r.items()}
        for world in 'CEB':
            op=n[world+'_zero_present_causal'];bp=n[world+'_zero_present_blocked']
            for mode in ('removed','random'):
                oq=n[f'{world}_zero_{mode}_causal'];bq=n[f'{world}_zero_{mode}_blocked']
                v={f'{world}_{mode}_effect_open':oq-op,f'{world}_{mode}_effect_blocked':bq-bp,
                   f'{world}_{mode}_interaction':(bq-bp)-(oq-op)}
                for t,d in v.items():contrasts.append({'corpus':c,'identity':i,'window':w,'trial':t,'delta_nll':d})
            contrasts.append({'corpus':c,'identity':i,'window':w,'trial':world+'_mask_reference_drift','delta_nll':op-n[world+'_zero_native']})
    dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
    for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    paired=[]
    for (c,i,t),ds in cg.items():
        x=np.array(ds);assert len(x)==4;rng=np.random.default_rng(20261021);bs=x[rng.integers(0,4,(2000,4))].mean(1)
        paired.append({'corpus':c,'identity':i,'trial':t,'n':4,'mean_delta_nll':float(x.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',paired)
    checks=json.loads((p/'checks.json').read_text());ops=json.loads((p/'operation_checks.json').read_text())
    assert len(checks)==680 and len(ops)==776
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    assert all(o['projection_checks']==o['query_checks']==o['context_checks']==8 and o['writer_locality_checks']==30 and o['max_context_error']<=1e-4 for o in ops)
    dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':error,'checks':len(checks),'attention_head_rows':len(attrows),
                        'max_context_error':max(o['max_context_error'] for o in ops),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['identity']=='natural' and r['trial'].startswith('C_zero_'):print(r['corpus'],r['trial'],round(r['ppl_ratio'],6),round(r['mean_agreement'],6))
if __name__=='__main__':main()
