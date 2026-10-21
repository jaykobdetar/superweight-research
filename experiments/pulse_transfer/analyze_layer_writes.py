"""Exact numeric replay and layer-local conditional comparisons."""
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
    traces=torch.load(p/'traces.pt',weights_only=True,map_location='cpu')
    manifest=json.loads((p/'manifest.json').read_text());assert len(rows)==len(raw)==len(traces)==manifest['records']==manifest['forward_passes']==776
    assert manifest['model_hash_before']==manifest['model_hash_after'] and manifest['parameter_versions_unchanged']
    groups=defaultdict(list);byinput=defaultdict(dict);error=0.;geometry=[]
    for r in rows:
        key=r['key'];prefix='__'.join(key.split('__')[:5]);z=raw[key];b=raw[prefix+'__E0B30BL0'];j=1
        assert r['score_index']==z['score_index']==j and r['scored_tokens']==510 and z['token_sha256']==b['token_sha256']==r['token_sha256']
        vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),
              'kl':float(z['kl'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean())}
        for k,v in vals.items():assert math.isfinite(v);error=max(error,abs(v-r[k]));assert abs(v-r[k])<1e-12
        groups[r['corpus'],r['identity'],r['trial']].append(r);byinput[r['corpus'],r['identity'],r['window']][r['trial']]=r
        for l,parts in traces[key]['capture']['writers'].items():
            for component,vec in parts.items():
                geometry.append({'corpus':r['corpus'],'identity':r['identity'],'window':r['window'],'trial':r['trial'],
                                 'layer':l,'component':component,'before_coords':vec['before'][manifest['coordinates']].tolist(),
                                 'after_coords':vec['after'][manifest['coordinates']].tolist(),'after_norm':float(vec['after'].norm())})
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','identity')]
    dump(p/'summary.json',summary);dump(p/'geometry.json',geometry)
    contrasts=[]
    for (c,i,w),r in byinput.items():
        v={'block3_givenE':r['E1B31BL0']['nll']-r['E1B30BL0']['nll'],
           'blocklater_givenE':r['E1B30BL1']['nll']-r['E1B30BL0']['nll'],
           'block3_interactionE':r['E1B31BL0']['nll']-r['E1B30BL0']['nll']-r['E0B31BL0']['nll']+r['E0B30BL0']['nll'],
           'blocklater_interactionE':r['E1B30BL1']['nll']-r['E1B30BL0']['nll']-r['E0B30BL1']['nll']+r['E0B30BL0']['nll']}
        for label,d in v.items():contrasts.append({'corpus':c,'identity':i,'window':w,'trial':label,'delta_nll':d})
    dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
    for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    paired=[]
    for (c,i,t),ds in cg.items():
        x=np.array(ds);rng=np.random.default_rng(20261021);bs=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
        paired.append({'corpus':c,'identity':i,'trial':t,'n':len(x),'mean_delta_nll':float(x.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',paired)
    checks=json.loads((p/'checks.json').read_text());assert len(checks)==40 and all(c['loss_exact'] and c['pred_exact'] for c in checks)
    ops=json.loads((p/'operation_checks.json').read_text());conditions=json.loads((p/'conditions.json').read_text())
    cc={c['trial']:c for c in conditions['conditions']};assert len(ops)==768
    assert all(o['locality_checks']==30 and o['writer_zero_checks']==len(cc[o['trial']]['blocked_layers']) for o in ops)
    dump(p/'audit.json',{'status':'passed','records':len(rows),'max_scalar_error':error,'closures':len(checks),'locality_assertions':sum(o['locality_checks'] for o in ops),'writer_zero_checks':sum(o['writer_zero_checks'] for o in ops),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['identity']=='natural' and (r['trial'].startswith('E1_only') or r['trial'] in ('E1B30BL0','E1B31BL0','E1B30BL1','E1B31BL1')):
            print(r['corpus'],r['trial'],round(r['ppl_ratio'],5),round(r['mean_agreement'],5))

if __name__=='__main__':main()
