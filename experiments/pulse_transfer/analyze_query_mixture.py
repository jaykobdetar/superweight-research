"""Audit and paired summaries for the query allocation/mixture factorial."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from run_transfer import dump,sha
from analyze_transfer import aggregate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);p=ap.parse_args().run
    rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()];raw=torch.load(p/'per_token.pt',weights_only=True,map_location='cpu');m=json.loads((p/'manifest.json').read_text())
    assert len(rows)==len(raw)==m['records']==m['forward_passes']==520 and 'completed_utc' in m and m.get('status')!='failed'
    assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged']
    groups=defaultdict(list);inputs=defaultdict(dict);error=0.
    for r in rows:
        z=raw[r['key']];key='__'.join(r['key'].split('__')[:5]);b=raw[key+'__C']
        assert r['score_index']==z['score_index']==1 and r['scored_tokens']==510 and z['token_sha256']==r['token_sha256']==b['token_sha256']
        values={'nll':float(z['loss'][1:].double().mean()),'baseline_nll':float(b['loss'][1:].double().mean()),'kl':float(z['kl'][1:].double().mean()),'agreement':float(z['pred'][1:].eq(b['pred'][1:]).double().mean())}
        for k,v in values.items():error=max(error,abs(v-r[k]));assert math.isfinite(v) and abs(v-r[k])<1e-12
        assert len(z['final_hidden_sha256'])==64
        groups[r['corpus'],r['identity'],r['trial']].append(r);inputs[r['corpus'],r['identity'],r['window']][r['trial']]=r
    summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rr)} for (c,i,t),rr in groups.items() if t not in ('repeat','identity')];dump(p/'summary.json',summary)
    contrasts=[]
    for (c,i,w),r in inputs.items():
        n={t:z['nll'] for t,z in r.items()};pp=n['P_GP_DP'];pr=n['P_GP_DR'];rp=n['P_GR_DP'];rr=n['P_GR_DR']
        vals={'mass_at_DP':rp-pp,'mass_at_DR':rr-pr,'mixture_at_GP':pr-pp,'mixture_at_GR':rr-rp,'interaction':rr-rp-pr+pp,
              'bias_removed':n['R']-n['P'],'block_P':n['P_blocked']-n['P'],'block_R':n['R_blocked']-n['R']}
        contrasts.extend({'corpus':c,'identity':i,'window':w,'trial':t,'delta_nll':v} for t,v in vals.items())
    dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
    for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
    paired=[]
    for (c,i,t),ds in cg.items():
        x=np.array(ds);assert len(x)==4;rng=np.random.default_rng(20261021);bs=x[rng.integers(0,4,(2000,4))].mean(1)
        paired.append({'corpus':c,'identity':i,'trial':t,'n':4,'mean_delta_nll':float(x.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist()})
    dump(p/'paired_summary.json',paired)
    checks=json.loads((p/'checks.json').read_text());ops=json.loads((p/'operation_checks.json').read_text());assert len(checks)==840 and len(ops)==520
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    maximums=defaultdict(float);minmass=1.;mixcount=0
    for op in ops:
        assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
        assert op['max_context_error']<=1e-4;maximums['native_context_error']=max(maximums['native_context_error'],op['max_context_error'])
        for ch in op['mixture_checks'].values():
            assert ch['finite'] and ch['source_context_exact'] and ch['min_mass']>0;minmass=min(minmass,ch['min_mass']);mixcount+=1
            for k in ('conditional_error','native_reconstruction_error','mixed_reconstruction_error','formula_error','max_gain'):
                if k!='max_gain':assert ch[k]<=1e-4
                maximums[k]=max(maximums[k],ch[k])
    assert mixcount==32*15*8
    donor_hashes=json.loads((p/'donor_files.json').read_text());assert len(donor_hashes)==32;geom=[]
    for name,h in donor_hashes.items():
        file=p/'donors'/name;assert sha(file)==h;ds=torch.load(file,weights_only=True,map_location='cpu');_,c,w,_,identity=name[:-3].split('__')
        for l in range(4,12):
            a,b=ds['P'][l],ds['R'][l]
            for head in range(12):
                ap=a['mass'][head,:-1].double();ar=b['mass'][head,:-1].double()
                assert ap.numel()==ar.numel()==510
                up=a['stable_mixture'][head,:-1].double();ur=b['stable_mixture'][head,:-1].double()
                geom.append({'corpus':c,'window':int(w),'identity':identity,'layer':l,'head':head,'queries':510,
                    'P_mean_nonsource_mass':float(ap.mean()),'R_mean_nonsource_mass':float(ar.mean()),'P_min_mass':float(ap.min()),'R_min_mass':float(ar.min()),
                    'max_R_over_P':float((ar/ap).max()),'max_P_over_R':float((ap/ar).max()),
                    'mean_conditional_difference_norm':float((ur-up).norm(dim=-1).mean())})
        del ds
    dump(p/'donor_geometry.json',geom);dump(p/'audit.json',{'status':'passed','records':len(rows),'summaries':len(summary),'contrasts':len(contrasts),'paired_summaries':len(paired),
        'max_scalar_error':error,'checks':len(checks),'mixture_checks':mixcount,'min_mass':minmass,'maxima':dict(maximums),'donor_files':len(donor_hashes),'donor_geometry_rows':len(geom),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    for r in summary:
        if r['trial'] in ('P','R','P_GP_DR','P_GR_DP'):print(r['corpus'],r['identity'],r['trial'],round(r['ppl_ratio'],6),round(r['mean_agreement'],6))

if __name__=='__main__':main()
