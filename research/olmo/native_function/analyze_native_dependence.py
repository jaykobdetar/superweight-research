"""Independent descriptive reductions of commit-pinned OLMo public outputs.

No model is loaded. A target is an observed next-token target, not a semantic
task label. Window-weighted summaries preserve the original sampling unit.
"""
from pathlib import Path
import csv, hashlib, itertools, json, math, resource
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'repo/results/pulse_continuation'
OUT=ROOT/'analysis'; OUT.mkdir(exist_ok=True)

def load(run):
    p=DATA/run
    meta=json.loads((p/'per_token_index.json').read_text())
    arrays=np.load(p/'per_token_numeric.npz')
    rows=[json.loads(s) for s in (p/'windows.jsonl').read_text().splitlines()]
    byrow={(r['group']+'__'+r['trial'],r['window']):r for r in rows}
    result={}; errors=[]
    for m in meta:
        key=(m['condition'],m['window']); row=byrow[key]
        x={k:arrays[str(m['index'])+'_'+k].astype(np.float64) for k in ('loss','kl')}
        assert m['token_sha256']==row['token_sha256']
        assert len(x['loss'])==row['scored_tokens']==len(x['kl'])
        errors.extend([abs(x['loss'].mean()-row['nll']),abs(x['kl'].mean()-row['kl_clean_to_condition'])])
        x.update(row=row,sha=m['token_sha256']);result[key]=x
    assert max(errors)<1e-6,(run,max(errors))
    return result,{'run':run,'rows_checked':len(meta),'max_scalar_reduction_error':max(errors)}

def stats(delta,kl):
    pos=np.maximum(delta,0)
    q=np.quantile(delta,[.05,.25,.5,.75,.95,.99])
    topn=max(1,int(np.ceil(len(delta)*.1)))
    total=pos.sum()
    return {'delta_nll':float(delta.mean()),'mean_abs_delta_nll':float(np.abs(delta).mean()),
        'kl':float(kl.mean()),'fraction_harmed':float((delta>0).mean()),
        **{f'fraction_harmed_gt_{t:g}_nat':float((delta>t).mean()) for t in (.1,.5,1,2)},
        **{f'fraction_helped_gt_{t:g}_nat':float((delta< -t).mean()) for t in (.1,.5,1)},
        'top_10pct_targets_share_positive_loss':float(np.sort(pos)[-topn:].sum()/total) if total else 0.,
        **{f'delta_quantile_{p:g}':float(v) for p,v in zip((.05,.25,.5,.75,.95,.99),q)},
        **{f'delta_quarter_{i}':float(a.mean()) for i,a in enumerate(np.array_split(delta,4),1)}}

all_data={}; validation=[]; perwindow=[]; summaries=[]
for p in sorted(DATA.iterdir()):
    d,v=load(p.name); all_data[p.name]=d;validation.append(v)
    bycondition={}
    for (cond,window),x in d.items():
        group,trial=cond.rsplit('__',1)
        base=d[(group+'__clean',window)]
        assert x['sha']==base['sha'] and x['row']['score_start']==base['row']['score_start']
        delta=x['loss']-base['loss']
        m={'run':p.name,'group':group,'trial':trial,'window':window,'targets':len(delta),
           'score_start':x['row']['score_start'],'agreement':x['row']['agreement'],**stats(delta,x['kl'])}
        perwindow.append(m);bycondition.setdefault((group,trial),[]).append(m)
    for (group,trial),ms in bycondition.items():
        keys=[k for k in ms[0] if k not in ('run','group','trial','window','targets','score_start')]
        s={'run':p.name,'group':group,'trial':trial,'windows':len(ms),'targets_total':sum(m['targets'] for m in ms)}
        s.update({k:float(np.mean([m[k] for m in ms])) for k in keys});s['ppl_ratio']=math.exp(s['delta_nll'])
        dv=np.array([m['delta_nll'] for m in ms]);s['window_delta_min']=float(dv.min());s['window_delta_max']=float(dv.max())
        if len(ms)<=4:
            means=np.array([np.mean(sample) for sample in itertools.product(dv,repeat=len(dv))]);lo,hi=np.quantile(means,[.025,.975])
            s['descriptive_empirical_bootstrap_low']=float(lo);s['descriptive_empirical_bootstrap_high']=float(hi)
        summaries.append(s)

def writecsv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
writecsv(OUT/'per_window_native_dependence.csv',perwindow)
writecsv(OUT/'condition_native_dependence.csv',summaries)

# Same-window ranking/overlap checks, descriptive rather than independent tokens.
pairs=[]
comparisons={
 'column_factorial_run':[('delete_row','delete_column')],
 'confirmation_run':[('deleted','remove_source'),('remove_source','mean_KV')],
 'later_uses_run':[('row_source','column_source'),('row_later','column_later')],
 'newline_broadcast_run':[('row','clean_block_p_all'),('cleanK_zero_p_values','clean_block_p_all')],
 'quiet_anchor_run':[('deleted','remove_source_all_clean'),('K_mean_V_zero_clean','remove_source_all_clean')],
}
for run,contrasts in comparisons.items():
    d=all_data[run];groups=sorted({k[0].rsplit('__',1)[0] for k in d})
    for group in groups:
        for a,b in contrasts:
            windows=sorted(w for c,w in d if c==group+'__'+a)
            if not windows or (group+'__'+b,windows[0]) not in d:continue
            values=[]
            for w in windows:
                clean=d[(group+'__clean',w)]['loss'];da=d[(group+'__'+a,w)]['loss']-clean;db=d[(group+'__'+b,w)]['loss']-clean
                ma,mb=da>1,db>1
                m={'run':run,'group':group,'a':a,'b':b,'window':w,'n':len(da),
                   'a_harmed_gt1':int(ma.sum()),'b_harmed_gt1':int(mb.sum()),
                   'overlap_gt1':int((ma&mb).sum()),'a_only_gt1':int((ma&~mb).sum()),
                   'jaccard_gt1':float((ma&mb).sum()/(ma|mb).sum()) if (ma|mb).any() else None,
                   'corr_delta_nll':float(np.corrcoef(da,db)[0,1]) if np.std(da)>0 and np.std(db)>0 else None}
                pairs.append(m)
writecsv(OUT/'paired_target_overlap.csv',pairs)

# Integrity and scope checks, including source block closure under row/column damage.
closures=[]
for run,pairspec in {
 'newline_broadcast_run':[('clean_block_p_all','row_block_p_all'),('clean_block_p_all','column_block_p_all')],
 'quiet_anchor_run':[('remove_source_all_clean','remove_source_all_deleted'),('K_mean_V_mean_clean','K_mean_V_mean_deleted')]
}.items():
    d=all_data[run]
    for a,b in pairspec:
        for (c,w),x in d.items():
            if c.endswith('__'+a):
                other=d[(c.rsplit('__',1)[0]+'__'+b,w)]
                assert all(np.array_equal(x[k],other[k]) for k in ('loss','kl'))
                closures.append({'run':run,'a':a,'b':b,'group':c.rsplit('__',1)[0],'window':w,'exact':True})

report={'scope':'Reanalysis of already recorded final-model outputs, not fresh inference or semantic-task identification.',
        'sampling':'Every summary is an equal average of per-window statistics. Per-token thresholds are descriptive; tokens are not treated as independent experimental units.',
        'commit':'0bf1747a8d1b4e6a0adfa3aab72005f7b13df096',
        'array_reduction_checks':validation,'exact_output_closures':len(closures),
        'memory_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'new_primary_claims':'None. This is exploratory localization of recorded native-dependence and lesion effects.'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

# A compact parent-facing selection.
for s in summaries:
    if ((s['run']=='confirmation_run' and s['group'].endswith('_natural') and s['trial'] in ('mean_KV','remove_source','deleted'))
        or (s['run']=='later_uses_run' and s['group'].endswith('_natural') and s['trial'] in ('row_later','column_later','column_source'))
        or (s['run']=='newline_broadcast_run' and s['trial'] in ('row','clean_block_p_all','cleanK_zero_p_values'))):
        print(s['run'],s['group'],s['trial'],{k:round(s[k],5) for k in ('ppl_ratio','agreement','kl','fraction_harmed_gt_1_nat','fraction_harmed_gt_0.1_nat','top_10pct_targets_share_positive_loss')})
