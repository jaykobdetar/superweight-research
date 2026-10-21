"""CPU result auditing helpers; never imports or loads a model."""
import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

def interval(vals):
    x=np.asarray(vals);ix=np.random.default_rng(20261021).integers(0,len(x),(2000,len(x)))
    a,b=np.quantile(x[ix].mean(1),[.025,.975]);return {'mean':float(x.mean()),'low':float(a),'high':float(b)}

def load_audit(out,expected_windows=8,require_tables=True):
    raw=torch.load(out/'per_token.pt',weights_only=True,map_location='cpu')
    raw={k:{r['window']:r for r in v} for k,v in raw.items()}
    rows=[json.loads(s) for s in (out/'windows.jsonl').read_text().splitlines()];groups=defaultdict(dict)
    for row in rows:
        key=row['group']+'__'+row['trial'];assert row['window'] not in groups[key];groups[key][row['window']]=row
    assert set(raw)==set(groups)
    for key,items in raw.items():
        assert len(items)==expected_windows;clean=raw[key.split('__')[0]+'__clean']
        for i,z in items.items():
            row=groups[key][i]
            assert z['token_sha256']==row['token_sha256']==clean[i]['token_sha256']
            assert torch.isfinite(z['loss']).all() and torch.isfinite(z['kl']).all()
            assert abs(float(z['loss'].double().mean())-row['nll'])<1e-12
            assert abs(float(z['kl'].double().mean())-row['kl_clean_to_condition'])<1e-12
            assert abs(float(clean[i]['loss'].double().mean())-row['baseline_nll'])<1e-12
            assert abs(float(z['pred'].eq(clean[i]['pred']).double().mean())-row['agreement'])<1e-12
            assert len(z['loss'])==row['scored_tokens']==row['input_tokens']-1-row['score_start']
    summaries=list(csv.DictReader((out/'summary.csv').open())) if require_tables or (out/'summary.csv').exists() else []
    if summaries:assert len(summaries)==len(groups)
    for row in summaries:
        a=groups[row['group']+'__'+row['trial']];rs=[a[i] for i in sorted(a)]
        for vals,fields in [([r['nll']-r['baseline_nll'] for r in rs],['delta_nll','delta_low','delta_high']),([r['agreement'] for r in rs],['agreement','agreement_low','agreement_high']),([r['kl_clean_to_condition'] for r in rs],['kl','kl_low','kl_high'])]:
            stats=interval(vals)
            for field,k in zip(fields,['mean','low','high']):assert abs(float(row[field])-stats[k])<1e-12
    contrasts=list(csv.DictReader((out/'contrasts.csv').open())) if require_tables or (out/'contrasts.csv').exists() else []
    for row in contrasts:
        stats=effect(groups,row['group'],row['trial'],row['reference'])
        for field,k in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[field])-stats[k])<1e-12
    return raw,groups,{'rows':len(rows),'groups':len(groups),'summary_rows_replayed':len(summaries),'contrast_rows_replayed':len(contrasts),'all_raw_metrics_match':True}

def effect(groups,group,trial,reference):
    a,b=groups[group+'__'+trial],groups[group+'__'+reference]
    assert set(a)==set(b)
    assert all(a[i]['token_sha256']==b[i]['token_sha256'] for i in a)
    return interval([a[i]['nll']-b[i]['nll'] for i in sorted(a)])

def compare(raw,groups,group,trial,reference):
    a,b=raw[group+'__'+trial],raw[group+'__'+reference]
    return {'nll':effect(groups,group,trial,reference),'max_token_loss_difference':max(float((a[i]['loss']-b[i]['loss']).abs().max()) for i in a),'prediction_mismatches':sum(int(a[i]['pred'].ne(b[i]['pred']).sum()) for i in a),'tokens':sum(len(a[i]['pred']) for i in a)}
