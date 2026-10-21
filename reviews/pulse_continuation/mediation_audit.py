"""CPU-only mediation audit: scoring scope, exact cuts, calibration, summaries."""
import ast,csv,hashlib,json,math,os,resource
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'mediation_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'mediation.py').read_bytes()).hexdigest()
assert meta['tokens_sha256']==hashlib.sha256((ROOT/'gate_run/tokens.pt').read_bytes()).hexdigest()
raw=torch.load(OUT/'per_token.pt',weights_only=True,map_location='cpu')
raw={k:{r['window']:r for r in v} for k,v in raw.items()}
rows=[json.loads(s) for s in (OUT/'windows.jsonl').read_text().splitlines()]
groups=defaultdict(dict)
for row in rows:
    key=row['group']+'__'+row['trial'];assert row['window'] not in groups[key];groups[key][row['window']]=row
assert len(rows)==1920 and len(groups)==240 and meta['forward_passes']==1936
assert set(raw)==set(groups)
for key,items in raw.items():
    assert len(items)==8;clean=raw[key.split('__')[0]+'__clean']
    for i,z in items.items():
        row=groups[key][i];assert row['score_start']==1 and row['scored_tokens']==510
        assert z['token_sha256']==row['token_sha256']==clean[i]['token_sha256']
        assert torch.isfinite(z['loss']).all() and torch.isfinite(z['kl']).all()
        assert abs(float(z['loss'].double().mean())-row['nll'])<1e-12
        assert abs(float(z['kl'].double().mean())-row['kl_clean_to_condition'])<1e-12
        assert abs(float(clean[i]['loss'].double().mean())-row['baseline_nll'])<1e-12
        assert abs(float(z['pred'].eq(clean[i]['pred']).double().mean())-row['agreement'])<1e-12
        assert len(z['loss'])==510
gate=torch.load(ROOT/'gate_run/per_token.pt',weights_only=True,map_location='cpu')
for corpus in ['wiki','hume','plato']:
    group='natural_'+corpus
    for direction,state in [('rescue','clean'),('damage','deleted')]:
        for i in range(8):
            a,b=raw[group+'__all_2_15_'+direction][i],raw[group+'__'+state][i]
            for key in ['loss','pred','kl']:assert torch.equal(a[key],b[key])
    for state in ['clean','deleted']:
        for z in gate[group+'__'+state]:
            actual=raw[group+'__'+state][z['window']]
            for key in ['loss','pred','kl']:assert torch.equal(actual[key],z[key][1:])
cal=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu')
gatecal=torch.load(ROOT/'gate_run/calibration.pt',weights_only=True,map_location='cpu')
for corpus in ['wiki','hume']:
    assert set(cal[corpus])==set(range(16))
    for layer in range(16):
        for kind in ['k','v']:
            x=cal[corpus][layer][kind];assert x.shape==(1,1,2048) and torch.isfinite(x).all()
            if layer==2:assert torch.equal(x,gatecal[corpus][kind])

def ci(vals):
    x=np.asarray(vals);ix=np.random.default_rng(20261021).integers(0,len(x),(2000,len(x)))
    a,b=np.quantile(x[ix].mean(1),[.025,.975]);return {'mean':float(x.mean()),'low':float(a),'high':float(b)}
def effect(group,trial,reference):
    a,b=groups[group+'__'+trial],groups[group+'__'+reference]
    return ci([a[i]['nll']-b[i]['nll'] for i in sorted(a)])
summaries=list(csv.DictReader((OUT/'summary.csv').open()))
assert len(summaries)==240
for row in summaries:
    a=groups[row['group']+'__'+row['trial']];rs=[a[i] for i in sorted(a)]
    for vals,fields in [([r['nll']-r['baseline_nll'] for r in rs],['delta_nll','delta_low','delta_high']),([r['agreement'] for r in rs],['agreement','agreement_low','agreement_high']),([r['kl_clean_to_condition'] for r in rs],['kl','kl_low','kl_high'])]:
        stats=ci(vals)
        for field,k in zip(fields,['mean','low','high']):assert abs(float(row[field])-stats[k])<1e-12
contrasts=list(csv.DictReader((OUT/'contrasts.csv').open()))
for row in contrasts:
    stats=effect(row['group'],row['trial'],row['reference'])
    for field,k in [('delta_nll','mean'),('low','low'),('high','high')]:assert abs(float(row[field])-stats[k])<1e-12
result={'rows':len(rows),'conditions':len(groups),'forward_passes':meta['forward_passes'],'scored_predictors':'1..510 inclusive; target token indices2..511','all_raw_metrics_match':True,'exact_all_layer_cuts_loss_prediction_kl_both_directions':True,'clean_deleted_exactly_gate_panel_with_first_loss_removed':True,'calibration_layer2_matches_gate_calibration':True,'summary_rows_replayed':len(summaries),'contrast_rows_replayed':len(contrasts),'cuts':{},'constants':{},'later_heads':{}}
cuts=['L2_H15','L2_other_heads','L2_all','L2_keys','L2_values','early_2_3','early_2_5','early_2_8','early_2_11','all_2_15','late_3_15','late_6_15','late_12_15']
for corpus in ['wiki','hume','plato']:
    group='natural_'+corpus
    result['cuts'][corpus]={'deletion_vs_clean':effect(group,'deleted','clean')}
    for direction,state in [('damage','clean'),('rescue','deleted')]:
        for name in cuts:result['cuts'][corpus][name+'_'+direction]={'vs_state':effect(group,name+'_'+direction,state),'vs_clean':effect(group,name+'_'+direction,'clean')}
    for key,values in groups.items():
        if not key.startswith(group+'__'):continue
        trial=key.split('__')[1];rs=list(values.values())
        dest=result['constants'] if trial.startswith('constant_') else result['later_heads'] if trial.startswith(('L12H','L15H')) else None
        if dest is not None:
            state=trial.split('_')[-1]
            if state in ['rescue','damage']:state='deleted' if state=='rescue' else 'clean'
            dest[corpus+'_'+trial]={'vs_state':effect(group,trial,state),'vs_clean':effect(group,trial,'clean'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rs])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rs]))}
for corpus in ['wiki','hume','plato']:
    for i in range(8):
        paths=torch.load(OUT/f'paths_{corpus}_{i}.pt',weights_only=True,map_location='cpu')
        for layer in [0,1]:
            for kind in ['k','v']:assert torch.equal(paths['clean'][layer][kind],paths['deleted'][layer][kind])
result['pre_source_layers_kv0_exactly_equal']=True

# Exercise hook localization independently on synthetic CPU tensors.
source=(ROOT/'mediation.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Paths')
env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<Paths AST>','exec'),env)
hook=env['Paths'].__new__(env['Paths']);hook.capture=False
x=torch.randn(1,4,2048).half();donor=torch.randn(1,1,2048).half()
hook.patch={'type':'cut','layers':[2],'heads':[15],'kinds':'kv','donor':{2:{'k':donor,'v':donor}}}
y=x.clone();y[:,:1,1920:]=donor[:,:,1920:]
assert torch.equal(hook.proj(2,'k',x),y) and torch.equal(hook.proj(2,'q',x),x) and torch.equal(hook.proj(3,'k',x),x)
d=torch.randn(1,4,128).half();hook.patch={'type':'candidate','layer':12,'head':5,'kinds':'q','donor':{'candidate':{12:{5:{'q':d}}}}}
y=x.clone();y[:,:,640:768]=d
assert torch.equal(hook.proj(12,'q',x),y) and torch.equal(hook.proj(12,'v',x),x)
result['synthetic_hook_localization_passed']=True
result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['cuts','constants','later_heads']},indent=2))
