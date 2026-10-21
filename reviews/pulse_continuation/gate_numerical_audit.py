"""CPU-only audit of gate_run, including native-reference approximation errors."""
import hashlib,json,math,os,resource
from pathlib import Path
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'gate_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'gate.py').read_bytes()).hexdigest()
raw=torch.load(OUT/'per_token.pt',weights_only=True,map_location='cpu')
rows=[json.loads(s) for s in (OUT/'windows.jsonl').read_text().splitlines()]
groups=defaultdict(dict)
for row in rows:
    key=row['group']+'__'+row['trial'];assert row['window'] not in groups[key];groups[key][row['window']]=row
assert len(rows)==1344 and len(groups)==168 and meta['forward_passes']==1360
assert set(raw)==set(groups)
raw={k:{r['window']:r for r in v} for k,v in raw.items()}
for key,items in raw.items():
    assert len(items)==8;clean=raw[key.split('__')[0]+'__clean']
    for i,z in items.items():
        row=groups[key][i]
        assert z['token_sha256']==row['token_sha256']==clean[i]['token_sha256']
        assert torch.isfinite(z['loss']).all() and torch.isfinite(z['kl']).all()
        assert abs(float(z['loss'].double().mean())-row['nll'])<1e-12
        assert abs(float(z['kl'].double().mean())-row['kl_clean_to_condition'])<1e-12
        assert abs(float(clean[i]['loss'].double().mean())-row['baseline_nll'])<1e-12
        assert abs(float(z['pred'].eq(clean[i]['pred']).double().mean())-row['agreement'])<1e-12
        assert len(z['loss'])==511
tokens=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu')
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
data=json.loads((OUT/'data_manifest.json').read_text())
prior=json.loads((ROOT/'factorial_run/data_manifest.json').read_text())
for corpus in ['wiki','hume']:
    excluded=[(s,s+1024) for split in old['starts'][corpus].values() for s in split]
    excluded += [(s-512,s+512) for s in prior[corpus]['starts']]
    for x,s in zip(tokens[corpus]['calibration'],data[corpus]['calibration_starts']):
        assert torch.equal(x,old['corpora'][corpus][s:s+512])
    for x,s in zip(tokens[corpus]['evaluation'],data[corpus]['evaluation_starts']):
        assert all(s+512<=a or s>=b for a,b in excluded)
        assert torch.equal(x,old['corpora'][corpus][s:s+512])
for corpus in ['wiki','hume','plato']:
    for i,x in enumerate(tokens[corpus]['evaluation']):
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==raw['natural_'+corpus+'__clean'][i]['token_sha256']
cal=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu')
assert set(cal)=={'wiki','hume'} and 'calibration' not in tokens['plato']
for c in cal.values():
    for k in ['k','v']:assert c[k].shape==(1,1,2048) and torch.isfinite(c[k]).all()
    for h in [0,15]:
        assert 0<=c['gate'][h]<=1 and c['gate_position'][h].shape==(512,)
        assert torch.all((c['gate_position'][h]>=0)&(c['gate_position'][h]<=1))
        assert c['gate_position'][h][0]==1
result={'rows':len(rows),'conditions':len(groups),'forward_passes':meta['forward_passes'],'raw_metrics_match':True,'calibration_and_evaluation_isolation_passed':True,'calibration_replay_limitation':'Raw calibration activations were not saved; means cannot be independently recomputed without model inference.'}
def ci(vals):
    x=np.asarray(vals);ix=np.random.default_rng(20261021).integers(0,len(x),(2000,len(x)))
    a,b=np.quantile(x[ix].mean(1),[.025,.975]);return {'mean':float(x.mean()),'low':float(a),'high':float(b)}
def contrast(group,trial,reference):
    a,b=groups[group+'__'+trial],groups[group+'__'+reference]
    return ci([a[i]['nll']-b[i]['nll'] for i in sorted(a)])
def comparison(group,trial,reference):
    a,b=raw[group+'__'+trial],raw[group+'__'+reference]
    return {'nll':contrast(group,trial,reference),'max_token_loss_difference':max(float((a[i]['loss']-b[i]['loss']).abs().max()) for i in a),'prediction_mismatches':sum(int(a[i]['pred'].ne(b[i]['pred']).sum()) for i in a),'tokens':sum(len(a[i]['pred']) for i in a)}
result['reconstruction_controls']={};result['oracle_vs_native']={};result['replacements']={};result['gate_basics']={}
for corpus in ['wiki','hume','plato']:
    group='natural_'+corpus
    for h in [0,15]:
        for state in ['clean','deleted']:
            a,b=raw[group+f'__H{h}_identity_{state}'],raw[group+'__'+state]
            for i in range(8):
                for k in ['loss','pred','kl']:assert torch.equal(a[i][k],b[i][k])
            result['reconstruction_controls'][f'{corpus}_H{h}_{state}']=comparison(group,f'H{h}_reconstruction_{state}',state)
    for state,direction in [('clean','damage'),('deleted','rescue')]:
        for operation,native in [('donor_gate','k'),('donor_value','v'),('donor_both','kv')]:
            result['oracle_vs_native'][f'{corpus}_{operation}_{state}']=comparison(group,f'H15_{operation}_{state}',f'H15_native_{native}_{direction}')
        for h in [0,15]:
            for operation in ['zero_value','remove_source','constant_gate_zero_value','constant_kv']:
                trial=f'H{h}_{operation}_{state}';vals=list(groups[group+'__'+trial].values())
                result['replacements'][f'{corpus}_{trial}']={'vs_state':contrast(group,trial,state),'vs_clean':contrast(group,trial,'clean'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in vals])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in vals]))}
    for operation in ['constant_value','constant_gate','position_gate_zero_value','hume_gate_zero_value','hume_position_gate_zero_value','constant_key_zero_value','hume_constant_kv','zero_head']:
        trial=f'H15_{operation}_clean';vals=list(groups[group+'__'+trial].values())
        result['replacements'][f'{corpus}_{trial}']={'vs_clean':contrast(group,trial,'clean'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in vals])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in vals]))}
    metrics=[]
    for i in range(8):
        d=torch.load(OUT/f'decomposition_{corpus}_{i}.pt',weights_only=True,map_location='cpu')
        for state in ['clean','deleted']:
            for h in [0,15]:
                p=d[state][h]
                assert p['a'][0]==1 and torch.equal(p['u'][0],torch.zeros(128))
                assert torch.equal(p['y'],p['a'][:,None]*p['v']+(1-p['a'][:,None])*p['u'])
                if state=='deleted':assert torch.equal(p['u'],d['clean'][h]['u'])
                metrics.append({'window':i,'state':state,'head':h,'mean_a':float(p['a'][1:].mean()),'v_norm':float(p['v'].norm()),'u_rms':float(p['u'][1:].square().sum(-1).mean().sqrt())})
    result['gate_basics'][corpus]=metrics
result['identity_loss_prediction_kl_exact']=True
result['conditional_mixture_unchanged_clean_to_deleted']=True
result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'conditions':len(groups),'identity_exact':True,'max_abs_reconstruction_mean_nll':max(abs(x['nll']['mean']) for x in result['reconstruction_controls'].values()),'max_abs_oracle_vs_native_mean_nll':max(abs(x['nll']['mean']) for x in result['oracle_vs_native'].values()),'max_oracle_native_token_loss_difference':max(x['max_token_loss_difference'] for x in result['oracle_vs_native'].values()),'peak_rss_kib':result['peak_rss_kib']},indent=2))
