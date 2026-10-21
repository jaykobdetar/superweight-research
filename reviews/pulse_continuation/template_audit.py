"""CPU-only static-template geometry, unused-span, and result audit."""
import hashlib,json,math,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'template_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'template.py').read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==672 and result['groups']==84 and meta['forward_passes']==672
data=json.loads((OUT/'data_manifest.json').read_text());tokens=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu')
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
for corpus in ['wiki','hume','plato']:
    starts=data[corpus]['starts'];excluded=data[corpus]['excluded_spans']
    assert len(starts)==8 and len(set(starts))==8
    for i,s in enumerate(starts):
        assert all(s+512<=a or s>=b for a,b in excluded)
        assert all(i==j or s+512<=t or s>=t+512 for j,t in enumerate(starts))
        x=tokens[corpus][i];assert len(x)==512
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==raw['natural_'+corpus+'__clean'][i]['token_sha256']
        if corpus in ['wiki','hume']:assert torch.equal(x,old['corpora'][corpus][s:s+512])
# Independently verify that exclusion lists include every declared prior span.
fm=json.loads((ROOT/'factorial_run/data_manifest.json').read_text());gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text())
for corpus in ['wiki','hume']:
    expected=[(s,s+1024) for split in old['starts'][corpus].values() for s in split]
    expected += [(s-512,s+512) for s in fm[corpus]['starts']]+[(s,s+1024) for s in gm[corpus]['evaluation_starts']]
    assert set(expected)==set(map(tuple,data[corpus]['excluded_spans']))
assert set(map(tuple,data['plato']['excluded_spans']))=={(s,s+1024) for s in gm['plato']['evaluation_starts']}
assert data['plato']['source_sha256']==hashlib.sha256(Path(gm['plato']['path']).read_bytes()).hexdigest()
fit=json.loads((OUT/'template_fit.json').read_text());assert fit['calibration_sha256']==hashlib.sha256((ROOT/'normalization_run/calibration.pt').read_bytes()).hexdigest()
bundle=torch.load(OUT/'templates.pt',weights_only=True,map_location='cpu');cal=torch.load(ROOT/'normalization_run/calibration.pt',weights_only=True,map_location='cpu')
mean=torch.stack([cal['wiki'][l]['norm'][0,0].float() for l in range(2,16)])
ideal=torch.ones(2048)/math.sqrt(2047);ideal[1764]=-math.sqrt(2047)
base=mean[:,1764,None]/ideal[1764]*ideal[None];res=mean-base
assert torch.equal(mean,bundle['mean_vectors']) and torch.equal(ideal,bundle['analytic_axis'])
assert torch.equal(base,bundle['coordinate_matched_axis']) and torch.equal(res,bundle['residual'])
assert abs(float(ideal.double().sum()))<1e-5 and abs(float(ideal.double().square().mean())-1)<1e-6
u,s,v=bundle['uncentered_svd'];uc,sc,vc=bundle['centered_across_layers_svd']
assert res.shape==(14,2048) and u.shape==(14,14) and v.shape==(14,2048)
assert torch.allclose((u*s)@v,res,rtol=1e-4,atol=1e-5)
assert torch.allclose((uc*sc)@vc,res-res.mean(0),rtol=1e-4,atol=1e-5)
assert torch.allclose(u.T@u,torch.eye(14),atol=2e-6) and torch.allclose(v@v.T,torch.eye(14),atol=2e-6)
energy=[float(s[:r].square().sum()/s.square().sum()) for r in [1,2,4,8]]
assert np.allclose(energy,fit['uncentered_energy'],rtol=0,atol=1e-12)
expected={'ideal':ideal.expand(14,-1).clone(),'coordinate_matched_axis':base,'residual_only':res,'wiki_full_mean':mean,'hume_full_mean':torch.stack([cal['hume'][l]['norm'][0,0].float() for l in range(2,16)]),'shared_residual':base+res.mean(0)}
for rank in [1,2,4,8]:expected[f'residual_rank{rank}']=base+(u[:,:rank]*s[:rank])@v[:rank]
random_stats={}
for seed in [20261021,20261022]:
    gen=torch.Generator().manual_seed(seed);z=torch.randn((14,2048),generator=gen);z[:,1764]=0
    mask=torch.ones(2048,dtype=torch.bool);mask[1764]=False;z[:,mask]-=z[:,mask].mean(1,keepdim=True)
    z*=res.norm(dim=1,keepdim=True)/z.norm(dim=1,keepdim=True)
    assert torch.equal(z[:,1764],torch.zeros(14)) and torch.allclose(z[:,mask].mean(1),torch.zeros(14),atol=1e-7)
    assert torch.allclose(z.norm(dim=1),res.norm(dim=1),rtol=1e-6)
    expected[f'random_residual_{seed}']=base+z
    random_stats[str(seed)]={'max_layer_norm_relative_error':float(((z.norm(dim=1)-res.norm(dim=1))/res.norm(dim=1)).abs().max()),'matching_target':'full learned residual before FP16 cast'}
wrong=torch.ones(2048)/math.sqrt(2047);wrong[1763]=-math.sqrt(2047);expected['wrong_axis']=wrong.expand(14,-1).clone()
assert set(expected)==set(bundle['templates'])==set(fit['templates'])
for name,x in expected.items():
    for l in range(2,16):assert torch.equal(bundle['templates'][name][l]['norm'],x[l-2][None,None].half()),(name,l)
result.update(unused_token_span_checks_passed=True,template_geometry_and_serialized_values_replayed=True,static_across_layer_svd_verified=True,svd_matrix_shape=[14,2048],uncentered_residual_energy=energy,random_controls=random_stats,results={},random_vs_matched_axis={})
for corpus in ['wiki','hume','plato']:
    group='natural_'+corpus
    for name in expected:
        a,b=raw[group+'__'+name+'_clean'],raw[group+'__'+name+'_deleted']
        for i in range(8):
            for k in ['loss','pred','kl']:assert torch.equal(a[i][k],b[i][k])
        rows=list(groups[group+'__'+name+'_clean'].values())
        result['results'][corpus+'_'+name]={'vs_clean':effect(groups,group,name+'_clean','clean'),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rows])),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rows]))}
    for seed in [20261021,20261022]:result['random_vs_matched_axis'][corpus+'_'+str(seed)]=effect(groups,group,f'random_residual_{seed}_clean','coordinate_matched_axis_clean')
    for key,rows in groups.items():
        if key.startswith(group+'__'):assert all(r['score_start']==1 and r['scored_tokens']==510 for r in rows.values())
result['clean_deleted_template_identity_loss_prediction_kl']=True;result['uniform_predictors_1_to_510']=True
result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['results','random_vs_matched_axis']},indent=2))
