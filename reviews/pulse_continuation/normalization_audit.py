"""CPU cross-layer normalization audit; no model imports or loading."""
import ast,hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'normalization_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'normalization.py').read_bytes()).hexdigest()
assert meta['tokens_sha256']==hashlib.sha256((ROOT/'gate_run/tokens.pt').read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==816 and result['groups']==102 and meta['forward_passes']==832
prior=torch.load(ROOT/'mediation_run/per_token.pt',weights_only=True,map_location='cpu');prior={k:{r['window']:r for r in v} for k,v in prior.items()}
result['oracle']={};result['constants']={};norm_error=0
for corpus in ['wiki','hume','plato']:
    group='natural_'+corpus
    for state in ['clean','deleted']:
        for i in range(8):
            for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__'+state][i][k],prior[group+'__'+state][i][k])
    for direction,state in [('rescue','clean'),('damage','deleted')]:
        for i in range(8):
            for k in ['loss','pred','kl']:
                assert torch.equal(raw[group+'__all_2_15_full_'+direction][i][k],raw[group+'__'+state][i][k])
                assert torch.equal(raw[group+'__L2_full_'+direction][i][k],prior[group+'__L2_all_'+direction][i][k])
        for layer in ['L2','all_2_15']:
            for mode in ['coordinate','complement','scale','full']:
                trial=f'{layer}_{mode}_{direction}';rs=list(groups[group+'__'+trial].values())
                result['oracle'][corpus+'_'+trial]={'vs_clean':effect(groups,group,trial,'clean'),'vs_deleted':effect(groups,group,trial,'deleted'),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rs])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rs]))}
    for cal in ['wiki','hume']:
        for mode in ['coordinate','complement','scale','full']:
            for state in ['clean','deleted']:
                trial=f'constant_{cal}_{mode}_{state}';rs=list(groups[group+'__'+trial].values())
                result['constants'][corpus+'_'+trial]={'vs_clean':effect(groups,group,trial,'clean'),'vs_state':effect(groups,group,trial,state),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rs])),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rs]))}
    for i in range(8):
        snap=torch.load(OUT/f'norms_{corpus}_{i}.pt',weights_only=True,map_location='cpu')
        for layer in range(16):
            if layer<2:
                for k in ['input','norm','sigma']:assert torch.equal(snap['clean'][layer][k],snap['deleted'][layer][k])
            for state in ['clean','deleted']:
                c=snap[state][layer];sigma=c['input'][:,0].var(-1,unbiased=False).add(1e-5).sqrt()
                assert torch.equal(sigma,c['sigma'])
                n=torch.nn.functional.layer_norm(c['input'],(2048,),eps=1e-5).to(c['norm'].dtype)
                norm_error=max(norm_error,float((n.float()-c['norm'].float()).abs().max()))
                assert torch.allclose(n,c['norm'],rtol=.002,atol=.001)
    for row in [r for k,v in groups.items() if k.startswith(group+'__') for r in v.values()]:assert row['score_start']==1 and row['scored_tokens']==510
cal=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu')
assert set(cal)=={'wiki','hume'}
for co in cal.values():
    assert set(co)==set(range(16))
    for c in co.values():assert c['norm'].shape==(1,1,2048) and c['sigma'].shape==(1,) and c['sigma'][0]>0 and torch.isfinite(c['norm']).all()

source=(ROOT/'normalization.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Norms')
env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<Norms AST>','exec'),env);h=env['Norms'].__new__(env['Norms']);h.capture=False
x=torch.randn(1,4,2048).half();o=torch.nn.functional.layer_norm(x.float(),(2048,),eps=1e-5).half();don=torch.randn(1,1,2048).half();sigma=torch.tensor([2.])
for mode in ['coordinate','complement','scale','full']:
    h.patch={'layers':[2],'mode':mode,'donor':{2:{'norm':don,'sigma':sigma}}};a=h.hook(2,(x,),o)
    assert torch.equal(a[:,1:],o[:,1:]) and torch.equal(h.hook(3,(x,),o),o)
    expected=o.clone()
    if mode=='coordinate':expected[:,0,1764]=don[:,0,1764]
    if mode=='complement':expected[:,0]=don[:,0];expected[:,0,1764]=o[:,0,1764]
    if mode=='full':expected[:,0]=don[:,0]
    if mode=='scale':expected[:,0]=((x[:,0].float()-x[:,0].float().mean(-1,keepdim=True))/sigma[:,None]).half()
    assert torch.equal(a,expected)
result.update(exact_all_full_cut_both_directions=True,L2_full_matches_native_all_head_KV0_cut=True,baseline_matches_mediation=True,uniform_predictors_1_to_510=True,sigma_recomputation_passed=True,synthetic_hook_localization_passed=True,max_cpu_gpu_norm_absolute_difference=norm_error,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['oracle','constants']},indent=2))
