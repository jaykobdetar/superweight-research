"""CPU development-selection, matched-control, heldout and channel-cut audit."""
import ast,collections,hashlib,json,os,resource
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
torch.set_num_threads(1)
from audit_common import load_audit,effect,interval
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'residual_paths_run'
meta=json.loads((OUT/'manifest.json').read_text());assert 'completed_utc' in meta
assert meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
assert meta['script_sha256']==hashlib.sha256((ROOT/'residual_paths.py').read_bytes()).hexdigest()
assert meta['template_sha256']==hashlib.sha256((ROOT/'template_run/templates.pt').read_bytes()).hexdigest()
raw,groups,result=load_audit(OUT);assert result['rows']==3000 and result['groups']==375 and meta['forward_passes']==3008
selection=json.loads((OUT/'selection.json').read_text());assert selection['selected_before_heldout']
ranking=selection['ranking'];assert len(ranking)==224
improvements={}
for entry in ranking:
    l,h=entry['layer'],entry['head'];vals=[]
    for i in range(8):
        a=raw['development_wiki__axis'][i]['loss'];z=raw[f'development_wiki__L{l}H{h}_rescue'][i]['loss']
        vals.append(float((a-z).double().mean()))
    assert vals==entry['window_improvements'] and float(np.mean(vals))==entry['mean_dev_improvement']
    improvements[l,h]=vals
order=sorted(improvements,key=lambda e:(-np.mean(improvements[e]),e))
assert order==[(r['layer'],r['head']) for r in ranking]
rng=np.random.default_rng(20261021);controls={}
for k in [1,2,4,8,16,32]:
    selected=order[:k];counts=collections.Counter(l for l,h in selected);control=[]
    for l,n in sorted(counts.items()):
        unused=[h for h in range(16) if (l,h) not in selected];pool=unused if len(unused)>=n else list(range(16))
        control.extend((l,int(h)) for h in rng.choice(pool,n,replace=False))
    saved=selection['sets'][str(k)]
    assert list(map(tuple,saved['selected']))==selected and list(map(tuple,saved['control']))==control
    assert len(set(control))==k and collections.Counter(l for l,h in control)==counts
    overlap=len(set(selected)&set(control));assert overlap==saved['control_overlap']==0
    controls[str(k)]={'layer_counts':dict(counts),'overlap':overlap}
sets=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');data=json.loads((OUT/'data_manifest.json').read_text())
old=torch.load(ROOT.parent/'pulse_causal_tranche/run/tokens.pt',weights_only=True,map_location='cpu')
fm=json.loads((ROOT/'factorial_run/data_manifest.json').read_text());gm=json.loads((ROOT/'gate_run/data_manifest.json').read_text());tm=json.loads((ROOT/'template_run/data_manifest.json').read_text())
for corpus in ['wiki','hume']:
    exclusion=[(s,s+1024) for split in old['starts'][corpus].values() for s in split]
    exclusion += [(s-512,s+512) for s in fm[corpus]['starts']]+[(s,s+1024) for s in gm[corpus]['evaluation_starts']]+[(s,s+512) for s in tm[corpus]['starts']]
    assert set(exclusion)==set(map(tuple,data[corpus]['excluded_spans']))
    for x,s in zip(sets[corpus]['evaluation'],data[corpus]['evaluation_starts']):
        assert all(s+512<=a or s>=b for a,b in exclusion)
        assert torch.equal(x,old['corpora'][corpus][s:s+512])
for x,s in zip(sets['wiki']['development'],old['starts']['wiki']['development']):assert torch.equal(x,old['corpora']['wiki'][s:s+512])
for corpus in ['wiki','hume','plato','python']:
    starts=data[corpus]['evaluation_starts'];assert len(starts)==8
    assert all(b-a>=512 for a,b in zip(starts,starts[1:]))
    if corpus=='python':exclusion=[(s,s+512) for s in data[corpus]['calibration_starts']]
    else:exclusion=data[corpus]['excluded_spans']
    for i,(x,s) in enumerate(zip(sets[corpus]['evaluation'],starts)):
        assert all(s+512<=a or s>=b for a,b in exclusion) and len(x)==512
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==raw['heldout_'+corpus+'__clean'][i]['token_sha256']
for source in data['python']['sources']:assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
assert data['plato']['source_sha256']==hashlib.sha256(Path(gm['plato']['path']).read_bytes()).hexdigest()
pycal=torch.load(OUT/'python_calibration.pt',weights_only=True,map_location='cpu')
assert set(pycal)==set(range(2,16))
for c in pycal.values():assert c['norm'].shape==(1,1,2048) and torch.isfinite(c['norm']).all()
for key,items in groups.items():assert all(r['score_start']==1 and r['scored_tokens']==510 for r in items.values())
result.update(development_ranking_exactly_replayed=True,seeded_layer_matched_controls_replayed=controls,new_evaluation_exclusions_passed=True,python_source_hashes_and_calibration_separation_passed=True,selected_top8=selection['sets']['8']['selected'],anchors={},curves={},selected_vs_control={},top8_components={})
for corpus in ['wiki','hume','plato','python']:
    group='heldout_'+corpus
    for direction,state in [('rescue','full_mean'),('damage','axis')]:
        for i in range(8):
            for k in ['loss','pred','kl']:assert torch.equal(raw[group+'__all224_'+direction][i][k],raw[group+'__'+state][i][k])
    for trial in ['axis','full_mean','python_full_mean','residual_rank2','ideal','deleted']:
        rows=list(groups[group+'__'+trial].values())
        result['anchors'][corpus+'_'+trial]={'vs_clean':effect(groups,group,trial,'clean'),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rows])),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rows]))}
    for direction,reference in [('rescue','axis'),('damage','full_mean')]:
        for k in [1,2,4,8,16,32]:
            for label in ['selected','control']:
                trial=f'top{k}_{label}_{direction}';rows=list(groups[group+'__'+trial].values())
                result['curves'][corpus+'_'+trial]={'vs_recipient':effect(groups,group,trial,reference),'vs_clean':effect(groups,group,trial,'clean'),'mean_clean_agreement':float(np.mean([r['agreement'] for r in rows])),'mean_clean_kl':float(np.mean([r['kl_clean_to_condition'] for r in rows]))}
            result['selected_vs_control'][f'{corpus}_top{k}_{direction}']=effect(groups,group,f'top{k}_selected_{direction}',f'top{k}_control_{direction}')
        for kinds in ['k','v']:
            trial=f'top8_{kinds}_{direction}'
            result['top8_components'][corpus+'_'+trial]={'vs_recipient':effect(groups,group,trial,reference),'vs_clean':effect(groups,group,trial,'clean')}

# Edge hook tests without importing model packages.
class StubPaths:
    def proj(self,l,k,o):return o
source=(ROOT/'residual_paths.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Edges')
env={'torch':torch,'Paths':StubPaths};exec(compile(ast.Module(body=[node],type_ignores=[]),'<Edges AST>','exec'),env)
h=env['Edges'].__new__(env['Edges']);x=torch.randn(1,4,2048).half();d=torch.randn(1,1,2048).half()
h.patch={'type':'edges','edges':[(2,2),(2,9),(3,7)],'kinds':'kv','donor':{2:{'k':d,'v':d},3:{'k':d,'v':d}}}
y=x.clone();y[:,0,256:384]=d[:,0,256:384];y[:,0,1152:1280]=d[:,0,1152:1280]
assert torch.equal(h.proj(2,'k',x),y) and torch.equal(h.proj(2,'q',x),x) and torch.equal(h.proj(4,'k',x),x)
result.update(all224_exact_loss_prediction_kl_both_directions=True,synthetic_edge_localization_passed=True,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['anchors','curves','selected_vs_control','top8_components']},indent=2))
