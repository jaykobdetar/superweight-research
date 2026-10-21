"""CPU-only localization audit, paired contrasts and descriptive nonadditivity.

This analyzer checks saved records and implementations; it is not an independent
scientific review. In particular source/hash closures refer to recorded states.
"""
import argparse
import json
import math
import resource
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analyze_transfer import aggregate
from gpt2_gate_subspace import ERROR_TOLERANCE, LAYERS, maps_digest, tensor_hash
from gpt2_gate_localization import (CONDITIONS, SNAPSHOT_MODES, FAMILIES, SETS,
    CORPUS_COUNTS, NORM_TOLERANCE, INPUT_HASHES, mode_spec, localization_map)
from run_transfer import dump, sha
from gpt2_gate_subspace_v2 import fingerprint

ACTIVE = tuple(t for t in CONDITIONS if mode_spec(t)[0] is not None)
CONTRASTS = tuple(pair for layer_set in SETS for pair in (
    ('remove_pc1_'+layer_set+'_minus_P','remove_pc1_'+layer_set,'P'),
    ('remove_pc1_'+layer_set+'_minus_orthogonal_control','remove_pc1_'+layer_set,'orthogonal_control_'+layer_set),
    ('position_'+layer_set+'_minus_P','position_'+layer_set,'P')))


def load(path):
    return torch.load(path, weights_only=True, map_location='cpu')


def paired(values):
    x = np.asarray(values, dtype=np.float64)
    assert len(x) in (3,4) and np.isfinite(x).all()
    rng = np.random.default_rng(20261021)
    bs = x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return {'n': len(x), 'mean_delta_nll': float(x.mean()),
            'ci95_delta_nll': np.quantile(bs, [.025, .975]).tolist(),
            'per_window_delta_nll': x.tolist()}


def dual_aggregate(rows):
    native = aggregate(rows)
    altered = aggregate([{**r, 'baseline_nll': r['p_baseline_nll'],
                          'kl': r['kl_p'], 'agreement': r['agreement_p']} for r in rows])
    return {**native, **{'p_'+k: v for k, v in altered.items()}}


def scalars(z, c, p, sl):
    return {'nll': float(z['loss'][sl].double().mean()),
            'baseline_nll': float(c['loss'][sl].double().mean()),
            'p_baseline_nll': float(p['loss'][sl].double().mean()),
            'kl': float(z['kl'][sl].double().mean()),
            'kl_p': float(z['kl_p'][sl].double().mean()),
            'agreement': float(z['pred'][sl].eq(c['pred'][sl]).double().mean()),
            'agreement_p': float(z['pred'][sl].eq(p['pred'][sl]).double().mean())}


def audit_suffix(p,raw,inputs,mainops,maps,directions):
    checks=json.loads((p/'suffix_checks.json').read_text());ops=json.loads((p/'suffix_operations.json').read_text())
    files=json.loads((p/'suffix_files.json').read_text());assert len(checks)==len(ops)==930 and len(files)==30
    def identity(d):return d['corpus'],d['window'],d['identity'],d['trial']
    ci={identity(d):d for d in checks};oi={identity(d):d for d in ops}
    mi={identity(d):d for d in mainops if d['length']==512};assert len(ci)==len(oi)==930
    tokens=load(p/'tokens.pt');tokenmap={(d['corpus'],d['window']):d['tokens']for d in tokens}
    bos=json.loads((p/'data_manifest.json').read_text())['literals']['start_special']['id']
    from run_transfer import th
    source=load(p/'source_states.pt');controls=gatechecks=0;max_map_error=0.;minimum_mass=1.;maxima=defaultdict(float)
    required={'logits_exact','hidden_exact','gates_exact','source_q_exact','source_kv_exact','mask_exact','contexts_exact',
              'loss_exact','pred_exact','kl_exact','kl_p_exact','agreement_p_exact'}
    for filename,h in files.items():
        path=p/'suffix'/filename;assert sha(path)==h;_,c,w,i=path.stem.split('__');w=int(w)
        records=load(path);assert set(records)==set(CONDITIONS)
        prefix=load(p/'prefix'/filename);stem=f'gate__{c}__{w}__512__{i}'
        originals=load(p/'gates'/(stem+'.pt'));x=tokenmap[c,w].clone()
        if i=='bos':x[0]=bos
        y=x.clone();y[128:]=torch.roll(y[128:],1)
        assert torch.equal(x[:128],y[:128])and not torch.equal(x[128:],y[128:])
        for trial,d in records.items():
            controls+=1;ident=(c,w,i,trial);check=ci[ident];op=oi[ident];originalop=mi[ident];z=raw[stem+'__'+trial]
            assert all(check[k]is True for k in required)
            assert check['original_token_sha256']==th(x)==z['token_sha256']
            assert check['control_token_sha256']==d['token_sha256']==th(y)
            for name in('logits_reference','logits_control'):
                f=check[name];assert f['shape']==[128,50257]and f['dtype']=='torch.float32'and len(f['sha256'])==64
            assert check['logits_reference']==check['logits_control']
            for field,expected in(('reference_context_hashes',originalop['prefix_context_hashes']),
                                   ('control_context_hashes',op['prefix_context_hashes'])):
                assert check[field]==expected
                for layer in LAYERS:
                    for kind in('current','delivered'):
                        f=expected[str(layer)][kind]
                        assert f['shape']==[1,128,768]and f['dtype']=='torch.float32'and len(f['sha256'])==64
            assert check['reference_context_hashes']==check['control_context_hashes']
            assert op['mask_records']==originalop['mask_records']
            assert torch.equal(d['hidden'],prefix[trial]['hidden'])and d['hidden'].shape==(1,128,768)
            assert d['score_index']==1 and d['scored_tokens']==126
            for k,v in d['raw'].items():
                count=127 if k=='loss'else 128
                assert v.shape==(count,)and torch.isfinite(v).all()and torch.equal(v,z[k][:count])
            assert set(d['raw'])=={'loss','pred','kl','kl_p','agreement_p'}
            st=source[stem+'__'+trial]
            assert all(torch.equal(v,d['source']['q'][l])for l,v in st['q'].items())
            assert all(torch.equal(v,d['source']['kv'][l][k])for l,q in st['kv'].items()for k,v in q.items())
            assert set(d['gates'])==(set()if trial=='C'else set(LAYERS))
            for l,g in d['gates'].items():
                b,a,t=g['b'],g['mass'],g['target_mass'];assert b.shape==a.shape==t.shape==(12,511)
                assert torch.isfinite(b).all()and torch.isfinite(a).all()and (a>0).all()and torch.isfinite(t).all()
                minimum_mass=min(minimum_mass,float(a.min()))
                tb,ta,_,_,_,ne=localization_map(trial,l,b,a,{k:v.float()for k,v in maps[l].items()}, {k:directions[l][k].float() for k in ('u','v')});assert ne<=NORM_TOLERANCE
                for expected,actual in((tb,g['target_b']),(ta,t)):
                    if expected is None:assert actual is None
                    else:
                        error=float((expected-actual).abs().max());max_map_error=max(max_map_error,error);assert error<=ERROR_TOLERANCE
                for k,v in g.items():
                    ref=originals[trial][l][k]
                    assert(v is None and ref is None)or(v is not None and torch.equal(v[:,:127],ref[:,:127]))
                if l in mode_spec(trial)[1]:assert (t>=0).all()and(t<=1).all()
            assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
            assert math.isfinite(op['max_context_error'])and op['max_context_error']<=ERROR_TOLERANCE
            for ch in op['gate_checks'].values():
                gatechecks+=1;assert ch['finite']and ch['source_context_exact']and ch['unaffected_context_exact']and ch['min_mass']>0
                assert ch['norm_match_error']<=NORM_TOLERANCE
                for k,v in ch.items():
                    if k.endswith('_error'):assert math.isfinite(v)and v<=ERROR_TOLERANCE;maxima[k]=max(maxima[k],v)
        del records,prefix,originals
    assert controls==930 and gatechecks==7200
    return {'records':controls,'gate_context_checks':gatechecks,'raw_hidden_gate_metric_source_exact':True,
        'runtime_and_hash_logit_context_closures':930,'raw_successful_logits_contexts_available':False,
        'snapshot_subset_available':True,'max_cpu_map_error':max_map_error,'min_current_mass':minimum_mass,
        'maxima':dict(maxima),'changed_suffix_scores_present':False}


def nonadditivity(inputs):
    rows=[];groups=defaultdict(list)
    for (c,i,n,w),rs in inputs.items():
        for family in FAMILIES:
            joint=rs[family+'_all']['nll']-rs['P']['nll']
            singles=[rs[family+'_L'+str(l)]['nll']-rs['P']['nll'] for l in LAYERS]
            delta=joint-sum(singles)
            rows.append({'corpus':c,'identity':i,'length':n,'window':w,'family':family,
                'joint_delta_nll':joint,'singleton_delta_nll':singles,'sum_singleton_delta_nll':sum(singles),
                'joint_minus_sum_delta_nll':delta,'interpretation':'descriptive nonadditivity; not additive mediation'})
            groups[c,i,n,family].append(delta)
    summary=[{'corpus':c,'identity':i,'length':n,'family':f,**paired(v)} for(c,i,n,f),v in groups.items()]
    return rows,summary


def audit_norms(p,inputs,maps,directions):
    files=json.loads((p/'norm_files.json').read_text());assert len(files)==30
    rows=[];count=0;max_error=0.;max_guard=0.
    fields={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm'}
    for name,h in files.items():
        path=p/'local_norms'/name;assert sha(path)==h
        _,c,w,n,i=path.stem.split('__');w,n=int(w),int(n)
        norms=load(path);gs=load(p/'gates'/name)
        assert set(norms)==set(gs)==set(inputs[c,i,n,w])
        for trial,ls in norms.items():
            assert set(ls)==(set() if trial in ('C','repeat','identity') else set(LAYERS))
            family,affected=mode_spec(trial) if trial in CONDITIONS else (None,())
            if len(affected)==1 and family in ('remove_pc1','orthogonal_control'):
                l=affected[0]
                assert torch.equal(ls[l]['s'],norms['P'][l]['s']),('singleton saved s mismatch',name,trial,l)
            for l,d in ls.items():
                count+=1;assert set(d)==fields and all(v.shape==(511,) and torch.isfinite(v).all() for v in d.values())
                state=gs[trial][l]
                _,_,_,_,expected,guard=localization_map(trial,l,state['b'],state['mass'],
                    {k:v.float() for k,v in maps[l].items()},{k:directions[l][k].float() for k in ('u','v')})
                expected['logit_delta_norm']=(state['target_b']-state['b']).T.double().norm(dim=-1)
                expected['mass_delta_norm']=(state['target_mass']-state['mass']).T.double().norm(dim=-1)
                max_guard=max(max_guard,guard);assert guard<=NORM_TOLERANCE
                for k,e in expected.items():
                    err=float((e-d[k]).abs().max());max_error=max(max_error,err);assert err<=ERROR_TOLERANCE,(name,trial,l,k,err)
                assert (d['context_delta_norm']>=0).all()
                family,affected=mode_spec(trial)
                if l not in affected:assert torch.count_nonzero(d['context_delta_norm'])==0
                if trial=='P' or l in affected:
                    for view,sl in [('full',slice(0,510)),('early',slice(0,126)),('later',slice(126,510))]:
                        rows.append({'corpus':c,'window':w,'identity':i,'length':n,'trial':trial,'layer':l,'view':view,
                            'scored_queries':len(d['s'][sl]),
                            **{k:{'mean':float(v[sl].double().mean()),'rms':float(v[sl].double().square().mean().sqrt()),
                                  'max':float(v[sl].abs().max())} for k,v in d.items()},
                            'all_query_norm_match_guard_max':guard})
        del norms,gs
    assert count==7200 and len(rows)==5040
    dump(p/'norm_summary.json',rows)
    return {'saved_layer_states':count,'score_aligned_rows':len(rows),'max_cpu_scalar_error':max_error,
        'all_query_norm_match_guard_max':max_guard,'context_norms_full_raw_context_available':False,
        'note':'Context norms are runtime observations with fixed raw snapshot subset; guard includes unscored query511.'}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path)
    p = ap.parse_args().run; torch.set_num_threads(4)
    dump(p/'audit.json',{'status':'checking','note':'A new audit has started; only a final passed state confirms completion.'})
    m = json.loads((p/'manifest.json').read_text())
    rows = [json.loads(s) for s in (p/'windows.jsonl').read_text().splitlines()]
    raw = load(p/'per_token.pt'); traces = load(p/'traces.pt')
    assert len(rows) == len(raw) == len(traces) == m['records'] == m['evaluation_forwards'] == 938
    assert m['forward_passes']==m['forward_attempts']==1868 and m['calibration_forwards']==0 and m['inherited_calibration_forwards']==8 and m['suffix_forwards']==930
    assert m['phase'] == 'completed' and m.get('status') == 'completed' and 'completed_utc' in m
    assert m['model_hash_before'] == m['model_hash_after'] and m['parameter_versions_unchanged']
    assert m['calibration_maps_unchanged'] and m['calibration_frozen_utc'] <= m['completed_utc']
    assert m['context_tolerance']==ERROR_TOLERANCE and m['study']=='gate_localization'
    assert m['physical_execution_lengths']==[512] and m['cross_length_comparison']is False
    assert m['score_views']=={'full':[2,511],'early':[2,127],'later':[128,511]}
    assert not any(k in m for k in('local_relative_tolerance','global_relative_tolerance','loss_prefix_tolerance'))
    assert list(CONDITIONS) == json.loads((p/'conditions.json').read_text()) and len(CONTRASTS) == 27
    for name, h in m['source_hashes'].items(): assert sha(Path(__file__).parent/name) == h, name
    for name, field in (('conditions.json','conditions_sha256'), ('calibration_inputs.json','calibration_inputs_sha256'),
                        ('data_manifest.json','evaluation_inputs_sha256'), ('calibration_maps.pt','calibration_maps_sha256'),
                        ('calibration_raw.pt','calibration_raw_sha256')): assert sha(p/name) == m[field]
    assert sha(p/'calibration_dependency.json')==m['calibration_dependency_sha256']
    dependency=json.loads((p/'calibration_dependency.json').read_text())
    assert dependency['inherited_calibration_forwards']==8 and dependency['new_calibration_forwards']==0
    assert dependency['map_tensor_digest']==m['calibration_maps_tensor_digest']
    for name,h in dependency['copied_files'].items():assert sha(p/name)==h,name
    directions=load(p/'directions.pt')
    assert m['passages_per_corpus']==CORPUS_COUNTS and m['direction_sha256']==INPUT_HASHES['directions.pt']
    assert all(sha(p/n)==h for n,h in m['input_dependencies'].items())
    assert all(m['input_dependencies'][n]==h for n,h in INPUT_HASHES.items())
    assert len({r['key'] for r in rows}) == len(rows) and set(raw) == set(traces) == {r['key'] for r in rows}
    del traces
    groups = defaultdict(list); segments = defaultdict(list); inputs = defaultdict(dict); scalar_error = 0.
    for r in rows:
        assert r['exposed_passage'] is False
        z = raw[r['key']]; stem = r['key'].rsplit('__', 1)[0]; c, ref = raw[stem+'__C'], raw[stem+'__P']
        n = r['length']; assert n == 512 and r['input_tokens'] == n
        assert r['score_index'] == z['score_index'] == 1 and r['target_start'] == 2 and r['scored_tokens'] == n-2
        assert z['token_sha256'] == r['token_sha256'] == c['token_sha256'] == ref['token_sha256']
        assert len(z['final_hidden_sha256']) == 64
        for field in ('loss', 'kl', 'kl_p', 'pred', 'agreement_p'):
            assert z[field].shape == (n-1,) and torch.isfinite(z[field]).all()
        assert torch.equal(z['agreement_p'], z['pred'].eq(ref['pred']))
        for field, val in scalars(z, c, ref, slice(1,None)).items():
            assert math.isfinite(val); scalar_error = max(scalar_error, abs(val-r[field]))
            assert abs(val-r[field]) < 1e-12, (r['key'], field)
        if r['trial'] == 'P': assert torch.count_nonzero(z['kl_p']) == 0
        if r['trial'] == 'C': assert torch.count_nonzero(z['kl']) == 0
        if r['trial'] in ('current','fullrank','repeat','identity'):
            closure = ref if r['trial'] in ('current','fullrank') else c
            assert z['final_hidden_sha256'] == closure['final_hidden_sha256']
            assert all(torch.equal(z[k][1:],closure[k][1:]) for k in ('loss','pred','kl','kl_p','agreement_p'))
        inputs[r['corpus'],r['identity'],n,r['window']][r['trial']] = r
        if r['trial'] in CONDITIONS:
            groups[r['corpus'],r['identity'],n,r['trial']].append(r)
            for segment, sl in [('early_2_127',slice(1,127)),('later_128_511',slice(127,None))]:
                segments[r['corpus'],r['identity'],n,r['trial'],segment].append({**scalars(z,c,ref,sl),'window':r['window'],'exposed_passage':r['exposed_passage']})
    assert len(segments)==496
    assert len(inputs) == 30 and len(groups) == 248 and all(len(v)==CORPUS_COUNTS[k[0]] for k,v in groups.items())
    for (c,i,n,w), rs in inputs.items():
        expected = set(CONDITIONS) | ({'repeat','identity'} if i == 'natural' and w == 0 else set())
        assert set(rs) == expected
    summary = [{'corpus':c,'identity':i,'length':n,'trial':t,**dual_aggregate(rr)} for (c,i,n,t),rr in groups.items()]
    segment_summary = [{'corpus':c,'identity':i,'length':n,'trial':t,'segment':s,
                        'scored_tokens_per_window':126 if s == 'early_2_127' else 384,
                        **dual_aggregate(rr)} for (c,i,n,t,s),rr in segments.items()]
    dump(p/'summary.json', summary); dump(p/'segment_summary.json', segment_summary)
    contrasts = []; cg = defaultdict(list)
    for (c,i,n,w), rs in inputs.items():
        for name, a, b in CONTRASTS:
            delta = rs[a]['nll']-rs[b]['nll']
            contrasts.append({'corpus':c,'identity':i,'length':n,'window':w,'trial':name,'delta_nll':delta})
            cg[c,i,n,name].append(delta)
    ps = [{'corpus':c,'identity':i,'length':n,'trial':t,**paired(ds)} for (c,i,n,t),ds in cg.items()]
    assert len(contrasts) == 810 and len(ps) == 216
    dump(p/'paired_contrasts.json', contrasts); dump(p/'paired_summary.json', ps)
    nonadd,nonadd_summary=nonadditivity(inputs)
    assert len(nonadd)==90 and len(nonadd_summary)==24
    dump(p/'nonadditivity.json',nonadd);dump(p/'nonadditivity_summary.json',nonadd_summary)

    checks = json.loads((p/'checks.json').read_text()); ops = json.loads((p/'operation_checks.json').read_text())
    calops = json.loads((p/'calibration_operations.json').read_text())
    assert len(checks) == 1688 and len(ops) == 938 and len(calops) == 8
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    assert sum(ch['trial'].endswith('_source_identity') for ch in checks) == 900
    assert sum(ch['trial'].endswith('_singleton') for ch in checks)==720
    for ch in checks:
        if ch['trial'].endswith('_singleton'):assert ch['reference_context']==ch['trial_context']
    maxima = defaultdict(float); minmass = 1.; gatecount = 0; endpoints = {'zeros':0,'ones':0}
    errors = ('conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error',
              'fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error')
    for op in ops:
        assert op['projection_checks'] == op['query_checks'] == op['context_checks'] == 8 and op['writer_locality_checks'] == 30
        assert math.isfinite(op['max_context_error']) and op['max_context_error'] <= ERROR_TOLERANCE
        maxima['sdpa_manual_error'] = max(maxima['sdpa_manual_error'],op['max_context_error'])
        for ch in op['gate_checks'].values():
            gatecount += 1; assert ch['finite'] and ch['source_context_exact'] and ch['unaffected_context_exact'] and ch['min_mass'] > 0
            assert ch['norm_match_error']<=NORM_TOLERANCE
            minmass = min(minmass,ch['min_mass'])
            for k in errors+('max_gain',):
                assert math.isfinite(ch[k]); maxima[k] = max(maxima[k],ch[k])
                if k != 'max_gain': assert ch[k] <= ERROR_TOLERANCE
            if ch['active']:
                assert 0 <= ch['min_target_mass'] <= ch['max_target_mass'] <= 1
                endpoints['zeros'] += ch['target_zeros']; endpoints['ones'] += ch['target_ones']
    assert gatecount == 7200
    maps = load(p/'calibration_maps.pt'); cal = load(p/'calibration_raw.pt'); cm = load(p/'calibration_metrics.pt')
    assert len(cal) == len(cm) == 8 and maps_digest(maps) == m['calibration_maps_tensor_digest']
    for name,z in cm.items():
        assert z['kl_reference'] == 'self/no external C reference' and torch.count_nonzero(z['kl']) == 0
    bounds = {}
    for l in LAYERS:
        b = torch.stack([d[l]['b'][:,:510].T for d in cal.values()]).double()
        a = torch.stack([d[l]['mass'][:,:510].T for d in cal.values()]).double()
        assert b.shape == a.shape == (8,510,12) and torch.isfinite(b).all() and (a > 0).all()
        r = b-torch.arange(1,511,dtype=torch.float64).log()[None,:,None]
        assert torch.equal(maps[l]['mean_b'],b.reshape(-1,12).mean(0))
        assert torch.equal(maps[l]['alpha'],r.reshape(-1,12).mean(0))
        assert torch.equal(maps[l]['mean_mass'],a.reshape(-1,12).mean(0))
        bounds[l] = (b.reshape(-1,12).amin(0),b.reshape(-1,12).amax(0))
    specs = json.loads((p/'calibration_spectra.json').read_text())
    for l in LAYERS:
        for space in ('mass','b','r','position_centered_b'):
            s = specs[str(l)][space]
            assert s['vectors'] == 4080 and s['covariance_denominator'] == 4079
            assert np.allclose(np.square(s['singular_values'])/4079,s['eigenvalues'],rtol=1e-12,atol=1e-12)
        assert len(specs[str(l)]['leave_one_passage_out']) == 8
    filehashes = json.loads((p/'gate_files.json').read_text()); assert len(filehashes) == 30
    gate_geometry = []; map_error = 0.; exact_gate_copies = 0
    for name,h in filehashes.items():
        file = p/'gates'/name; assert sha(file) == h
        _,c,w,ns,i = file.stem.split('__'); n = int(ns); gs = load(file)
        assert set(gs) == set(inputs[c,i,n,int(w)])
        for trial,layers in gs.items():
            assert set(layers) == (set() if trial in ('C','repeat','identity') else set(LAYERS))
            family,affected=mode_spec(trial) if trial in CONDITIONS else (None,())
            if len(affected)==1:
                l=affected[0]
                for field in ('b','mass'):
                    assert torch.equal(layers[l][field],gs['P'][l][field]),('singleton saved prepatch mismatch',name,trial,l,field)
            for l,d in layers.items():
                b,a,t = d['b'],d['mass'],d['target_mass']
                assert b.shape == a.shape == t.shape == (12,n-1)
                assert torch.isfinite(b).all() and torch.isfinite(a).all() and torch.isfinite(t).all() and (a > 0).all()
                mb,ma,_,_,_,ne = localization_map(trial,l,b,a,{k:v.float() for k,v in maps[l].items()}, {k:directions[l][k].float() for k in ('u','v')})
                assert ne<=NORM_TOLERANCE
                for expected,actual in ((mb,d['target_b']),(ma,t)):
                    if expected is None: assert actual is None
                    else:
                        assert actual is not None and torch.isfinite(actual).all()
                        error = float((expected-actual).abs().max()); map_error = max(map_error,error)
                        assert error <= ERROR_TOLERANCE, (name,trial,l,error)
                if trial in ('P','current','fullrank'):
                    assert torch.equal(a,t); exact_gate_copies += 1
                if l in mode_spec(trial)[1]: assert (t >= 0).all() and (t <= 1).all()
                lo,hi = bounds[l]; score = slice(None,-1)
                gate_geometry.append({'corpus':c,'window':int(w),'identity':i,'length':n,'trial':trial,'layer':l,
                    'scored_queries':n-2,'mean_mass_by_head':a[:,score].double().mean(1).tolist(),
                    'mean_target_mass_by_head':t[:,score].double().mean(1).tolist(),
                    'mean_abs_mass_change_by_head':(t[:,score]-a[:,score]).abs().double().mean(1).tolist(),
                    'current_b_outside_calibration_count':int(((b[:,score].T.double()<lo)|(b[:,score].T.double()>hi)).sum()),
                    'target_zeros':int((t[:,score]==0).sum()),'target_ones':int((t[:,score]==1).sum()),
                    'min_current_mass':float(a.min()),'max_gain':float((t/a).max())})
        del gs
    assert len(gate_geometry) == 7200 and exact_gate_copies == 720
    dump(p/'gate_geometry.json',gate_geometry)

    sources = load(p/'source_states.pt'); assert set(sources) == set(raw)
    source_count = 0
    for key,st in sources.items():
        trial = key.rsplit('__',1)[1]
        if trial in ('C','repeat','identity'): continue
        ref = sources[key.rsplit('__',1)[0]+'__P']
        assert set(st['q']) == set(st['kv']) == set(LAYERS)
        assert all(torch.equal(v,ref['kv'][l][k]) for l,d in st['kv'].items() for k,v in d.items())
        assert all(torch.equal(v,ref['q'][l]) for l,v in st['q'].items()); source_count += 1
    assert source_count == 900; del sources
    snaps = load(p/'attention_snapshots.pt')
    assert len(snaps)==2*len(SNAPSHOT_MODES) and sum(map(len,snaps.values()))==52
    assert all(set(s) == {4,11} for s in snaps.values()); del snaps

    pf=json.loads((p/'prefix_files.json').read_text());assert len(pf)==30
    prefix_reference_count=0
    for name,h in pf.items():
        file=p/'prefix'/name;assert sha(file)==h
        _,c,w,i=file.stem.split('__');rs=load(file)
        assert set(rs)==set(inputs[c,i,512,int(w)])
        for trial,d in rs.items():
            assert set(d)=={'hidden'} and d['hidden'].shape==(1,128,768) and torch.isfinite(d['hidden']).all()
            prefix_reference_count+=1
    assert prefix_reference_count==938
    audit = {'status':'passed','records':len(raw),'forwards':m['forward_passes'],'calibration_forwards':0,'inherited_calibration_forwards':8,
             'summaries':len(summary),'segment_summaries':len(segment_summary),'contrasts':len(contrasts),'paired_summaries':len(ps),
             'checks':len(checks),'gate_context_checks':gatecount,'source_identity_records':source_count,
             'exact_gate_copy_records':exact_gate_copies,'gate_geometry_rows':len(gate_geometry),
             'same_shape_prefix_references':938,'cross_length_pairs':0,'snapshot_states':52,'calibration_vectors_per_layer':4080,
             'max_scalar_error':scalar_error,'max_cpu_projection_error':map_error,'min_current_mass':minmass,
             'active_target_endpoints_all_queries':endpoints,'maxima':dict(maxima),
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'limitations':'Saved-state audit, not independent review or pretrained re-execution. Bootstrap intervals condition on fixed calibration; early/later score views share passages and one fixed-512 execution.'}
    audit['suffix_audit']=audit_suffix(p,raw,inputs,ops,maps,directions)
    audit['total_new_gate_context_checks']=gatecount+audit['suffix_audit']['gate_context_checks']
    assert audit['total_new_gate_context_checks']==14400
    audit['nonadditivity_records']=len(nonadd);audit['nonadditivity_summaries']=len(nonadd_summary)
    audit['norm_audit']=audit_norms(p,inputs,maps,directions)
    audit['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(p/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__': main()
