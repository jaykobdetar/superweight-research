"""CPU-only V2 audit, fixed contrasts and exposure-aware summaries.

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
from gpt2_gate_subspace import (CONDITIONS, ERROR_TOLERANCE, LAYERS,
                               PREFIX_TOLERANCE, RANKS, SNAPSHOT_MODES,
                               mapped_gate, maps_digest, tensor_hash)
from run_transfer import dump, sha
from gpt2_gate_subspace_v2 import relative_prefix, fingerprint, LOCAL_RELATIVE_TOLERANCE, GLOBAL_RELATIVE_TOLERANCE

ACTIVE = tuple(t for t in CONDITIONS if t not in ('C', 'P_native', 'P', 'current', 'fullrank'))
CONTRASTS = tuple((t+'_minus_P', t, 'P') for t in ACTIVE) + tuple(
    (f'pca_{action}{k}_minus_qr_{action}{k}', f'pca_{action}{k}', f'qr_{action}{k}')
    for action in ('keep', 'remove') for k in RANKS) + tuple(
    (f'pca_{action}{k}_minus_position', f'pca_{action}{k}', 'position')
    for action in ('keep', 'remove') for k in RANKS) + (
    ('global1_minus_pca_keep1', 'global1', 'pca_keep1'),
    ('meanmass_minus_position', 'meanmass', 'position'),
    ('meanlogit_minus_position', 'meanlogit', 'position'), ('P_minus_C', 'P', 'C'))


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


def audit_suffix(p,raw,inputs,mainops,maps):
    checks=json.loads((p/'suffix_checks.json').read_text());ops=json.loads((p/'suffix_operations.json').read_text())
    files=json.loads((p/'suffix_files.json').read_text());assert len(checks)==len(ops)==672 and len(files)==32
    def identity(d):return d['corpus'],d['window'],d['identity'],d['trial']
    ci={identity(d):d for d in checks};oi={identity(d):d for d in ops}
    mi={identity(d):d for d in mainops if d['length']==512};assert len(ci)==len(oi)==672
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
            assert torch.equal(d['hidden'],prefix[trial]['long_prefix_hidden'])and d['hidden'].shape==(1,128,768)
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
                tb,ta,_,_=mapped_gate(trial,b,a,{k:v.float()for k,v in maps[l].items()})
                for expected,actual in((tb,g['target_b']),(ta,t)):
                    if expected is None:assert actual is None
                    else:
                        error=float((expected-actual).abs().max());max_map_error=max(max_map_error,error);assert error<=ERROR_TOLERANCE
                for k,v in g.items():
                    ref=originals[trial][l][k]
                    assert(v is None and ref is None)or(v is not None and torch.equal(v[:,:127],ref[:,:127]))
                if trial in ACTIVE:assert (t>=0).all()and(t<=1).all()
            assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
            assert math.isfinite(op['max_context_error'])and op['max_context_error']<=ERROR_TOLERANCE
            for ch in op['gate_checks'].values():
                gatechecks+=1;assert ch['finite']and ch['source_context_exact']and ch['min_mass']>0
                for k,v in ch.items():
                    if k.endswith('_error'):assert math.isfinite(v)and v<=ERROR_TOLERANCE;maxima[k]=max(maxima[k],v)
        del records,prefix,originals
    assert controls==672 and gatechecks==5120
    return {'records':controls,'gate_context_checks':gatechecks,'raw_hidden_gate_metric_source_exact':True,
        'runtime_and_hash_logit_context_closures':672,'raw_successful_logits_contexts_available':False,
        'snapshot_subset_available':True,'max_cpu_map_error':max_map_error,'min_current_mass':minimum_mass,
        'maxima':dict(maxima),'changed_suffix_scores_present':False}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path)
    p = ap.parse_args().run; torch.set_num_threads(4)
    dump(p/'audit.json',{'status':'checking','note':'A new audit has started; only a final passed state confirms completion.'})
    m = json.loads((p/'manifest.json').read_text())
    rows = [json.loads(s) for s in (p/'windows.jsonl').read_text().splitlines()]
    raw = load(p/'per_token.pt'); traces = load(p/'traces.pt')
    assert len(rows) == len(raw) == len(traces) == m['records'] == m['evaluation_forwards'] == 1360
    assert m['forward_passes']==2032 and m['calibration_forwards']==0 and m['inherited_calibration_forwards']==8 and m['suffix_forwards']==672
    assert m['phase'] == 'completed' and m.get('status') == 'completed' and 'completed_utc' in m
    assert m['model_hash_before'] == m['model_hash_after'] and m['parameter_versions_unchanged']
    assert m['calibration_maps_unchanged'] and m['calibration_frozen_utc'] <= m['completed_utc']
    assert m['context_tolerance'] == ERROR_TOLERANCE and m['loss_prefix_tolerance'] == PREFIX_TOLERANCE
    assert m['local_relative_tolerance']==LOCAL_RELATIVE_TOLERANCE and m['global_relative_tolerance']==GLOBAL_RELATIVE_TOLERANCE
    assert list(CONDITIONS) == json.loads((p/'conditions.json').read_text()) and len(CONTRASTS) == 32
    for name, h in m['source_hashes'].items(): assert sha(Path(__file__).parent/name) == h, name
    for name, field in (('conditions.json','conditions_sha256'), ('calibration_inputs.json','calibration_inputs_sha256'),
                        ('data_manifest.json','evaluation_inputs_sha256'), ('calibration_maps.pt','calibration_maps_sha256'),
                        ('calibration_raw.pt','calibration_raw_sha256')): assert sha(p/name) == m[field]
    assert sha(p/'calibration_dependency.json')==m['calibration_dependency_sha256']
    dependency=json.loads((p/'calibration_dependency.json').read_text())
    assert dependency['inherited_calibration_forwards']==8 and dependency['new_calibration_forwards']==0
    assert dependency['map_tensor_digest']==m['calibration_maps_tensor_digest']
    for name,h in dependency['copied_files'].items():assert sha(p/name)==h,name
    assert len({r['key'] for r in rows}) == len(rows) and set(raw) == set(traces) == {r['key'] for r in rows}
    del traces
    groups = defaultdict(list); segments = defaultdict(list); inputs = defaultdict(dict); scalar_error = 0.
    for r in rows:
        assert r['exposed_passage']==(r['corpus']=='wiki_validation' and r['window']==0)
        z = raw[r['key']]; stem = r['key'].rsplit('__', 1)[0]; c, ref = raw[stem+'__C'], raw[stem+'__P']
        n = r['length']; assert n in (128, 512) and r['input_tokens'] == n
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
        if n == 128: assert r['q_contribution'] is None
        inputs[r['corpus'],r['identity'],n,r['window']][r['trial']] = r
        if r['trial'] in CONDITIONS:
            groups[r['corpus'],r['identity'],n,r['trial']].append(r)
            for segment, sl in [('shared_2_127',slice(1,127))]+([('later_128_511',slice(127,None))] if n == 512 else []):
                segments[r['corpus'],r['identity'],n,r['trial'],segment].append({**scalars(z,c,ref,sl),'window':r['window'],'exposed_passage':r['exposed_passage']})
    assert len(inputs) == 64 and len(groups) == 336 and all(len(v) == 4 for v in groups.values())
    for (c,i,n,w), rs in inputs.items():
        expected = set(CONDITIONS) | ({'repeat','identity'} if i == 'natural' and w == 0 else set())
        assert set(rs) == expected
    summary = [{'corpus':c,'identity':i,'length':n,'trial':t,**dual_aggregate(rr)} for (c,i,n,t),rr in groups.items()]
    segment_summary = [{'corpus':c,'identity':i,'length':n,'trial':t,'segment':s,
                        'scored_tokens_per_window':126 if s == 'shared_2_127' else 384,
                        **dual_aggregate(rr)} for (c,i,n,t,s),rr in segments.items()]
    dump(p/'summary.json', summary); dump(p/'segment_summary.json', segment_summary)
    unexposed=[{'corpus':c,'identity':i,'length':n,'trial':t,'view':'unexposed_passages',
        **dual_aggregate([r for r in rr if not r['exposed_passage']])} for (c,i,n,t),rr in groups.items()]
    unexposed_segments=[{'corpus':c,'identity':i,'length':n,'trial':t,'segment':s,'view':'unexposed_passages',
        'scored_tokens_per_window':126 if s=='shared_2_127' else 384,
        **dual_aggregate([r for r in rr if not r['exposed_passage']])} for (c,i,n,t,s),rr in segments.items()]
    assert all(r['windows']==(3 if r['corpus']=='wiki_validation' else 4)for r in unexposed+unexposed_segments)
    dump(p/'unexposed_summary.json',unexposed);dump(p/'unexposed_segment_summary.json',unexposed_segments)
    contrasts = []; cg = defaultdict(list)
    for (c,i,n,w), rs in inputs.items():
        for name, a, b in CONTRASTS:
            delta = rs[a]['nll']-rs[b]['nll']
            contrasts.append({'corpus':c,'identity':i,'length':n,'window':w,'trial':name,'delta_nll':delta})
            cg[c,i,n,name].append(delta)
    ps = [{'corpus':c,'identity':i,'length':n,'trial':t,**paired(ds)} for (c,i,n,t),ds in cg.items()]
    assert len(contrasts) == 2048 and len(ps) == 512
    dump(p/'paired_contrasts.json', contrasts); dump(p/'paired_summary.json', ps)
    ucg=defaultdict(list)
    for r in contrasts:
        if not(r['corpus']=='wiki_validation'and r['window']==0):ucg[r['corpus'],r['identity'],r['length'],r['trial']].append(r['delta_nll'])
    ups=[{'corpus':c,'identity':i,'length':n,'trial':t,'view':'unexposed_passages',**paired(ds)}for(c,i,n,t),ds in ucg.items()]
    assert len(ups)==512 and all(r['n']==(3 if r['corpus']=='wiki_validation' else 4)for r in ups)
    dump(p/'unexposed_paired_summary.json',ups)

    checks = json.loads((p/'checks.json').read_text()); ops = json.loads((p/'operation_checks.json').read_text())
    calops = json.loads((p/'calibration_operations.json').read_text())
    assert len(checks) == 1488 and len(ops) == 1360 and len(calops) == 8
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    assert sum(ch['trial'].endswith('_source_identity') for ch in checks) == 1280
    assert sum(ch['trial'] == 'mask_reference' for ch in checks) == 64
    maxima = defaultdict(float); minmass = 1.; gatecount = 0; endpoints = {'zeros':0,'ones':0}
    errors = ('conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error',
              'fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error')
    for op in ops:
        assert op['projection_checks'] == op['query_checks'] == op['context_checks'] == 8 and op['writer_locality_checks'] == 30
        assert math.isfinite(op['max_context_error']) and op['max_context_error'] <= ERROR_TOLERANCE
        maxima['sdpa_manual_error'] = max(maxima['sdpa_manual_error'],op['max_context_error'])
        for ch in op['gate_checks'].values():
            gatecount += 1; assert ch['finite'] and ch['source_context_exact'] and ch['min_mass'] > 0
            minmass = min(minmass,ch['min_mass'])
            for k in errors+('max_gain',):
                assert math.isfinite(ch[k]); maxima[k] = max(maxima[k],ch[k])
                if k != 'max_gain': assert ch[k] <= ERROR_TOLERANCE
            if op['trial'] in ACTIVE:
                assert 0 <= ch['min_target_mass'] <= ch['max_target_mass'] <= 1
                endpoints['zeros'] += ch['target_zeros']; endpoints['ones'] += ch['target_ones']
    assert gatecount == 10240
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
    filehashes = json.loads((p/'gate_files.json').read_text()); assert len(filehashes) == 64
    gate_geometry = []; map_error = 0.; exact_gate_copies = 0
    for name,h in filehashes.items():
        file = p/'gates'/name; assert sha(file) == h
        _,c,w,ns,i = file.stem.split('__'); n = int(ns); gs = load(file)
        assert set(gs) == set(inputs[c,i,n,int(w)])
        for trial,layers in gs.items():
            assert set(layers) == (set() if trial in ('C','repeat','identity') else set(LAYERS))
            for l,d in layers.items():
                b,a,t = d['b'],d['mass'],d['target_mass']
                assert b.shape == a.shape == t.shape == (12,n-1)
                assert torch.isfinite(b).all() and torch.isfinite(a).all() and torch.isfinite(t).all() and (a > 0).all()
                mb,ma,_,_ = mapped_gate(trial,b,a,{k:v.float() for k,v in maps[l].items()})
                for expected,actual in ((mb,d['target_b']),(ma,t)):
                    if expected is None: assert actual is None
                    else:
                        assert actual is not None and torch.isfinite(actual).all()
                        error = float((expected-actual).abs().max()); map_error = max(map_error,error)
                        assert error <= ERROR_TOLERANCE, (name,trial,l,error)
                if trial in ('P','P_native','current','fullrank'):
                    assert torch.equal(a,t); exact_gate_copies += 1
                if trial in ACTIVE: assert (t >= 0).all() and (t <= 1).all()
                lo,hi = bounds[l]; score = slice(None,-1)
                gate_geometry.append({'corpus':c,'window':int(w),'identity':i,'length':n,'trial':trial,'layer':l,
                    'scored_queries':n-2,'mean_mass_by_head':a[:,score].double().mean(1).tolist(),
                    'mean_target_mass_by_head':t[:,score].double().mean(1).tolist(),
                    'mean_abs_mass_change_by_head':(t[:,score]-a[:,score]).abs().double().mean(1).tolist(),
                    'current_b_outside_calibration_count':int(((b[:,score].T.double()<lo)|(b[:,score].T.double()>hi)).sum()),
                    'target_zeros':int((t[:,score]==0).sum()),'target_ones':int((t[:,score]==1).sum()),
                    'min_current_mass':float(a.min()),'max_gain':float((t/a).max())})
        del gs
    assert len(gate_geometry) == 10240 and exact_gate_copies == 2048
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
    assert source_count == 1280; del sources
    snaps = load(p/'attention_snapshots.pt')
    assert len(snaps)==3*len(SNAPSHOT_MODES) and sum(map(len,snaps.values()))==60
    assert all(set(s) == {4,11} for s in snaps.values()); del snaps

    pf = json.loads((p/'prefix_files.json').read_text()); pc = json.loads((p/'prefix_checks.json').read_text())
    assert len(pf) == 32 and len(pc) == 680
    pci = {(v['corpus'],v['window'],v['identity'],v['trial']):v for v in pc}; assert len(pci) == 680
    for name,h in pf.items():
        file = p/'prefix'/name; assert sha(file) == h
        _,c,w,i = file.stem.split('__'); w = int(w); pairs = load(file)
        sstem = f'gate__{c}__{w}__128__{i}'; lstem = f'gate__{c}__{w}__512__{i}'
        sg,lg = load(p/'gates'/(sstem+'.pt')),load(p/'gates'/(lstem+'.pt'))
        assert set(pairs) == set(inputs[c,i,128,w])
        for trial,pair in pairs.items():
            z,y = raw[sstem+'__'+trial],raw[lstem+'__'+trial]; record = pci[c,w,i,trial]
            short,long = pair['short_hidden'],pair['long_prefix_hidden']; assert short.shape == long.shape == (1,128,768)
            assert tensor_hash(short) == z['final_hidden_sha256']
            vals = {'max_loss_drift':float((z['loss'][1:].double()-y['loss'][1:127].double()).abs().max()),
                    **relative_prefix(short,long),
                    'argmax_agreement':float(z['pred'][1:].eq(y['pred'][1:127]).double().mean()),
                    'max_gate_b_drift':max((float((d['b']-lg[trial][l]['b'][:,:127]).abs().max()) for l,d in sg[trial].items()),default=0.),
                    'max_target_mass_drift':max((float((d['target_mass']-lg[trial][l]['target_mass'][:,:127]).abs().max()) for l,d in sg[trial].items()),default=0.)}
            sm,lm=pair['short_prefix_margins'],pair['long_prefix_margins']
            assert sm.shape==lm.shape==(127,) and torch.isfinite(sm).all() and torch.isfinite(lm).all() and(sm>=0).all() and(lm>=0).all()
            mismatch=z['pred'][1:].ne(y['pred'][1:127])
            vals.update(scored_argmax_mismatches=int(mismatch.sum()),minimum_short_scored_margin=float(sm[1:].min()),
                minimum_long_scored_margin=float(lm[1:].min()),mismatch_queries=(mismatch.nonzero().flatten()+1).tolist(),
                short_margins_at_mismatch=sm[1:][mismatch].tolist(),long_margins_at_mismatch=lm[1:][mismatch].tolist())
            for k,v in vals.items():
                assert v == record[k],(name,trial,k,v,record[k])
                if isinstance(v,(float,int)):
                    assert math.isfinite(v);maxima['prefix_'+k]=max(maxima['prefix_'+k],v)
            assert vals['max_loss_drift']<=PREFIX_TOLERANCE and vals['max_query_relative_l2']<=LOCAL_RELATIVE_TOLERANCE and vals['whole_prefix_relative_l2']<=GLOBAL_RELATIVE_TOLERANCE
        del pairs,sg,lg
    audit = {'status':'passed','records':len(raw),'forwards':m['forward_passes'],'calibration_forwards':0,'inherited_calibration_forwards':8,
             'summaries':len(summary),'segment_summaries':len(segment_summary),'contrasts':len(contrasts),'paired_summaries':len(ps),
             'checks':len(checks),'gate_context_checks':gatecount,'source_identity_records':source_count,
             'exact_gate_copy_records':exact_gate_copies,'gate_geometry_rows':len(gate_geometry),
             'prefix_pairs':680,'snapshot_states':60,'calibration_vectors_per_layer':4080,
             'max_scalar_error':scalar_error,'max_cpu_projection_error':map_error,'min_current_mass':minmass,
             'active_target_endpoints_all_queries':endpoints,'maxima':dict(maxima),
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'limitations':'Saved-state audit, not independent review or pretrained re-execution. Bootstrap intervals condition on fixed calibration; lengths share prefixes.'}
    audit['suffix_audit']=audit_suffix(p,raw,inputs,ops,maps)
    audit['total_new_gate_context_checks']=gatecount+audit['suffix_audit']['gate_context_checks']
    assert audit['total_new_gate_context_checks']==15360
    audit['old_absolute_hidden_threshold_exceedance_pairs']=sum(c['hidden_old_absolute_exceedance_count']>0 for c in pc)
    audit['old_absolute_hidden_threshold_exceedance_entries']=sum(c['hidden_old_absolute_exceedance_count']for c in pc)
    audit['unexposed_summaries']=len(unexposed);audit['unexposed_segment_summaries']=len(unexposed_segments);audit['unexposed_paired_summaries']=len(ups)
    audit['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    dump(p/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__': main()
