"""CPU-only validation and predeclared summaries for calibrated gate maps.

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
    assert len(x) == 4 and np.isfinite(x).all()
    rng = np.random.default_rng(20261021)
    bs = x[rng.integers(0, 4, (2000, 4))].mean(1)
    return {'n': 4, 'mean_delta_nll': float(x.mean()),
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run', type=Path)
    p = ap.parse_args().run; torch.set_num_threads(4)
    dump(p/'audit.json',{'status':'checking','note':'A new audit has started; only a final passed state confirms completion.'})
    m = json.loads((p/'manifest.json').read_text())
    rows = [json.loads(s) for s in (p/'windows.jsonl').read_text().splitlines()]
    raw = load(p/'per_token.pt'); traces = load(p/'traces.pt')
    assert len(rows) == len(raw) == len(traces) == m['records'] == m['evaluation_forwards'] == 1360
    assert m['forward_passes'] == 1368 and m['calibration_forwards'] == 8
    assert m['phase'] == 'completed' and m.get('status') != 'failed' and 'completed_utc' in m
    assert m['model_hash_before'] == m['model_hash_after'] and m['parameter_versions_unchanged']
    assert m['calibration_maps_unchanged'] and m['calibration_frozen_utc'] <= m['completed_utc']
    assert m['context_tolerance'] == ERROR_TOLERANCE and m['prefix_tolerance'] == PREFIX_TOLERANCE
    assert list(CONDITIONS) == json.loads((p/'conditions.json').read_text()) and len(CONTRASTS) == 32
    for name, h in m['source_hashes'].items(): assert sha(Path(__file__).parent/name) == h, name
    for name, field in (('conditions.json','conditions_sha256'), ('calibration_inputs.json','calibration_inputs_sha256'),
                        ('data_manifest.json','evaluation_inputs_sha256'), ('calibration_maps.pt','calibration_maps_sha256'),
                        ('calibration_raw.pt','calibration_raw_sha256')): assert sha(p/name) == m[field]
    assert len({r['key'] for r in rows}) == len(rows) and set(raw) == set(traces) == {r['key'] for r in rows}
    del traces
    groups = defaultdict(list); segments = defaultdict(list); inputs = defaultdict(dict); scalar_error = 0.
    for r in rows:
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
                segments[r['corpus'],r['identity'],n,r['trial'],segment].append(scalars(z,c,ref,sl))
    assert len(inputs) == 64 and len(groups) == 336 and all(len(v) == 4 for v in groups.values())
    for (c,i,n,w), rs in inputs.items():
        expected = set(CONDITIONS) | ({'repeat','identity'} if i == 'natural' and w == 0 else set())
        assert set(rs) == expected
    summary = [{'corpus':c,'identity':i,'length':n,'trial':t,**dual_aggregate(rr)} for (c,i,n,t),rr in groups.items()]
    segment_summary = [{'corpus':c,'identity':i,'length':n,'trial':t,'segment':s,
                        'scored_tokens_per_window':126 if s == 'shared_2_127' else 384,
                        **dual_aggregate(rr)} for (c,i,n,t,s),rr in segments.items()]
    dump(p/'summary.json', summary); dump(p/'segment_summary.json', segment_summary)
    contrasts = []; cg = defaultdict(list)
    for (c,i,n,w), rs in inputs.items():
        for name, a, b in CONTRASTS:
            delta = rs[a]['nll']-rs[b]['nll']
            contrasts.append({'corpus':c,'identity':i,'length':n,'window':w,'trial':name,'delta_nll':delta})
            cg[c,i,n,name].append(delta)
    ps = [{'corpus':c,'identity':i,'length':n,'trial':t,**paired(ds)} for (c,i,n,t),ds in cg.items()]
    assert len(contrasts) == 2048 and len(ps) == 512
    dump(p/'paired_contrasts.json', contrasts); dump(p/'paired_summary.json', ps)

    checks = json.loads((p/'checks.json').read_text()); ops = json.loads((p/'operation_checks.json').read_text())
    calops = json.loads((p/'calibration_operations.json').read_text())
    assert len(checks) == 1488 and len(ops) == 1360 and len(calops) == 8
    assert all(v for ch in checks for k,v in ch.items() if k.endswith('_exact'))
    assert sum(ch['trial'].endswith('_source_identity') for ch in checks) == 1280
    assert sum(ch['trial'] == 'mask_reference' for ch in checks) == 64
    maxima = defaultdict(float); minmass = 1.; gatecount = 0; endpoints = {'zeros':0,'ones':0}
    errors = ('conditional_error','native_reconstruction_error','mixed_reconstruction_error','sigmoid_context_error',
              'fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error')
    for op in ops+calops:
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
    assert gatecount == 10304
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
    assert len(snaps) == 2*len(SNAPSHOT_MODES) and sum(map(len,snaps.values())) == 40
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
            vals = {'max_loss_drift':float((z['loss'][1:]-y['loss'][1:127]).abs().max()),
                    'max_hidden_drift':float((short-long).abs().max()),
                    'argmax_agreement':float(z['pred'][1:].eq(y['pred'][1:127]).double().mean()),
                    'max_gate_b_drift':max((float((d['b']-lg[trial][l]['b'][:,:127]).abs().max()) for l,d in sg[trial].items()),default=0.),
                    'max_target_mass_drift':max((float((d['target_mass']-lg[trial][l]['target_mass'][:,:127]).abs().max()) for l,d in sg[trial].items()),default=0.)}
            for k,v in vals.items():
                assert math.isfinite(v) and v == record[k]; maxima['prefix_'+k] = max(maxima['prefix_'+k],v)
            assert vals['max_loss_drift'] <= PREFIX_TOLERANCE and vals['max_hidden_drift'] <= PREFIX_TOLERANCE
        del pairs,sg,lg
    audit = {'status':'passed','records':len(raw),'forwards':m['forward_passes'],'calibration_forwards':8,
             'summaries':len(summary),'segment_summaries':len(segment_summary),'contrasts':len(contrasts),'paired_summaries':len(ps),
             'checks':len(checks),'gate_context_checks':gatecount,'source_identity_records':source_count,
             'exact_gate_copy_records':exact_gate_copies,'gate_geometry_rows':len(gate_geometry),
             'prefix_pairs':680,'snapshot_states':40,'calibration_vectors_per_layer':4080,
             'max_scalar_error':scalar_error,'max_cpu_projection_error':map_error,'min_current_mass':minmass,
             'active_target_endpoints_all_queries':endpoints,'maxima':dict(maxima),
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'limitations':'Saved-state audit, not independent review or pretrained re-execution. Bootstrap intervals condition on fixed calibration; lengths share prefixes.'}
    dump(p/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__': main()
