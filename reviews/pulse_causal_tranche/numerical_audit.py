"""Independent result audit: CPU tensors only, no model imports or loading."""
import csv, hashlib, json, math, os, resource
from collections import defaultdict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'run'
manifest = json.loads((OUT / 'manifest.json').read_text())
assert 'completed_utc' in manifest, 'Wait for completed manifest before final audit'
assert manifest['model_unchanged'] and manifest['parameter_versions_unchanged']
assert manifest['code_sha256']['run_tranche.py'] == hashlib.sha256((ROOT / 'run_tranche.py').read_bytes()).hexdigest()
raw = torch.load(OUT / 'per_token.pt', map_location='cpu', weights_only=True)
rows = [json.loads(line) for line in (OUT / 'windows.jsonl').read_text().splitlines()]
groups = defaultdict(dict)
for row in rows:
    key = row['group'] + '__' + row['trial']
    assert row['window'] not in groups[key]
    groups[key][row['window']] = row
assert set(raw) == set(groups)
checks = {'completed_utc': manifest['completed_utc'], 'rows': len(rows), 'conditions': len(groups)}
for key, items in raw.items():
    assert len(items) == len(groups[key])
    for item in items:
        row = groups[key][item['window']]
        assert item['token_sha256'] == row['token_sha256']
        assert torch.isfinite(item['loss']).all()
        assert abs(item['loss'].double().mean().item() - row['nll']) < 1e-12
raw = {key: {r['window']: r for r in vals} for key, vals in raw.items()}

def interval(values):
    values = np.array(values, dtype=float)
    indices = np.random.default_rng(20261021).integers(0, len(values), (2000, len(values)))
    limits = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return {'mean': float(values.mean()), 'low': float(limits[0]), 'high': float(limits[1])}

summary = list(csv.DictReader((OUT / 'summary.csv').open()))
assert len(summary) == len(groups)
for row in summary:
    items = groups[row['group'] + '__' + row['trial']]
    ordered = [items[i] for i in sorted(items)]
    ci = interval([r['nll'] - r['baseline_nll'] for r in ordered])
    for field, other in [('delta_nll', 'mean'), ('delta_nll_low', 'low'), ('delta_nll_high', 'high')]:
        assert abs(float(row[field]) - ci[other]) < 1e-12
    assert abs(float(row['ppl_ratio']) - math.exp(ci['mean'])) < 1e-9
contrasts = list(csv.DictReader((OUT / 'contrasts.csv').open()))
for row in contrasts:
    key = row['group'] + '__' + row['trial']
    refgroup = 'causal' if row['group'] in ('directions', 'replacements') else row['group']
    refkey = refgroup + '__' + row['reference']
    assert set(groups[key]) == set(groups[refkey])
    vals = []
    for i in sorted(groups[key]):
        a, b = groups[key][i], groups[refkey][i]
        assert a['token_sha256'] == b['token_sha256']
        vals.append(a['nll'] - b['nll'])
    ci = interval(vals)
    for field, other in [('delta_nll', 'mean'), ('low', 'low'), ('high', 'high')]:
        assert abs(float(row[field]) - ci[other]) < 1e-12
checks['summary_and_contrast_replay'] = {'summary_rows': len(summary), 'contrast_rows': len(contrasts), 'passed': True}

def effect(group, trial, reference, refgroup='causal'):
    a, b = groups[group + '__' + trial], groups[refgroup + '__' + reference]
    return interval([a[i]['nll'] - b[i]['nll'] for i in sorted(a)])

for state in ['clean', 'deleted']:
    for i, a in raw['causal__identity_' + state].items():
        b = raw['causal__' + state][i]
        assert torch.equal(a['loss'], b['loss']) and torch.equal(a['pred'], b['pred'])
checks['identity_controls_bitwise_equal'] = True
for key, items in raw.items():
    if not key.startswith('directions__'): continue
    reference = 'deleted' if key.endswith('_rescue') else 'clean'
    for i, a in items.items():
        b = raw['causal__' + reference][i]
        assert torch.equal(a['loss'][:9], b['loss'][:9])
        assert torch.equal(a['pred'][:9], b['pred'][:9])
checks['direction_first9_bitwise_unchanged'] = True
checks['final_layer_full_patch'] = {}
for direction, reference in [('rescue', 'clean'), ('damage', 'deleted')]:
    ds, ps = [], []
    for i, a in raw['directions__L15_full_' + direction].items():
        b = raw['causal__' + reference][i]
        ds.append((a['loss'][9:].double() - b['loss'][9:].double()).abs().max().item())
        ps.append(int(a['pred'][9:].ne(b['pred'][9:]).sum()))
    checks['final_layer_full_patch'][direction] = {'max_abs_later9_token_nll_error': max(ds), 'prediction_mismatches': sum(ps)}

checks['L2H15_zero'] = {}
for direction, reference in [('rescue', 'deleted'), ('damage', 'clean')]:
    for kind in ['k', 'v', 'kv']:
        name = f'L2H15_{kind}_zero_{direction}'
        checks['L2H15_zero'][name] = effect('causal', name, reference)
    trials = [groups[f'causal__L2H15_{kind}_zero_{direction}'] for kind in ['kv', 'k', 'v']]
    baseline = groups['causal__' + reference]
    interaction = [trials[0][i]['nll'] - trials[1][i]['nll'] - trials[2][i]['nll'] + baseline[i]['nll'] for i in sorted(baseline)]
    checks['L2H15_zero'][direction + '_additive_interaction_nats'] = interval(interaction)
checks['late_heads'] = {}
for head in ['L12H5', 'L15H11']:
    for direction, reference in [('rescue', 'deleted'), ('damage', 'clean')]:
        for kind, positions in [('k', 'zero'), ('v', 'zero'), ('kv', 'zero'), ('k', 'all'), ('v', 'all'), ('kv', 'all'), ('context', 'all')]:
            name = f'{head}_{kind}_{positions}_{direction}'
            checks['late_heads'][name] = effect('causal', name, reference)
checks['replacements'] = {}
for key in groups:
    if key.startswith('replacements__'):
        trial = key.split('__')[1]
        checks['replacements'][trial] = {'against_state': effect('replacements', trial, trial.split('_')[-1]), 'against_clean': effect('replacements', trial, 'clean')}

checks['covariance'] = {}
for layer in [2, 12, 15]:
    bundle = torch.load(OUT / f'covariance_L{layer}.pt', map_location='cpu', weights_only=True)
    d = torch.cat([v[0, 9:] for v in bundle['window_deltas']])
    mu, cov, eig, vec = [bundle[k] for k in ['mean', 'covariance', 'eigenvalues', 'top_vectors']]
    centered = d - mu
    assert torch.equal(mu, d.mean(0))
    assert torch.allclose(cov, cov.T, atol=1e-7)
    assert torch.allclose(vec.T @ vec, torch.eye(32), atol=2e-6)
    columns = torch.tensor([0, 7, 128, 721, 1710, 1764, 2047])
    sampled = centered[:, columns].T @ centered[:, columns] / (len(d) - 1)
    assert torch.allclose(cov[columns[:, None], columns], sampled, rtol=2e-5, atol=1e-7)
    trace, eigtotal = cov.diag().double().sum().item(), eig.double().sum().item()
    assert abs(trace-eigtotal)/trace < 1e-5
    assert torch.allclose(bundle['singular_values'].square(), eig*(len(d)-1), rtol=2e-6, atol=1e-5)
    checks['covariance'][str(layer)] = {'mean_energy_fraction': float(len(d)*mu.double().square().sum()/d.double().square().sum()), 'trace_eigenvalue_relative_error': abs(trace-eigtotal)/trace, 'sampled_covariance_entries_passed': True}
    del bundle, d, centered
retention = [json.loads(s) for s in (OUT / 'heldout_retention.jsonl').read_text().splitlines()]
for layer in [2, 12, 15]:
    rs = [r for r in retention if r['layer'] == layer]
    assert len(rs) == 12
    assert all(0 <= r['pc1_energy'] <= r['pc8_energy'] <= r['pc32_energy'] <= r['centered_energy'] * 1.00001 for r in rs)
    denominator = sum(r['centered_energy'] for r in rs)
    checks['covariance'][str(layer)]['heldout_centered_retention'] = {p: sum(r[p+'_energy'] for r in rs)/denominator for p in ['pc1','pc8','pc32']}
selection = json.loads((OUT / 'selection.json').read_text())
import ast
selected = [list(ast.literal_eval(h)) for h, values in selection['screen'].items() if np.mean(values['damage']) > .02 or -np.mean(values['rescue']) > .02]
assert selected == selection['selected']
checks['selection_replay_passed'] = True
checks['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
Path(__file__).with_name('numerical_audit.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
