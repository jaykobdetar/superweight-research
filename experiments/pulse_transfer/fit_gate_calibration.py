"""T18 sample-count generalization; inherited fitter remains unchanged."""
import torch
from gpt2_gate_subspace import spectrum, LAYERS, RANKS, QR_SEED

def fit_maps_n(calibration, n_expected):
    """Fixed n=4 or n=8; no duplication, evaluation data or outcome selection."""
    assert n_expected in (4, 8) and len(calibration) == n_expected
    n = n_expected
    generator = torch.Generator(device='cpu').manual_seed(QR_SEED)
    maps, reports = {}, {}
    logq = torch.arange(1, 511, dtype=torch.float64).log()[None, :, None]
    for layer in LAYERS:
        b = torch.stack([v[layer]['b'][:, :510].T for v in calibration.values()]).double()
        a = torch.stack([v[layer]['mass'][:, :510].T for v in calibration.values()]).double()
        assert b.shape == a.shape == (n, 510, 12)
        assert torch.isfinite(b).all() and torch.isfinite(a).all() and (a > 0).all()
        r = b - logq
        spaces = {'mass': a, 'b': b, 'r': r, 'position_centered_b': b-b.mean(0, keepdim=True)}
        specs = {name: spectrum(value) for name, value in spaces.items()}
        q, qr_r = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.float64))
        signs = torch.sign(qr_r.diag()); signs[signs == 0] = 1
        q = q * signs
        assert (q.T @ q-torch.eye(12, dtype=torch.float64)).abs().max() <= 1e-10
        maps[layer] = {'alpha': specs['r'][0], 'mean_b': specs['b'][0],
                       'mean_mass': specs['mass'][0], 'basis_r': specs['r'][1],
                       'basis_b': specs['b'][1], 'basis_qr': q}
        # Native summed mass may have tiny roundoff above1, but the fitted
        # constant gate must be in [0,1]. Never silently clip this mean.
        assert (maps[layer]['mean_mass'] > 0).all() and (maps[layer]['mean_mass'] <= 1).all()
        reports[layer] = {name: value[2] for name, value in specs.items()}
        reports[layer]['leave_one_passage_out'] = []
        for omit, name in enumerate(calibration):
            keep = [i for i in range(n) if i != omit]
            record = {'omitted_passage': name, 'spaces': {}}
            for space in ('mass', 'b', 'r', 'position_centered_b'):
                values = spaces[space][keep]
                if space == 'position_centered_b':
                    values = b[keep]-b[keep].mean(0, keepdim=True)
                _, ub, report = spectrum(values)
                report['principal_cosines_to_full'] = {
                    str(k): torch.linalg.svdvals(specs[space][1][:, :k].T @ ub[:, :k]).tolist()
                    for k in RANKS}
                record['spaces'][space] = report
            reports[layer]['leave_one_passage_out'].append(record)
    return maps, reports

