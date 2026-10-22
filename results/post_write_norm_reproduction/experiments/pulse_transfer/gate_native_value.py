"""Prospective T19 native-value boundary hook. No model loading or calibration.

Bits U/L/D: 0 retains live projected V0, 1 zeros V0 at L4:9/L10/L11.
Only L10's pre-c_proj context can receive the frozen FF operation. CPU fixtures
exercise this module without importing transformers or loading any checkpoint.
"""
import hashlib
import math
import torch

LAYERS = tuple(range(4, 12))
BOUNDARIES = tuple(f'{i:03b}' for i in range(8))
OPERATIONS = ('baseline', 'current', 'remove', 'orth', 'remove_scale', 'orth_scale')
CONTEXT_TOLERANCE = 1e-4
NORM_TOLERANCE = 1e-6
# Summed FP32 softmax probabilities can slightly exceed one. Never clip them.
MASS_ROUNDOFF_TOLERANCE = 1e-6


def cpu(x):
    return x.detach().cpu().clone()


def fingerprint(x):
    value = x.detach().cpu().contiguous()
    return {'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'shape': list(value.shape), 'dtype': str(value.dtype)}


def error_metrics(a, b):
    """FP64 diagnostics; a zero norm is handled explicitly, never divided by."""
    a, b = a.double(), b.double()
    delta = a-b
    denominator = max(float(a.norm()), float(b.norm()))
    return {'max_abs': float(delta.abs().max()), 'rms': float(delta.square().mean().sqrt()),
            'symmetric_relative_l2': float(delta.norm())/denominator if denominator else 0.}


def ff_map(b, alpha, u, v, operation):
    """Exact T18 factorial_map FF arithmetic; b has heads x non-source-query shape."""
    assert operation in OPERATIONS
    assert b.ndim == 2 and alpha.shape == u.shape == v.shape == (b.shape[0],)
    logq = torch.arange(1, b.shape[1]+1, device=b.device, dtype=b.dtype).log()[:, None]
    s = (b.T-logq-alpha) @ u
    du = s[:, None]*u
    dv = s[:, None]*v
    nu, nv = du.double().norm(dim=-1), dv.double().norm(dim=-1)
    norm_error = float(((nu-nv).abs()/torch.maximum(torch.ones_like(nu), torch.maximum(nu, nv))).max())
    target_b = b if operation in ('baseline', 'current') else (b.T-(du if operation.startswith('remove') else dv)).T
    return target_b, s, nu, nv, norm_error


def deliver_context(actual, mass, target_mass, mixture, v0, *, scaling=False, unchanged=False):
    """Head/query/feature arithmetic, shared by fixtures and the actual hook.

    actual excludes q0. Native delta accepts finite saturated mass without division.
    The caller enforces the original bridge conditional reconstruction guard.
    Scaling reproduces T18 local arithmetic on this incoming state/mask; it does
    not claim replication of the old full T18 explicit-mask execution protocol.
    """
    assert actual.shape == mixture.shape and mass.shape == target_mass.shape == actual.shape[:-1]
    assert v0.shape == (actual.shape[0], 1, actual.shape[-1])
    assert all(bool(torch.isfinite(z).all()) for z in (actual, mass, target_mass, mixture, v0))
    assert bool((mass >= 0).all() and (mass <= 1+MASS_ROUNDOFF_TOLERANCE).all())
    assert bool((target_mass >= 0).all() and (target_mass <= 1+MASS_ROUNDOFF_TOLERANCE).all())
    if scaling:
        assert bool((mass > 0).all()), 'legacy scaling requires positive diagnostic mass'
        ratio = target_mass/mass
        assert bool(torch.isfinite(ratio).all()), 'legacy scaling gain is nonfinite'
    if unchanged:
        return actual
    result = actual*(target_mass/mass)[..., None] if scaling else actual+(target_mass-mass)[..., None]*(mixture-v0)
    assert bool(torch.isfinite(result).all()), 'nonfinite delivered context'
    return result


class GateNativeValue:
    """GPT2 hook for one live, uncached, FP32 batch of fixed physical length.

    Constructor takes frozen F layer10 alpha/u/v FP64 CPU vectors and a provenance
    digest. No fitting occurs. expected_positions=512 is the production default;
    a smaller explicit value is for synthetic CPU fixtures only. Captures source
    Q/K/V and block-output q0 at all12 layers, full diagnostics at L4..11, and full
    L10 raw/actual/delivered tensors only when reset(..., snapshot=True).
    """
    def __init__(self, model, alpha, u, v, digest, *, expected_positions=512):
        self.width = model.config.n_embd
        self.heads = model.transformer.h[10].attn.num_heads
        self.head_dim = model.transformer.h[10].attn.head_dim
        assert len(model.transformer.h) == 12 and self.width == self.heads*self.head_dim
        assert self.heads == 12 and expected_positions >= 2
        assert isinstance(digest, str) and digest
        self.digest = digest
        self.expected_positions = expected_positions
        self.alpha, self.u, self.v = (cpu(z) for z in (alpha, u, v))
        assert all(z.dtype == torch.float64 and z.shape == (self.heads,) and bool(torch.isfinite(z).all()) for z in (self.alpha, self.u, self.v))
        assert abs(float(self.u.norm())-1) <= 1e-12 and abs(float(self.v.norm())-1) <= 1e-12
        assert abs(float(self.u@self.v)) <= 1e-12
        uf, vf = self.u.float().double(), self.v.float().double()
        assert abs(float(uf.norm()-vf.norm())) <= NORM_TOLERANCE and abs(float(uf@vf)) <= NORM_TOLERANCE
        self.device_map = {}
        self.handles = []
        self.reset('000', 'baseline')
        # Validate the whole architecture before installing any hooks.
        for i, block in enumerate(model.transformer.h):
            att = block.attn
            assert att.num_heads == self.heads and att.head_dim == self.head_dim
            assert att.scale_attn_weights and not att.scale_attn_by_inverse_layer_idx and not att.reorder_and_upcast_attn
            assert getattr(att, 'layer_idx', i) == i
            assert not att.training, 'attention must be in evaluation mode'
        for i, block in enumerate(model.transformer.h):
            self.handles.append(block.attn.register_forward_pre_hook(lambda mod, args, kwargs, i=i: self.attn(i, args, kwargs), with_kwargs=True))
            self.handles.append(block.attn.c_attn.register_forward_hook(lambda mod, args, output, i=i: self.combined(i, output)))
            self.handles.append(block.register_forward_hook(lambda mod, args, output, i=i: self.source_block(i, output)))
            if i in LAYERS:
                self.handles.append(block.attn.c_proj.register_forward_pre_hook(lambda mod, args, i=i: self.context(i, args[0])))

    def reset(self, boundary, operation, snapshot=False):
        assert boundary in BOUNDARIES and operation in OPERATIONS, (boundary, operation)
        assert not operation.endswith('_scale') or boundary == '111', 'scaling only permitted at111'
        self.boundary, self.operation, self.save_snapshot = boundary, operation, bool(snapshot)
        self.pending = {}; self.diag = {}; self.gate_cache = {}; self.gate_checks = {}
        self.snapshots = {}; self.source_states = {}; self.projection_records = {}
        self.prefix_contexts = {}; self.context_records = {}; self.mask_records = {}
        self.effective_masks = {}; self.localization_norms = {}; self.failure = None
        self.projection_checks = self.context_checks = self.query_checks = 0

    def zero_layer(self, layer):
        return (4 <= layer <= 9 and self.boundary[0] == '1') or (layer == 10 and self.boundary[1] == '1') or (layer == 11 and self.boundary[2] == '1')

    def _fail(self, stage, layer, reason, **arrays):
        # One affected layer only; never retain all full contexts or vocabulary rows.
        self.failure = {'stage': stage, 'layer': layer, 'reason': str(reason), 'boundary': self.boundary,
                        'operation': self.operation, 'digest': self.digest,
                        'arrays': {k: cpu(v) if isinstance(v, torch.Tensor) else v for k, v in arrays.items()}}
        raise AssertionError((stage, layer, str(reason)))

    def attn(self, layer, args, kwargs):
        x = kwargs.get('hidden_states', args[0] if args else None)
        assert x is not None and x.ndim == 3 and x.shape[:2] == (1, self.expected_positions)
        assert x.dtype == torch.float32 and not x.requires_grad, 'FP32 inference only'
        assert not kwargs.get('output_attentions', False), 'SDPA context required'
        assert not kwargs.get('use_cache', False), 'cached attention unsupported'
        assert all(kwargs.get(name) is None for name in ('layer_past', 'past_key_value', 'past_key_values', 'encoder_hidden_states', 'head_mask'))
        n = x.shape[1]
        old = kwargs.get('attention_mask')
        mask = torch.zeros((1, 1, n, n), dtype=x.dtype, device=x.device)
        mask.masked_fill_(torch.ones((n, n), dtype=torch.bool, device=x.device).triu(1), -float('inf'))
        if old is not None:
            if not (old.ndim == 4 and old.shape[:2] == (1, 1) and old.shape[-2] in (1, n) and old.shape[-1] == n and old.device == x.device):
                self._fail('mask', layer, 'invalid mask shape/device', mask=old)
            if old.dtype == torch.bool:
                mask = mask.masked_fill(~old, -float('inf'))
            else:
                if old.dtype != x.dtype or bool(torch.isnan(old).any() or torch.isposinf(old).any()):
                    self._fail('mask', layer, 'invalid additive mask', mask=old)
                mask = old+mask  # Exact T18 order: old external mask + explicit causal.
        if bool(torch.isneginf(mask).all(-1).any()):
            self._fail('mask', layer, 'fully masked query', mask=mask)
        if layer in LAYERS:
            self.effective_masks[layer] = mask
            self.mask_records[layer] = {'kind': 'implicit_causal' if old is None else 'preserved_explicit',
                'model_argument_preserved': True, 'incoming': None if old is None else fingerprint(old),
                'diagnostic_effective': fingerprint(mask)}
        # Observer/current controls preserve the model's original mask argument
        # (including implicit causal SDPA); the explicit effective mask is diagnostic.
        return None

    def source_block(self, layer, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.source_states[layer]['hidden'] = cpu(hidden[:, 0])

    def combined(self, layer, output):
        assert output.dtype == torch.float32 and output.shape == (1, self.expected_positions, 3*self.width)
        if not bool(torch.isfinite(output).all()):
            self._fail('projection', layer, 'nonfinite incoming projection', projected=output)
        before = output
        if self.zero_layer(layer):
            output = output.clone(); output[:, 0, 2*self.width:] = 0
        assert torch.equal(output[:, :, :2*self.width], before[:, :, :2*self.width])
        assert torch.equal(output[:, 1:], before[:, 1:])
        assert self.zero_layer(layer) or torch.equal(output, before)
        if self.zero_layer(layer):
            assert int(torch.count_nonzero(output[:, 0, 2*self.width:])) == 0
        source = output[:, 0].split(self.width, dim=-1)
        self.source_states[layer] = {name: cpu(value) for name, value in zip(('q', 'k', 'v'), source)}
        if layer not in LAYERS:
            return output
        pre = {'q': before[:, :, :self.width], 'k': before[:, :, self.width:2*self.width],
               'nonsource_v': before[:, 1:, 2*self.width:], 'source_v': before[:, 0, 2*self.width:]}
        post = {'q': output[:, :, :self.width], 'k': output[:, :, self.width:2*self.width],
                'nonsource_v': output[:, 1:, 2*self.width:], 'source_v': output[:, 0, 2*self.width:]}
        self.projection_records[layer] = {'before': {k: fingerprint(z) for k, z in pre.items()},
                                          'after': {k: fingerprint(z) for k, z in post.items()}, 'zeroed': self.zero_layer(layer)}
        self.projection_checks += 1; self.query_checks += 1
        n = self.expected_positions
        q, k, value = [z.reshape(1, n, self.heads, self.head_dim).transpose(1, 2) for z in output.split(self.width, dim=-1)]
        scores = q@k.transpose(-1, -2)/math.sqrt(self.head_dim)
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), -float('inf'))
        mask = self.effective_masks[layer]
        scores = scores+mask
        prob = scores.softmax(-1)
        expected = (prob@value).transpose(1, 2).reshape(1, n, self.width)
        mass = prob[0, :, 1:, 1:].sum(-1)
        b = (scores[:, :, 1:, 1:].logsumexp(-1)-scores[:, :, 1:, 0])[0]
        stable = (scores[:, :, 1:, 1:].softmax(-1)@value[:, :, 1:])[0]
        if not all(bool(torch.isfinite(z).all()) for z in (expected, mass, b, stable)):
            self._fail('reconstruction', layer, 'nonfinite diagnostic operands', q=q, k=k, v=value, mask=mask, b=b, mass=mass, stable_mixture=stable)
        self.pending[layer] = dict(q=q, k=k, v=value, mask=mask, expected=expected, b=b, mass=mass,
                                   stable=stable, source_probability=prob[0, :, 1:, 0])
        return output

    def context(self, layer, actual):
        assert actual.shape == (1, self.expected_positions, self.width) and actual.dtype == torch.float32
        state = self.pending.pop(layer)
        b, mass, stable = state['b'], state['mass'], state['stable']
        n = actual.shape[1]
        c = actual.reshape(1, n, self.heads, self.head_dim)[0].transpose(0, 1)[:, 1:]
        v0 = state['v'][0, :, :1]
        decomposition = mass[..., None]*stable+(1-mass)[..., None]*v0
        explicit_decomposition = mass[..., None]*stable+state['source_probability'][..., None]*v0
        checks = {'context_max_abs_error': float((actual-state['expected']).abs().max()),
                  'native_reconstruction_error': float((c-decomposition).abs().max()),
                  'explicit_source_reconstruction_error': float((c-explicit_decomposition).abs().max()),
                  'source_complement_max_abs': float((state['source_probability']-(1-mass)).abs().max()),
                  'ideal_mass_max_abs': float((b.sigmoid()-mass).abs().max()),
                  'decomposition_fp64': error_metrics(c, mass.double()[..., None]*stable.double()+(1-mass.double())[..., None]*v0.double()),
                  'min_mass': float(mass.min()), 'max_mass': float(mass.max()),
                  'mass_zeros': int((mass == 0).sum()), 'mass_ones': int((mass == 1).sum()),
                  'mass_above_one_roundoff': int((mass > 1).sum())}
        self.diag[layer] = {**checks, 'source_q': cpu(state['q'][0, :, 0]),
                            'source_probability': cpu(state['source_probability']),
                            'source_vector_norm': cpu((state['source_probability'][..., None]*v0).double().norm(dim=-1)),
                            'nonsource_vector_norm': cpu((mass[..., None]*stable).double().norm(dim=-1))}
        self.context_checks += 1
        guard_names = ('context_max_abs_error', 'native_reconstruction_error', 'explicit_source_reconstruction_error')
        if not bool(torch.isfinite(actual).all()) or any(not math.isfinite(checks[k]) or checks[k] > CONTEXT_TOLERANCE for k in guard_names):
            self._fail('context', layer, checks, **state, actual_context=actual)
        if not bool((mass >= 0).all() and (mass <= 1+MASS_ROUNDOFF_TOLERANCE).all()):
            self._fail('mass', layer, 'invalid diagnostic mass', mass=mass)
        result = actual
        if layer == 10:
            key = (b.device, b.dtype)
            if key not in self.device_map:
                self.device_map[key] = tuple(z.to(device=b.device, dtype=b.dtype) for z in (self.alpha, self.u, self.v))
            alpha, u, v = self.device_map[key]
            target_b, s, nu, nv, norm_error = ff_map(b, alpha, u, v, self.operation)
            unchanged = self.operation in ('baseline', 'current') or torch.equal(target_b, b)
            proposed_mass = target_b.sigmoid()
            target_mass = mass if unchanged else proposed_mass
            scaling = self.operation.endswith('_scale')
            mixed = None
            try:
                assert norm_error <= NORM_TOLERANCE and bool(torch.isfinite(target_b).all())
                mixed = deliver_context(c, mass, target_mass, stable, v0, scaling=scaling, unchanged=unchanged)
                legacy_errors = {}
                if scaling:
                    # Unchanged T18 positive-mass/conditional/mixed/sigmoid guards.
                    legacy_errors = {'conditional_error': float((c/mass[..., None]-stable).abs().max()),
                        'native_reconstruction_error': float((c-mass[..., None]*stable).abs().max()),
                        'mixed_reconstruction_error': float((mixed-target_mass[..., None]*stable).abs().max()),
                        'sigmoid_context_error': float((c-b.sigmoid()[..., None]*stable).abs().max())}
                    assert all(math.isfinite(z) and z <= CONTEXT_TOLERANCE for z in legacy_errors.values()), legacy_errors
                if not unchanged:
                    result = actual.clone()
                    result[:, 1:] = mixed.transpose(0, 1).reshape(1, n-1, self.width)
                assert torch.equal(result[:, 0], actual[:, 0])
            except AssertionError as error:
                self._fail('gate', layer, error, **state, actual_context=actual,
                           target_b=target_b, target_mass=target_mass, delivered_heads=mixed)
            delivered = result.reshape(1, n, self.heads, self.head_dim)[0].transpose(0, 1)[:, 1:]
            delta = delivered.double()-c.double()
            prescribed_delta = torch.zeros_like(delta) if unchanged else (target_mass.double()-mass.double())[..., None]*(stable.double()-v0.double())
            direction = u if self.operation.startswith('remove') else v
            mapped = error_metrics((target_b.double()-b.double()).T, torch.zeros_like(b.T).double() if self.operation in ('baseline', 'current') else -s.double()[:, None]*direction.double())
            self.gate_cache[10] = {k: cpu(z) for k, z in {'b': b, 'mass': mass, 'target_b': target_b,
                'target_mass': target_mass, 'proposed_target_mass': proposed_mass, 's': s}.items()}
            self.gate_checks[10] = {'active': not unchanged, 'unchanged_bypass': unchanged,
                'source_context_exact': torch.equal(result[:, 0], actual[:, 0]),
                'unaffected_context_exact': not unchanged or torch.equal(result, actual),
                'norm_match_error': norm_error, 'mapped_gate_fp64': mapped,
                'delta_formula_fp64': error_metrics(delta, prescribed_delta),
                'max_gain': float((target_mass/mass).max()) if bool((mass > 0).all()) else None,
                'legacy_errors': legacy_errors, **checks}
            self.localization_norms[10] = {'s': cpu(s), 'hypothetical_pc1_norm': cpu(nu),
                'hypothetical_orthogonal_norm': cpu(nv),
                'logit_delta_norm': cpu((target_b-b).T.double().norm(dim=-1)),
                'mass_delta_norm': cpu((target_mass-mass).T.double().norm(dim=-1)),
                'context_delta_norm': cpu(delta.transpose(0, 1).reshape(n-1, self.width).norm(dim=-1)),
                'b_norm': cpu(b.T.double().norm(dim=-1)),
                'target_b_norm': cpu(target_b.T.double().norm(dim=-1)),
                'mass_norm': cpu(mass.T.double().norm(dim=-1)),
                'target_mass_norm': cpu(target_mass.T.double().norm(dim=-1)),
                'actual_context_norm': cpu(c.transpose(0, 1).reshape(n-1, self.width).double().norm(dim=-1)),
                'delivered_context_norm': cpu(delivered.transpose(0, 1).reshape(n-1, self.width).double().norm(dim=-1)),
                'stable_mixture_norm': cpu(stable.transpose(0, 1).reshape(n-1, self.width).double().norm(dim=-1)),
                'source_v_norm': cpu(v0[:, 0].double().norm(dim=-1))}
            if self.save_snapshot:
                self.snapshots[10] = {k: cpu(state[k]) for k in ('q', 'k', 'v', 'mask')}
                self.snapshots[10].update(gate_state=self.gate_cache[10], stable_mixture=cpu(stable),
                    actual_context=cpu(actual), delivered_context=cpu(result))
        self.prefix_contexts[layer] = {'current': cpu(actual[:, :128]), 'delivered': cpu(result[:, :128])}
        self.context_records[layer] = {'current': fingerprint(actual), 'delivered': fingerprint(result)}
        return None if result is actual else (result,)

    def finish(self):
        """Call after every model forward, including baseline/current and suffix."""
        assert self.projection_checks == self.query_checks == self.context_checks == 8
        assert set(self.source_states) == set(range(12)) and all('hidden' in v for v in self.source_states.values())
        assert not self.pending and set(self.gate_checks) == {10}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
