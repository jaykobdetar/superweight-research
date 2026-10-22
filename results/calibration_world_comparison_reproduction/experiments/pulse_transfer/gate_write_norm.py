"""T20 post-c_proj write interpolation. No model loading or fitting.

Composes the immutable T19 hook. All matching arithmetic is explicit rowwise
NumPy CPU FP64; the only float32 rounding is the final conversion of interior
rows. Endpoint/source and device-copy checks compare raw bytes, including -0.
"""
from contextlib import contextmanager
import copy
import hashlib
import math
import numpy as np
import torch
from gate_native_value import GateNativeValue, cpu, fingerprint

ARMS = ('baseline', 'current', 'remove_full', 'orth_full',
        'remove_write_matched', 'orth_write_matched')
VARIANTS = ('main', 'suffix')
U32, ETA32, U64 = 2.**-24, 2.**-149, 2.**-53
RMS_RELATIVE_LIMIT = 1e-5


def byte_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and fingerprint(a) == fingerprint(b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(byte_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(byte_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def clone_tree(value):
    if isinstance(value, torch.Tensor): return cpu(value)
    if isinstance(value, np.ndarray): return value.copy()
    if isinstance(value, dict): return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [clone_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def row_norm(value):
    x = np.asarray(value, dtype=np.float64)
    return np.sqrt(np.sum(x*x, axis=-1, dtype=np.float64))


def runtime_behavior():
    tiny = np.array([2.**-149, -2.**-149, 0., -0.], dtype=np.float64).astype(np.float32)
    assert tiny.view(np.uint32).tolist() == [1, 0x80000001, 0, 0x80000000], 'subnormal/signed-zero conversion'
    tie = np.array([1+2.**-24, 1+3*2.**-24], dtype=np.float64).astype(np.float32)
    assert tie.view(np.uint32).tolist() == [0x3f800000, 0x3f800002], 'round-to-nearest-even conversion'
    return {'numpy_version': np.__version__, 'torch_version': torch.__version__,
            'subnormal_and_signed_zero_conversion': True, 'round_to_nearest_even_witness': True,
            'cpu_fp64_row_sum_squares': True, 'interior_multiply_then_add_then_one_float32_cast': True}


def interpolate_writes(base, left, right):
    """Pure arithmetic helper; returns both candidates and complete diagnostics.

    Input shape is queries x width, float32. Production uses512x768; smaller
    explicit fake architectures are useful for hook fixtures only.
    """
    assert base.ndim == 2 and base.shape == left.shape == right.shape and base.shape[0] >= 3
    assert all(x.dtype == np.float32 and np.isfinite(x).all() for x in (base, left, right)), 'finite float32 writes'
    assert byte_equal(base[0], left[0]) and byte_equal(base[0], right[0]), 'full source write bytes'
    q, width = base.shape
    b = base.astype(np.float64)
    delta = [x.astype(np.float64)-b for x in (left, right)]
    norms = [row_norm(x) for x in delta]
    target = np.minimum(norms[0], norms[1])
    k64 = (32*width*U64)/(1-32*width*U64)
    values, scales, bounds, actual = [], [], [], []
    for full, dd, nn in zip((left, right), delta, norms):
        scale = np.zeros(q, dtype=np.float64)
        positive = nn > 0
        scale[positive] = target[positive]/nn[positive]
        assert np.isfinite(scale).all() and ((scale >= 0) & (scale <= 1)).all(), 'lambda range'
        value = base.copy()
        one = scale == 1
        middle = (scale > 0) & ~one
        value[one] = full[one]
        product = np.multiply(scale[middle, None], dd[middle])
        summed = np.add(b[middle], product)
        value[middle] = summed.astype(np.float32)
        assert byte_equal(value[scale == 0], base[scale == 0]), 'lambda zero exact bytes'
        assert byte_equal(value[one], full[one]), 'lambda one exact bytes'
        assert byte_equal(value[0], base[0]), 'source exact bytes'
        assert np.isfinite(value).all(), 'finite returned write'
        bound = U32*row_norm(np.abs(b)+scale[:, None]*np.abs(dd)) + .5*math.sqrt(width)*ETA32 + k64*(row_norm(b)+nn+target)
        values.append(value); scales.append(scale); bounds.append(bound)
        actual.append(row_norm(value.astype(np.float64)-b))
    result = {'target': target, 'values': values, 'scales': scales, 'bounds': bounds,
              'actual_norms': actual, 'full_norms': norms, 'deltas': delta,
              'width': width, 'kappa64': k64}
    try:
        result['quality'] = audit_matching(result)
    except AssertionError as error:
        error.matching_evidence = result
        raise
    return result


def audit_matching(result, actual_norms=None):
    """Frozen analytical guards AND distinct all/scored RMS quality guards."""
    target = result['target']; norms = result['actual_norms'] if actual_norms is None else actual_norms
    assert target.ndim == 1 and len(target) >= 3
    assert all(x.shape == target.shape and np.isfinite(x).all() for x in (*norms, *result['bounds'], target))
    a, b = norms; ba, bb = result['bounds']; gap = np.abs(a-b)
    for value, bound in zip(norms, (ba, bb)):
        assert np.all(np.abs(value-target) <= bound), 'per-arm analytical norm bound'
    assert np.all(gap <= ba+bb), 'paired analytical norm bound'
    domains = {}
    for name, sl in (('all_nonsource', slice(1, len(target))), ('primary_scored', slice(1, len(target)-1))):
        rms_target = float(np.sqrt(np.mean(target[sl]*target[sl], dtype=np.float64)))
        rms_gap = float(np.sqrt(np.mean(gap[sl]*gap[sl], dtype=np.float64)))
        if rms_target == 0:
            assert np.count_nonzero(a[sl]) == np.count_nonzero(b[sl]) == 0, name+' zero RMS exact'
        else:
            assert rms_gap <= RMS_RELATIVE_LIMIT*rms_target, name+' RMS norm-match quality'
        domains[name] = {'RMS_target': rms_target, 'RMS_gap': rms_gap,
                         'relative_RMS_gap': rms_gap/rms_target if rms_target else None}
    positive = target > 0
    relative = np.full_like(target, np.nan)
    relative[positive] = gap[positive]/target[positive]
    return {'domains': domains, 'maximum_positive_target_relative_gap': float(relative[positive].max()) if positive.any() else None,
            'quantized_to_zero_positive_targets': [int(np.count_nonzero((x == 0) & positive)) for x in norms],
            'zero_target_queries': int(np.count_nonzero(~positive)),
            'maximum_absolute_pair_gap': float(gap.max()),
            'maximum_per_arm_target_error': max(float(np.abs(x-target).max()) for x in norms)}


class GateWriteNorm:
    """Same-state frozen full edits followed by actual attention-write matching.

    start_input(metadata,boundary) binds one identity and clears both variants.
    reset(boundary,arm,variant,token_sha256,snapshot=False,observe_only=False)
    follows arm-major main/suffix order. Call finish after each complete forward.
    unhooked() removes all native and post-write hooks for the native repeat.
    """
    def __init__(self, model, alpha, u, v, digest, freeze_digest, *, expected_positions=512):
        assert isinstance(freeze_digest, str) and freeze_digest
        self.model = model; self.params = (cpu(alpha), cpu(u), cpu(v), digest)
        self.freeze_digest = freeze_digest; self.expected_positions = expected_positions
        self.runtime = runtime_behavior(); self.handles = []; self.active = False; self.closed = False
        self.write_cache = {}; self.identity = None; self.failure = None; self.order_index = 0
        self._install()

    def _install(self):
        self.native = GateNativeValue(self.model, *self.params, expected_positions=self.expected_positions)
        self.handles = [self.model.register_forward_pre_hook(self._input, with_kwargs=True),
            self.model.transformer.h[10].attn.c_proj.register_forward_pre_hook(self._prewrite),
            self.model.transformer.h[10].attn.c_proj.register_forward_hook(self._postwrite),
            self.model.transformer.h[10].attn.register_forward_hook(self._attention_write)]

    def __getattr__(self, name):
        native = self.__dict__.get('native')
        if native is not None: return getattr(native, name)
        raise AttributeError(name)

    def _fail(self, reason, **extra):
        # Preserve COMPLETE caches and current native tensors, not a subset of
        # mismatches. Writer must account for this full worst-case tree.
        self.failure = {'reason': str(reason), 'identity': clone_tree(self.identity),
            'boundary': self.boundary if self.identity else None,
            'arm': getattr(self, 'arm', None), 'variant': getattr(self, 'variant', None),
            'freeze_digest': self.freeze_digest, 'F_digest': self.params[3],
            'order_index': self.order_index, 'write_cache': clone_tree(self.write_cache),
            'current': clone_tree(getattr(self, 'write_snapshot', {})),
            'native_snapshots': clone_tree(self.native.snapshots),
            'native_pending': clone_tree(self.native.pending),
            'native_source_states': clone_tree(self.native.source_states),
            'native_gate_cache': clone_tree(self.native.gate_cache),
            'native_gate_checks': clone_tree(self.native.gate_checks),
            'native_localization_norms': clone_tree(self.native.localization_norms),
            'native_mask_records': clone_tree(self.native.mask_records),
            'native_effective_masks': clone_tree(self.native.effective_masks),
            'native_projection_records': clone_tree(self.native.projection_records),
            'native_context_records': clone_tree(self.native.context_records),
            'native_prefix_contexts': clone_tree(self.native.prefix_contexts),
            'native_diag': clone_tree(self.native.diag),
            'native_failure': clone_tree(self.native.failure), 'extra': clone_tree(extra)}
        raise AssertionError(str(reason))

    def start_input(self, metadata, boundary):
        assert not self.closed and not self.active
        if self.identity is not None and self.order_index != 12:
            self._fail('cannot discard incomplete boundary/input caches')
        assert boundary in ('000', '111')
        assert set(metadata) == {'corpus', 'window', 'identity', 'length', 'token_sha256'}
        assert metadata['identity'] in ('natural', 'bos') and metadata['length'] == self.expected_positions
        assert type(metadata['window']) is int and metadata['window'] >= 0
        assert isinstance(metadata['corpus'], str) and metadata['corpus']
        assert isinstance(metadata['token_sha256'], str) and len(metadata['token_sha256']) == 64
        self.identity = copy.deepcopy(metadata); self.boundary = boundary
        self.write_cache = {'main': {}, 'suffix': {}}; self.variant_hashes = {'main': metadata['token_sha256']}
        self.order_index = 0; self.failure = None

    def reset(self, boundary, arm, variant='main', token_sha256=None, snapshot=False, observe_only=False):
        assert not self.closed and not self.active and self.identity is not None
        if boundary != self.boundary or arm not in ARMS or variant not in VARIANTS:
            self._fail('identity/boundary/arm/variant mismatch')
        if not isinstance(token_sha256, str) or len(token_sha256) != 64:
            self._fail('actual token fingerprint required')
        if variant in self.variant_hashes and token_sha256 != self.variant_hashes[variant]:
            self._fail('variant token fingerprint changed')
        if observe_only:
            if not (boundary == '000' and arm == 'baseline' and variant == 'main' and 'baseline' in self.write_cache['main']):
                self._fail('invalid writer-identity observer')
        else:
            if self.order_index >= 12 or (arm, variant) != (ARMS[self.order_index//2], VARIANTS[self.order_index%2]):
                self._fail('arm/variant order violation')
        self.variant_hashes[variant] = token_sha256
        if arm.endswith('write_matched') and not all(k in self.write_cache[variant] for k in ('baseline', 'remove_full', 'orth_full')):
            self._fail('both full candidates and baseline required')
        self.arm, self.variant, self.observe_only, self.save_snapshot = arm, variant, observe_only, bool(snapshot)
        self.cache_key = {**self.identity, 'boundary': boundary, 'variant': variant,
            'actual_token_sha256': token_sha256, 'F_digest': self.params[3], 'freeze_digest': self.freeze_digest}
        operation = 'remove' if arm.startswith('remove') else 'orth' if arm.startswith('orth') else arm
        self.native.reset(boundary, operation, snapshot=True)
        self.write_norms = {}; self.write_checks = {}; self.write_snapshot = {}; self.prefix_writes = None
        self._input_seen = self._pre_seen = self._post_seen = self._attention_seen = False; self.active = True

    def _input(self, module, args, kwargs):
        if not self.active or self._input_seen: self._fail('model invocation without one reset')
        tokens = kwargs.get('input_ids', args[0] if args else None)
        if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or tokens.shape != (1, self.expected_positions):
            self._fail('actual input_ids int64 fixed shape required', input_ids=tokens)
        if fingerprint(tokens)['sha256'] != self.cache_key['actual_token_sha256']:
            self._fail('actual input_ids fingerprint mismatch', input_ids=tokens)
        self._input_seen = True

    def _prewrite(self, module, args):
        if not self.active or not self._input_seen or self._pre_seen: self._fail('pre-write hook ordering')
        snap = self.native.snapshots.get(10)
        if snap is None or not byte_equal(args[0], snap['delivered_context']):
            self._fail('frozen native context hook must run before write observer', proposed=args[0])
        self.write_snapshot = {'incoming_reference': clone_tree(snap),
            'mask_reference': clone_tree(self.native.mask_records[10]), 'cache_key': copy.deepcopy(self.cache_key)}
        self._pre_seen = True

    def _postwrite(self, module, args, output):
        if not self._pre_seen or self._post_seen: self._fail('post-write hook ordering')
        if not byte_equal(args[0], self.write_snapshot['incoming_reference']['delivered_context']):
            self._fail('projection input changed after native/pre-write hook', actual_projection_input=args[0])
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or output.shape != (1, self.expected_positions, self.native.width) or not bool(torch.isfinite(output).all()):
            self._fail('finite FP32 fixed-shape projected write required', output=output)
        cache = self.write_cache[self.variant]; full = cpu(output); record = clone_tree(self.write_snapshot)
        record['full_write'] = full
        if self.arm == 'baseline' and not self.observe_only:
            cache['baseline'] = record
        else:
            base = cache['baseline']; incoming = record['incoming_reference']; reference = base['incoming_reference']
            if record['cache_key'] != base['cache_key']:
                self._fail('baseline cache identity/freeze mismatch', current=record)
            for key in ('q', 'k', 'v', 'mask', 'actual_context', 'stable_mixture'):
                if not byte_equal(incoming[key], reference[key]): self._fail('baseline incoming-state identity: '+key, current=record)
            for key in ('b', 'mass', 's'):
                if not byte_equal(incoming['gate_state'][key], reference['gate_state'][key]):
                    self._fail('baseline incoming-gate identity: '+key, current=record)
            if not byte_equal(record['mask_reference'], base['mask_reference']): self._fail('native mask identity', current=record)
            if not byte_equal(full[:, 0], base['full_write'][:, 0]): self._fail('source projected write bytes', current=record)
            if self.arm in ('baseline', 'current'):
                if not byte_equal(full, base['full_write']): self._fail('baseline/current/observer write bytes', current=record)
            elif self.arm.endswith('_full'):
                cache[self.arm] = record
        returned = output
        if self.arm.endswith('write_matched'):
            refname = 'remove_full' if self.arm.startswith('remove') else 'orth_full'; which = 0 if refname == 'remove_full' else 1
            if not byte_equal(record, cache[refname]): self._fail('cached same-state full proposal/write identity', current=record, reference=cache[refname])
            if 'match' not in cache:
                try:
                    cache['match'] = interpolate_writes(*(cache[k]['full_write'][0].numpy() for k in ('baseline', 'remove_full', 'orth_full')))
                except (AssertionError, FloatingPointError) as error:
                    self._fail(error, current=record, arithmetic=getattr(error, 'matching_evidence', None))
            matching = cache['match']; desired = matching['values'][which]
            returned = torch.from_numpy(desired.copy()).unsqueeze(0).to(device=output.device)
            actual = cpu(returned)
            if not byte_equal(actual[0].numpy(), desired): self._fail('device round-trip bytes', expected=desired, actual=actual)
            cache.setdefault('delivered', {})[self.arm] = actual
            actual_norm = row_norm(actual[0].numpy().astype(np.float64)-cache['baseline']['full_write'][0].numpy().astype(np.float64))
            if not byte_equal(actual_norm, matching['actual_norms'][which]): self._fail('actual delivered norm bytes', actual_norm=actual_norm)
            pair_complete = len(cache['delivered']) == 2
            if pair_complete:
                actual_pair = [row_norm(cache['delivered'][k][0].numpy().astype(np.float64)-cache['baseline']['full_write'][0].numpy().astype(np.float64)) for k in ARMS[-2:]]
                try: quality = audit_matching(matching, actual_pair)
                except AssertionError as error: self._fail(error, actual_norms=actual_pair)
            else: quality = matching['quality']
            self.write_checks.update(matching_quality=quality, paired_actual_delivery_complete=pair_complete,
                counterpart_norm_scope='actual_delivered' if pair_complete else 'expected_CPU_rounded_not_yet_delivered')
            self.write_norms.update({k: torch.from_numpy(x.copy()) for k, x in {
                'target_norm': matching['target'], 'lambda': matching['scales'][which],
                'own_full_norm': matching['full_norms'][which], 'other_full_norm': matching['full_norms'][1-which],
                'rounding_budget': matching['bounds'][which], 'other_rounding_budget': matching['bounds'][1-which],
                'signed_target_residual': actual_norm-matching['target'],
                'absolute_pair_gap': np.abs(matching['actual_norms'][0]-matching['actual_norms'][1])}.items()})
        base = cache['baseline']['full_write']; delivered = cpu(returned)
        snap = record['incoming_reference']
        self.write_norms.update({k: torch.from_numpy(row_norm(x).copy()) for k, x in {
            'actual_write_norm': delivered[0].numpy(),
            'actual_write_delta_norm': delivered[0].double().numpy()-base[0].double().numpy(),
            'proposed_full_write_delta_norm': full[0].double().numpy()-base[0].double().numpy(),
            'proposed_pre_context_delta_norm': snap['delivered_context'][0].double().numpy()-snap['actual_context'][0].double().numpy()}.items()})
        self.write_checks.update(source_write_bytes_exact=byte_equal(delivered[:, 0], base[:, 0]),
            proposed_full_write=fingerprint(full), returned_write=fingerprint(delivered), baseline_write=fingerprint(base),
            actual_norms_include_q0=True, source_q0_norm=float(self.write_norms['actual_write_delta_norm'][0]),
            proposed_context_and_actual_write_distinct=True, native_mask_preserved=True)
        self.write_snapshot.update(baseline_write=cpu(base), proposed_full_write=full, returned_write=delivered,
                                   interpolation=clone_tree(self.write_norms))
        self.prefix_writes = cpu(delivered[:, :128]); self._post_seen = True
        self._returned_write = delivered
        return None if returned is output else returned

    def _attention_write(self, module, args, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if not self._post_seen or self._attention_seen or not byte_equal(value, self._returned_write):
            self._fail('actual attention output differs from returned c_proj write', actual_attention_output=value,
                       expected_write=getattr(self, '_returned_write', None))
        self._attention_seen = True

    def finish(self):
        if not self.active or not self._input_seen or not self._pre_seen or not self._post_seen or not self._attention_seen: self._fail('incomplete model/write hooks')
        try: self.native.finish()
        except AssertionError as error: self._fail(error)
        cache = self.write_cache[self.variant]
        if self.arm == 'baseline' and not self.observe_only:
            cache['baseline']['source_states'] = clone_tree(self.native.source_states)
        elif not byte_equal(self.native.source_states, cache['baseline']['source_states']):
            self._fail('all-layer source QKV/hidden byte identity', source_states=self.native.source_states)
        self.active = False
        if not self.observe_only: self.order_index += 1
        if not self.save_snapshot:
            self.write_snapshot = {}
            self.native.snapshots = {}

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles = []; self.native.close(); self.closed = True

    def capture_failure(self, reason, **extra):
        """Runner exception path: preserve frozen-hook failures plus all caches.

        This raises after constructing .failure. It never drops arrays or truncates
        them to fit a storage allowance; the complete writer must fail closed.
        """
        self._fail(reason, **extra)

    @contextmanager
    def unhooked(self):
        assert not self.closed and not self.active
        if self.identity is None or self.boundary != '000' or self.order_index != 1:
            self._fail('unhooked anchor only after000 baseline main')
        for handle in self.handles: handle.remove()
        self.handles = []; self.native.close()
        try: yield
        finally: self._install()
