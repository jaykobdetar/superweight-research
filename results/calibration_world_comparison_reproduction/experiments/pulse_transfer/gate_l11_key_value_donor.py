"""T22 prospective earlier-L11 K/V donor hook. No model loading or fitting.

The explicit twelve-arm machine reuses frozen T21 early-support delivery and
T20 interpolation. Source-only candidate: CPU witnesses and model execution
require separate review. All indices are zero-based; later starts at127.
"""
import copy
from contextlib import contextmanager
import numpy as np
import torch
from gate_query_write_timing import GateQueryWriteTiming, LATER_QUERY_START
from gate_write_norm import GateWriteNorm, VARIANTS, byte_equal, row_norm
from gate_native_value import cpu, fingerprint

ARMS = ('baseline', 'current', 'remove_full', 'orth_full',
        'remove_write_matched_early', 'remove_l11_kv',
        'orth_write_matched_early', 'orth_l11_kv',
        'remove_l11_k', 'remove_l11_v', 'orth_l11_k', 'orth_l11_v')
DIRECTIONS = ('remove', 'orth')
DONORS = tuple(d+'_write_matched_early' for d in DIRECTIONS)
RECIPIENTS = tuple(d+'_l11_'+path for d in DIRECTIONS for path in ('kv', 'k', 'v'))
SINGLES = tuple(d+'_l11_'+path for d in DIRECTIONS for path in ('k', 'v'))
TIMING_NODES = ('post_attention', 'mlp_output', 'block_output', 'l11_qkv')
L11_FIELDS = ('c_attn_input', 'original_projection', 'expected_qkv', 'delivered_qkv',
              'attention_context', 'attention_write', 'block_hidden', 'final_hidden')
CLOSURE_FIELDS = ('attention_context', 'attention_write', 'block_hidden', 'final_hidden', 'logits')
L11_NORM_KEYS = ('q_norm', 'k_norm', 'v_norm', 'q_delta_baseline_norm',
                 'k_delta_baseline_norm', 'v_delta_baseline_norm',
                 'q_delta_original_norm', 'k_delta_original_norm', 'v_delta_original_norm')
LAYOUT_VERSION = 'T22_c_attn_Q_K_V_contiguous_allheads_E1_126_v1'


def _owned_cpu(value):
    """Freeze an evidence tree without cloning already-owned CPU array storage.

    Containers are new; CPU tensors/NumPy arrays are attached, not deep-cloned.
    GPU tensors are copied once at this handoff. This is not an alias/dedup codec:
    the unchanged writer still serializes every occurrence in the supplied tree.
    The aborted producer must never mutate/reset these arrays after handoff.
    """
    if isinstance(value, torch.Tensor):
        return value.detach() if value.device.type == 'cpu' else cpu(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {k: _owned_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_owned_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_owned_cpu(v) for v in value)
    return copy.deepcopy(value)


class GateL11KeyValueDonor(GateQueryWriteTiming):
    """Native000 donor/recipient experiment; model(...), capture, then finish().

    reset(...,snapshot=True) retains the L10 panel only. Runner copies the
    purpose-specific l11_current/timing_current panels BEFORE finish. A bound
    model-end observer owns closure verification; no caller-supplied logits can
    substitute for actual model output. This class creates no files.
    """
    def __init__(self, *args, expected_vocab_size=50257, **kwargs):
        assert type(expected_vocab_size) is int and expected_vocab_size > 0
        self.expected_vocab_size = expected_vocab_size
        self.failed = False
        self.output_observer = None
        self.output_events = []
        self.model_output_observed = False
        self.l11_current = {}; self.l11_records = {}; self.l11_norms = {}
        self.baseline_l11 = {}; self.donor_cache = {}; self.closure_references = {}
        self.closure_checks = {}; self.baseline_logits = {}
        self._observed_logits = None
        self._postwrite_operand = None
        self._output_validation_complete = False
        self._identity_hook_ids = set()
        super().__init__(*args, **kwargs)

    def _install(self):
        # Parent installs frozen native observers, L10 write/timing hooks and
        # the dynamically dispatched model-end observer. Prepend below means
        # native.combined sees the actual delivered QKV, never stale baseline.
        super()._install()
        self._model_end_handle = self.handles[-1]
        self.native._fail = self._native_fail
        attn = self.model.transformer.h[11].attn
        self._transplant_handle = attn.c_attn.register_forward_hook(self._transplant, prepend=True)
        self.handles.extend([
            self._transplant_handle,
            attn.c_proj.register_forward_pre_hook(self._l11_context),
            attn.c_proj.register_forward_hook(self._l11_write),
            attn.register_forward_hook(self._l11_attention),
            self.model.transformer.h[11].register_forward_hook(self._l11_block),
            self.model.transformer.ln_f.register_forward_hook(self._final_hidden),
        ])
        if next(iter(attn.c_attn._forward_hooks)) != self._transplant_handle.id:
            self._fail('transplant must prepend before native combined observer')
        owned_ids = {handle.id for handle in self.handles+self.native.handles}
        self._owned_roster = []
        for module in self.model.modules():
            for kind in ('_forward_pre_hooks', '_forward_hooks'):
                self._owned_roster.extend((module, kind, key) for key in getattr(module, kind) if key in owned_ids)
        if len(self._owned_roster) != len(owned_ids): self._fail('owned hook handles missing/duplicated')
        self._installed_roster = self._roster()

    def _roster(self):
        return {(id(module), kind, key) for module in self.model.modules()
                for kind in ('_forward_pre_hooks', '_forward_hooks') for key in getattr(module, kind)}

    def allow_identity_hooks(self, handles):
        """Explicit owned LayerWrites handles for the single identity anchor.

        Caller closes those handles before clearing this allowlist. No arbitrary
        hook addition/removal is admitted during scientific arms.
        """
        if self.failed or self.closed or self.active or self.order_index != 1:
            self._fail('identity hook registration outside anchor boundary')
        ids = {handle.id for handle in handles}
        actual = self._roster()
        additions = actual-self._installed_roster
        if self._installed_roster-actual or {x[2] for x in additions} != ids:
            self._fail('identity hook allowlist must equal actual additional handles')
        allowed_modules = {id(module) for block in self.model.transformer.h
                           for module in (block, block.mlp.c_proj, block.ln_1, block.attn.c_proj)}
        if any(owner not in allowed_modules or kind != '_forward_hooks' for owner, kind, key in additions):
            self._fail('identity writer hooks must be known LayerWrites module outputs')
        self._identity_hook_ids = ids

    def _input(self, module, args, kwargs):
        if self.failed: raise AssertionError('T22 producer frozen after failure')
        actual = self._roster(); additions = actual-self._installed_roster
        allowed = (self.observe_only and self.arm == 'baseline' and self.variant == 'main'
                   and {x[2] for x in additions} == self._identity_hook_ids)
        if (self.native._fail != self._native_fail
                or any(key not in getattr(owner, kind) for owner, kind, key in self._owned_roster)
                or self._installed_roster-actual or (additions and not allowed)
                or (self._identity_hook_ids and not allowed)
                or next(iter(self.model.transformer.h[11].attn.c_attn._forward_hooks)) != self._transplant_handle.id
                or next(reversed(self.model._forward_hooks)) != self._model_end_handle.id):
            self._fail('owned/native failure dispatch or hook roster/order changed')
        # Controlled block identity-writer hooks can coexist for observe_only;
        # the immutable owned handles above must remain installed.
        return GateWriteNorm._input(self, module, args, kwargs)

    def set_output_observer(self, callback):
        if (self.failed or self.closed or (self.active and self._input_seen)
                or (callback is not None and not callable(callback))):
            self._fail('output observer registration state/type')
        self.output_observer = callback

    def _native_fail(self, stage, layer, reason, **arrays):
        # Dispatch BEFORE frozen GateWriteNorm/GateQueryWriteTiming clone_tree.
        # A popped native context's operands arrive here in arrays, intact.
        self.native.failure = {'stage': stage, 'layer': layer, 'reason': str(reason),
                               'arrays': _owned_cpu(arrays)}
        self._fail(reason, native_stage=stage, native_layer=layer)

    def _fail(self, reason, **extra):
        if self.failed:
            raise AssertionError(str(reason))
        self.failed = True
        # Inherited arithmetic passes a local full record as current and a
        # cached full record as reference. Give the actual current record one
        # owner and name the already-owned reference; never clone/cache-union.
        if 'current' in extra and isinstance(extra['current'], dict):
            self.write_snapshot = extra.pop('current')
        if (getattr(self, '_postwrite_operand', None) is not None
                and not any(k in self.write_snapshot for k in ('full_write', 'proposed_full_write'))):
            self.write_snapshot = {**self.write_snapshot, 'full_write': self._postwrite_operand}
        if (self.model_output_observed and not self._output_validation_complete
                and 'actual_logits' not in extra):
            extra['actual_logits'] = self._observed_logits
        if getattr(self, 'write_norms', {}) and 'interpolation' not in self.write_snapshot:
            self.write_snapshot = {**self.write_snapshot, 'interpolation': self.write_norms}
        if 'reference' in extra:
            reference = extra.pop('reference')
            names = [name for name, value in self.write_cache.get(getattr(self, 'variant', None), {}).items()
                     if value is reference]
            if names:
                extra['reference_owner_path'] = ['write_cache', self.variant, names[0]]
            else:
                extra['reference'] = reference
        # Comparison paths are metadata rooted in the complete capsule, never
        # an alias codec or a dependency on an unsaved future state.
        for name, path in tuple(extra.items()):
            if name.endswith('_owner_path') and isinstance(path, list) and path and path[0] not in ('owned', 'extra'):
                extra[name] = ['owned']+path
        native_names = ('snapshots', 'pending', 'source_states', 'gate_cache', 'gate_checks',
                        'localization_norms', 'mask_records', 'effective_masks',
                        'projection_records', 'context_records', 'prefix_contexts', 'diag', 'failure')
        owned = {name: getattr(self, name, {}) for name in (
            'write_cache', 'write_snapshot', 'write_checks',
            'timing_cache', 'timing_current', 'timing_snapshot', 'timing_checks',
            'baseline_l11', 'donor_cache', 'closure_references', 'closure_checks',
            'l11_current', 'l11_records', 'l11_norms')}
        owned.update({'native_'+name: getattr(self.native, name, {}) for name in native_names})
        self.failure = _owned_cpu({
            'reason': str(reason), 'identity': self.identity,
            'boundary': getattr(self, 'boundary', None), 'arm': getattr(self, 'arm', None),
            'variant': getattr(self, 'variant', None), 'order_index': self.order_index,
            'cache_key': getattr(self, 'cache_key', None), 'F_digest': self.params[3],
            'freeze_digest': self.freeze_digest, 'layout_version': LAYOUT_VERSION,
            'model_output_observed': self.model_output_observed,
            'output_events': self.output_events, 'owned': owned, 'extra': extra})
        # GPU baseline logits are not generic failure operands. If actually
        # compared in a failed control, its complete pair is in extra. Retain
        # their object references until close, but never serialize unrelated
        # all-vocabulary caches speculatively.
        self.active = False
        raise AssertionError(str(reason))

    def capture_failure(self, reason, **extra):
        """Durable handoff tree; first failure stays primary, CPU arrays not cloned.

        Runner supplies only its own reached-stage fields/metrics/failed operands,
        not another whole wrapper-cache copy. Same-format writer owns durability.
        """
        if self.failed:
            if extra:
                if 'runner_handoff' in self.failure:
                    raise AssertionError('runner failure evidence already handed off')
                self.failure['runner_handoff'] = _owned_cpu(extra)
                self.failure['runner_reason'] = str(reason)
            raise AssertionError(str(reason))
        self._fail(reason, **extra)

    def start_input(self, metadata, boundary='000'):
        if self.failed or self.closed or self.active:
            self._fail('start_input after failure/close or during active forward')
        if self.identity is not None and self.order_index != 24:
            self._fail('cannot discard incomplete T22 input caches')
        if boundary != '000' or set(metadata) != {'corpus', 'window', 'identity', 'length', 'token_sha256'}:
            self._fail('T22 metadata/native boundary mismatch')
        if (metadata['identity'] not in ('natural', 'bos') or metadata['length'] != self.expected_positions
                or type(metadata['window']) is not int or metadata['window'] < 0
                or not isinstance(metadata['corpus'], str) or not metadata['corpus']
                or not self._sha(metadata['token_sha256'])):
            self._fail('invalid fixed input identity')
        self.identity = copy.deepcopy(metadata); self.boundary = boundary
        self.write_cache = {v: {} for v in VARIANTS}; self.timing_cache = {v: {} for v in VARIANTS}
        self.baseline_l11 = {v: {} for v in VARIANTS}
        self.donor_cache = {d: {} for d in DIRECTIONS}
        self.closure_references = {d: {} for d in DIRECTIONS}
        self.closure_checks = {}; self.baseline_logits = {}; self.output_events = []
        self.variant_hashes = {'main': metadata['token_sha256']}; self.order_index = 0
        self.failure = None

    @staticmethod
    def _sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    def reset(self, boundary, arm, variant='main', token_sha256=None, snapshot=False, observe_only=False):
        if self.failed or self.closed or self.active or self.identity is None:
            self._fail('reset state violation')
        if boundary != '000' or arm not in ARMS or variant not in VARIANTS or not self._sha(token_sha256):
            self._fail('native identity/arm/variant/token mismatch')
        if variant in self.variant_hashes and token_sha256 != self.variant_hashes[variant]:
            self._fail('variant token fingerprint changed')
        if observe_only:
            if not (arm == 'baseline' and variant == 'main' and self.order_index == 1
                    and 'baseline' in self.write_cache['main']):
                self._fail('invalid T22 identity observer')
        elif self.order_index >= 24 or (arm, variant) != (ARMS[self.order_index//2], VARIANTS[self.order_index%2]):
            self._fail('T22 arm/variant order violation')
        if arm in SINGLES:
            if set(self.closure_checks) != {d+'/'+v for d in DIRECTIONS for v in VARIANTS}:
                self._fail('all four actual KV closures required before singles')
            for d in DIRECTIONS:
                for v in VARIANTS:
                    receipt = self.closure_checks[d+'/'+v]
                    donor = self.donor_cache[d].get(v, {})
                    if (receipt.get('cache_key') != donor.get('cache_key') or receipt.get('passed') is not True
                            or receipt.get('all_byte_exact') is not True or set(receipt.get('checks', {})) != set(CLOSURE_FIELDS)
                            or any(x.get('byte_exact') is not True or x.get('actual') != x.get('donor') for x in receipt['checks'].values())
                            or self.closure_references[d]):
                        self._fail('four KV closure receipts/provenance/retirement required before singles')
        if arm in DONORS and not all(k in self.write_cache[variant] for k in ('baseline', 'remove_full', 'orth_full')):
            self._fail('both full proposals required before genuine matched donor')
        if arm in RECIPIENTS and variant not in self.donor_cache[arm.split('_')[0]]:
            self._fail('actual same-variant donor required before recipient')
        self.variant_hashes[variant] = token_sha256
        self.arm, self.variant = arm, variant
        self.observe_only, self.save_snapshot = bool(observe_only), bool(snapshot)
        self.support = 'early' if arm in DONORS else None
        self.cache_key = {**self.identity, 'boundary': '000', 'variant': variant,
            'actual_token_sha256': token_sha256, 'F_digest': self.params[3], 'freeze_digest': self.freeze_digest}
        operation = arm.split('_')[0] if arm in DONORS or arm.endswith('_full') else 'current' if arm == 'current' else 'baseline'
        self.native.reset('000', operation, snapshot=True)
        self.write_norms = {}; self.write_checks = {}; self.write_snapshot = {}; self.prefix_writes = None
        self.timing_current = {}; self.timing_snapshot = {}; self.timing_checks = {
            'support': self.support, 'later_query_start': LATER_QUERY_START,
            'same_full_matched_array_across_supports': self.support is not None}
        self.timing_checks['donor_later_passed'] = False if arm in DONORS else None
        self.l11_current = {}; self.l11_norms = {}; self.l11_records = {
            'cache_key': copy.deepcopy(self.cache_key), 'arm': arm, 'layout_version': LAYOUT_VERSION,
            'head_count': self.native.heads, 'head_dim': self.native.head_dim,
            'earlier_query_slice': [1, LATER_QUERY_START], 'later_query_slice': [LATER_QUERY_START, self.expected_positions],
            'natural_projection_distinct_from_delivered': True, 'native_observer_sees_delivered': True,
            'native_mask_preserved': True, 'closure_required': arm.endswith('_l11_kv')}
        self.l11_records['recipient_upstream_passed'] = False if arm in RECIPIENTS else None
        self.model_output_observed = False
        self._observed_logits = None
        self._output_validation_complete = False
        self._postwrite_operand = self._returned_write = None
        self._l11_attention_seen = False
        self._input_seen = self._pre_seen = self._post_seen = self._attention_seen = False
        self.active = True

    def _postwrite(self, module, args, output):
        # Frozen inner cache/recipe checks can throw without passing their local
        # projected output to _fail. Bind it before delegation; failure attaches
        # it once as the current full_write when no complete record exists yet.
        self._postwrite_operand = output
        returned = super()._postwrite(module, args, output)
        if self.arm in RECIPIENTS:
            if not byte_equal(self._returned_write, self.write_cache[self.variant]['baseline']['full_write']):
                self._fail('recipient actual L10 delivered write must equal baseline',
                    actual_owner_path=['write_snapshot', 'returned_write'],
                    baseline_owner_path=['write_cache', self.variant, 'baseline', 'full_write'])
            self.write_checks['recipient_actual_L10_write_baseline_bytes_exact'] = True
        return returned

    def _field(self, name, output, width=None):
        width = self.native.width if width is None else width
        if (not self.active or name in self.l11_current or not isinstance(output, torch.Tensor)
                or output.dtype != torch.float32 or output.shape != (1, self.expected_positions, width)
                or not bool(torch.isfinite(output).all())):
            self._fail('finite fixed-shape ordered L11 field: '+name, actual=output)
        value = cpu(output); self.l11_current[name] = value
        return value

    def _baseline(self, name):
        record = self.baseline_l11[self.variant]
        value = self.timing_cache[self.variant].get('l11_qkv') if name == 'original_projection' else record.get(name)
        if value is None or fingerprint(value) != record.get('fingerprints', {}).get(name):
            self._fail('baseline L11 reference absent/changed', node=name)
        if record.get('cache_key') != self.cache_key:
            self._fail('baseline L11 provenance mismatch', node=name)
        return value

    @property
    def l11_effective_mask(self):
        """Actual frozen diagnostic L11 mask; never substitute the L10 mask.

        Selected baseline normal payload copies this once. All actual arms bind
        fingerprints and byte guards; failure owns both native current mask and
        the variant baseline mask through explicit, counted fields.
        """
        return self.native.effective_masks[11]

    def _remember_baseline(self, name, value):
        record = self.baseline_l11[self.variant]
        if name != 'original_projection': record[name] = value
        record.setdefault('fingerprints', {})[name] = fingerprint(value)
        record['cache_key'] = copy.deepcopy(self.cache_key)

    def _guard(self, name, actual, baseline, sl=slice(None)):
        exact = byte_equal(actual[:, sl], baseline[:, sl])
        self.l11_records.setdefault('protected', {})[name] = {'byte_exact': exact,
            'query_slice': [sl.start, sl.stop], 'actual': fingerprint(actual[:, sl]),
            'reference': fingerprint(baseline[:, sl])}
        if not exact:
            # Full current is already owned. Only the failed baseline operand
            # is named by its existing cache path; no duplicate whole bundles.
            self._fail('L11 baseline byte guard: '+name, comparison=name,
                       actual_owner_path=['l11_current', name],
                       baseline_owner_path=['timing_cache', self.variant, 'l11_qkv'] if name == 'original_projection' else ['baseline_l11', self.variant, name],
                       query_slice=[sl.start, sl.stop])

    def _transplant(self, module, args, output):
        if not self._post_seen or set(self.timing_current) != set(TIMING_NODES[:3]):
            self._fail('L11 transplant ordering')
        if next(iter(module._forward_hooks)) != self._transplant_handle.id:
            self._fail('L11 prepend hook order changed')
        incoming = self._field('c_attn_input', args[0])
        original = self._field('original_projection', output, 3*self.native.width)
        baseline_run = self.arm == 'baseline' and not self.observe_only
        if baseline_run:
            self._remember_baseline('c_attn_input', incoming)
            self._remember_baseline('original_projection', original)
        else:
            sl = slice(None) if self.arm in RECIPIENTS or self.arm in ('baseline', 'current') else slice(LATER_QUERY_START, None) if self.arm in DONORS else slice(0, 1)
            self._guard('c_attn_input', incoming, self._baseline('c_attn_input'), sl)
            self._guard('original_projection', original, self._baseline('original_projection'), sl)
        mask = self.l11_effective_mask
        if mask.dtype != torch.float32 or mask.shape != (1, 1, self.expected_positions, self.expected_positions):
            self._fail('actual L11 effective diagnostic mask shape/dtype', actual_mask=mask)
        if baseline_run:
            self._remember_baseline('effective_mask', mask)
        elif not byte_equal(mask, self._baseline('effective_mask')):
            self._fail('actual L11 effective mask differs from own baseline',
                actual_owner_path=['native_effective_masks', 11],
                reference_owner_path=['baseline_l11', self.variant, 'effective_mask'])
        self.l11_records['effective_mask'] = fingerprint(mask)
        self.l11_records['effective_mask_baseline_byte_exact'] = True
        expected = original
        returned = output
        if self.arm in RECIPIENTS or self.arm == 'current':
            expected = original.clone()
            path = self.arm.rsplit('_', 1)[-1] if self.arm in RECIPIENTS else 'kv'
            donor = self._donor(self.arm.split('_')[0])['original_projection'] if self.arm in RECIPIENTS else self._baseline('original_projection')
            w = self.native.width
            if 'k' in path: expected[:, 1:LATER_QUERY_START, w:2*w] = donor[:, 1:LATER_QUERY_START, w:2*w]
            if 'v' in path: expected[:, 1:LATER_QUERY_START, 2*w:] = donor[:, 1:LATER_QUERY_START, 2*w:]
            returned = expected.to(device=output.device)
        delivered = cpu(returned)
        self.l11_current['expected_qkv'] = expected
        self.l11_current['delivered_qkv'] = delivered
        if not byte_equal(delivered, expected):
            self._fail('L11 device-copy/splice bytes', expected_owner_path=['l11_current', 'expected_qkv'], actual_owner_path=['l11_current', 'delivered_qkv'])
        w = self.native.width
        protected = byte_equal(delivered[:, :, :w], original[:, :, :w]) and byte_equal(delivered[:, 0], original[:, 0]) and byte_equal(delivered[:, LATER_QUERY_START:], original[:, LATER_QUERY_START:])
        if not protected: self._fail('Q/source/later projection bytes protected')
        if self.arm in RECIPIENTS:
            path = self.arm.rsplit('_', 1)[-1]
            if path == 'k' and not byte_equal(delivered[:, :, 2*w:], original[:, :, 2*w:]): self._fail('K-only recipient V changed')
            if path == 'v' and not byte_equal(delivered[:, :, w:2*w], original[:, :, w:2*w]): self._fail('V-only recipient K changed')
        self.l11_records.update(original_projection=fingerprint(original), delivered_qkv=fingerprint(delivered),
            expected_splice_bytes_exact=True, all_Q_preserved=True, source_projection_preserved=True, later_KV_preserved=True)
        if self.arm in RECIPIENTS: self.l11_records['recipient_upstream_passed'] = True
        base = original if baseline_run else self._baseline('original_projection')
        for label, sl in zip(('q', 'k', 'v'), (slice(0, w), slice(w, 2*w), slice(2*w, 3*w))):
            a, b, o = delivered[0, :, sl], base[0, :, sl], original[0, :, sl]
            for suffix, x in (('_norm', a.double().numpy()), ('_delta_baseline_norm', a.double().numpy()-b.double().numpy()), ('_delta_original_norm', a.double().numpy()-o.double().numpy())):
                self.l11_norms[label+suffix] = torch.from_numpy(row_norm(x).copy())
        self.l11_records['norm_scope'] = 'actual_delivered_QKV; deltas_from_own_baseline_and_original_natural_projection'
        self.l11_records['active_summary'] = {'query_slice': [1, LATER_QUERY_START],
            'query_count': LATER_QUERY_START-1,
            'norms': {name: {'mean': float(value[1:LATER_QUERY_START].mean()),
                           'RMS': float(value[1:LATER_QUERY_START].square().mean().sqrt()),
                           'max': float(value[1:LATER_QUERY_START].max())}
                      for name, value in self.l11_norms.items()}}
        return None if returned is output else returned

    def _timing(self, name, output):
        width = 3*self.native.width if name == 'l11_qkv' else self.native.width
        if (not self.active or not self._post_seen or name in self.timing_current
                or not isinstance(output, torch.Tensor) or output.dtype != torch.float32
                or output.shape != (1, self.expected_positions, width) or not bool(torch.isfinite(output).all())):
            self._fail('finite fixed-shape ordered timing node', node=name, actual=output)
        value = cpu(output); self.timing_current[name] = value
        reference = self.timing_cache[self.variant]
        if self.arm == 'baseline' and not self.observe_only:
            reference[name] = value
        else:
            if name not in reference: self._fail('missing baseline timing reference', node=name)
            sl = (slice(None) if self.arm in ('baseline', 'current') or (self.arm in RECIPIENTS and name != 'l11_qkv')
                  else slice(LATER_QUERY_START, None) if self.arm in DONORS else slice(0, 1))
            exact = byte_equal(value[:, sl], reference[name][:, sl])
            self.timing_checks[name] = {'query_slice': [sl.start, sl.stop], 'byte_exact': exact,
                'actual': fingerprint(value[:, sl]), 'baseline': fingerprint(reference[name][:, sl])}
            if not exact:
                self._fail('timing baseline byte guard: '+name, node=name,
                    actual_owner_path=['timing_current', name], baseline_owner_path=['timing_cache', self.variant, name])
        if name == 'l11_qkv' and not byte_equal(value, self.l11_current.get('delivered_qkv')):
            self._fail('timing must observe actual post-transplant QKV')
        self.timing_checks.setdefault('observed_nodes', []).append(name)
        if name == 'l11_qkv' and self.arm in DONORS:
            self.timing_checks['donor_later_passed'] = True

    def _l11_context(self, module, args):
        if set(self.timing_current) != set(TIMING_NODES): self._fail('L11 context before delivered-QKV observer')
        self._field('attention_context', args[0])

    def _l11_write(self, module, args, output):
        if 'attention_context' not in self.l11_current or not byte_equal(args[0], self.l11_current['attention_context']):
            self._fail('L11 actual c_proj context changed', actual=args[0])
        self._field('attention_write', output)

    def _l11_attention(self, module, args, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        if self._l11_attention_seen or not byte_equal(value, self.l11_current.get('attention_write')):
            self._fail('L11 delivered attention write differs from actual c_proj', actual=value)
        self._l11_attention_seen = True

    def _l11_block(self, module, args, output):
        if not self._l11_attention_seen: self._fail('L11 block before actual attention write')
        self._field('block_hidden', output[0] if isinstance(output, (tuple, list)) else output)

    def _final_hidden(self, module, args, output):
        if 'block_hidden' not in self.l11_current: self._fail('final hidden before L11 block')
        self._field('final_hidden', output)

    def _donor(self, direction):
        record = self.donor_cache[direction].get(self.variant)
        expected_key = {**self.cache_key, 'direction': direction, 'donor_arm': direction+'_write_matched_early',
                        'support': 'early', 'layout_version': LAYOUT_VERSION}
        if (record is None or record['cache_key'] != expected_key
                or fingerprint(record['original_projection']) != record['fingerprint']):
            self._fail('donor QKV cache provenance/bytes mismatch', direction=direction)
        return record

    @staticmethod
    def _device_bytes_equal(a, b):
        return (a.dtype == b.dtype and a.shape == b.shape and a.device == b.device
                and bool(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))))

    def _readout_observer(self, module, args, output):
        if not self.active or self.model_output_observed:
            self._fail('actual model-output observer ordering')
        # Own the actual raw logits BEFORE any journal callback can throw.
        # Successful normal records do not serialize this full tensor.
        self._observed_logits = getattr(output, 'logits', None)
        self.model_output_observed = True
        event = {'event': 'model_output_observed', 'arm': self.arm, 'variant': self.variant,
                 'actual_token_sha256': self.cache_key['actual_token_sha256'],
                 'cache_key': copy.deepcopy(self.cache_key), 'observe_only': self.observe_only}
        self.output_events.append(event)
        if self.output_observer is not None:
            try: self.output_observer(copy.deepcopy(event))
            except BaseException as error:
                self._fail('model-output observation callback: '+str(error), actual_logits=self._observed_logits)
        # An output was observed even if its validation/closure now throws;
        # runner's model-returned counter must remain separate.
        logits = self._observed_logits
        if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32 or logits.ndim != 3
                or logits.shape != (1, self.expected_positions, self.expected_vocab_size) or not bool(torch.isfinite(logits).all())):
            self._fail('actual finite fixed-shape FP32 model readout required', actual_logits=logits)
        if set(self.l11_current) != set(L11_FIELDS): self._fail('incomplete actual L11/end-state observers')
        self._output_validation_complete = True
        if self.arm == 'baseline' and not self.observe_only:
            for name in ('attention_context', 'attention_write', 'block_hidden', 'final_hidden'):
                self._remember_baseline(name, self.l11_current[name])
            self.baseline_logits[self.variant] = {'tensor': logits[0].detach(), 'fingerprint': fingerprint(logits[0]),
                                                'cache_key': copy.deepcopy(self.cache_key)}
        else:
            record = self.baseline_logits.get(self.variant)
            if record is None or record['cache_key'] != self.cache_key or fingerprint(record['tensor']) != record['fingerprint']:
                self._fail('baseline actual readout reference absent/changed',
                    baseline_logits=None if record is None else record['tensor'],
                    expected_fingerprint=None if record is None else record['fingerprint'])
            control = self.arm in ('baseline', 'current')
            sl = slice(None) if control else slice(0, 1)
            if not self._device_bytes_equal(logits[0, sl], record['tensor'][sl]):
                self._fail('actual identity/source model readout bytes', actual_logits=logits[0, sl], baseline_logits=record['tensor'][sl])
            self.l11_records['actual_readout_control'] = {'byte_exact': True, 'query_slice': [sl.start, sl.stop]}
            for name in ('attention_context', 'attention_write', 'block_hidden', 'final_hidden'):
                self._guard(name, self.l11_current[name], self._baseline(name), slice(None) if control else slice(0, 1))
        if self.arm in DONORS:
            direction = self.arm.split('_')[0]
            if self.closure_references[direction] and self.variant in self.closure_references[direction]:
                self._fail('donor closure reference overwrite')
            projection = self.l11_current['original_projection']
            self.donor_cache[direction][self.variant] = {'original_projection': projection,
                'fingerprint': fingerprint(projection), 'cache_key': {**self.cache_key, 'direction': direction,
                    'donor_arm': self.arm, 'support': 'early', 'layout_version': LAYOUT_VERSION}}
            refs = {name: cpu(self.l11_current[name][:, LATER_QUERY_START:]) for name in CLOSURE_FIELDS[:-1]}
            refs['logits'] = cpu(logits[0, LATER_QUERY_START:])
            self.closure_references[direction][self.variant] = {'arrays': refs,
                'fingerprints': {k: fingerprint(v) for k, v in refs.items()},
                'cache_key': copy.deepcopy(self.donor_cache[direction][self.variant]['cache_key'])}
        elif self.arm.endswith('_l11_kv'):
            self._verify_closure(logits[0])
        self.l11_records['fields'] = {name: fingerprint(value) for name, value in self.l11_current.items()}
        self.l11_records['actual_model_output_observed'] = True
        self.l11_records['actual_output_shape_dtype_finite_validated'] = True

    def _verify_closure(self, logits):
        direction = self.arm.split('_')[0]; donor = self._donor(direction)
        ref = self.closure_references[direction].get(self.variant)
        if ref is None or ref['cache_key'] != donor['cache_key']:
            self._fail('actual donor closure references absent/provenance mismatch')
        checks = {}
        for name in CLOSURE_FIELDS:
            expected = ref['arrays'][name]
            if fingerprint(expected) != ref['fingerprints'][name]: self._fail('actual donor closure reference changed', node=name)
            actual = cpu(logits[LATER_QUERY_START:]) if name == 'logits' else self.l11_current[name][:, LATER_QUERY_START:]
            checks[name] = {'byte_exact': byte_equal(actual, expected),
                            'actual': fingerprint(actual), 'donor': fingerprint(expected)}
            if not checks[name]['byte_exact']:
                # Actual non-logit full operands are in l11_current. Only the
                # logits operand is not otherwise a CPU owner and is attached.
                self._fail('actual later combined-KV closure: '+name, closure_node=name,
                    closure_actual_logits=actual if name == 'logits' else None,
                    actual_owner_path=['l11_current', name] if name != 'logits' else ['extra', 'closure_actual_logits'],
                    reference_owner_path=['closure_references', direction, self.variant, 'arrays', name],
                    reached_closure_checks=checks)
        key = direction+'/'+self.variant
        if key in self.closure_checks: self._fail('closure receipt overwrite')
        self.closure_checks[key] = {'direction': direction, 'variant': self.variant,
            'cache_key': copy.deepcopy(donor['cache_key']), 'query_slice': [LATER_QUERY_START, self.expected_positions],
            'checks': checks, 'actual_model_output_observed': True, 'all_byte_exact': True,
            'passed': True,
            'references_retired_after_all_checks': True}
        self.l11_records['combined_KV_closure'] = copy.deepcopy(self.closure_checks[key])
        # Prospective successful lifetime end, never a saved-file deletion. A
        # failure above leaves every expected/actual operand live and owned.
        self.closure_references[direction].pop(self.variant)

    def finish(self):
        if self.failed or not self.active or not self.model_output_observed:
            self._fail('finish requires actual model-output observation')
        if set(self.timing_current) != set(TIMING_NODES) or set(self.l11_current) != set(L11_FIELDS):
            self._fail('incomplete timing/L11 observations')
        if set(self.l11_norms) != set(L11_NORM_KEYS): self._fail('incomplete nine L11 norm arrays')
        if self.arm.endswith('_l11_kv') and self.arm.split('_')[0]+'/'+self.variant not in self.closure_checks:
            self._fail('finish requires actual combined-KV closure receipt')
        # Explicit24-call state, no T20/T21 order spoofing. The frozen base
        # finish only checks native/source/write completeness and increments1.
        saved = self.save_snapshot
        # Keep reached-state operands until the NEXT reset. A runner resource
        # guard/journal can fail after finish; its capsule still needs actual
        # full current L10/L11/timing arrays, not merely normal prefix samples.
        self.save_snapshot = True
        try: GateWriteNorm.finish(self)
        finally: self.save_snapshot = saved
        self._observed_logits = None

    @contextmanager
    def unhooked(self):
        if self.failed: raise AssertionError('T22 producer frozen after failure')
        # Dynamic _install restores dispatch, prepend order and owned roster.
        with GateWriteNorm.unhooked(self):
            yield

