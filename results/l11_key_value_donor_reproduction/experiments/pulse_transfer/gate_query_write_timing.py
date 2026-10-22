"""T21 prospective native000 support gate. Implementation-only, no model loading.

Uses immutable T20 arithmetic once per main/suffix variant. Selecting support
copies already-rounded matched rows; it never refits/recomputes matching.
Early queries1..126, later127..last (production511); sourceq0 always baseline.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from gate_write_norm import (GateWriteNorm, VARIANTS, byte_equal, clone_tree,
                             interpolate_writes, audit_matching, row_norm)
from gate_native_value import cpu, fingerprint

ARMS = ('baseline', 'current', 'remove_full', 'orth_full',
        'remove_write_matched_early', 'remove_write_matched_later', 'remove_write_matched_both',
        'orth_write_matched_early', 'orth_write_matched_later', 'orth_write_matched_both')
SUPPORTS = ('early', 'later', 'both')
LATER_QUERY_START = 127


def support_mask(positions, support):
    assert positions >= 129 and support in SUPPORTS
    mask = np.zeros(positions, dtype=bool)
    if support in ('early', 'both'): mask[1:LATER_QUERY_START] = True
    if support in ('later', 'both'): mask[LATER_QUERY_START:] = True
    return mask


def splice_support(base, matched, support):
    """Copy fixed FP32 rows, preserving signed zero/subnormal bytes exactly."""
    assert base.dtype == matched.dtype == np.float32 and base.ndim == 2 and base.shape == matched.shape
    assert np.isfinite(base).all() and np.isfinite(matched).all()
    assert byte_equal(base[0], matched[0]), 'matched source bytes'
    active = support_mask(len(base), support)
    out = base.copy(); out[active] = matched[active]
    assert byte_equal(out[~active], base[~active]) and byte_equal(out[active], matched[active])
    if support == 'both': assert byte_equal(out, matched), 'both is exact full matched array'
    return out, active


def arithmetic_commitment(value):
    """Hash all fixed recipe bytes without retaining a duplicate array tree."""
    def encode(x):
        if isinstance(x, np.ndarray):
            return {'array_dtype': str(x.dtype), 'shape': list(x.shape),
                    'sha256': hashlib.sha256(x.tobytes()).hexdigest()}
        if isinstance(x, dict): return {str(k):encode(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [encode(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class GateQueryWriteTiming(GateWriteNorm):
    """T21 10-arm wrapper; inherited frozen native/T20 hooks remain unmodified.

    start_input(metadata,boundary='000'), reset(boundary,arm,variant,...),
    model(...), verify_readout(logits,own_baseline_logits) for later-only arms,
    finish(). Logit tensors must be [positions,vocab] FP32. An absent later-only
    readout check is a hard failure. Runner still owns all other scoring/control
    checks and evidence serialization; this class creates no files.
    """
    def __init__(self, *args, **kwargs):
        positions = kwargs.get('expected_positions',512)
        assert type(positions) is int and positions >= 129
        self.timing_cache = {}; self.timing_current = {}; self.timing_snapshot = {}
        super().__init__(*args, **kwargs)

    def _install(self):
        block = self.model.transformer.h[10]
        assert hasattr(block, 'ln_2') and hasattr(block, 'mlp'), 'GPT2 rowwise LN2/MLP required'
        super()._install()
        self.handles.extend([
            block.ln_2.register_forward_pre_hook(lambda m, a: self._timing('post_attention', a[0])),
            block.mlp.register_forward_hook(lambda m, a, o: self._timing('mlp_output', o)),
            block.register_forward_hook(lambda m, a, o: self._timing('block_output', o[0] if isinstance(o, (tuple, list)) else o)),
            self.model.transformer.h[11].attn.c_attn.register_forward_hook(lambda m, a, o: self._timing('l11_qkv', o)),
            self.model.register_forward_hook(self._readout_observer),
        ])

    def _fail(self, reason, **extra):
        try: super()._fail(reason, **extra)
        finally:
            if isinstance(getattr(self, 'failure', None), dict):
                self.failure.update(timing_cache=clone_tree(self.timing_cache),
                    timing_current=clone_tree(self.timing_current),
                    timing_checks=clone_tree(getattr(self, 'timing_checks', {})),
                    timing_snapshot=clone_tree(self.timing_snapshot))

    def start_input(self, metadata, boundary='000'):
        if self.identity is not None and self.order_index != 20:
            self._fail('cannot discard incomplete T21 input caches')
        # Parent validation expects a completed12-call T20 boundary. The parent
        # method is not used here: T21 order is explicitly20, never forged.
        assert not self.closed and not self.active and boundary == '000'
        assert set(metadata) == {'corpus', 'window', 'identity', 'length', 'token_sha256'}
        assert metadata['identity'] in ('natural', 'bos') and metadata['length'] == self.expected_positions
        assert type(metadata['window']) is int and metadata['window'] >= 0
        assert isinstance(metadata['corpus'], str) and metadata['corpus']
        assert isinstance(metadata['token_sha256'], str) and len(metadata['token_sha256']) == 64
        self.identity = copy.deepcopy(metadata); self.boundary = boundary
        self.write_cache = {'main': {}, 'suffix': {}}; self.timing_cache = {'main': {}, 'suffix': {}}
        self.variant_hashes = {'main': metadata['token_sha256']}; self.order_index = 0; self.failure = None

    def reset(self, boundary, arm, variant='main', token_sha256=None, snapshot=False, observe_only=False):
        assert not self.closed and not self.active and self.identity is not None
        if boundary != '000' or arm not in ARMS or variant not in VARIANTS:
            self._fail('T21 native identity/arm/variant mismatch')
        if not isinstance(token_sha256, str) or len(token_sha256) != 64:
            self._fail('actual token fingerprint required')
        if variant in self.variant_hashes and token_sha256 != self.variant_hashes[variant]:
            self._fail('variant token fingerprint changed')
        if observe_only:
            if not (arm == 'baseline' and variant == 'main' and 'baseline' in self.write_cache['main'] and self.order_index == 1):
                self._fail('invalid T21 writer-identity observer')
        elif self.order_index >= 20 or (arm, variant) != (ARMS[self.order_index//2], VARIANTS[self.order_index%2]):
            self._fail('T21 arm/variant order violation')
        if '_write_matched_' in arm and not all(k in self.write_cache[variant] for k in ('baseline','remove_full','orth_full')):
            self._fail('both full candidates and baseline required')
        self.variant_hashes[variant] = token_sha256
        self.arm, self.variant, self.observe_only, self.save_snapshot = arm, variant, bool(observe_only), bool(snapshot)
        self.support = arm.rsplit('_', 1)[-1] if '_write_matched_' in arm else None
        self.cache_key = {**self.identity, 'boundary': '000', 'variant': variant,
            'actual_token_sha256': token_sha256, 'F_digest': self.params[3], 'freeze_digest': self.freeze_digest}
        operation = 'remove' if arm.startswith('remove') else 'orth' if arm.startswith('orth') else arm
        self.native.reset('000', operation, snapshot=True)
        self.write_norms = {}; self.write_checks = {}; self.write_snapshot = {}; self.prefix_writes = None
        self._current_readout_prefix = None
        self.timing_current = {}; self.timing_snapshot = {}; self.timing_checks = {
            'support': self.support, 'later_query_start': LATER_QUERY_START,
            'same_full_matched_array_across_supports': self.support is not None,
            'later_only_prefix_logits': 'pending' if self.support == 'later' else 'not_required'}
        self._input_seen = self._pre_seen = self._post_seen = self._attention_seen = False; self.active = True

    def _postwrite(self, module, args, output):
        if self.support is None:
            return super()._postwrite(module, args, output)
        if not self._pre_seen or self._post_seen: self._fail('post-write hook ordering')
        if not byte_equal(args[0], self.write_snapshot['incoming_reference']['delivered_context']):
            self._fail('projection input changed after frozen context hook', actual_projection_input=args[0])
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or output.shape != (1,self.expected_positions,self.native.width) or not bool(torch.isfinite(output).all()):
            self._fail('finite FP32 fixed-shape projected write required', output=output)
        cache = self.write_cache[self.variant]; full = cpu(output); record = clone_tree(self.write_snapshot); record['full_write'] = full
        refname = 'remove_full' if self.arm.startswith('remove') else 'orth_full'; which = 0 if refname == 'remove_full' else 1
        if not byte_equal(record, cache[refname]):
            self._fail('cached same-state full proposal/write identity', current=record, reference=cache[refname])
        if 'match' not in cache:
            try: cache['match'] = interpolate_writes(*(cache[k]['full_write'][0].numpy() for k in ('baseline','remove_full','orth_full')))
            except (AssertionError,FloatingPointError) as error:
                self._fail(error, current=record, arithmetic=getattr(error,'matching_evidence',None))
            cache['match_commitment'] = arithmetic_commitment(cache['match'])
            cache['match_origin'] = {k:fingerprint(cache[k]['full_write']) for k in ('baseline','remove_full','orth_full')}
        if cache.get('match_commitment') != arithmetic_commitment(cache['match']):
            self._fail('fixed matched arithmetic cache changed between supports')
        origin = {k:fingerprint(cache[k]['full_write']) for k in ('baseline','remove_full','orth_full')}
        if cache.get('match_origin') != origin: self._fail('fixed matched arithmetic origin changed between supports',actual_origin=origin)
        matching = cache['match']; base = cache['baseline']['full_write']
        desired, active = splice_support(base[0].numpy(), matching['values'][which], self.support)
        returned = torch.from_numpy(desired.copy()).unsqueeze(0).to(device=output.device); delivered = cpu(returned)
        if not byte_equal(delivered[0].numpy(), desired): self._fail('device round-trip bytes', expected=desired, actual=delivered)
        if not byte_equal(delivered[:, ~active], base[:, ~active]): self._fail('inactive/source write bytes', actual=delivered, baseline=base)
        actual_norm = row_norm(delivered[0].double().numpy()-base[0].double().numpy())
        expected_norm = np.where(active, matching['actual_norms'][which], 0.)
        if not byte_equal(actual_norm, expected_norm): self._fail('support actual delivered norm bytes', actual_norm=actual_norm, expected_norm=expected_norm)
        pair = cache.setdefault('support_delivered', {}).setdefault(self.support, {})
        pair[which] = delivered
        expected_pair = [np.where(active,x,0.) for x in matching['actual_norms']]
        pair_complete = set(pair) == {0,1}
        actual_pair = [row_norm(pair[j][0].double().numpy()-base[0].double().numpy()) for j in (0,1)] if pair_complete else expected_pair
        masked = {'target':np.where(active,matching['target'],0.),
                  'bounds':[np.where(active,x,0.) for x in matching['bounds']], 'actual_norms':expected_pair}
        try: quality = audit_matching(masked,actual_pair)
        except AssertionError as error: self._fail(error, masked_recipe=masked, actual_norms=actual_pair)
        self.write_checks.update(matching_quality=quality, full_recipe_matching_quality=matching['quality'],
            paired_actual_delivery_complete=pair_complete,
            counterpart_norm_scope='actual_delivered_same_support' if pair_complete else 'expected_CPU_rounded_not_yet_delivered_same_support',
            target_scope='same_frozen_full_recipe_target_on_active_queries_zero_elsewhere',
            active_write_bytes_exact=True, inactive_write_bytes_exact=True,
            full_matched_array=fingerprint(torch.from_numpy(matching['values'][which])), support=self.support)
        self.write_norms.update({k:torch.from_numpy(x.copy()) for k,x in {
            'target_norm':masked['target'], 'lambda':np.where(active,matching['scales'][which],0.),
            'own_full_norm':matching['full_norms'][which], 'other_full_norm':matching['full_norms'][1-which],
            'rounding_budget':masked['bounds'][which], 'other_rounding_budget':masked['bounds'][1-which],
            'signed_target_residual':actual_norm-masked['target'],
            'absolute_pair_gap':np.abs(expected_pair[0]-expected_pair[1])}.items()})
        snap = record['incoming_reference']
        self.write_norms.update({k:torch.from_numpy(row_norm(x).copy()) for k,x in {
            'actual_write_norm':delivered[0].numpy(),
            'actual_write_delta_norm':delivered[0].double().numpy()-base[0].double().numpy(),
            'proposed_full_write_delta_norm':full[0].double().numpy()-base[0].double().numpy(),
            'proposed_pre_context_delta_norm':snap['delivered_context'][0].double().numpy()-snap['actual_context'][0].double().numpy()}.items()})
        self.write_checks.update(source_write_bytes_exact=byte_equal(delivered[:,0],base[:,0]),
            proposed_full_write=fingerprint(full),returned_write=fingerprint(delivered),baseline_write=fingerprint(base),
            actual_norms_include_q0=True,source_q0_norm=float(actual_norm[0]),
            proposed_context_and_actual_write_distinct=True,native_mask_preserved=True)
        self.write_snapshot.update(baseline_write=cpu(base),proposed_full_write=full,returned_write=delivered,
            interpolation=clone_tree(self.write_norms),support_active=torch.from_numpy(active.copy()))
        self.prefix_writes = cpu(delivered[:,:128]); self._post_seen = True; self._returned_write = delivered
        return returned

    def _timing(self, name, output):
        width = 3*self.native.width if name == 'l11_qkv' else self.native.width
        if not self.active or not self._post_seen or name in self.timing_current:
            self._fail('timing observer ordering', node=name, output=output)
        if not isinstance(output,torch.Tensor) or output.dtype != torch.float32 or output.shape != (1,self.expected_positions,width) or not bool(torch.isfinite(output).all()):
            self._fail('finite fixed-shape timing observer required', node=name, output=output)
        value = cpu(output); self.timing_current[name] = value
        reference = self.timing_cache[self.variant]
        if self.arm == 'baseline' and not self.observe_only:
            reference[name] = value
        else:
            if name not in reference: self._fail('missing baseline timing reference',node=name)
            sl = slice(LATER_QUERY_START,None) if self.support == 'early' else slice(None,LATER_QUERY_START) if self.support == 'later' else slice(None) if self.arm in ('baseline','current') else slice(0,1)
            exact = byte_equal(value[:,sl],reference[name][:,sl])
            self.timing_checks[name] = {'query_slice':[sl.start,sl.stop], 'byte_exact':exact,
                'actual':fingerprint(value[:,sl]),'baseline':fingerprint(reference[name][:,sl])}
            if not exact:self._fail('timing baseline byte guard: '+name,node=name,query_slice=[sl.start,sl.stop],actual=value,baseline=reference[name])
        self.timing_checks.setdefault('observed_nodes',[]).append(name)
        return None

    def _readout_observer(self, module, args, output):
        # Hash only the compared prefix. No duplicate full vocabulary tensor is
        # retained. Explicit verify_readout still binds the runner's operands.
        if self.arm != 'baseline' and self.support != 'later': return None
        logits = getattr(output,'logits',None)
        if not isinstance(logits,torch.Tensor) or logits.dtype != torch.float32 or logits.ndim != 3 or logits.shape[:2] != (1,self.expected_positions):
            self._fail('model readout observer fixed FP32 shape required',actual_logits=logits)
        actual = cpu(logits[0,:LATER_QUERY_START])
        if not bool(torch.isfinite(actual).all()): self._fail('finite model readout prefix required',actual=actual)
        reference = self.timing_cache[self.variant]
        current = fingerprint(actual)
        if self.arm == 'baseline' and not self.observe_only:
            reference['baseline_readout_prefix'] = current
        elif current != reference.get('baseline_readout_prefix'):
            self._fail('actual model later/identity prefix logits byte guard',actual_prefix=actual,baseline_prefix_fingerprint=reference.get('baseline_readout_prefix'))
        self.timing_checks['observed_model_readout_prefix'] = current
        self.timing_checks['baseline_model_readout_prefix'] = reference['baseline_readout_prefix']
        self._current_readout_prefix = current
        return None

    def verify_readout(self, actual_logits, baseline_logits):
        if not self.active or self.support != 'later' or len(self.timing_current) != 4:
            self._fail('later-only readout guard ordering/scope')
        if not all(isinstance(x,torch.Tensor) and x.dtype == torch.float32 and x.ndim == 2 and x.shape[0] == self.expected_positions for x in (actual_logits,baseline_logits)) or actual_logits.shape != baseline_logits.shape:
            self._fail('full fixed-shape FP32 readout operands required',actual=actual_logits,baseline=baseline_logits)
        actual = cpu(actual_logits[:LATER_QUERY_START]); baseline = cpu(baseline_logits[:LATER_QUERY_START])
        if fingerprint(actual) != self._current_readout_prefix:
            self._fail('readout operand not actual model prefix',actual=actual,observed_prefix_fingerprint=self._current_readout_prefix)
        if fingerprint(baseline) != self.timing_cache[self.variant].get('baseline_readout_prefix'):
            self._fail('readout baseline operand not cached native prefix',baseline=baseline,expected_prefix_fingerprint=self.timing_cache[self.variant].get('baseline_readout_prefix'))
        exact = byte_equal(actual,baseline)
        self.timing_checks['later_only_prefix_logits'] = {'query_slice':[0,LATER_QUERY_START], 'byte_exact':exact,
            'actual':fingerprint(actual),'baseline':fingerprint(baseline)}
        if not exact:self._fail('later-only prefix logits byte guard',actual=actual,baseline=baseline)

    def finish(self):
        if set(self.timing_current) != {'post_attention','mlp_output','block_output','l11_qkv'}:
            self._fail('incomplete timing observers')
        if self.support == 'later' and not isinstance(self.timing_checks['later_only_prefix_logits'],dict):
            self._fail('later-only prefix readout guard required before finish')
        super().finish()
        if self.save_snapshot:self.timing_snapshot = clone_tree(self.timing_current)
        self.timing_current = {}
