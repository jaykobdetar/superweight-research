"""T23 cell-aware native000 transfer operator. SOURCE HOLD; no execution here.

Frozen native delivery, actual-write interpolation and support copying are used
unchanged. This file adds only explicit cell binding, order/cache lifetimes,
compact-panel ownership and the already reviewed model-end/failure discipline.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import numpy as np
import torch
from gate_native_value import cpu, fingerprint, NORM_TOLERANCE
from gate_write_norm import GateWriteNorm, byte_equal, VARIANTS
from gate_query_write_timing import GateQueryWriteTiming, arithmetic_commitment
from gate_l11_key_value_donor import GateL11KeyValueDonor, _owned_cpu

EVALUATION_EXECUTION_APPROVED=False
CELLS=('PP','PC','CP','CC')
CELL_WORLDS={'PP':('PE','PE'),'PC':('PE','CE'),'CP':('CE','PE'),'CC':('CE','CE')}
LOCAL_ARMS=('remove_full','orth_full','remove_write_matched_early','orth_write_matched_early')
ARMS=('baseline','current')+tuple(cell+'_'+arm for cell in CELLS for arm in LOCAL_ARMS)
ORDER=tuple((arm,variant) for arm in ARMS for variant in VARIANTS)
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
CALIBRATION_ACCEPTANCE_SHA='87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300'
WRITE_COMMON_KEYS=('actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm')
WRITE_MATCH_KEYS=('target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap')


def _digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _hex(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


class GateCalibrationTransfer(GateQueryWriteTiming):
 """18 arms x MAIN/SUFFIX. No selection, fit, serialization or model loading.

 Capture current fields before finish; finish keeps them until reset. Call
 retire_cell only after the final suffix evidence/controls/journal commit.
 """
 def __init__(self,model,cells,fit_digests,freeze_digest,calibration_acceptance_sha256,*,expected_positions=512,expected_vocab_size=50257):
  assert set(cells)==set(CELLS) and set(fit_digests)=={'PE','CE'}
  assert fit_digests['PE']==F_DIGEST and all(_hex(v) for v in fit_digests.values())
  assert calibration_acceptance_sha256==CALIBRATION_ACCEPTANCE_SHA and _hex(freeze_digest)
  assert expected_positions==512 and expected_vocab_size==50257
  self.fit_digests=dict(fit_digests);self.calibration_acceptance_sha256=calibration_acceptance_sha256
  self._parameters={};self.parameter_records={}
  for cell in CELLS:
   assert set(cells[cell])=={'alpha','u','v'}
   values={k:cpu(v) for k,v in cells[cell].items()}
   assert all(z.dtype==torch.float64 and z.shape==(12,) and bool(torch.isfinite(z).all()) for z in values.values())
   u,v=values['u'],values['v']
   assert abs(float(u.norm())-1)<=1e-12 and abs(float(v.norm())-1)<=1e-12 and abs(float(u@v))<=1e-12
   uf,vf=u.float().double(),v.float().double()
   assert abs(float(uf.norm()-vf.norm()))<=NORM_TOLERANCE and abs(float(uf@vf))<=NORM_TOLERANCE
   center,basis=CELL_WORLDS[cell];self._parameters[cell]=values
   record={'cell':cell,'center_world':center,'basis_world':basis,'center_map_digest':fit_digests[center],'basis_map_digest':fit_digests[basis],'vectors':{k:fingerprint(z) for k,z in values.items()}}
   record['parameter_digest']=_digest(record);self.parameter_records[cell]=record
  assert byte_equal(self._parameters['PP']['alpha'],self._parameters['PC']['alpha']) and byte_equal(self._parameters['CP']['alpha'],self._parameters['CC']['alpha'])
  for name in ('u','v'):
   assert byte_equal(self._parameters['PP'][name],self._parameters['CP'][name]) and byte_equal(self._parameters['PC'][name],self._parameters['CC'][name])
  self._parameter_commitment=_digest(self.parameter_records)
  self._fit_commitment=_digest(self.fit_digests)
  self.cell_device_maps={c:{v:{} for v in VARIANTS} for c in CELLS}
  self.parameter_cell='PP';self.cell=None;self.local_arm='baseline';self.external_arm='baseline'
  self.failed=False;self.output_observer=None;self.output_events=[];self._identity_hook_ids=set()
  self.common_baseline={};self.shared_snapshot=None;self.active_cell=None;self.cell_records={};self.retirement_receipts={}
  self.current_hidden=None;self.output_fields={};self._observed_logits=None;self.model_output_observed=False;self._postwrite_operand=None
  self.expected_vocab_size=expected_vocab_size
  p=self._parameters['PP']
  super().__init__(model,p['alpha'],p['u'],p['v'],self.parameter_records['PP']['parameter_digest'],freeze_digest,expected_positions=expected_positions)

 _roster=GateL11KeyValueDonor._roster
 allow_identity_hooks=GateL11KeyValueDonor.allow_identity_hooks

 def _install(self):
  super()._install()
  self._model_end_handle=self.handles[-1];self.native._fail=self._native_fail
  self.handles.append(self.model.transformer.ln_f.register_forward_hook(self._final_hidden))
  self._bind_parameters(self.parameter_cell,getattr(self,'variant','main'))
  ids={h.id for h in self.handles+self.native.handles}
  self._owned_roster=[(m,k,i) for m in self.model.modules() for k in ('_forward_pre_hooks','_forward_hooks') for i in getattr(m,k) if i in ids]
  assert len(self._owned_roster)==len(ids)
  self._installed_roster=self._roster()

 def _assert_parameters(self):
  if _digest(self.fit_digests)!=self._fit_commitment or _digest(self.parameter_records)!=self._parameter_commitment:self._fail('parameter provenance changed')
  for cell in CELLS:
   if any(fingerprint(z)!=self.parameter_records[cell]['vectors'][k] for k,z in self._parameters[cell].items()):self._fail('immutable cell vector changed',cell=cell)
   for mapping in self.cell_device_maps[cell].values():
    for key,values in mapping.items():
     if not isinstance(key,tuple) or len(key)!=2 or not isinstance(values,tuple) or len(values)!=3:self._fail('cell/variant device-map ABI changed',cell=cell)
     device,dtype=key
     if any(z.device!=device or z.dtype!=dtype or not byte_equal(cpu(z),self._parameters[cell][k].to(dtype=dtype)) for k,z in zip(('alpha','u','v'),values)):self._fail('immutable cell/variant device vector changed',cell=cell,actual_device_values=values)
  for name in ('alpha','u','v'):
   if not byte_equal(getattr(self.native,name),self._parameters[self.parameter_cell][name]):self._fail('native parameter binding changed',parameter=name)
  if self.native.device_map is not self.cell_device_maps[self.parameter_cell][getattr(self,'variant','main')]:self._fail('native cell/variant device-map owner changed')

 def _bind_parameters(self,cell,variant):
  self.parameter_cell=cell;p=self._parameters[cell];digest=self.parameter_records[cell]['parameter_digest']
  self.params=tuple(cpu(p[k]) for k in ('alpha','u','v'))+(digest,)
  self.native.alpha,self.native.u,self.native.v=self.params[:3];self.native.digest=digest
  self.native.device_map=self.cell_device_maps[cell][variant]

 def start_input(self,metadata,boundary='000'):
  assert not self.failed and not self.closed and not self.active and boundary=='000'
  if self.identity is not None and (self.order_index!=36 or set(self.retirement_receipts)!=set(CELLS)):self._fail('cannot discard incomplete input/cell caches')
  assert set(metadata)=={'corpus','window','identity','length','token_sha256'}
  assert metadata['identity'] in ('natural','bos') and metadata['length']==512 and type(metadata['window']) is int and metadata['window']>=0
  assert isinstance(metadata['corpus'],str) and metadata['corpus'] and _hex(metadata['token_sha256'])
  self.identity=copy.deepcopy(metadata);self.boundary='000';self.order_index=0;self.failure=None
  self.write_cache={v:{} for v in VARIANTS};self.common_baseline={};self.timing_cache={v:{} for v in VARIANTS}
  self.variant_hashes={'main':metadata['token_sha256']};self.shared_snapshot=None;self.active_cell=None;self.retirement_receipts={};self.cell_records={}

 def reset(self,boundary,arm,variant='main',token_sha256=None,snapshot=False,observe_only=False):
  if self.failed:raise AssertionError('operator frozen after failure')
  if self.closed or self.active or self.identity is None or boundary!='000' or arm not in ARMS or variant not in VARIANTS:self._fail('native identity/arm/variant state')
  if not _hex(token_sha256) or (variant in self.variant_hashes and self.variant_hashes[variant]!=token_sha256):self._fail('variant token binding changed')
  if observe_only:
   if not (arm=='baseline' and variant=='main' and self.order_index==1 and 'main' in self.common_baseline):self._fail('identity anchor outside baseline-main boundary')
  elif self.order_index>=36 or ORDER[self.order_index]!=(arm,variant):self._fail('18-arm/variant order violation',requested_arm=arm,requested_variant=variant)
  cell=None if arm in ('baseline','current') else arm.split('_',1)[0]
  local=arm if cell is None else arm[len(cell)+1:]
  if cell is not None and cell!=self.active_cell:
   previous=CELLS.index(cell)-1
   if self.active_cell is not None or (previous>=0 and CELLS[previous] not in self.retirement_receipts):self._fail('previous cell lacks successful retirement')
   if local!='remove_full' or variant!='main':self._fail('cell must begin with MAIN full-remove')
   self.active_cell=cell;self.write_cache={v:{} for v in VARIANTS}
  self.arm=self.external_arm=arm;self.local_arm=local;self.cell=cell;self.variant=variant;self.variant_hashes[variant]=token_sha256
  self.observe_only=bool(observe_only);self.save_snapshot=bool(snapshot);self.support='early' if local.endswith('_early') else None
  self._bind_parameters(cell or 'PP',variant);self._assert_parameters()
  self.cache_key={**self.identity,'boundary':'000','variant':variant,'actual_token_sha256':token_sha256,'cell':cell,'parameter_digest':self.parameter_records[cell or 'PP']['parameter_digest'],'fit_digests':dict(self.fit_digests),'freeze_digest':self.freeze_digest,'calibration_acceptance_sha256':self.calibration_acceptance_sha256}
  operation='remove' if local.startswith('remove') else 'orth' if local.startswith('orth') else local
  self.native.reset('000',operation,snapshot=True)
  self.write_norms={};self.write_checks={};self.write_snapshot={};self.prefix_writes=None;self._returned_write=None;self._postwrite_operand=None
  self.timing_current={};self.timing_snapshot={};self.timing_checks={'support':self.support,'later_query_start':127,'later_only_prefix_logits':'not_required'}
  self.cell_records={'cell':cell,'baseline_parameter_scope':'PP_reference_only' if cell is None else None,'parameter_record':copy.deepcopy(self.parameter_records[cell or 'PP'])}
  self.current_hidden=None;self.output_fields={};self._observed_logits=None;self.model_output_observed=False
  self._input_seen=self._pre_seen=self._post_seen=self._attention_seen=False;self.active=True

 def _input(self,module,args,kwargs):
  self._assert_parameters();actual=self._roster();additions=actual-self._installed_roster
  allowed=self.observe_only and self.arm=='baseline' and self.variant=='main' and {x[2] for x in additions}==self._identity_hook_ids
  if self.native._fail!=self._native_fail or self._installed_roster-actual or (additions and not allowed) or (self._identity_hook_ids and not allowed) or any(i not in getattr(m,k) for m,k,i in self._owned_roster) or next(reversed(self.model._forward_hooks))!=self._model_end_handle.id:self._fail('owned hook/failure-dispatch roster changed')
  return GateWriteNorm._input(self,module,args,kwargs)

 def _local(self,function,*args):
  saved=self.arm;self.arm=self.local_arm
  try:return function(self,*args)
  finally:self.arm=saved

 def _common_incoming(self,incoming,reference):
  for key in ('q','k','v','mask','actual_context','stable_mixture'):
   if not byte_equal(incoming[key],reference[key]):self._fail('shared native incoming state changed: '+key,current=incoming,reference=reference)
  for key in ('b','mass'):
   if not byte_equal(incoming['gate_state'][key],reference['gate_state'][key]):self._fail('shared native gate changed: '+key,current=incoming,reference=reference)

 def _postwrite(self,module,args,output):
  self._postwrite_operand=output;self._assert_parameters()
  if self.cell is not None:
   if self.variant not in self.common_baseline:self._fail('missing actual variant baseline')
   common=self.common_baseline[self.variant];incoming=self.write_snapshot['incoming_reference']
   self._common_incoming(incoming,common['incoming_reference'])
   if not byte_equal(self.write_snapshot['mask_reference'],common['mask_reference']):self._fail('common actual mask binding changed')
   cache=self.write_cache[self.variant];s=incoming['gate_state']['s']
   if 'baseline' not in cache:
    if self.local_arm!='remove_full':self._fail('cell score must originate in first full-remove')
    if self.cell=='PP' and not byte_equal(s,common['incoming_reference']['gate_state']['s']):self._fail('PP score differs from actual baseline PE score',actual=s,expected=common['incoming_reference']['gate_state']['s'])
    gate={**common['incoming_reference']['gate_state'],'s':s}
    cache['baseline']={**common,'cache_key':copy.deepcopy(self.cache_key),'incoming_reference':{**common['incoming_reference'],'gate_state':gate}}
   elif not byte_equal(s,cache['baseline']['incoming_reference']['gate_state']['s']):self._fail('same-cell score changed',actual=s,expected=cache['baseline']['incoming_reference']['gate_state']['s'])
   self.cell_records.update(common_incoming_byte_exact=True,score=fingerprint(s),score_origin='same_variant_first_actual_cell_full_remove_ff_map',PP_baseline_s_exact=True if self.cell=='PP' else 'not_applicable')
  result=self._local(GateQueryWriteTiming._postwrite,module,args,output)
  self.write_checks.update(cell=self.cell,parameter_digest=self.parameter_records[self.cell or 'PP']['parameter_digest'],full_proposal_and_actual_delivery_distinct=True)
  expected=set(WRITE_COMMON_KEYS)|(set(WRITE_MATCH_KEYS) if self.support else set())
  if set(self.write_norms)!=expected:self._fail('write-norm ABI changed')
  return result

 def _final_hidden(self,module,args,output):
  if not self.active or not isinstance(output,torch.Tensor) or output.dtype!=torch.float32 or output.shape!=(1,512,self.native.width) or not bool(torch.isfinite(output).all()):self._fail('finite actual final hidden required',actual_hidden=output)
  self.current_hidden=cpu(output)

 def set_output_observer(self,callback):
  if self.failed or self.closed or (self.active and self._input_seen) or (callback is not None and not callable(callback)):self._fail('output observer registration state')
  self.output_observer=callback

 def _readout_observer(self,module,args,output):
  self._observed_logits=getattr(output,'logits',None)
  if self.model_output_observed:self._fail('duplicate model-end observation')
  self.model_output_observed=True
  event={'arm':self.external_arm,'variant':self.variant,'actual_token_sha256':self.cache_key['actual_token_sha256']};self.output_events.append(event)
  if self.output_observer is not None:self.output_observer(dict(event))
  value=self._observed_logits
  if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.shape!=(1,512,self.expected_vocab_size) or not bool(torch.isfinite(value).all()):self._fail('known actual full readout ABI/finite guard',actual_logits=value)
  if self.current_hidden is None:self._fail('model end before final hidden observation')
  self.output_fields={'final_hidden':fingerprint(self.current_hidden),'logits':fingerprint(value)}

 def finish(self):
  if self.failed or not self.active or not self.model_output_observed or set(self.timing_current)!=set(TIMING_NODES):self._fail('incomplete actual model/timing observation')
  saved=self.save_snapshot;self.save_snapshot=True
  try:self._local(GateWriteNorm.finish)
  finally:self.save_snapshot=saved
  if self.external_arm=='baseline' and not self.observe_only:
   self.common_baseline[self.variant]=self.write_cache[self.variant]['baseline']
   if self.variant=='main':
    snap=self.common_baseline['main']['incoming_reference']
    self.shared_snapshot={k:snap[k] for k in ('q','k','v','mask','actual_context','stable_mixture','gate_state')}
    self.shared_snapshot.update(baseline_write=self.common_baseline['main']['full_write'],mask_records=self.common_baseline['main']['mask_reference'],s_scope='PP_reference_only')

 def panel_snapshot(self):
  if self.variant!='main' or self.shared_snapshot is None or not self._post_seen:self._fail('selected MAIN panel capture ordering')
  snap=self.write_snapshot;incoming=snap['incoming_reference'];self._common_incoming(incoming,self.shared_snapshot)
  return {'proposed_context':incoming['delivered_context'],'proposed_full_write':snap.get('proposed_full_write',snap.get('full_write')),'returned_write':self._returned_write,'gate_state':incoming['gate_state'],'interpolation':self.write_norms,'cell_records':copy.deepcopy(self.cell_records),'cache_key':copy.deepcopy(self.cache_key)}

 def retire_cell(self,cell):
  if cell not in CELLS or self.failed or self.active or cell!=self.active_cell or self.order_index!=4+8*(CELLS.index(cell)+1) or self.external_arm!=cell+'_orth_write_matched_early' or self.variant!='suffix':self._fail('retirement outside successfully completed cell boundary')
  variants={}
  for variant,cache in self.write_cache.items():
   if set(cache.get('support_delivered',{}).get('early',{}))!={0,1}:self._fail('retirement before actual matched pair complete',variant=variant)
   variants[variant]={'score':fingerprint(cache['baseline']['incoming_reference']['gate_state']['s']),'full_candidates':{k:fingerprint(cache[k]['full_write']) for k in ('remove_full','orth_full')},'matching_commitment':arithmetic_commitment(cache['match']),'delivered':{str(k):fingerprint(v) for k,v in cache['support_delivered']['early'].items()}}
  receipt={'cell':cell,'order_index':self.order_index,'parameter_record':copy.deepcopy(self.parameter_records[cell]),'variants':variants,'retirement_scope':'after runner final-suffix evidence/controls/per-call journal and cell_completed commit; reached current operands retained until next reset'}
  self.retirement_receipts[cell]=receipt;self.write_cache={v:{} for v in VARIANTS};self.active_cell=None
  return copy.deepcopy(receipt)

 def _owned_state(self):
  native=('snapshots','pending','source_states','gate_cache','gate_checks','localization_norms','mask_records','effective_masks','projection_records','context_records','prefix_contexts','diag','failure')
  devices={c:{v:[{'device_dtype_key':repr(key),'values':values} for key,values in mapping.items()] for v,mapping in variants.items()} for c,variants in self.cell_device_maps.items()}
  return {'common_baseline':self.common_baseline,'active_cell_cache':self.write_cache,'write_snapshot':getattr(self,'write_snapshot',{}),'write_checks':getattr(self,'write_checks',{}),'write_norms':getattr(self,'write_norms',{}),'prefix_writes':getattr(self,'prefix_writes',None),'returned_write':getattr(self,'_returned_write',None),'timing_cache':self.timing_cache,'timing_current':self.timing_current,'timing_checks':getattr(self,'timing_checks',{}),'current_hidden':self.current_hidden,'parameters':self._parameters,'bound_native_parameters':{k:getattr(self.native,k) for k in ('alpha','u','v')},'device_maps':devices,'parameter_records':self.parameter_records,'retirement_receipts':self.retirement_receipts,'native':{name:getattr(self.native,name,{}) for name in native}}

 def resource_owners(self):return [self._owned_state(),self.failure,self._observed_logits,self._postwrite_operand]

 def _native_fail(self,stage,layer,reason,**arrays):
  self.native.failure={'stage':stage,'layer':layer,'reason':str(reason),'arrays':_owned_cpu(arrays)}
  self._fail(reason,native_stage=stage,native_layer=layer)

 def _fail(self,reason,**extra):
  if self.failed:raise AssertionError(str(reason))
  self.failed=True
  if isinstance(extra.get('current'),dict):self.write_snapshot=extra.pop('current')
  if self._postwrite_operand is not None and not any(k in getattr(self,'write_snapshot',{}) for k in ('full_write','proposed_full_write')):self.write_snapshot={**getattr(self,'write_snapshot',{}),'current_projection_output':self._postwrite_operand}
  if self._observed_logits is not None and 'actual_logits' not in extra:extra['actual_logits']=self._observed_logits
  self.failure=_owned_cpu({'reason':str(reason),'identity':self.identity,'arm':self.external_arm,'local_arm':self.local_arm,'cell':self.cell,'variant':getattr(self,'variant',None),'order_index':self.order_index,'cache_key':getattr(self,'cache_key',None),'freeze_digest':self.freeze_digest,'fit_digests':self.fit_digests,'calibration_acceptance_sha256':self.calibration_acceptance_sha256,'model_output_observed':self.model_output_observed,'output_events':self.output_events,'owned':self._owned_state(),'extra':extra})
  self.active=False;raise AssertionError(str(reason))

 def capture_failure(self,reason,**extra):self._fail(reason,**extra)

 @contextmanager
 def unhooked(self):
  if self.failed:raise AssertionError('operator frozen after failure')
  with GateWriteNorm.unhooked(self):yield
