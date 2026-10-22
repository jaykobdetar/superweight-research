"""T22 same-schema immutable evidence storage; SOURCE ONLY, admission not witnessed.

The frozen codec is unchanged. Failure fallback writes its exact member schema
straight to a fresh ZIP instead of allocating a second complete archive buffer.
No alias codec, deduplication, omitted operands or old-artifact modification.
"""
import hashlib,io,json,math,os,zipfile
from pathlib import Path
import numpy as np
import torch
from post_write_norm_storage import (EvidenceWriter as FrozenEvidenceWriter,NORMAL_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,
 READOUT_ROWS,FAILURE_ROWS,sha,json_write,codec_tree,byte_exact,nested_fp,summarize_diag,
 trim_gate,trim_gate_norm,logit_rows,source_closure,read_npz)
from post_write_norm_storage import empty_payload as old_empty_payload
from gate_wiki_storage import _encode,dumps_npz,SCHEMA as CODEC_SCHEMA

SCHEMA='l11_key_value_donor_explicit_v1'
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched_early','remove_l11_kv',
 'orth_write_matched_early','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
DONORS=('remove_write_matched_early','orth_write_matched_early')
RECIPIENTS=('remove_l11_kv','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
L10_PANEL=('baseline','remove_full','orth_full',*DONORS)
L11_PANEL=('baseline',*DONORS,*RECIPIENTS)
CLOSURE_KEYS=('remove/main','remove/suffix','orth/main','orth/suffix')
L11_NORM_KEYS=('q_norm','k_norm','v_norm','q_delta_baseline_norm','k_delta_baseline_norm','v_delta_baseline_norm',
 'q_delta_original_norm','k_delta_original_norm','v_delta_original_norm')
TIMING_ROWS=(1,126,127,510,511)
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')
FAILURE_LIMIT=1024**3
RSS_LIMIT=3*1024**3
NON_EVIDENCE_LIMIT=1152*1024**2
WORKSPACE_ALLOWANCE=64*1024**2
NORMALIZATION_ALLOWANCE=32*1024**2
HASH_CHUNK_BYTES=1024**2
FAILURE_JOURNAL_RECORD_LIMIT=64*1024
FAILURE_JOURNAL_DISK_ALLOWANCE=3*FAILURE_JOURNAL_RECORD_LIMIT
# Header-phase admission only: v2 largest raw tree plus one complete actual
# readout and finalhidden conservatively bounds the new return-I/O ownership
# amendment. This is not a new NPZ/RSS encoding certificate.
KNOWN_FAILURE_HEADER_RAW_UPPER=655801638+102926336+1572864
KNOWN_LARGEST_IDENTITY_BYTES=512*50257*4
RAW_METADATA_LIMIT={'normal':8*1024**2,'failure':16*1024**2}
COMPRESSED_METADATA_LIMIT={'normal':1024**2,'failure':8*1024**2}

def empty_payload(metadata):
 value=old_empty_payload(metadata);value['schema']=SCHEMA
 for key in ('l11_norms','suffix_l11_norms','l11_panels','timing_samples'):value[key]={}
 return value

def failure_bundle(bundle):
 """Already-owned CPU references only. Never clone the complete live caches."""
 if bundle is None:return None
 return {key:bundle[key] for key in ('metrics','hidden','source','gates','gate_norms','write_norms','l11_norms',
  'prefix_contexts','prefix_writes','returned_write','operations') if key in bundle}

def suffix_operands(left,right):
 """Exact operands actually compared by suffix_check, not whole unrelated sides."""
 def side(bundle):
  if bundle is None:return None
  return {'hidden':bundle['hidden'][:,:128], 'source':bundle['source'], 'gates':trim_gate(bundle['gates']),
   'gate_norms':trim_gate_norm(bundle['gate_norms']), 'write_norms':{k:v[:128] for k,v in bundle['write_norms'].items()},
   'l11_norms':{k:v[:128] for k,v in bundle['l11_norms'].items()},'prefix_contexts':bundle['prefix_contexts'],
   'prefix_writes':bundle['prefix_writes'],'metrics':{k:v[:127 if k=='loss' else 128] for k,v in bundle['metrics'].items() if isinstance(v,torch.Tensor)},
   'mask_records':bundle['operations']['mask_records'],'l11_prefix':bundle['l11_prefix']}
 return {'expected':side(left),'actual_owner':'hook_failure/current_and_native_prefix_fields',
  'actual_metrics':right['metrics'],'actual_l11_norms':right['l11_norms'],'actual_write_norms':right['write_norms']}

def current_rss_bytes():
 return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024

def array_terms(value):
 """Count serialized occurrences, preserving the frozen codec's no-alias format."""
 if isinstance(value,torch.Tensor):return [(value.numel()*value.element_size(),str(value.dtype),tuple(value.shape))]
 if isinstance(value,np.ndarray):return [(value.nbytes,str(value.dtype),tuple(value.shape))]
 if isinstance(value,dict):return [item for k,v in value.items() for child in (k,v) for item in array_terms(child)]
 if isinstance(value,(list,tuple)):return [item for child in value for item in array_terms(child)]
 return []

def zbound(n):return n+(n>>12)+(n>>14)+(n>>25)+13

def owned_cpu_bytes(values):
 """RAM accounting only: union backing storage intervals, never codec aliases.

 Serialized occurrences remain fully separate. This does not replace/change any
 saved artifact, introduce reference members, or omit a comparison operand.
 """
 ranges=[]
 def walk(value):
  if isinstance(value,torch.Tensor):
   if value.device.type=='cpu' and value.numel():
    storage=value.untyped_storage();ranges.append((storage.data_ptr(),storage.data_ptr()+storage.nbytes()))
  elif isinstance(value,np.ndarray):
   root=value
   while isinstance(root.base,np.ndarray):root=root.base
   if root.size:ranges.append(tuple(map(int,np.byte_bounds(root))))
  elif isinstance(value,dict):
   for k,v in value.items():walk(k);walk(v)
  elif isinstance(value,(list,tuple)):
   for v in value:walk(v)
 walk(values);total=0;last_end=0
 for lo,hi in sorted(ranges):
  if hi>last_end:total+=hi-max(lo,last_end);last_end=hi
 return total

def numpy_copy_bytes(value):
 if isinstance(value,np.ndarray):return value.nbytes
 if isinstance(value,dict):return sum(numpy_copy_bytes(k)+numpy_copy_bytes(v) for k,v in value.items())
 if isinstance(value,(list,tuple)):return sum(numpy_copy_bytes(v) for v in value)
 return 0

def tensor_contiguous_copy_bytes(value):
 # Frozen _encode retains one C-order copy PER noncontiguous CPU occurrence.
 if isinstance(value,torch.Tensor):return value.numel()*value.element_size() if value.device.type=='cpu' and not value.is_contiguous() else 0
 if isinstance(value,dict):return sum(tensor_contiguous_copy_bytes(k)+tensor_contiguous_copy_bytes(v) for k,v in value.items())
 if isinstance(value,(list,tuple)):return sum(tensor_contiguous_copy_bytes(v) for v in value)
 return 0

def tensor_identity_contiguous_temporary_bytes(value):
 # tensor_identity independently normalizes its operand a second time, then
 # creates its byte string. The second normalization is transient, not retained.
 if isinstance(value,torch.Tensor):return value.numel()*value.element_size() if value.device.type=='cpu' and not value.is_contiguous() else 0
 if isinstance(value,dict):return max((max(tensor_identity_contiguous_temporary_bytes(k),tensor_identity_contiguous_temporary_bytes(v)) for k,v in value.items()),default=0)
 if isinstance(value,(list,tuple)):return max((tensor_identity_contiguous_temporary_bytes(v) for v in value),default=0)
 return 0

def c_order_chunks(array,chunk_bytes=HASH_CHUNK_BYTES):
 """Bounded C-order bytes including noncontiguous/F/negative-stride views."""
 if array.size==0:return
 if array.flags.c_contiguous:
  view=memoryview(array).cast('B')
  for start in range(0,view.nbytes,chunk_bytes):yield view[start:start+chunk_bytes]
 else:
  iterator=np.nditer(array,flags=['external_loop','buffered','zerosize_ok'],op_flags=['readonly'],
   order='C',buffersize=max(1,chunk_bytes//array.dtype.itemsize))
  for chunk in iterator:yield chunk.tobytes(order='C')

def bounded_encode(value,arrays):
 """Same frozen metadata/schema with bounded tensor SHA/C-order traversal.

 This is a new preservation implementation, NOT the previously validated
 frozen encoder. CPU storage is attached once per occurrence without aliasing
 members or normalization copies. Byte-exact codec witness remains mandatory.
 """
 if isinstance(value,(torch.Tensor,np.ndarray)):
  if isinstance(value,torch.Tensor):
   assert value.device.type=='cpu','Failure handoff must own CPU arrays before encoding'
   array=value.detach().numpy();dtype=str(value.dtype)
  else:
   array=value;dtype=str(torch.from_numpy(np.empty((0,),dtype=array.dtype)).dtype)
  assert array.dtype.kind in 'biufc' and not array.dtype.hasobject
  h=hashlib.sha256()
  for chunk in c_order_chunks(array):h.update(chunk)
  name='a'+str(len(arrays));arrays[name]=array
  return {'type':'tensor','array':name,'dtype':dtype,'shape':list(array.shape),'sha256':h.hexdigest()}
 if isinstance(value,np.generic):value=value.item()
 if isinstance(value,dict):return {'type':'dict','items':[[bounded_encode(k,arrays),bounded_encode(v,arrays)] for k,v in value.items()]}
 if isinstance(value,(list,tuple)):return {'type':type(value).__name__,'items':[bounded_encode(v,arrays) for v in value]}
 assert value is None or type(value) in (bool,int,float,str),type(value)
 return {'type':'scalar','value':value}

def write_C_order_npy(member,array):
 # Frozen codec normalizes to C order. Explicit header+chunk traversal gives
 # identical logical NPY content without a full contiguous copy for views.
 header={'descr':np.lib.format.dtype_to_descr(array.dtype),'fortran_order':False,'shape':array.shape}
 np.lib.format.write_array_header_1_0(member,header)
 for chunk in c_order_chunks(array):member.write(chunk)

def admission(value,*,failure=False,owners=()):
 terms=array_terms(value);raw=sum(n for n,_,_ in terms);kind='failure' if failure else 'normal'
 upper=sum(zbound(n+256)+384 for n,_,_ in terms)+COMPRESSED_METADATA_LIMIT[kind]+640
 rss=current_rss_bytes();owned=owned_cpu_bytes((value,owners));base=max(0,rss-owned);copies=numpy_copy_bytes(value)
 normalization=max(NORMALIZATION_ALLOWANCE,copies)
 contiguous_copies=tensor_contiguous_copy_bytes(value);identity_contiguous=tensor_identity_contiguous_temporary_bytes(value)
 identity_bytes=max((n for n,_,_ in terms),default=0)
 header_peak=rss+contiguous_copies+identity_contiguous+identity_bytes+WORKSPACE_ALLOWANCE+normalization
 buffered_peak=rss+contiguous_copies+math.ceil(2.25*upper)+WORKSPACE_ALLOWANCE+normalization
 return {'current_rss_bytes':rss,'array_occurrences':len(terms),'serialized_raw_array_bytes':raw,'npz_upper_bytes':upper,
  'owned_CPU_backing_storage_bytes':owned,'non_evidence_observed_bytes':base,'non_evidence_allowance_bytes':NON_EVIDENCE_LIMIT,
  'non_evidence_budget_passed':base<=NON_EVIDENCE_LIMIT,'numpy_normalization_bytes':copies,'normalization_allowance_bytes':NORMALIZATION_ALLOWANCE,'normalization_budget_passed':copies<=NORMALIZATION_ALLOWANCE,'tensor_contiguous_retained_copy_bytes':contiguous_copies,'tensor_identity_contiguous_temporary_bytes':identity_contiguous,'tensor_identity_byte_string_upper_bytes':identity_bytes,'header_phase_upper_bytes':header_peak,'buffered_phase_upper_bytes':buffered_peak,'encoding_peak_upper_bytes':max(header_peak,buffered_peak),
  'rss_limit_bytes':RSS_LIMIT,'buffered_admitted':header_peak<=RSS_LIMIT and buffered_peak<=RSS_LIMIT and base<=NON_EVIDENCE_LIMIT and copies<=NORMALIZATION_ALLOWANCE,
  'stream_admitted':header_peak<=RSS_LIMIT and copies<=NORMALIZATION_ALLOWANCE,
  'bounded_stream_phase_upper_bytes':rss+WORKSPACE_ALLOWANCE+HASH_CHUNK_BYTES,
  'bounded_stream_admitted':rss+WORKSPACE_ALLOWANCE+HASH_CHUNK_BYTES<=RSS_LIMIT,
  'ownership_scope':'Existing CPU backing intervals for declared live owners only; no codec/storage deduplication.'}

class EvidenceWriter(FrozenEvidenceWriter):
 def __init__(self,out,*,normal_limit=NORMAL_LIMIT,failure_limit=FAILURE_LIMIT,reserve=DISK_RESERVE,outstanding_other_caps=0):
  super().__init__(out,normal_limit=normal_limit,failure_limit=failure_limit,reserve=reserve,outstanding_other_caps=outstanding_other_caps)
  self.resource_owners=lambda:()
  self.failure_journal_error_index=0;self.failure_journal_errors=[]
 def set_resource_owners(self,callback):
  assert callable(callback);self.resource_owners=callback
 def _failure_journal(self,event):
  """Failure-only best effort: append refusal cannot discard the owned tree.

  Normal journal calls still raise. Fresh sidecars commit the refusal and event;
  no failed append is relabeled a successful journal event/completed study.
  """
  try:self.journal(event);return
  except OSError as error:
   self.failure_journal_error_index+=1;index=self.failure_journal_error_index
   # Only bounded scalar pre-encode/stream/commit events reach this wrapper.
   # At most one failure save plus its MemoryError fallback emits four calls.
   message=str(error);encoded_event=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
   record={'status':'failed_journal_preservation_continues','sequence':index,'event':event,
    'error_type':type(error).__name__[:256],'error':message[:4096],'complete_error_sha256':hashlib.sha256(message.encode(errors='backslashreplace')).hexdigest(),
    'error_text_prefix_only':len(message)>4096,'journal_chain_unchanged':self.chain,'failure_write_only':True,
    'journal_append_outcome':'Unconfirmed after exception; original journal bytes remain preserved.'}
   data=(json.dumps(record,indent=2,allow_nan=False)+'\n').encode()
   if len(data)>FAILURE_JOURNAL_RECORD_LIMIT:
    record['event']={'event':event.get('event'),'complete_event_sha256':hashlib.sha256(encoded_event).hexdigest(),
     'scope':'Oversized auxiliary journal metadata committed by hash; no scientific capsule fields removed.'}
   name='journal_preservation_error_'+format(index,'06d')+'.json'
   receipt={'path':name,'event':event.get('event'),'journal_error_type':type(error).__name__,'status':record['status']}
   try:
    json_write(self.out/name,record);receipt.update(bytes=(self.out/name).stat().st_size,sha256=sha(self.out/name))
   except OSError as sidecar_error:
    # Permanent raw/error-file I/O failure cannot be guaranteed recoverable.
    # Still attempt the complete raw capsule; final failed metadata gets this.
    receipt['sidecar_error']=repr(sidecar_error)
   self.failure_journal_errors.append(receipt)
 def _write_event(self,event,*,failure=False):
  if failure:self._failure_journal(event)
  else:self.journal(event)
 def guard_known_header(self,stage):
  owners=self.resource_owners();rss=current_rss_bytes();owned=owned_cpu_bytes(owners);base=max(0,rss-owned)
  predicted=base+2*KNOWN_FAILURE_HEADER_RAW_UPPER+2*KNOWN_LARGEST_IDENTITY_BYTES+WORKSPACE_ALLOWANCE+NORMALIZATION_ALLOWANCE
  record={'stage':stage,'current_rss_bytes':rss,'owned_CPU_backing_storage_bytes':owned,'non_evidence_observed_bytes':base,
   'non_evidence_limit_bytes':NON_EVIDENCE_LIMIT,'known_failure_header_raw_upper_bytes':KNOWN_FAILURE_HEADER_RAW_UPPER,
   'known_tensor_contiguous_retained_copy_upper_bytes':KNOWN_FAILURE_HEADER_RAW_UPPER,'known_tensor_identity_contiguous_temporary_upper_bytes':KNOWN_LARGEST_IDENTITY_BYTES,'known_largest_identity_temporary_bytes':KNOWN_LARGEST_IDENTITY_BYTES,'known_identity_byte_string_upper_bytes':KNOWN_LARGEST_IDENTITY_BYTES,'known_header_phase_upper_bytes':predicted,
   'rss_limit_bytes':RSS_LIMIT,'passed':base<=NON_EVIDENCE_LIMIT and predicted<=RSS_LIMIT}
  self.journal({'event':'known_header_admission',**record})
  assert record['passed'],('Stop before model/next forward: non-evidence/header assumptions failed',record)
  return record
 def save(self,name,value,*,failure=False):
  assert self.owns() and not Path(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/'))
  record=admission(value,failure=failure,owners=self.resource_owners())
  self._write_event({'event':'pre_encode_admission','path':name,'failure_write':failure,**record},failure=failure)
  if not record['buffered_admitted']:
   if not failure:raise MemoryError(('Normal encoding admission rejected; owned evidence retained for failure fallback',record))
   # Refusal cannot discard the sole RAM failure tree: directly preserve it.
   return self.save_failure_stream(name,value,record)
  # Exactly the frozen dumps_npz(codec_tree(value)) representation. This local
  # commit adds pre-encode admission/member checks, not a scientific/codec delta.
  try:data=dumps_npz(codec_tree(value))
  except MemoryError as error:
   error.__traceback__=None
   if failure:return self.save_failure_stream(name,value,record)
   raise
  kind='failure' if failure else 'normal';terms=array_terms(value)
  with zipfile.ZipFile(io.BytesIO(data)) as archive:
   infos=archive.infolist();assert [v.filename for v in infos]==['a'+str(i)+'.npy' for i in range(len(terms))]+['metadata.npy']
   assert all(info.file_size<=term[0]+256 for info,term in zip(infos,terms))
   assert infos[-1].file_size<=RAW_METADATA_LIMIT[kind]+256 and infos[-1].compress_size<=COMPRESSED_METADATA_LIMIT[kind]
  limit=self.failure_limit if failure else self.normal_limit-self.used();assert len(data)<=limit,('Serialized cap',name,len(data),limit)
  self.guard('write:'+name,len(data)+(FAILURE_JOURNAL_DISK_ALLOWANCE if failure else 0),failure=failure);path=self.out/name;assert not path.exists() and not path.is_symlink();path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  entry={'bytes':len(data),'sha256':sha(path)};self.inventory[name]=entry;self._write_event({'event':'committed','path':name,**entry},failure=failure);return entry
 def save_failure_stream(self,name,value,record):
  assert self.owns() and not Path(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/'))
  path=self.out/name;assert not path.exists() and not path.is_symlink()
  arrays={};bounded=not record['stream_admitted']
  # A header/normalization refusal must not enter the same oversized header
  # route. Keep frozen _encode only when its identity temporary is admitted.
  if bounded:header={'schema':CODEC_SCHEMA,'object':bounded_encode(value,arrays)}
  else:
   normalized=codec_tree(value);header={'schema':CODEC_SCHEMA,'object':_encode(normalized,arrays)}
  self._failure_journal({'event':'failure_stream_admission','path':name,'bounded_header':bounded,
   'bounded_stream_admitted':record['bounded_stream_admitted'],'stream_admitted':record['stream_admitted'],
   'admission_breach':not record['bounded_stream_admitted'],
   'scope':'Unexpected admission breach is failed resource provenance; complete owned evidence is preserved with bounded chunks, never silently relabeled admitted.'})
  metadata=json.dumps(header,separators=(',',':')).encode();assert len(metadata)<=RAW_METADATA_LIMIT['failure']
  arrays['metadata']=np.frombuffer(metadata,dtype=np.uint8)
  # Full raw-metadata compressionBound is used here, avoiding any empirical
  # compression assumption. Failure cap/reserve are checked before file creation.
  upper=sum(zbound(a.nbytes+256)+384 for a in arrays.values())+256
  assert upper<=self.failure_limit,('Complete streaming failure bound exceeds cap',upper,self.failure_limit)
  self.guard('stream_failure:'+name,upper+FAILURE_JOURNAL_DISK_ALLOWANCE,failure=True);path.parent.mkdir(parents=True,exist_ok=True)
  try:
   with path.open('xb') as raw:
    with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
     for key,array in arrays.items():
      with archive.open(key+'.npy','w',force_zip64=True) as member:
       if bounded:write_C_order_npy(member,array)
       else:np.lib.format.write_array(member,array,allow_pickle=False)
    raw.flush();os.fsync(raw.fileno())
   assert path.stat().st_size<=self.failure_limit
   entry={'bytes':path.stat().st_size,'sha256':sha(path)};self.inventory[name]=entry
   self._failure_journal({'event':'committed','path':name,**entry,'encoding':'same_schema_bounded_header_npz' if bounded else 'same_schema_direct_npz','admission':record});return entry
  except BaseException as error:
   json_write(self.out/'failure_stream_error.json',{'status':'failed','path':name,'error_type':type(error).__name__,'error':str(error),
    'partial_file_preserved':path.exists(),'partial_bytes':path.stat().st_size if path.exists() else 0,'admission':record})
   raise
