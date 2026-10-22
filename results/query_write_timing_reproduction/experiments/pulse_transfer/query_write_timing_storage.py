"""T21 immutable lossless storage, reusing the unchanged reviewed T20 codec/writer.

Reviewed resource-only caps:512MiB normal/1GiB failure; frozen T20 defaults stay384MiB.
No model loading, truncation, independent static tensor counts, or deduplication.
"""
from post_write_norm_storage import (EvidenceWriter as T20EvidenceWriter,NORMAL_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,
 READOUT_ROWS,FAILURE_ROWS,sha,json_write,codec_tree,byte_exact,nested_fp,summarize_diag,
 trim_gate,trim_gate_norm,logit_rows,source_closure,read_npz)
from post_write_norm_storage import empty_payload as old_empty_payload,failure_bundle as old_failure_bundle
from gate_query_write_timing import ARMS
SCHEMA='query_write_timing_explicit_v1'
TIMING_ROWS=(1,126,127,510,511)
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')

def empty_payload(metadata):
 value=old_empty_payload(metadata);value['schema']=SCHEMA;value['timing_samples']={};return value

def failure_bundle(bundle):
 value=old_failure_bundle(bundle)
 if value is not None:
  value.update({k:bundle[k] for k in ('timing_current',) if k in bundle})
 return value

# Explicit local constructor defaults are required: rebinding a module constant
# cannot change the already-defined frozen T20 class's384MiB default argument.
FAILURE_LIMIT=1024**3
class EvidenceWriter(T20EvidenceWriter):
 def __init__(self,out,*,normal_limit=NORMAL_LIMIT,failure_limit=FAILURE_LIMIT,reserve=DISK_RESERVE,outstanding_other_caps=0):
  super().__init__(out,normal_limit=normal_limit,failure_limit=failure_limit,reserve=reserve,outstanding_other_caps=outstanding_other_caps)
