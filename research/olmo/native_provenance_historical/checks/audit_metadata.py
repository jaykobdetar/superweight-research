"""Offline-only consistency audit. Never requests or deserializes model payloads."""
from pathlib import Path
import json,math,hashlib,collections
R=Path(__file__).resolve().parents[1];S=R/'sources'
m=json.loads((S/'native954k_model_metadata.txt').read_text())['tensors']
o=json.loads((S/'native954k_optim_metadata.txt').read_text())['tensors']
expected=json.loads((R/'deliverables/native_metadata_audit.json').read_text())
assert len(m)==66 and len(o)==201
assert sum(math.prod(v['shape']) for v in m.values())==1279787008
assert all(v['dtype']=='F32' for v in m.values())
assert not any(any(s in k for s in ('expert','router','moe')) for k in m)
for k,v in m.items():
 ranges=sorted((int(a),int(b)) for a,b in v['flattened_offsets_per_file'].values() if a!=b)
 assert ranges[0][0]==0 and ranges[-1][1]==math.prod(v['shape'])
 assert all(a[1]==b[0] for a,b in zip(ranges,ranges[1:]))
 for s in ('__exp_avg','__exp_avg_sq'):
  z=o['state.'+k+'.'+s]
  assert z['shape']==v['shape'] and z['dtype']=='F32'
  assert z['flattened_offsets_per_file']==v['flattened_offsets_per_file']
v=m['transformer.blocks.1.ff_out.weight'];idx=1764*8192+1710
rank=next(k for k,(a,b) in v['flattened_offsets_per_file'].items() if a<=idx<b)
assert rank=='rank_9.safetensors' and idx-v['flattened_offsets_per_file'][rank][0]==8808110
assert list(o['state.transformer.blocks.1.ff_out.weight.__step']['flattened_offsets_per_file'])==['rank_8.safetensors']
verified=[]
for receipt in json.loads((R/'linked_receipts.json').read_text()):
 if receipt['name'].startswith('native954k_'):
  b=(S/(receipt['name']+'.txt')).read_bytes()
  assert len(b)==receipt['bytes'] and hashlib.sha256(b).hexdigest()==receipt['sha256']
  assert receipt['status']==200;verified.append(receipt['name'])
assert len(verified)==3
heads=json.loads((R/'head_receipts.json').read_text())
assert len(heads)==1 and heads[0]['status']==403 and heads[0]['body_bytes_read']==0
print(json.dumps({'offline_checks':'passed','verified_metadata_objects':verified,'model_parameters':1279787008,'moment_shapes_and_partition_maps':'all 66 match','tensor_payload_downloads':0,'head_access_restriction':'recorded, no retry'},indent=2))
