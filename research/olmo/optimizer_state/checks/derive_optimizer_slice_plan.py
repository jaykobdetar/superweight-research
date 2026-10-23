#!/usr/bin/env python3
"""Derive a minimal read-only Adam diagnostic request. No network or tensor load."""
import hashlib, json, pathlib, math
ROOT=pathlib.Path(__file__).resolve().parents[1]
P=ROOT/'inputs/archive_recovery/native954k_optim_metadata.txt'
U=ROOT/'inputs/user_optimizer_metadata/metadata.json'
assert P.read_bytes()==U.read_bytes()
md=json.loads(P.read_text())['tensors']
base='https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/optim/'
param='state.transformer.blocks.1.ff_out.weight.__'
row_start=1764*8192; row_end=row_start+8192
assert (row_start,row_end)==(14450688,14458880)
items=[]
for suffix in ('exp_avg','exp_avg_sq'):
    k=param+suffix; x=md[k]
    assert x['dtype']=='F32' and x['shape']==[2048,8192]
    for f,(a,b) in x['flattened_offsets_per_file'].items():
        lo,hi=max(a,row_start),min(b,row_end)
        if lo<hi:
            items.append(dict(tensor=k,file=f,url=base+f,global_elements=[lo,hi],
                 local_elements=[lo-a,hi-a],tensor_relative_bytes=[4*(lo-a),4*(hi-a)],bytes=4*(hi-lo)))
for k in (param+'step','num_param_groups','param_group0','state_keys'):
    x=md[k]; n=math.prod(x['shape']); size={'F32':4,'I64':8,'U8':1}[x['dtype']]
    for f,(a,b) in x['flattened_offsets_per_file'].items():
        if a<b:
            assert a==0 and b==n
            items.append(dict(tensor=k,file=f,url=base+f,global_elements=[a,b],
                 local_elements=[0,n],tensor_relative_bytes=[0,n*size],bytes=n*size))
payload={}
for x in md.values():
    size={'F32':4,'I64':8,'U8':1}[x['dtype']]
    for f,(a,b) in x['flattened_offsets_per_file'].items():payload[f]=payload.get(f,0)+(b-a)*size
files=sorted({x['file'] for x in items})
columns=[]
for j in (1710,8041):
    for suffix in ('exp_avg','exp_avg_sq'):
        k=param+suffix; m=md[k]
        for f,(a,b) in m['flattened_offsets_per_file'].items():
            indices=[i*8192+j for i in range(2048) if a<=i*8192+j<b]
            if indices:
                assert all(y-x==8192 for x,y in zip(indices,indices[1:]))
                columns.append(dict(tensor=k,column=j,file=f,url=base+f,
                    first_global_element=indices[0],first_local_element=indices[0]-a,
                    element_stride=8192,count=len(indices),bytes=4*len(indices),
                    caution='Discontiguous four-byte reads; naive HTTP ranges have high request overhead. Not included in minimal row plan.'))
spec=dict(checkpoint='native OLMo-1B-0724 step954000',
    metadata_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),
    metadata_bytes=P.stat().st_size,
    user_uploaded_copy_exactly_matches_archived_official=True,
    status='Frozen acquisition specification, subsequently executed within cap; see inputs/optimizer_slices/receipt.json. This derivation script itself makes no network requests.',
    minimal_row_requests=items,
    minimal_payload_bytes=sum(x['bytes'] for x in items),
    files={f:dict(url=base+f,tensor_payload_bytes_excluding_header=payload[f]) for f in files},
    header_rule='For each file read its first 8 bytes as little-endian u64 H, reject H>1MiB, then read exactly H bytes of JSON. Validate all tensor offsets, dtypes, shapes and file bounds. Each absolute interval equals 8+H+header[tensor].data_offsets[0]+tensor_relative_bytes. Never reuse model-shard offsets for optimizer shards.',
    serialization='param_group0 and state_keys are U8 pickle bytes in the verified May 2024 official serializer. Inspect with pickletools and a primitive-only decoder; never use pickle.loads, torch.load, GLOBAL, REDUCE, BUILD or arbitrary imports. Reject unsupported opcodes.',
    minimal_hard_cap_including_headers_bytes=512*1024,
    optional_columns_not_authorized_by_this_plan=columns,
    optional_column_tensor_bytes=sum(x['bytes'] for x in columns),
    optional_columns_require_matching_weight_columns=True,
    interpretation='One late-state observation can diagnose current adaptive scaling, not establish historical causality or an authentic next gradient. Saved moments reconstruct the stored direction, not a new batch update.',
    this_script_attempts_no_network=True)
assert spec['minimal_payload_bytes']==72465
(ROOT/'results/optimizer_slice_plan.json').write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps({k:spec[k] for k in ('metadata_sha256','minimal_payload_bytes','files','optional_column_tensor_bytes')},indent=2))
