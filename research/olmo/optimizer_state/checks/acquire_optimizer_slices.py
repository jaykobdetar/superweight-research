#!/usr/bin/env python3
"""One bounded, separately approved public-range retrieval. NO retries or training.

Official optimizer rank9/rank8/rank0 only. The metadata URL is never requested.
Reruns are blocked by an on-disk receipt even if the first request failed.
"""
import json, pathlib, struct, urllib.request, urllib.error, hashlib, math, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'inputs/optimizer_slices'; OUT.mkdir(exist_ok=True)
receipt=OUT/'receipt.json'
if receipt.exists(): raise SystemExit('Receipt already exists; no retry is authorized by this script.')
spec=json.loads((ROOT/'results/optimizer_slice_plan.json').read_text())
md=json.loads((ROOT/'inputs/archive_recovery/native954k_optim_metadata.txt').read_text())['tensors']
CAP=512*1024; used=0
audit={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'status':'started','body_bytes':0,'cap_bytes':CAP,'requests':[],
       'scope':'Official optimizer headers and 72465 selected tensor bytes only'}
def save():receipt.write_text(json.dumps(audit,indent=2)+'\n')
save()
def get(url,a,b,name):
    global used
    n=b-a; assert 0<n<=CAP-used
    rec={'url':url,'inclusive_range':[a,b-1],'expected_bytes':n,'local_file':name,'status':'pending'}
    audit['requests'].append(rec);save()
    req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b-1}','Accept-Encoding':'identity','User-Agent':'OLMo-training-origin-read-only-research/1.0'})
    try:
        with urllib.request.urlopen(req,timeout=60) as r:
            rec['http_status']=r.status;rec['content_range']=r.headers.get('Content-Range');save()
            if r.status!=206:raise ValueError('Server did not honor the exact range; body not read.')
            cr=rec['content_range']; assert cr and cr.startswith(f'bytes {a}-{b-1}/'),cr
            total=int(cr.split('/')[-1]); assert total>=b
            data=r.read(n);used+=len(data);audit['body_bytes']=used
            assert len(data)==n
    except urllib.error.HTTPError as e:
        rec['http_status']=e.code;raise
    path=OUT/name;path.write_bytes(data)
    rec.update(status='complete',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),remote_file_bytes=total);save()
    return data,total
try:
    for rank in (9,8,0):
        f=f'rank_{rank}.safetensors';url=spec['files'][f]['url']
        pre,total=get(url,0,8,f'rank{rank}_header_length.bin')
        h=struct.unpack('<Q',pre)[0];assert 2<=h<=min(1024*1024,CAP-used)
        raw,total2=get(url,8,8+h,f'rank{rank}_header.json');assert total==total2
        header=json.loads(raw); assert isinstance(header,dict)
        intervals=[]
        for key,ent in header.items():
            if key=='__metadata__':continue
            assert key in md
            dtype=ent['dtype'];assert dtype==md[key]['dtype']
            size={'F32':4,'I64':8,'U8':1}[dtype]
            a,b=ent['data_offsets'];assert 0<=a<=b<=total-8-h
            global_interval=md[key]['flattened_offsets_per_file'].get(f,[0,0])
            count=global_interval[1]-global_interval[0]
            assert math.prod(ent['shape'])==count
            assert b-a==count*size
            if a<b:intervals.append((a,b))
        intervals.sort();assert all(a[1]==b[0] for a,b in zip(intervals,intervals[1:]))
        assert intervals[0][0]==0 and intervals[-1][1]==total-8-h
        for item in spec['minimal_row_requests']:
            if item['file']!=f:continue
            key=item['tensor']; ent=header[key];a,b=item['tensor_relative_bytes'];o=8+h+ent['data_offsets'][0]
            assert b<=ent['data_offsets'][1]-ent['data_offsets'][0]
            name=('row1764_'+key.rsplit('__',1)[-1]+'.f32') if key.endswith(('__exp_avg','__exp_avg_sq')) else ('step.f32' if key.endswith('__step') else key+'.bin')
            _,checktotal=get(url,o+a,o+b,name);assert checktotal==total
    audit['status']='complete'
except Exception as e:
    audit['status']='stopped';audit['error']=repr(e)
    if audit['requests'] and audit['requests'][-1]['status']=='pending':audit['requests'][-1]['status']='failed'
    save();raise
finally:
    audit['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
print(json.dumps(audit,indent=2))
