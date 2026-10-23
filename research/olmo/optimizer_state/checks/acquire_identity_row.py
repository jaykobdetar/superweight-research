#!/usr/bin/env python3
"""One approved HF row, <=65,000 body bytes. Never contacts native shard URL.
No retries, no model loading, no unbounded response reads.
"""
import hashlib,json,pathlib,struct,urllib.request,urllib.error
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'inputs/native_identity'; OUT.mkdir(exist_ok=True)
RECEIPTS=OUT/'receipts.json'; COMMIT='fee32c35e310e1bc659181564529d157b8093ebf'
BASE='https://huggingface.co/allenai/OLMo-1B-0724-hf/'
if RECEIPTS.exists(): raise SystemExit('Receipt file exists: audit prior/uncertain outcome before any new request.')
receipts=[]; used=0
def save(): RECEIPTS.write_text(json.dumps({'bytes_read':used,'receipts':receipts},indent=2)+'\n')
def get(url,filename,maxbytes,byte_range=None):
    global used
    assert used+maxbytes<=65000
    headers={}
    if byte_range:
        a,b=byte_range; headers['Range']=f'bytes={a}-{b}'; url+=f'?slice={a}-{b}'
    rec={'url':url,'requested_file':filename,'bytes_read':0,'state':'request_started'}
    receipts.append(rec); save()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as r:
            rec['status']=r.status; rec['content_range']=r.headers.get('Content-Range'); save()
            if byte_range:
                assert r.status==206 and rec['content_range'].startswith(f'bytes {a}-{b}/'), 'Server did not honor exact range; no body read.'
            else: assert r.status==200
            size=r.headers.get('Content-Length')
            if size: assert int(size)<=maxbytes
            raw=r.read(maxbytes)
            used+=len(raw); rec['bytes_read']=len(raw)
            if byte_range: assert len(raw)==b-a+1
            else: assert len(raw)<maxbytes, 'Metadata hit body bound; stop rather than expand.'
            rec['sha256']=hashlib.sha256(raw).hexdigest(); rec['state']='saved'
            (OUT/filename).write_bytes(raw); save(); return raw
    except Exception as e:
        rec['state']='stopped'; rec['error']=repr(e)
        if isinstance(e,urllib.error.HTTPError): rec['status']=e.code
        save(); raise
idx=json.loads(get(BASE+f'raw/{COMMIT}/model.safetensors.index.json','index.json',12000))
key='model.layers.1.mlp.down_proj.weight'; shard=idx['weight_map'][key]
url=BASE+f'resolve/{COMMIT}/{shard}'
length=struct.unpack('<Q',get(url,'header_length.bin',8,(0,7)))[0]
assert 0<length<20000
h=json.loads(get(url,'header.json',length,(8,7+length)))
t=h[key]; assert t['dtype']=='F32' and t['shape']==[2048,8192]
offset=8+length+t['data_offsets'][0]+1764*8192*4
raw=get(url,'down_row1764.f32',32768,(offset,offset+32767))
result={'model':'allenai/OLMo-1B-0724-hf','branch':'step954000-tokens2000B','commit':COMMIT,
        'tensor':key,'row':1764,'dtype':'F32','shape':[8192],'bytes':len(raw),
        'sha256':hashlib.sha256(raw).hexdigest(),'hf_shard':shard,'hf_file_offset':offset,
        'samples':{str(j):struct.unpack_from('<f',raw,j*4)[0] for j in [0,1710,4820,5755,8041,8191]},
        'scope':'One row only. A match is not whole-checkpoint identity or historical training provenance.'}
(OUT/'identity_row.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2)); print('Total body bytes:',used)
