"""Acquire only three pinned final L1 MLP tensors, never the full checkpoint."""
from pathlib import Path
import json, hashlib, urllib.request, time, sys, resource, os
R=Path(__file__).resolve().parent
B=Path(os.environ.get('OLMO_SOURCE_ROOT',str(R/'inputs/olmo_training_origins')))
S=B/'sources/trajectory/main'
OUT=R/'tensors';OUT.mkdir(exist_ok=True)
REV='d7cbab742d80589e714b1a2d7f838dcd21cbe143'
h=json.loads((S/'header.json').read_text());idx=json.loads((S/'index.json').read_text());offset=8+(S/'header.json').stat().st_size
jobs=[]
for name in ('gate_proj','up_proj','down_proj'):
 key=f'model.layers.1.mlp.{name}.weight';m=h[key];assert m['dtype']=='F32'
 assert m['shape']==([2048,8192] if name=='down_proj' else [8192,2048])
 a,b=[offset+v for v in m['data_offsets']]
 jobs.append({'key':key,'shape':m['shape'],'file':name+'.f32','url':f'https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/{REV}/'+idx['weight_map'][key],'start':a,'end':b-1,'size':b-a})
assert sum(j['size'] for j in jobs)==201326592
plan={'revision':REV,'created_unix':time.time(),'download_cap_bytes':201326592,'jobs':jobs,'scope':'Only L1 MLP; cached prefix states; no later layers; no downstream behavioral claim.'}
pf=R/'acquisition_plan.json'
if not pf.exists():pf.write_text(json.dumps(plan,indent=2))
for j in jobs:
 p=OUT/j['file'];rp=Path(str(p)+'.receipt.json')
 if p.exists():
  rr=json.loads(rp.read_text());assert p.stat().st_size==j['size']
  assert all(rr[k]==j[k] for k in ('key','shape','file','url','start','end','size')),'Existing receipt does not identify the requested immutable tensor range'
  hh=hashlib.sha256()
  with p.open('rb') as f:
   while b:=f.read(1048576):hh.update(b)
  assert hh.hexdigest()==rr['sha256'];print('Verified existing',p,flush=True);continue
 tmp=Path(str(p)+'.partial');assert not tmp.exists(),'Do not overwrite uncertain partial'
 a,b=j['start'],j['end'];req=urllib.request.Request(j['url']+f'?slice={a}-{b}',headers={'Range':f'bytes={a}-{b}','User-Agent':'OLMo-signal-production-bounded/1.0'})
 start=time.time();total=0;sha=hashlib.sha256()
 with urllib.request.urlopen(req,timeout=120) as resp:
  assert resp.status==206 and resp.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'),(resp.status,dict(resp.headers))
  cr=resp.headers.get('Content-Range')
  with tmp.open('wb') as f:
   while total<j['size']:
    block=resp.read(min(1048576,j['size']-total));assert block,'Short response'
    f.write(block);sha.update(block);total+=len(block)
   assert not resp.read(1),'Response exceeds authorized range'
 assert total==j['size'];tmp.rename(p)
 rp.write_text(json.dumps({**j,'sha256':sha.hexdigest(),'content_range':cr,'received_bytes':total,'seconds':time.time()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
 print('COMPLETE',j['key'],total,sha.hexdigest(),flush=True)
