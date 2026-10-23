"""Read-only official pinned tensor acquisition. Existing L1 reused; no model code executed."""
from pathlib import Path
import json,urllib.request,struct,hashlib,time,resource,shutil,os,sys
R=Path(__file__).resolve().parent
S=Path('/workspace/shared/olmo_recovered/olmo_training_development_complete/olmo_training_origins/sources/trajectory/main')
REV='d7cbab742d80589e714b1a2d7f838dcd21cbe143'
BASE=f'https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/{REV}/'
CAP=5_120_000_000
resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
assert shutil.disk_usage(R).free>6_000_000_000
idx=json.loads((S/'index.json').read_text());assert idx['metadata']['total_size']==5119148032
for name in ['config.json','index.json']:
 p=R/'sources'/name
 if not p.exists():p.write_bytes((S/name).read_bytes())
def getrange(url,a,b):
 req=urllib.request.Request(url+f'?slice={a}-{b}',headers={'Range':f'bytes={a}-{b}','User-Agent':'OLMo-native-function-bounded/1.0'})
 resp=urllib.request.urlopen(req,timeout=120)
 assert resp.status==206 and resp.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'),(resp.status,dict(resp.headers))
 return resp
headers={};headerbytes=0
for shard in sorted(set(idx['weight_map'].values())):
 p=R/'sources'/(shard+'.header.json')
 if not p.exists():
  with getrange(BASE+shard,0,7) as r:n=struct.unpack('<Q',r.read(8))[0]
  assert n<1_000_000
  with getrange(BASE+shard,8,7+n) as r:data=r.read(n);assert len(data)==n
  p.write_bytes(data);headerbytes+=8+n
 data=p.read_bytes();headers[shard]=(json.loads(data),8+len(data))
jobs=[]
for key,shard in idx['weight_map'].items():
 h,off=headers[shard];m=h[key];assert m['dtype']=='F32'
 a,b=[off+v for v in m['data_offsets']];size=b-a
 reuse=Path('/workspace/shared/olmo_signal_production/tensors')/(key.split('.')[-2]+'.f32') if key.startswith('model.layers.1.mlp.') else None
 if reuse and reuse.exists():
  receipt=json.loads(Path(str(reuse)+'.receipt.json').read_text())
  assert receipt['key']==key and receipt['start']==a and receipt['end']==b-1 and reuse.stat().st_size==size
  hsh=hashlib.sha256()
  with reuse.open('rb') as f:
   while block:=f.read(1048576):hsh.update(block)
  assert hsh.hexdigest()==receipt['sha256']
 filename=str(reuse) if reuse and reuse.exists() else str(R/'weights'/(key+'.f32'))
 jobs.append({'key':key,'shape':m['shape'],'file':filename,'url':BASE+shard,'start':a,'end':b-1,'size':size,'reuse':bool(reuse and reuse.exists())})
assert sum(j['size'] for j in jobs)==5119148032
remaining=sum(j['size'] for j in jobs if not j['reuse']);assert remaining+headerbytes<CAP
plan={'revision':REV,'tensor_bytes':5119148032,'reuse_bytes':5119148032-remaining,'new_tensor_bytes':remaining,'download_cap_bytes':CAP,'rss_hard_cap_bytes':4_000_000_000,'jobs':jobs,'created_unix':time.time()}
p=R/'protocols/acquisition_plan.json'
if p.exists():assert json.loads(p.read_text())['jobs']==jobs
else:p.write_text(json.dumps(plan,indent=2))
print('PLAN',json.dumps({k:v for k,v in plan.items() if k!='jobs'}),flush=True)
if '--plan-only' in sys.argv:sys.exit()
started=time.time();received=0
for n,j in enumerate(jobs):
 p=Path(j['file']);rp=Path(str(p)+'.receipt.json')
 if p.exists():
  rr=json.loads(rp.read_text());assert p.stat().st_size==j['size'] and rr['key']==j['key']
  print('REUSE',n,j['key'],flush=True);continue
 tmp=Path(str(p)+'.partial');offset=tmp.stat().st_size if tmp.exists() else 0
 assert offset<j['size']
 sha=hashlib.sha256()
 if offset:
  with tmp.open('rb') as f:
   while block:=f.read(1048576):sha.update(block)
 a=j['start']+offset;b=j['end'];t=time.time()
 try:
  with getrange(j['url'],a,b) as resp:
   cr=resp.headers['Content-Range']
   with tmp.open('ab') as f:
    while offset<j['size']:
     block=resp.read(min(1048576,j['size']-offset));assert block,'Short response'
     f.write(block);sha.update(block);offset+=len(block);received+=len(block)
    assert not resp.read(1)
  assert offset==j['size'];tmp.rename(p)
  rp.write_text(json.dumps({**j,'sha256':sha.hexdigest(),'received_bytes':offset,'content_range':cr,'seconds':time.time()-t,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2))
  print('DONE',n+1,len(jobs),j['key'],j['size'],'elapsed',round(time.time()-started,1),'new_MB',round(received/1e6,1),flush=True)
 except Exception as e:
  (R/'logs/acquisition_failure.json').write_text(json.dumps({'key':j['key'],'error':repr(e),'partial_bytes':tmp.stat().st_size if tmp.exists() else 0,'unix':time.time()},indent=2))
  raise
print('COMPLETE',received,'seconds',time.time()-started,flush=True)
