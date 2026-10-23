#!/usr/bin/env python3
"""Bounded preselected one-column moment sample, no model/optimizer execution."""
import pathlib,json,hashlib,urllib.request,urllib.error,threading,concurrent.futures,datetime,struct
ROOT=pathlib.Path(__file__).resolve().parent
FROZEN=pathlib.Path('/workspace/shared/olmo_training_origins_frozen_20261023T0615Z')
P=ROOT/'protocol.json';protocol=json.loads(P.read_text())
receipt=ROOT/'receipt.json'
if receipt.exists():raise SystemExit('Existing receipt: never rerun or blindly retry.')
assert hashlib.sha256(P.read_bytes()).hexdigest()=='b1b962fcc6cfa9faf4bac0270eb17f4dbf74ddc209220f7ea118104f0c210d7f'
old=json.loads((FROZEN/'inputs/optimizer_slices/receipt.json').read_text());assert old['status']=='complete'
md=json.loads((FROZEN/'inputs/archive_recovery/native954k_optim_metadata.txt').read_text())['tensors']
tasks=[]
for row in protocol['sample_rows']:
 for suffix in ('exp_avg','exp_avg_sq'):
  key='state.transformer.blocks.1.ff_out.weight.__'+suffix
  i=row*8192+1710;f=next(f for f,(a,b) in md[key]['flattened_offsets_per_file'].items() if a<=i<b)
  rank=int(f.split('_')[1].split('.')[0]);assert rank in (8,9)
  hp=FROZEN/f'inputs/optimizer_slices/rank{rank}_header.json';h=json.loads(hp.read_text())
  prev=next(r for r in old['requests'] if r['local_file']==hp.name)
  assert hashlib.sha256(hp.read_bytes()).hexdigest()==prev['sha256']
  start=8+len(hp.read_bytes())+h[key]['data_offsets'][0]+4*(i-md[key]['flattened_offsets_per_file'][f][0])
  tasks.append({'row':row,'moment':suffix,'url':prev['url'],'range':[start,start+3],
   'remote_bytes':prev['remote_file_bytes'],'status':'not_started'})
audit={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'status':'running','body_bytes':0,'requests':tasks}
lock=threading.Lock();stop=threading.Event()
def save():receipt.write_text(json.dumps(audit,indent=2)+'\n')
save()
def read(t):
 if stop.is_set():return
 with lock:t['status']='pending';save()
 a,b=t['range'];req=urllib.request.Request(t['url'],headers={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity','User-Agent':'OLMo-training-origin-read-only-research/1.0'})
 try:
  with urllib.request.urlopen(req,timeout=60) as r:
   t['http_status']=r.status;t['content_range']=r.headers.get('Content-Range')
   if r.status!=206:raise ValueError('Range not honored; body not read')
   assert t['content_range']==f'bytes {a}-{b}/{t["remote_bytes"]}'
   data=r.read(4);assert len(data)==4
  with lock:
   audit['body_bytes']+=len(data);assert audit['body_bytes']<=protocol['maximum_total_body_bytes']
   t.update(status='complete',data_hex=data.hex(),sha256=hashlib.sha256(data).hexdigest());save()
 except Exception as e:
  stop.set()
  with lock:
   t['status']='failed';t['error']=repr(e)
   if isinstance(e,urllib.error.HTTPError):t['http_status']=e.code
   audit['status']='stopped';save()
  raise
it=iter(tasks);errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 pending={pool.submit(read,next(it)) for _ in range(8)}
 while pending:
  done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
  for job in done:
   try:job.result()
   except Exception as e:errors.append(repr(e));stop.set()
  if not stop.is_set():
   for _ in done:
    t=next(it,None)
    if t is not None:pending.add(pool.submit(read,t))
audit['status']='stopped' if errors else 'complete';audit['errors']=errors
audit['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
if not errors:
 assert audit['body_bytes']==1024 and all(t['status']=='complete' for t in tasks)
 for suffix in ('exp_avg','exp_avg_sq'):
  (ROOT/f'{suffix}.f32').write_bytes(b''.join(bytes.fromhex(t['data_hex']) for t in tasks if t['moment']==suffix))
print(json.dumps({k:audit[k] for k in ('status','body_bytes','errors','finished_utc')},indent=2))
if errors:raise SystemExit(1)
