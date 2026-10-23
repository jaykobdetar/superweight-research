'''Official immutable HF tensors only; streamed bounded byte ranges, no full checkpoints.'''
from pathlib import Path
import json,hashlib,urllib.request,urllib.error,concurrent.futures,threading,time,sys,os,shutil
ROOT=Path(__file__).resolve().parents[1];P=json.loads((ROOT/'protocol.json').read_text());CAP=P['resources']['download_cap_bytes'];LOCK=threading.Lock();LOG=ROOT/'sources/download_events.jsonl'
used=sum(p.stat().st_size for p in (ROOT/'sources').glob('*') if p.is_file() and p.suffix in ['.json','.py'])
if LOG.exists():used+=sum(json.loads(s).get('read_bytes',0) for s in LOG.read_text().splitlines())
def log(v):
 with LOCK:
  with LOG.open('a') as f:f.write(json.dumps(v)+'\n')
def transfer(job):
 global used
 path,url,a,b=job;recpath=Path(str(path)+'.receipt.json')
 if path.exists():
  assert recpath.exists();r=json.loads(recpath.read_text());assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];return r
 assert not Path(str(path)+'.partial').exists(),'Uncertain partial download: inspect before resuming'
 req=urllib.request.Request(url+f'?slice={a}-{b}',headers={'Range':f'bytes={a}-{b}','User-Agent':'OLMo-bounded-prefix/1.0'})
 n=b-a+1;sha=hashlib.sha256();read=0;start=time.time();target=Path(str(path)+'.partial')
 with urllib.request.urlopen(req,timeout=90) as r:
  assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'),(r.status,dict(r.headers))
  with target.open('wb') as f:
   while read<n:
    data=r.read(min(1024*1024,n-read));assert data,'Unexpected early EOF'
    with LOCK:
     assert used+len(data)<=CAP,'Download budget exceeded';used+=len(data)
    f.write(data);sha.update(data);read+=len(data)
    log({'url':url,'range':[a,b],'read_bytes':len(data)})
   assert not r.read(1),'Oversized range response'
 receipt={'url':url,'range':[a,b],'content_range':r.headers.get('Content-Range'),'bytes':read,'sha256':sha.hexdigest(),'seconds':time.time()-start}
 assert read==n;target.rename(path);recpath.write_text(json.dumps(receipt,indent=2));return receipt
name=sys.argv[1] if len(sys.argv)>1 else 'main';ref=next(r for r in P['checkpoints'] if r['name']==name);out=ROOT/'sources'/name;out.mkdir(exist_ok=True)
prior=ROOT.parent/'sources/trajectory'/name;h=json.loads((prior/'header.json').read_text());idx=json.loads((prior/'index.json').read_text());offset=8+(prior/'header.json').stat().st_size
base=f"https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/{ref['targetCommit']}/"
jobs=[];tensors={}
for key,t in h.items():
 if key.startswith('model.layers.0.') or key.startswith('model.layers.1.self_attn.'):
  assert t['dtype']=='F32';a,b=t['data_offsets'];path=out/(key+'.f32');jobs.append((path,base+idx['weight_map'][key],offset+a,offset+b-1));tensors[key]={'file':path.name,'shape':t['shape']}
for part in ['gate_proj','up_proj']:
 key=f'model.layers.1.mlp.{part}.weight';t=h[key]
 for row in P['neurons']:
  a=offset+t['data_offsets'][0]+row*2048*4;b=a+2048*4-1;path=out/f'{part}_{row}.f32';jobs.append((path,base+idx['weight_map'][key],a,b));tensors[f'{part}_{row}']={'file':path.name,'shape':[2048]}
# Merge nearby embedding IDs into bounded spans, avoiding hundreds of one-row requests.
groups=[]
for i in P['embedding_ids']:
 if groups and i-groups[-1][-1]<=6 and i-groups[-1][0]<64:groups[-1].append(i)
 else:groups.append([i])
key='model.embed_tokens.weight';t=h[key];eg=[]
for g in groups:
 lo,hi=g[0],g[-1];a=offset+t['data_offsets'][0]+lo*8192;b=offset+t['data_offsets'][0]+(hi+1)*8192-1;path=out/f'embedding_{lo}_{hi}.f32';jobs.append((path,base+idx['weight_map'][key],a,b));eg.append({'file':path.name,'lo':lo,'hi':hi})
predicted=sum(b-a+1 for _,_,a,b in jobs);print(name,'jobs',len(jobs),'bytes',predicted,'all8',predicted*8,'used',used,flush=True);assert predicted*8+2200000<=CAP
receipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 fs={pool.submit(transfer,j):j[0] for j in jobs}
 for f in concurrent.futures.as_completed(fs):
  receipts.append(f.result())
  if len(receipts)%20==0:print(name,'done',len(receipts),'/',len(jobs),'used',used,flush=True)
manifest={'name':name,'commit':ref['targetCommit'],'tensors':tensors,'embedding_groups':eg,'copied_down_row_sha256':hashlib.sha256((prior/'down_row1764.f32').read_bytes()).hexdigest(),'bytes':predicted,'receipts':receipts}
(out/'acquisition.json').write_text(json.dumps(manifest,indent=2));print('COMPLETE',name,'used',used,flush=True)
