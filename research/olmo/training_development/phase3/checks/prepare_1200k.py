'''Read only bounded slices of official safetensors checkpoints. No Torch/model loading. All hashes are slice hashes, not whole-shard verification.'''
import urllib.request,urllib.error,json,struct,re,pathlib,hashlib,concurrent.futures,time,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]; OUT=ROOT/'sources'/'trajectory';OUT.mkdir(exist_ok=True)
refs=json.loads((ROOT/'sources/hf_refs.json').read_text())['branches']
by_step={int(re.search(r'step(\d+)',x['name'])[1]):x for x in refs if x['name'].startswith('step')}
steps=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 12000, 16000, 25000, 32000, 48000, 56000, 64000, 72000, 80000, 88000, 96000, 104000, 112000, 120000, 126000, 128000, 144000, 160000, 192000, 256000, 312000, 400000, 512000, 644000, 800000, 1000000, 1431000, 1454000]
if len(sys.argv)>1:steps=[int(s) for s in sys.argv[1].split(',')]
selected=[by_step[s] for s in steps]+[next(x for x in refs if x['name']=='main')]
def get(url,limit=1000000,brange=None):
 headers={'User-Agent':'OLMo-training-origins-read-only/1.0'}
 if brange:
  a,b=brange;headers['Range']=f'bytes={a}-{b}';url+=f'?slice={a}-{b}';limit=b-a+1
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=70) as r:
    if brange:
     assert r.status==206,(url,r.status)
     assert r.headers.get('Content-Range','').startswith(f'bytes {a}-{b}/'),(url,dict(r.headers))
    data=r.read(limit+1);assert len(data)<=limit
    if brange:assert len(data)==limit
    return data,{'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'content_range':r.headers.get('Content-Range')}
  except (TimeoutError,urllib.error.URLError) as e:
   if isinstance(e,urllib.error.HTTPError) and e.code in (401,403):raise
   if attempt==2:raise
   time.sleep(attempt+1)
def run(ref):
 name=ref['name'];sha=ref['targetCommit'];folder=OUT/name;folder.mkdir(exist_ok=True)
 if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
 base=f'https://huggingface.co/allenai/OLMo-1B-0724-hf/';receipts=[]
 data,rec=get(base+f'raw/{sha}/model.safetensors.index.json');receipts.append(rec);(folder/'index.json').write_bytes(data);idx=json.loads(data)
 data,rec=get(base+f'raw/{sha}/config.json');receipts.append(rec);(folder/'config.json').write_bytes(data)
 key='model.layers.1.mlp.down_proj.weight';fname=idx['weight_map'][key];url=base+f'resolve/{sha}/{fname}'
 data,rec=get(url,brange=(0,7));receipts.append(rec);n=struct.unpack('<Q',data)[0];assert n<1000000
 data,rec=get(url,brange=(8,7+n));receipts.append(rec);(folder/'header.json').write_bytes(data);h=json.loads(data)
 tensor=h[key];assert tensor['dtype']=='F32' and tensor['shape']==[2048,8192],tensor
 a=8+n+tensor['data_offsets'][0]+1764*8192*4;b=a+8192*4-1
 data,rec=get(url,brange=(a,b));receipts.append(rec);(folder/'down_row1764.f32').write_bytes(data);w=np.frombuffer(data,dtype='<f4').astype('float64')
 order=np.argsort(-abs(w));top=[{'column':int(i),'weight':float(w[i])} for i in order[:10]]
 result={'name':name,'commit':sha,'step':int(re.search(r'step(\d+)',name)[1]) if name!='main' else None,'primary':float(w[1710]),'secondary':float(w[8041]),'primary_abs_rank':int(np.where(order==1710)[0][0])+1,'secondary_abs_rank':int(np.where(order==8041)[0][0])+1,'row_rms':float(np.sqrt(np.mean(w*w))),'row_median_abs':float(np.median(abs(w))),'primary_row_energy_share':float(w[1710]**2/np.sum(w*w)),'pair_row_energy_share':float((w[1710]**2+w[8041]**2)/np.sum(w*w)),'top10':top,'receipts':receipts}
 (folder/'result.json').write_text(json.dumps(result,indent=2));print(name, 'primary',result['primary'],'rank',result['primary_abs_rank'],'secondary',result['secondary'],'rank',result['secondary_abs_rank'],'rms',result['row_rms'],flush=True);return result

assert len(sys.argv)==2 and sys.argv[1]=="1200000"
r=run(next(x for x in selected if x["name"].startswith("step1200000-")))
assert sum(q["bytes"] for q in r["receipts"]) <= 100000
print("Metadata bytes",sum(q["bytes"] for q in r["receipts"]))
