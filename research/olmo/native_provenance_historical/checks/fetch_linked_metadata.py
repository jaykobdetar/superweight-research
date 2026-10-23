import urllib.request,urllib.error,pathlib,json,hashlib,datetime
root=pathlib.Path(__file__).resolve().parents[1]
urls={
'native954k_config':'https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/config.yaml',
'native954k_model_metadata':'https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/model/metadata.json',
'native954k_optim_metadata':'https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/step954000/optim/metadata.json',
'wandb_dense_comparison':'https://wandb.ai/ai2-llm/olmoe/reports/Plot-OLMoE-1B-7B-vs-OLMo-7B-vs-OLMo-1B--Vmlldzo4OTcyMjEz',
'hf_eos_discussion':'https://huggingface.co/api/models/allenai/OLMo-1B-0724-hf/discussions/2',
'paper_html':'https://arxiv.org/html/2409.02060v2'}
receipts=[]
for name,url in urls.items():
 dest=root/'sources'/(name+'.txt')
 if dest.exists():print('already exists',name);continue
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OLMo-provenance-readonly'}),timeout=30) as r:
   b=r.read(2000001)
   if len(b)>2000000:raise RuntimeError('metadata exceeds 2 MB cap')
   dest.write_bytes(b)
   receipts.append(dict(name=name,url=url,resolved_url=r.url,status=r.status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
   print(name,r.status,len(b),flush=True)
 except urllib.error.HTTPError as e:
  receipts.append(dict(name=name,url=url,status=e.code));print(name,'HTTP',e.code,flush=True)
  if e.code in (401,403):break
 finally:(root/'linked_receipts.json').write_text(json.dumps(receipts,indent=2))
