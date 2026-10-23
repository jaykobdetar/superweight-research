import urllib.request,urllib.error,pathlib,json,hashlib,datetime
root=pathlib.Path(__file__).resolve().parents[1]
urls={
'issue901':'https://api.github.com/repos/allenai/OLMo/issues/901',
'issue901_comments':'https://api.github.com/repos/allenai/OLMo/issues/901/comments',
'olmoe_issue9_comments':'https://api.github.com/repos/allenai/OLMoE/issues/9/comments',
'olmoe_issue32_comments':'https://api.github.com/repos/allenai/OLMoE/issues/32/comments',
'current_readme':'https://raw.githubusercontent.com/allenai/OLMo/main/README.md',
'config_history':'https://api.github.com/repos/allenai/OLMo/commits?path=configs/official-0724/OLMo-1B.yaml&per_page=30',
'hf_discussions':'https://huggingface.co/api/models/allenai/OLMo-1B-0724-hf/discussions',
'olmoe_readme':'https://raw.githubusercontent.com/allenai/OLMoE/main/README.md'}
receipts=[]
for name,url in urls.items():
 dest=root/'sources'/(name+'.txt')
 if dest.exists(): print('already exists',name);continue
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'OLMo-provenance-readonly'}),timeout=30) as r:
   b=r.read(2000001)
   if len(b)>2000000: raise RuntimeError('metadata exceeds 2 MB cap')
   dest.write_bytes(b)
   receipt=dict(name=name,url=url,status=r.status,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
   receipts.append(receipt); print(name,r.status,len(b),flush=True)
 except urllib.error.HTTPError as e:
  receipts.append(dict(name=name,url=url,status=e.code));print(name,'HTTP',e.code,flush=True)
  if e.code in (401,403):break
 finally:
  (root/'receipts.json').write_text(json.dumps(receipts,indent=2))
