import urllib.request,urllib.error,pathlib,json,datetime
root=pathlib.Path(__file__).resolve().parents[1]
base='https://olmo-checkpoints.org/ai2-llm/olmo1b-0724/'
if (root/'head_receipts.json').exists():
 raise SystemExit('Historical probe already recorded. Do not retry a restricted target without new owner access clarification.')
# No payloads: these are HEAD requests, and no response body is read.
paths=['step954000/model/rank_9.safetensors','step954000/optim/rank_9.safetensors','step954000/train/rank0.pt','train_data/global_indices.npy']
out=[]
for p in paths:
 try:
  with urllib.request.urlopen(urllib.request.Request(base+p,method='HEAD'),timeout=30) as r:
   row={'path':p,'url':base+p,'status':r.status,'headers':dict(r.headers),'body_bytes_read':0}
 except urllib.error.HTTPError as e:
  row={'path':p,'url':base+p,'status':e.code,'body_bytes_read':0}
 out.append(row); print(p,row['status'],row.get('headers',{}).get('Content-Length'),flush=True)
 (root/'head_receipts.json').write_text(json.dumps(out,indent=2))
 if row['status'] in [401,403]:break
