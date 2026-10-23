from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parents[1];P=json.loads((R/'protocol.json').read_text());expected={x['name']:x['targetCommit'] for x in P['checkpoints']};details=[];total=0
for name,commit in expected.items():
 d=R/'sources'/name;m=json.loads((d/'acquisition.json').read_text());assert m['commit']==commit
 files={x['file'] for x in m['tensors'].values()}|{x['file'] for x in m['embedding_groups']};count=0;size=0
 for f in sorted(files):
  fp=d/f;rec=json.loads(Path(str(fp)+'.receipt.json').read_text());sha=hashlib.sha256()
  with fp.open('rb') as z:
   while block:=z.read(1024*1024):sha.update(block)
  assert sha.hexdigest()==rec['sha256'] and fp.stat().st_size==rec['bytes'];assert rec['url'].startswith('https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/'+commit+'/');a,b=rec['range'];assert rec['content_range'].startswith(f'bytes {a}-{b}/');assert rec['bytes']==b-a+1
  count+=1;size+=fp.stat().st_size
 assert size==m['bytes'];total+=size
 a=json.loads((R/'results'/name/'float32.json').read_text());b=json.loads((R/'results'/name/'fp16_approx.json').read_text());assert len(a['observations'])==len(b['observations'])==52;assert a['revision']==b['revision']==commit
 assert a['protocol_sha256']==b['protocol_sha256']==hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest();assert len(a['single_tokens'])==5;assert a['causal_prefix_max_scaled_error']<1e-4;assert b['causal_prefix_max_scaled_error']<1e-4;assert max(a['peak_rss_bytes'],b['peak_rss_bytes'])<P['resources']['process_rss_cap_bytes']
 details.append({'checkpoint':name,'commit':commit,'files_rehashed':count,'tensor_bytes':size,'results_each_precision':52,'single_tokens_each_precision':5,'peak_rss_bytes':max(a['peak_rss_bytes'],b['peak_rss_bytes'])})
assert json.loads((R/'results/endpoint_validation.json').read_text())['passed'];assert len(list((R/'results').glob('*/float32.json')))==8
logs=sum(json.loads(l)['read_bytes'] for l in (R/'sources/download_events.jsonl').read_text().splitlines());assert logs==total
extra=sum(f.stat().st_size for f in (R/'sources').glob('*') if f.is_file() and f.suffix in ['.json','.py'])
audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'initial_eight_separately_audited':True,'all_checks_passed':True,'protocol_sha256':hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest(),'tensor_payload_bytes':total,'downloaded_small_source_file_bytes':extra,'counted_download_bytes':total+extra,'cap_bytes':P['resources']['download_cap_bytes'],'max_rss_bytes':max(x['peak_rss_bytes'] for x in details),'checkpoints':details,'limits':'Hashes verify downloaded byte slices against their recorded receipts, not an independent whole-checkpoint hash. NumPy formula/source/pulse/invariant checks do not establish full Transformers/CUDA bit equivalence. No later adaptive results included.'}
assert total+extra<=P['resources']['download_cap_bytes'];(R/'deliverables/study_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='checkpoints'},indent=2))
