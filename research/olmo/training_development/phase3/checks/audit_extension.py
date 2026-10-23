"""Audit four adaptive checkpoints without modifying the initial study."""
from pathlib import Path
import hashlib,json,datetime,struct
R=Path(__file__).resolve().parents[1];B=R.parent;P=json.loads((R/'protocol.json').read_text());Q=json.loads((B/'phase2/protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B/'phase2/protocol.json')==P['original_protocol_sha256']
for k in ['panels','neurons','embedding_ids','q','p','worlds','edits','numerics']:assert P[k]==Q[k],k
for code in ['prefix_model.py','run_prefix.py']:assert sha(R/'checks'/code)==sha(B/'phase2/checks'/code),code
assert json.loads((B/'phase2/results/endpoint_validation.json').read_text())['passed']
assert not list((R/'sources').rglob('*.partial'))
assert len(list((R/'results').glob('*/float32.json')))==len(P['checkpoints'])==4
assert len(list((R/'results').glob('*/fp16_approx.json')))==4
rows=[];total=0
for ref in P['checkpoints']:
 name=ref['name'];commit=ref['targetCommit'];d=R/'sources'/name;m=json.loads((d/'acquisition.json').read_text());assert m['commit']==commit
 prior=B/'sources/trajectory'/name
 for k in ['config.json','index.json']:assert sha(prior/k)==sha(B/'sources/trajectory/main'/k),(name,k)
 assert sha(prior/'down_row1764.f32')==m['copied_down_row_sha256']
 files={t['file'] for t in m['tensors'].values()}|{t['file'] for t in m['embedding_groups']};nbytes=0
 for fn in files:
  f=d/fn;r=json.loads(Path(str(f)+'.receipt.json').read_text());h=hashlib.sha256()
  with f.open('rb') as stream:
   while block:=stream.read(1024*1024):h.update(block)
  assert h.hexdigest()==r['sha256'] and f.stat().st_size==r['bytes']
  assert r['url'].startswith('https://huggingface.co/allenai/OLMo-1B-0724-hf/resolve/'+commit+'/')
  a,b=r['range'];assert r['content_range'].startswith(f'bytes {a}-{b}/') and r['bytes']==b-a+1;nbytes+=r['bytes']
 assert nbytes==m['bytes']==339214336;total+=nbytes
 a=json.loads((R/'results'/name/'float32.json').read_text());b=json.loads((R/'results'/name/'fp16_approx.json').read_text())
 for z in [a,b]:
  assert z['revision']==commit and z['protocol_sha256']==sha(R/'protocol.json')
  assert len(z['observations'])==52 and len(z['single_tokens'])==5
  assert z['causal_prefix_max_scaled_error']<1e-4 and z['peak_rss_bytes']<P['resources']['process_rss_cap_bytes']
 rows.append({'checkpoint':name,'commit':commit,'files_rehashed':len(files),'tensor_bytes':nbytes,'peak_rss_bytes':max(a['peak_rss_bytes'],b['peak_rss_bytes'])})
logs=sum(json.loads(l)['read_bytes'] for l in (R/'sources/download_events.jsonl').read_text().splitlines());assert logs==total
newmeta=B/'sources/trajectory/step1200000-tokens2515B';rec=json.loads((newmeta/'result.json').read_text())['receipts'];metadata=sum(r['bytes'] for r in rec);assert metadata==55597
for f,r in zip(['index.json','config.json',None,'header.json','down_row1764.f32'],rec):
 data=(newmeta/f).read_bytes() if f else struct.pack('<Q',(newmeta/'header.json').stat().st_size)
 assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
initial=json.loads((B/'combined/initial_evidence_baseline.json').read_text())
for path,wanted in initial['files'].items():assert sha(B/path)==wanted,path
assert total+metadata<=P['resources']['download_cap_bytes']
initial_transfer=json.loads((B/'phase2/deliverables/study_audit.json').read_text())['counted_download_bytes']
audit={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'adaptive_extension_separately_audited':True,'all_checks_passed':True,'protocol_sha256':sha(R/'protocol.json'),'initial_protocol_sha256':sha(B/'phase2/protocol.json'),'same_frozen_panel_and_implementation':True,'initial_evidence_unchanged':True,'checkpoints':rows,'tensor_payload_bytes':total,'new_metadata_bytes':metadata,'counted_extension_download_bytes':total+metadata,'cap_bytes':P['resources']['download_cap_bytes'],'combined_initial_and_extension_transfer_bytes':initial_transfer+total+metadata,'max_rss_bytes':max(r['peak_rss_bytes'] for r in rows),'limits':'Adaptive checkpoint selection; original probes unchanged. Receipts do not establish independent whole-model hashes. Formula/reference/invariant checks do not prove full-framework bit equivalence. No loss, gradients, training data or optimizer states.'}
(R/'deliverables/study_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
