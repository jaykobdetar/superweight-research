"""Offline integrity and cohort consistency checks; no model loading or network."""
from pathlib import Path
import json,hashlib
B=Path(__file__).resolve().parents[2];M=json.loads((B/'BUNDLE_MANIFEST.json').read_text())
for rel,r in M['files'].items():
 f=B/rel;assert f.is_file(),rel;assert f.stat().st_size==r['bytes'],rel;assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],rel
count=0
for phase,n in [('phase2',8),('phase3',4)]:
 R=B/phase;P=json.loads((R/'protocol.json').read_text());assert len(P['checkpoints'])==n
 for ref in P['checkpoints']:
  for mode in ['float32','fp16_approx']:
   z=json.loads((R/'results'/ref['name']/(mode+'.json')).read_text());assert z['revision']==ref['targetCommit'];assert z['protocol_sha256']==hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest();assert len(z['observations'])==52 and len(z['single_tokens'])==5;count+=1
 assert json.loads((R/'deliverables/study_audit.json').read_text())['all_checks_passed']
A=json.loads((B/'phase2/protocol.json').read_text());D=json.loads((B/'phase3/protocol.json').read_text())
for key in ['panels','neurons','worlds','edits','numerics','q','p']:assert A[key]==D[key]
print('Verified',len(M['files']),'compact files and',count,'result datasets. Bulk tensor hashes require omitted payload.')
