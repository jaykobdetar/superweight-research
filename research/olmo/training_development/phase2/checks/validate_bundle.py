"""Verify compact evidence files offline. No model loading or network."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2]
M=json.loads((R/'BUNDLE_MANIFEST.json').read_text())
for rel,rec in M['files'].items():
 p=R/rel
 assert p.is_file(),rel
 assert p.stat().st_size==rec['bytes'],rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'],rel
print('Verified',len(M['files']),'compact evidence files; this is not a whole-model hash.')
