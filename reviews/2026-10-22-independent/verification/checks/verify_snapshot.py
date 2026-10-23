"""Verify a local snapshot against the recorded public Git tree (read-only)."""
import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parents[1]
tree=json.loads((root/'tree.json').read_text());bad=[];n=0;total=0
for entry in tree['tree']:
 if entry['type']!='blob':continue
 data=(root/'snapshot'/entry['path']).read_bytes();n+=1;total+=len(data)
 if len(data)!=entry['size'] or hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()!=entry['sha']:bad.append(entry['path'])
print(json.dumps({'tree':tree['sha'],'files':n,'bytes':total,'mismatches':bad},indent=2))
raise SystemExit(bool(bad))
