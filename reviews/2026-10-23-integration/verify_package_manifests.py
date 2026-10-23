from pathlib import Path
import json,hashlib,sys
r=Path(sys.argv[1]);results=[]
for directory,filename in [('training_development','BUNDLE_MANIFEST.json'),('signal_production','BUNDLE_MANIFEST.json'),('signal_continuation','MANIFEST.json'),('optimizer_state','BUNDLE_MANIFEST.json'),('optimizer_axes','BUNDLE_MANIFEST.json'),('native_function','EXPORT_MANIFEST.json'),('validated_continuation','EXPORT_MANIFEST.json')]:
 p=r/'research/olmo'/directory/filename
 if not p.exists():continue
 m=json.loads(p.read_text());f=m['files'];items=([dict(path=k,**v) for k,v in f.items()] if isinstance(f,dict) else f)
 for entry in items:
  q=p.parent/entry['path'];assert q.is_file(),str(q)
  assert hashlib.sha256(q.read_bytes()).hexdigest()==entry['sha256'],str(q)
 results.append({'package':directory,'files_verified':len(items)})
print(json.dumps({'all_pass':True,'packages':results},indent=2))
