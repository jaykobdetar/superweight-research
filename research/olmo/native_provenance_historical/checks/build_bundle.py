from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
files=[R/'README.md',R/'receipts.json',R/'linked_receipts.json',R/'head_receipts.json']
files += list((R/'checks').glob('*.py'))
files += [p for p in (R/'deliverables').iterdir() if p.suffix in ['.pdf','.md','.json']]
files += [R/'sources'/n for n in ['native954k_config.txt','native954k_model_metadata.txt','native954k_optim_metadata.txt','maintainer_release_excerpt.json','wandb_report_browser.txt','wandb_project_browser.txt']]
manifest={str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(R/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));files.append(R/'MANIFEST.json')
out=R/'deliverables/native_provenance_evidence.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in sorted(files):z.write(p,'native_provenance/'+str(p.relative_to(R)))
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for f,v in manifest.items():
  b=z.read('native_provenance/'+f);assert len(b)==v['bytes'];assert hashlib.sha256(b).hexdigest()==v['sha256']
print(json.dumps({'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(files),'manifest_checks':'passed'},indent=2))
