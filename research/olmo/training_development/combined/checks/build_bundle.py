"""Build a compact evidence bundle; never include bulk prefix payload."""
from pathlib import Path
import zipfile,hashlib,json
B=Path(__file__).resolve().parents[2];paths={}
def addtree(base):
 for p in (B/base).rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(B).as_posix()
  if any(x in rel for x in ['__pycache__','/pdf_review/','/bundle_smoke']):continue
  if p.suffix=='.zip' and rel!='phase2/deliverables/olmo_training_development_evidence.zip':continue
  if p.name.startswith('bundle_smoke_audit') or p.name in ['bundle_sha256.txt','replica_validation.json']:continue
  if p.suffix=='.f32' and base.startswith('phase') and not p.name.startswith(('gate_proj_','up_proj_')):continue
  if p.name=='download_events.jsonl':rel=rel.replace('/sources/download_events.jsonl','/results/recorded_download_events.jsonl')
  paths[rel]=p
for base in ['phase2/checks','phase2/results','phase2/sources','phase2/deliverables','phase3/checks','phase3/results','phase3/sources','phase3/deliverables','combined','sources/trajectory']:addtree(base)
for p in [*B.glob('phase2/*.json'),*B.glob('phase3/*.json'),B/'phase2/REPRODUCE.md',B/'sources/hf_refs.json',B/'checks/trajectory_results.json']:paths[p.relative_to(B).as_posix()]=p
readme=b'Completed OLMo-1B training-development investigation\n\nStart with combined/deliverables/training_development_report.pdf and combined/REPRODUCE.md. Eight initial checkpoints plus four adaptively selected later checkpoints are complete. Historical initial artifacts remain unchanged, including the earlier paused-extension description. No bulk prefix tensors are included.\n'
manifest={'description':'Completed eight initial plus four adaptive checkpoint studies. Bulk prefix tensors omitted. Historical initial ZIP retained unchanged. Receipt hashes are not independent whole-model hashes.','files':{r:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for r,p in sorted(paths.items())}}
manifest['files']['README.txt']={'bytes':len(readme),'sha256':hashlib.sha256(readme).hexdigest()}
out=B/'combined/deliverables/olmo_training_development_complete.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for r,p in sorted(paths.items()):z.write(p,'olmo_training_origins/'+r)
 z.writestr('olmo_training_origins/README.txt',readme);z.writestr('olmo_training_origins/BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out) as z:assert z.testzip() is None
h=hashlib.sha256(out.read_bytes()).hexdigest();(out.parent/'bundle_sha256.txt').write_text(h+'  '+out.name+'\n');print(json.dumps({'files':len(manifest['files']),'bytes':out.stat().st_size,'sha256':h},indent=2))
