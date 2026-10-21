"""Read-only exact-delta package audit; no models, tensor loads or array copies."""
import ast, hashlib, json, re, resource, subprocess, sys, time, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'superweight-causal-circuits'
OUT=Path(__file__).with_suffix('.json')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def git(*a):return subprocess.check_output(['git',*a],cwd=REPO).decode().split('\0')
start=time.time(); visible=set(filter(None,git('ls-files','--others','--exclude-standard','-z')+git('diff','HEAD','--name-only','-z')))
numeric=read(REPO/'manifests/wiki_calibration_numeric_export.json')
intended_numeric={'results/gate_wiki_calibration/'+n for n in numeric['exported_files']}
names=sorted(visible|intended_numeric)
files={};hits=[];broken=[];ext={};npy=0
patterns={'private_home':r'/home/(?!USER(?:/|\b)|user(?:/|\b))[^\s\"\'<>]+','library_id':r'libfile_[a-f0-9]{20,}','credential':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)'}
for name in names:
 p=REPO/name
 assert p.is_file() and not p.is_symlink(),name
 files[name]={'sha256':sha(p),'bytes':p.stat().st_size};ext[p.suffix]=ext.get(p.suffix,0)+1
 assert p.suffix not in {'.pt','.pth','.safetensors','.bin','.arrow','.parquet','.pkl','.pickle','.env'},name
 if p.suffix=='.npz':
  assert name.startswith('results/gate_wiki_calibration/'),name
  with zipfile.ZipFile(p) as z:
   for x in z.namelist():
    assert x.endswith('.npy') and not any(t in x.lower() for t in ['token','input_ids','pred','hidden','qkv','weight']), (name,x)
    npy+=1
 elif p.suffix in {'.md','.py','.json','.log','.txt','.svg'} or p.name=='README':
  s=p.read_text()
  for kind,pat in patterns.items():
   for m in re.finditer(pat,s):
    if kind=='private_home' and m.group().startswith('/home/[A-Za-z0-9_'):continue # literal privacy-validator regex
    hits.append({'file':name,'line':s.count('\n',0,m.start())+1,'kind':kind})
  if p.suffix=='.md':
   for m in re.finditer(r'(?<!!)\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)',s):
    target=(m.group(1) or m.group(2)).strip().split(' "')[0].strip('<>')
    if target.startswith(('http:','https:','mailto:','#','data:')):continue
    target=target.split('#')[0]
    if target and not (p.parent/target).exists():broken.append({'file':name,'target':target})
source=read(REPO/'manifests/wiki_calibration_source_export.json')
for n,h in numeric['exported_files'].items():assert sha(REPO/'results/gate_wiki_calibration'/n)==h,n
for n,h in source['files'].items():assert sha(REPO/'experiments/pulse_transfer'/n)==h,n
primary=read(REPO/'results/gate_wiki_calibration/manifest.json')
assert source['production_source_hashes']==primary['source_hashes']
for n,h in primary['source_hashes'].items():assert source['files'][n]==h,n
inventory=read(REPO/'results/gate_wiki_calibration_reproduction/source_inventory.json')
reviewdir=REPO/'reviews/gate_wiki_calibration'
original=reviewdir/'wiki_replay_source_manifest_reviewed_draft.json'
assert sha(original)==inventory['reviewed_draft_sha256']
assert read(original)['files']==inventory['files']
assert sha(REPO/'results/gate_wiki_calibration_reproduction/source_inventory.json')==sha(reviewdir/'wiki_replay_source_manifest.json')
rr=read(reviewdir/'wiki_replay_root_review.json')
assert rr['status']=='passed' and rr['source_inventory_sha256']==sha(original) and not rr['pretrained_replay']
assert inventory['status']=='independently_reviewed_preparation_and_wrapper_code_no_pretrained_replay'
for n,h in inventory['files'].items():assert sha(REPO/'experiments/pulse_transfer'/n)==h,n
# Check all statically visible local imports in staged replay closure.
for name in inventory['files']:
 if not name.endswith('.py'):continue
 for node in ast.walk(ast.parse((REPO/'experiments/pulse_transfer'/name).read_text())):
  imports=([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [])
  for module in imports:
   if (REPO/'experiments/pulse_transfer'/(module+'.py')).exists():assert module+'.py' in inventory['files'],(name,module)
meta=REPO/'results/gate_wiki_calibration_reproduction/metadata';identity=read(meta/'identity.json')
assert sha(meta/'identity.json')==rr['metadata_identity_sha256']
assert {p.name for p in meta.iterdir()}==set(identity['metadata_sha256'])|{'identity.json'}
for n,h in identity['metadata_sha256'].items():assert sha(meta/n)==h,n
# Actual token arrays/IDs must be absent. Names of canonical files and hashes are allowed.
for p in meta.glob('*.json'):
 def visit(x,path=''):
  if isinstance(x,dict):
   for k,v in x.items():
    assert k.lower() not in {'input_ids','token_ids','tokens','first_id','first_token_id','text','source_text','content'},(p.name,path,k)
    visit(v,path+'/'+k)
  elif isinstance(x,list):
   for i,v in enumerate(x):visit(v,path+'/'+str(i))
 visit(read(p))
report=REPO/'docs/GPT2_GATE_WIKI_CALIBRATION.md'; original_report=ROOT/'pulse_transfer/REPORT14.md'
a=original_report.read_text();b=report.read_text()
b=re.sub(r'\nRepository packaging update:.*?\n(?=\n)', '\n', b)
# Export changes figure and review-relative links; preserve substantive prose.
strip_links=lambda s:re.sub(r'\]\([^)]*\)','](LINK)',s)
report_prose_same=strip_links(a).split()==strip_links(b).split()
source_link_map={'experiments/pulse_transfer/GPT2_GATE_WIKI_CALIBRATION_REVISION.md':'review/gate_calibration_factorial_failure_review.md','experiments/pulse_transfer/GPT2_WIKI_CALIBRATION_REPLAY_PROPOSAL.md':'review/gate_wiki_replay_contract.json'}
guide=(REPO/'docs/GPT2_GATE_WIKI_CALIBRATION_REPRODUCING.md').read_text()
resolved=[]
for item in broken[:]:
 if source_link_map.get(item['file'])==item['target']:
  target=REPO/'reviews/gate_wiki_calibration'/Path(item['target']).name
  assert target.exists() and item['target'] in guide
  resolved.append(item);broken.remove(item)
adapt=read(REPO/'manifests/wiki_calibration_review_link_adaptations.json')
for item in adapt['files']:
 assert sha(REPO/item['file'])==item['exported_sha256']
 assert sha(ROOT/'pulse_transfer/review'/Path(item['file']).name)==item['original_sha256']
result={'status':'passed' if not hits and not broken and report_prose_same else 'needs_review','scope':'Exact working-tree delta plus all manifest-listed numeric files before final inventory; no model/GPU, tokenizers or scientific array loading; prior numeric audit reused','base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip(),'files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files.values()),'extensions':ext,'npz_members':npy,'manifest_numeric_files':len(intended_numeric),'numeric_files_absent_from_visible_git_delta':sorted(intended_numeric-visible),'source_exports':len(source['files']),'production_sources':len(primary['source_hashes']),'replay_sources':len(inventory['files']),'original_inventory_sha256':sha(original),'current_inventory_sha256':sha(REPO/'results/gate_wiki_calibration_reproduction/source_inventory.json'),'metadata_files':len(list(meta.iterdir())),'metadata_bytes':sum(p.stat().st_size for p in meta.iterdir()),'metadata_identity_sha256':sha(meta/'identity.json'),'report_prose_same_after_link_normalization':report_prose_same,'scan_hits':hits,'broken_local_links':broken,'frozen_source_links_mapped_in_guide':resolved,'historical_absolute_path_scope':'Frozen command/default examples use generic /tmp/superweight-venv and /tmp/superweight_experiment; these are not personal locators and the standalone route does not require those exact paths. Privacy-validation regex literals are not detected payloads.', 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.time()-start,'model_forwards':0,'gpu_used':False,'script_sha256':sha(Path(__file__))}
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
