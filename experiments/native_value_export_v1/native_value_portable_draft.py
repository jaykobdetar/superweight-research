"""Draft T19 research-only metadata/numerical export and tokenizer-only rebuild.

All destinations are fresh and outside the repository. Numerical export requires
completed composite+independent audit. No weights/model forwards/GPU here.
Reproduction staging modifies only INPUT_HASHES of the original frozen runner;
all historical source exposure is labeled inherited, not silently recertified.
"""
import argparse,ast,copy,hashlib,importlib,json,math,os,re,resource,shutil,stat,sys,time
from pathlib import Path,PurePosixPath
# CLI execution disables later import caches; child/replay processes inherit
# the environment flag and receive -B explicitly in recorded commands.
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
FROZEN_RUNNER_SHA='7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f'
FROZEN_CHECKPOINT_SHA='3da3253033f8d9a389ce4103a16143b2feb458bd5d0ab14499fd939fd092bb4a'
PHYSICAL=('wiki_validation','utilitarianism','ethics','adadelta','adamax','asgd')
DROP={'path','id','ids','tokens','pred','prediction','left_id','right_id','targets','input_ids','first_token_id','bos_id','evaluation_first_token_ids','delta_id','scale_id'}
METRICS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')
NORM_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
REFERENCE=('manifest.json','calibration_maps.npz','calibration_directions.npz')
RESERVE=2*1024**3
OWNED_OUT=None
PACKAGE_INVENTORY='metadata_export_inventory.json'
ADAPTER_NAME='native_value_portable_draft.py'
METADATA_NAMES={'data_manifest.template.json','exclusion_inventory.json','start_identity.template.json','historical_manifest_scan.json','historical_preflight.json','history_pins.json'}
FROZEN_SOURCE_HASHES={'GPT2_NATIVE_VALUE_BOUNDARY_PROPOSAL.md': '7a0726a1d8ca42ef2d2122282d1f8da2b0c33bfc951dc2e97677e3c5c49590b6', 'GPT2_NATIVE_VALUE_IMPLEMENTATION_ADDENDUM.md': 'b0c4a7adb9c5344d65f424340c38a217cd4ec860e049b534663a48ab596900f6', 'analyze_gate_native_value.py': '22f6526207034cc89f06dadde1877d35088836d010c5630bcb498a6e35f328d3', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_calibration_factorial_inputs.py': '810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_inputs.py': 'e0a6006772a2940aebf1e55da136ebeb835e4960d7df0a5635a6051bc3e7bf84', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}
METADATA_DEPENDENCY_HASHES={'gpt2_run/manifest.json': '60909cdc2af2240ba99bf103c3cbf62146bddffd43c4f8f6810d092632fb5515', 'gpt2_run/selection.json': '56bbb1076fc342ae1f504bf8bca0d72b521ff1b69269e51575375a144eba50ae', '../superweight-causal-circuits/manifests/transfer_models.json': '4cb4e6934a62ceccf9c062b8d85b171f2704d0891c40d8d2cd4f51ab1f75e90c'}


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')
def copy_exact(p,q,h=None):
 p,q=Path(p),Path(q);h=h or sha(p);assert p.is_file() and not p.is_symlink() and sha(p)==h
 q.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as a,q.open('xb') as b:shutil.copyfileobj(a,b)
 assert sha(q)==h

def fresh(out,allowance=128*1024**2):
 global OWNED_OUT
 assert not out.exists(),'Fresh immutable destination required'
 assert all(not p.is_symlink() for p in (out.absolute(),*out.absolute().parents)),'Symlink output/ancestor'
 assert not out.resolve().is_relative_to((ROOT.parent/'superweight-causal-circuits').resolve()),'Draft cannot mutate repository'
 assert shutil.disk_usage(out.parent).free>=RESERVE+allowance
 out.mkdir(parents=True);st=out.stat();OWNED_OUT=(out.resolve(),st.st_dev,st.st_ino)
def value_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def private_key(k):
 return k in DROP or str(k).endswith(('token_id','token_ids')) or k in {'password','api_key','access_token','refresh_token','authorization','credentials','secret'}
def clean(v):
 if isinstance(v,dict):return {k:clean(w) for k,w in v.items() if not private_key(k)}
 if isinstance(v,list):return [clean(w) for w in v]
 return v

def safe(v,path=()):
 if isinstance(v,dict):
  assert not any(private_key(k) for k in v),('Forbidden key',path)
  for k,w in v.items():safe(w,path+(str(k),))
 elif isinstance(v,(tuple,list)):
  for i,w in enumerate(v):safe(w,path+(str(i),))
 elif isinstance(v,str):
  if path and path[-1]=='pointer' and v.startswith('/'):
   assert re.fullmatch(r'/(?:[^/~]|~[01])*(?:/(?:[^/~]|~[01])*)*',v)
   assert not any(s in v for s in ('/home/','/tmp/','/Users/'))
  else:assert not v.startswith(('/','\\','~','file://')) and not re.match(r'^[A-Za-z]:[/\\]',v),('Absolute locator',path)
  assert '/home/' not in v and '/Users/' not in v
 else:assert v is None or type(v) in (bool,int,float) and (type(v)!=float or math.isfinite(v)),('Unexpected type/value',path,type(v))

def assigned(text,name):
 a=[n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
 assert len(a)==1 and len(a[0].targets)==1
 return a[0],ast.literal_eval(a[0].value)

def reference_audit(ref,dm):
 import torch
 from gate_wiki_storage import read_npz
 from gpt2_gate_subspace import maps_digest
 assert sha(ref/'manifest.json')==dm['calibration_reference_manifest_sha256']=='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060'
 m=read(ref/'manifest.json');assert m['status']=='completed' and m['forward_passes']==1548;safe(m)
 for n in REFERENCE[1:]:assert sha(ref/n)==dm['calibration_reference_files'][n]
 maps=read_npz(ref/'calibration_maps.npz');dirs=read_npz(ref/'calibration_directions.npz');arrays=0
 assert set(maps)==set(dirs)=={'O','F','M'}
 for label in maps:
  assert set(maps[label])==set(dirs[label])==set(range(4,12))
  for layer in maps[label]:
   d=maps[label][layer];assert set(d)=={'alpha','mean_b','mean_mass','basis_r','basis_b','basis_qr'}
   for k,t in d.items():
    assert t.dtype==torch.float64 and t.shape==((12,) if k in ('alpha','mean_b','mean_mass') else (12,12)) and torch.isfinite(t).all();arrays+=1
   d=dirs[label][layer];assert {'u','v'}<=set(d)
   for k,v in d.items():
    if isinstance(v,torch.Tensor):assert k in ('u','v') and v.dtype==torch.float64 and v.shape==(12,) and torch.isfinite(v).all();arrays+=1
    else:safe(v)
 assert maps_digest(maps['F'])==m['calibration_map_digests']['F']=='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
 return {'status':'passed','arrays':arrays,'schema':'Only fitted12-head vectors/12x12bases and direction metadata; no per-example activations.',
         'files':{n:{'sha256':sha(ref/n),'bytes':(ref/n).stat().st_size} for n in REFERENCE}}

def metadata_export(a):
 fresh(a.out);frozen=a.original/'frozen_sources'
 assert sha(a.original/'checkpoints/000018/manifest.json')==FROZEN_CHECKPOINT_SHA
 assert sha(frozen/'gpt2_gate_native_value.py')==FROZEN_RUNNER_SHA
 cp=read(a.original/'checkpoints/000018/manifest.json');assert cp['source_hashes']==FROZEN_SOURCE_HASHES and cp['metadata_dependencies']==METADATA_DEPENDENCY_HASHES
 _,expected=assigned((frozen/'gpt2_gate_native_value.py').read_text(),'INPUT_HASHES')
 assert all(sha(a.inputs/n)==h for n,h in expected.items())
 assert sha(ROOT/'gate_native_value_history_pins.json')==read(a.inputs/'preflight.json')['history_pin_file_sha256']
 dm=read(a.inputs/'data_manifest.json');inv=read(a.inputs/'exclusion_inventory.json');starts=read(a.inputs/'start_identity_summary.json')
 audit=reference_audit(a.reference,dm);md=a.out/'metadata';md.mkdir()
 template=clean(dm);template.pop('literals')
 objects={'data_manifest.template.json':template,'exclusion_inventory.json':clean(inv),'start_identity.template.json':clean(starts),
          'historical_manifest_scan.json':read(a.inputs/'historical_manifest_scan.json'),'historical_preflight.json':read(a.inputs/'preflight.json'),
          'history_pins.json':clean(read(ROOT/'gate_native_value_history_pins.json'))}
 for n,v in objects.items():safe(v);put(md/n,v)
 code=a.out/'experiments/pulse_transfer';code.mkdir(parents=True)
 for n,h in cp['source_hashes'].items():
  assert Path(n).name==n and sha(frozen/n)==h
  assert not re.search(r'/home/[A-Za-z0-9_.-]+/',(frozen/n).read_text()),('Personal path in scientific source',n)
  copy_exact(frozen/n,code/n,h)
 copy_exact(__file__,code/Path(__file__).name)
 for n in REFERENCE:copy_exact(a.reference/n,a.out/'calibration_reference'/n,audit['files'][n]['sha256'])
 deps={}
 for name,h in cp['metadata_dependencies'].items():
  src=ROOT/name;safe(read(src));dest=a.out/'metadata_dependencies'/name.removeprefix('../').replace('/','__')
  copy_exact(src,dest,h);deps[name]={'file':dest.name,'sha256':h}
 identity={'schema':'native_value_portable_metadata_draft_v2','canonical_input_sha256':expected,'frozen_source_sha256':cp['source_hashes'],
  'adapter_source_sha256':sha(__file__),'metadata_sha256':{n:sha(md/n) for n in objects},'metadata_dependencies':deps,'reference_audit':audit,
  'physical_sources':list(PHYSICAL),'private_value_commitments':{'start_identity':value_sha(starts),'literals':value_sha(dm['literals'])},
  'replay_input_metadata_may_change':sorted(set(expected)-{'tokens.pt'}),'tokens_pt_must_remain_byte_exact':True,
  'historical_scope':'Exposure/exclusions/history hashes and prior-start membership inherited from audited source. Six current source bytes, all382 candidates and final16windows are rechecked; no prior model/tokenizer reruns.',
  'replay_scope':'Same2184scheduled scientific forwards; historical composite2272actual returns and missing original guards remain original provenance, not reproduced historical failures.',
  'model_forwards':0,'calibration_forwards':0}
 safe(identity);put(a.out/'identity.json',identity)
 expected=package_members(identity);actual=scan_package(a.out,expected_dirs(expected));assert actual==set(expected)
 inventory={'schema':'native_value_metadata_inventory_v2','inventory_member':PACKAGE_INVENTORY,'files':{n:{'sha256':sha(a.out/n),'bytes':(a.out/n).stat().st_size} for n in sorted(actual)},'repository_mutated':False,'outcomes_exported':False}
 safe(inventory);put(a.out/PACKAGE_INVENTORY,inventory);validate_package(a.out)
 print('DRAFT_NATIVE_METADATA',a.out,flush=True)

def relative_member(name,*,inventory=False):
 assert type(name) is str and name and '\\' not in name and ':' not in name and '\x00' not in name
 assert not name.startswith(('/', '~')) and all(p not in ('','.','..') for p in name.split('/'))
 path=PurePosixPath(name);assert not path.is_absolute() and path.as_posix()==name
 assert inventory or PACKAGE_INVENTORY not in path.parts,'Inventory filename is reserved and cannot list itself'
 return name

def expected_dirs(files):
 return {parent.as_posix() for name in files for parent in PurePosixPath(name).parents if parent.as_posix()!='.'}

def scan_package(package,directories):
 package=Path(package).absolute()
 # Reject links before resolve/stat/open, including a link used as package root.
 assert all(not p.is_symlink() for p in (package,*package.parents)),'Symlink package root/ancestor'
 assert package.is_dir();root=package.resolve();files=set();dirs=set();pending=[package]
 while pending:
  directory=pending.pop()
  for entry in os.scandir(directory):
   path=Path(entry.path);st=entry.stat(follow_symlinks=False)
   assert not stat.S_ISLNK(st.st_mode),('Symlink package member',str(path))
   name=path.relative_to(package).as_posix();relative_member(name,inventory=(name==PACKAGE_INVENTORY))
   assert path.resolve().is_relative_to(root),('Escaping package member',name)
   if stat.S_ISDIR(st.st_mode):dirs.add(name);pending.append(path)
   else:assert stat.S_ISREG(st.st_mode),('Nonregular package member',name);files.add(name)
 assert dirs==set(directories),('Unexpected or missing directories',sorted(dirs-set(directories)),sorted(set(directories)-dirs))
 return files

def hash_value(value):
 assert type(value) is str and re.fullmatch('[a-f0-9]{64}',value)
 return value

def package_members(d):
 assert d['schema']=='native_value_portable_metadata_draft_v2';safe(d)
 assert d['frozen_source_sha256']==FROZEN_SOURCE_HASHES
 assert set(d['metadata_sha256'])==METADATA_NAMES
 assert set(d['reference_audit']['files'])==set(REFERENCE)
 assert set(d['metadata_dependencies'])==set(METADATA_DEPENDENCY_HASHES)
 members={'identity.json':None}
 def add(name,h):
  relative_member(name);assert name not in members;members[name]=hash_value(h)
 for n,h in d['frozen_source_sha256'].items():
  assert Path(n).name==n and Path(n).suffix in ('.py','.md');add('experiments/pulse_transfer/'+n,h)
 add('experiments/pulse_transfer/'+ADAPTER_NAME,d['adapter_source_sha256'])
 for n,h in d['metadata_sha256'].items():add('metadata/'+n,h)
 for n,v in d['reference_audit']['files'].items():
  assert set(v)=={'sha256','bytes'} and type(v['bytes']) is int and v['bytes']>0
  add('calibration_reference/'+n,v['sha256'])
 for n,v in d['metadata_dependencies'].items():
  filename=n.removeprefix('../').replace('/','__')
  assert set(v)=={'file','sha256'} and v['file']==filename and v['sha256']==METADATA_DEPENDENCY_HASHES[n]
  add('metadata_dependencies/'+filename,v['sha256'])
 return members

def validate_package(package):
 package=Path(package).absolute()
 # Scan once using only fixed directory names before opening identity/inventory.
 directories={'metadata','calibration_reference','metadata_dependencies','experiments','experiments/pulse_transfer'}
 actual=scan_package(package,directories)
 assert {'identity.json',PACKAGE_INVENTORY}<=actual
 d=read(package/'identity.json');expected=package_members(d)
 assert actual==set(expected)|{PACKAGE_INVENTORY},('Undeclared or missing package files',sorted(actual-set(expected)-{PACKAGE_INVENTORY}),sorted(set(expected)-actual))
 inventory=read(package/PACKAGE_INVENTORY);safe(inventory)
 assert set(inventory)=={'schema','inventory_member','files','repository_mutated','outcomes_exported'}
 assert inventory['schema']=='native_value_metadata_inventory_v2' and inventory['inventory_member']==PACKAGE_INVENTORY
 assert inventory['repository_mutated'] is False and inventory['outcomes_exported'] is False
 for name,entry in inventory['files'].items():
  relative_member(name);assert set(entry)=={'sha256','bytes'}
  hash_value(entry['sha256']);assert type(entry['bytes']) is int and entry['bytes']>0
 assert set(inventory['files'])==set(expected),'Inventory membership must match fixed declared package members exactly'
 for n,h in expected.items():
  entry=inventory['files'][n];assert sha(package/n)==entry['sha256'] and (package/n).stat().st_size==entry['bytes']
  if h is not None:assert entry['sha256']==h
 for n in METADATA_NAMES:safe(read(package/'metadata'/n))
 _,canonical=assigned((package/'experiments/pulse_transfer/gpt2_gate_native_value.py').read_text(),'INPUT_HASHES')
 assert d['canonical_input_sha256']==canonical
 return d


def rebuild(a):
 # Environment restriction applies only to this tokenizer-only command. The
 # separately executed replay command is not given a disabledCUDA environment.
 os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 fresh(a.out,32*1024**2);d=validate_package(a.package)
 import torch
 from datasets import Dataset
 from transformers import AutoTokenizer
 code=a.package/'experiments/pulse_transfer';sys.path.insert(0,str(code));selection=importlib.import_module('gate_calibration_factorial_inputs')
 assert Path(selection.__file__).resolve()==(code/'gate_calibration_factorial_inputs.py').resolve()
 torch.set_num_threads(1);md=a.package/'metadata';dm=read(md/'data_manifest.template.json');inv=read(md/'exclusion_inventory.json')
 cfg=read(a.config);gc=cfg['models']['gpt2'];model=read(a.package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2']
 assert gc['model_id']==model['model_id'] and gc['revision']==model['revision']
 tokenizer_path=Path(gc['path'])
 for n,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/n)==h
 tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True);assert tok.is_fast
 texts={}
 for c in PHYSICAL:
  p=Path(cfg['sources'][c]);spec=dm['sources'][c];assert sha(p)==spec['file_sha256']
  text='\n\n'.join(Dataset.from_file(str(p))['text']) if c=='wiki_validation' else selection.strip_book(p.read_text()) if c in ('utilitarianism','ethics') else p.read_text()
  assert len(text)==spec['characters'] and selection.textsha(text)==spec['processed_text_sha256'];texts[c]=text
 texts['optimizers_new']='\n\n'.join(texts[c] for c in ('adadelta','adamax','asgd'))
 assert len(texts['optimizers_new'])==dm['sources']['optimizers_new']['characters'] and selection.textsha(texts['optimizers_new'])==dm['sources']['optimizers_new']['processed_text_sha256']
 components=[];cursor=0
 for c in ('adadelta','adamax','asgd'):
  components.append({'source':c,'char_start':cursor,'char_end':cursor+len(texts[c]),'file_sha256':dm['sources'][c]['file_sha256']});cursor+=len(texts[c])+2
 assert components==dm['sources']['optimizers_new']['components']
 history=set(inv['history_token_hashes']);assert len(history)==577
 progress={};selection.select_panel(selection.EVALUATION,texts,tok,inv['exclusions'],history,components,False,progress)
 assert len(progress['items'])==16 and clean(progress['selected'])==dm['passages'] and progress['rejections']==dm['rejections']
 assert sum(map(len,progress['rejections'].values()))==366
 torch.save(progress['items'],a.out/'tokens.pt');assert sha(a.out/'tokens.pt')==d['canonical_input_sha256']['tokens.pt']
 copy_exact(md/'exclusion_inventory.json',a.out/'exclusion_inventory.json')
 # Historical scan is unchanged content; canonical serialization is checked, not
 # treated as a current private-history rescan.
 scan=read(md/'historical_manifest_scan.json');put(a.out/'historical_manifest_scan.json',scan)
 assert sha(a.out/'historical_manifest_scan.json')==d['canonical_input_sha256']['historical_manifest_scan.json']
 literals={name:{'literal':text,'ids':tok.encode(text,add_special_tokens=False)} for name,text in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]}
 literals['start_special']={'label':'bos','id':tok.bos_token_id};assert value_sha(literals)==d['private_value_commitments']['literals']
 private_dm=copy.deepcopy(dm);private_dm['passages']=progress['selected'];private_dm['literals']=literals;private_dm['exclusion_inventory_sha256']=sha(a.out/'exclusion_inventory.json')
 put(a.out/'data_manifest.json',private_dm)
 original_starts=read(md/'start_identity.template.json');starts={'evaluation_first_token_ids':[int(v['tokens'][0]) for v in progress['items']],**original_starts,'bos_id':tok.bos_token_id}
 assert value_sha(starts)==d['private_value_commitments']['start_identity'];put(a.out/'start_identity_summary.json',starts)
 put(a.out/'preflight.json',{'status':'passed','phase':'completed','kind':'portable_T19_tokenizer_only_reconstruction','model_forwards':0,'model_loaded':False,'gpu_used':False,'calibration_fitted':False,
  'evaluation_passages':16,'evaluation_inputs':32,'historical_exposure_is_inherited':True,'all382_current_candidates_rechecked':True,'original_preflight_sha256':d['canonical_input_sha256']['preflight.json'],
  'metadata_identity_sha256':sha(a.package/'identity.json'),'retained_calibration_map_digest':'8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'})
 regenerated={n:sha(a.out/n) for n in d['canonical_input_sha256']}
 put(a.out/'input_bindings.json',{'kind':'T19_input_metadata_only_binding','original':d['canonical_input_sha256'],'regenerated':regenerated,'metadata_identity_sha256':sha(a.package/'identity.json'),
  'exact_token_bytes':True,'current_sources_checked':list(PHYSICAL),'historical_exposure_is_inherited':True,'model_forwards':0})
 print('DRAFT_NATIVE_INPUTS_REBUILT',16,'model forwards0',flush=True)

def bind_runner(source,binding):
 node,expected=assigned(source,'INPUT_HASHES');assert expected==binding['original']
 assert set(expected)==set(binding['regenerated']) and binding['regenerated']['tokens.pt']==expected['tokens.pt']
 assert binding['regenerated']['historical_manifest_scan.json']==expected['historical_manifest_scan.json']
 lines=source.splitlines(keepends=True);changed=''.join(lines[:node.lineno-1])+'INPUT_HASHES='+repr(binding['regenerated'])+'\n'+''.join(lines[node.end_lineno:])
 def normalized(text):
  tree=ast.parse(text)
  for n in tree.body:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INPUT_HASHES' for t in n.targets):n.value=ast.Constant('ONLY_INPUT_HASHES_BINDING')
  return ast.dump(tree,include_attributes=False)
 assert normalized(source)==normalized(changed)
 return changed

def validate_private_inputs(a,d):
 import torch
 from transformers import AutoTokenizer
 expected=set(d['canonical_input_sha256'])|{'input_bindings.json'}
 assert scan_package(a.inputs,set())==expected,'Private reconstructed member universe differs'
 binding=read(a.inputs/'input_bindings.json')
 assert set(binding)=={'kind','original','regenerated','metadata_identity_sha256','exact_token_bytes','current_sources_checked','historical_exposure_is_inherited','model_forwards'}
 assert binding['kind']=='T19_input_metadata_only_binding' and binding['original']==d['canonical_input_sha256']
 assert set(binding['regenerated'])==set(d['canonical_input_sha256'])
 assert binding['metadata_identity_sha256']==sha(a.package/'identity.json') and binding['exact_token_bytes'] is True
 assert binding['current_sources_checked']==list(PHYSICAL) and binding['historical_exposure_is_inherited'] is True and binding['model_forwards']==0
 for n,h in binding['regenerated'].items():hash_value(h);assert sha(a.inputs/n)==h
 for n in ('tokens.pt','historical_manifest_scan.json'):assert binding['regenerated'][n]==d['canonical_input_sha256'][n]
 md=a.package/'metadata';template=read(md/'data_manifest.template.json')
 assert sha(a.inputs/'exclusion_inventory.json')==sha(md/'exclusion_inventory.json')
 # Derive starts/BOS/literals from exact canonical tensors and the pinned local
 # tokenizer, rather than accepting their values from editable binding JSON.
 cfg=read(a.config);model=read(a.package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2']
 gc=cfg['models']['gpt2'];assert gc['model_id']==model['model_id'] and gc['revision']==model['revision']
 for n,h in template['tokenizer_sha256'].items():assert sha(Path(gc['path'])/n)==h
 tok=AutoTokenizer.from_pretrained(gc['path'],local_files_only=True);assert tok.is_fast and tok.bos_token_id is not None
 items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu');assert len(items)==len(template['passages'])==16
 wanted=copy.deepcopy(template);first=[]
 for item,record in zip(items,wanted['passages']):
  assert set(item)=={'corpus','window','tokens'} and (item['corpus'],item['window'])==(record['corpus'],record['window'])
  t=item['tokens'];assert t.shape==(512,) and t.dtype==torch.int64
  assert hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()==record['window_token_sha256']
  record['first_token_id']=int(t[0]);first.append(int(t[0]))
 literals={name:{'literal':text,'ids':tok.encode(text,add_special_tokens=False)} for name,text in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]}
 literals['start_special']={'label':'bos','id':tok.bos_token_id}
 assert value_sha(literals)==d['private_value_commitments']['literals']
 wanted['literals']=literals;wanted['exclusion_inventory_sha256']=sha(a.inputs/'exclusion_inventory.json')
 assert read(a.inputs/'data_manifest.json')==wanted,'Canonical input metadata/start identities changed'
 starts={'evaluation_first_token_ids':first,**read(md/'start_identity.template.json'),'bos_id':tok.bos_token_id}
 assert read(a.inputs/'start_identity_summary.json')==starts and value_sha(starts)==d['private_value_commitments']['start_identity']
 preflight={'status':'passed','phase':'completed','kind':'portable_T19_tokenizer_only_reconstruction','model_forwards':0,'model_loaded':False,'gpu_used':False,'calibration_fitted':False,
  'evaluation_passages':16,'evaluation_inputs':32,'historical_exposure_is_inherited':True,'all382_current_candidates_rechecked':True,'original_preflight_sha256':d['canonical_input_sha256']['preflight.json'],
  'metadata_identity_sha256':sha(a.package/'identity.json'),'retained_calibration_map_digest':'8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'}
 assert read(a.inputs/'preflight.json')==preflight
 return binding

def stage_replay(a):
 # Reject editable binding/BOS/history changes before creating a staged tree.
 d=validate_package(a.package);binding=validate_private_inputs(a,d)
 fresh(a.out,64*1024**2)
 dest=a.out/'pulse_transfer';dest.mkdir()
 for n,h in d['frozen_source_sha256'].items():
  src=a.package/'experiments/pulse_transfer'/n
  if n=='gpt2_gate_native_value.py':
   changed=bind_runner(src.read_text(),binding)
   with (dest/n).open('x') as f:f.write(changed)
  else:copy_exact(src,dest/n,h)
 for name,v in d['metadata_dependencies'].items():copy_exact(a.package/'metadata_dependencies'/v['file'],dest/name,v['sha256'])
 for n in REFERENCE:copy_exact(a.package/'calibration_reference'/n,a.out/'calibration_reference'/n,d['reference_audit']['files'][n]['sha256'])
 command=[sys.executable,'-B',str((dest/'gpt2_gate_native_value.py').resolve()),'--config',str(a.config.resolve()),'--inputs',str(a.inputs.resolve()),'--calibration-reference',str((a.out/'calibration_reference').resolve()),'--out',str((a.out/'run').resolve())]
 put(a.out/'replay_preparation.json',{'status':'prepared_only','model_forwards':0,'calibration_forwards':0,'scheduled_forward_count':2184,'command':command,
  'canonical_runner_sha256':d['frozen_source_sha256']['gpt2_gate_native_value.py'],'adapted_runner_sha256':sha(dest/'gpt2_gate_native_value.py'),'adaptation':'Only the six-entry INPUT_HASHES assignment is rebound; all other AST is exact. Token and historical-scan bytes stay exact.',
  'original_results_execution_scope':'Original verified_composite_after_interruption; missing original after-run guards remain historical limitation. A future replay has its own final guards and execution record.'})
 print('DRAFT_NATIVE_REPLAY_PREPARED','zero model forwards',a.out,flush=True)

MANIFEST_FIELDS={'study','status','phase','execution_type','execution_description','forward_passes','forward_attempts','forward_counter_scope',
 'retained_scheduled_forwards','actual_cumulative_model_returns','actual_cumulative_model_attempts','actual_return_categories_by_execution','actual_return_category_totals',
 'continuity_validation_forwards','discarded_uncommitted_original_returns','original_model_returns','original_fully_processed_forwards',
 'original_final_guards_available','original_missing_final_guards','original_missing_guards','original_after_guards_available','original_final_guards_missing',
 'source_hashes','source_hashes_scope','executed_segment_source_hashes','segment_manifests','segment_final_guards_passed','model_id','revision','dtype','attention',
 'calibration_map_digest','calibration_fit','calibration_forwards','physical_execution_lengths','passages_per_corpus','bridge_policy','score_views','conditions','snapshot_modes',
 'runtime_versions','parameter_versions_unchanged','model_hash_before','model_hash_after','precision_settings_before','precision_settings_after',
 'original_observed_partial_checkpoint_resources','segment_complete_resources','cumulative_runtime_and_original_final_peak_unknown','continuity_scope','exact_continuity_passed',
 'recovery_adapter_sha256','recovery_segment','recovery_expected_counts','recovery_proposal_sha256','merger_sha256',
 'wall_seconds','peak_process_rss_kib','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','original_resource_measurements_scope'}

def safe_manifest(m):return {k:clean(v) for k,v in m.items() if k in MANIFEST_FIELDS}
def controls(v,arrays,path=()):
 import torch
 if isinstance(v,torch.Tensor):
  assert path[-1] in ('loss_difference','kl_delta_to_scale','kl_scale_to_delta') and v.ndim==1 and len(v) in (510,511) and torch.isfinite(v).all(),('Unexpected raw control array',path)
  key='control__'+'__'.join(map(str,path));arrays[key]=v.numpy();return {'numeric_array':key,'shape':list(v.shape),'dtype':str(v.dtype)}
 if isinstance(v,dict):return {k:controls(w,arrays,path+(k,)) for k,w in v.items() if not private_key(k)}
 if isinstance(v,list):return [controls(w,arrays,path+(i,)) for i,w in enumerate(v)]
 return v

def known_keys(value,allowed):
 if isinstance(value,dict):
  assert set(value)<=allowed,('Unknown public report fields',set(value)-allowed)
  for item in value.values():known_keys(item,allowed)
 elif isinstance(value,list):
  for item in value:known_keys(item,allowed)

def public_numerical(report):
 assert set(report)=={'status','audited_manifest_sha256','analysis_sha256','raw_inventory_sha256','audit_source_sha256','comparison_policy_sha256','created_utc','source_hashes','executed_segment_source_hashes','input_hashes','counts','provenance','bootstrap_contrast_exactness','all_numerical_comparisons','runtime_extrema','saved_snapshot_scope','evidence_scope','resources'}
 fields={'status','audited_manifest_sha256','analysis_sha256','raw_inventory_sha256','audit_source_sha256','comparison_policy_sha256','created_utc','counts','all_numerical_comparisons','evidence_scope','resources'}
 result={key:report[key] for key in fields}
 allowed=fields|set('retained_scientific_forwards actual_model_returns main_metric_records_including_anchors suffix_metric_records saved_array_identities_verified controls suffix source_q0 current arithmetic_bridge L10_incoming_causal_order D_causal_order direct_local_delta_identity native_repeat native_identity window_rows anchor_rows norm_rows summary_rows contrasts paired_start_contrasts independent_definitions bootstrap_repetitions shared_bootstrap_index_shape shared_bootstrap_index_sha256 comparisons exact maximum_absolute_error contrast_bootstrap contrast_independent_fsum ff_projection_reconstruction full_early_later_partition norm_aggregates paired_start_bootstrap paired_start_independent_fsum ppl_transforms saved_gate_norm_reconstruction saved_readout_kl saved_readout_nll sigmoid_reconstruction snapshot_FP32_delivery_formula snapshot_bridge_context snapshot_bridge_hidden snapshot_bridge_kl snapshot_context_fp64 snapshot_context_norms snapshot_delta_witness snapshot_gate_logit_fp64 snapshot_mass_fp64 snapshot_stable_mixture_fp64 summary_bootstrap window_metrics recomputed_from_actual_arrays recorded_runtime_witnesses model_weights interruption uncertainty wall_seconds peak_rss_kib model_forwards pretrained_weights_read CUDA_initialized'.split())
 known_keys(result,allowed);safe(result)
 result['public_omissions']='Private provenance and detailed runtime witness tables omitted; complete private numerical audit SHA256 is bound in export.json. Counts and recomputed comparison summaries retained.'
 return result

def public_readout(readout):
 # Explicit reviewed report view: prediction arrays and both mismatch-token IDs
 # remain private. Numeric margins/errors/KLs and query positions remain public.
 assert set(readout)=={'schema','status','preselected_input','scope','execution_provenance','source_sha256','run_manifest_sha256','inventory_sha256','snapshot_shard_sha256','checkpoint_sha256','weight_tensors_loaded','embedding_sha256','model_forwards','gpu_computation','cpu_threads','file_deletions','query_rows','target_positions','counts','prospective_offline_saved_metric_max_abs','arms','bridges','maxima','peak_rss_bytes','elapsed_seconds','resources'}
 arm_keys={'nll','prediction','margin','kl_c_to_trial','kl_trial_to_c','kl_boundary_to_trial','kl_trial_to_boundary','saved_fp32_metric_error','readout_gpu_vs_cpu_fp64','readout_gpu_vs_cpu_fp32','fp32_dot_product_bound','gpu_vs_cpu_fp64_argmax_mismatches','own_boundary_source_hidden_exact'}
 for arm in readout['arms'].values():
  assert set(arm)==arm_keys and len(arm['prediction'])==4 and all(type(x) is int for x in arm['prediction'])
 allowed=set(readout)|arm_keys|set('000__baseline 000__orth 000__remove 111__baseline 111__orth 111__orth_scale 111__remove 111__remove_scale remove orth all_query_mismatches_scope bridge_absolute_passage_mean_loss_difference bridge_context_max_abs bridge_independently_checked_argmax_mismatches bridge_recorded_all_query_argmax_mismatches bridge_scored_target_loss_max_abs context_passed corpus execution_type four_row_argmax_mismatches four_row_kl_delta_to_scale four_row_kl_scale_to_delta four_row_saved_logit_drift full_saved_context_drift full_saved_final_hidden_drift full_saved_hidden_queries_per_arm full_vocabulary_rows_per_arm gamma_2n gpu_vs_cpu_fp32_readout_max_abs gpu_vs_cpu_fp64_readout_max_abs identity independently_checked_bridges kl_boundary kl_c length max_abs max_abs_bound max_error_to_bound_ratio nll passage_mean_loss_difference passage_mean_passed prospective_bridge_criteria recorded_all_query_argmax_mismatches retained_forward_passes rms scored_target_loss_max_abs saved_kl_max_abs_error saved_nll_max_abs_error symmetric_relative_l2 target_loss_passed token_sha256 total_full_vocabulary_rows window query left_id right_id left_margin right_margin delta_id scale_id delta_margin scale_margin row_max_abs_drift compatible_with_observed_drift row_max_abs_logit_drift compatible_with_drift'.split())
 known_keys(readout,allowed)
 result=clean(readout);safe(result)
 result['public_omissions']='Prediction arrays and all mismatch token IDs omitted. The complete private report is bound by its SHA256 in export.json.'
 return result

def numerical_prerequisites(a):
 # These are two distinct external publication contracts. The numerical audit
 # has no schema field; the saved-readout audit retains its reviewed v2 schema.
 # Never reinterpret one contract as the other through a field-name fallback.
 m=read(a.run/'manifest.json');assert m['status']=='completed' and m['execution_type']=='verified_composite_after_interruption'
 assert m['retained_scheduled_forwards']==m['forward_passes']==2184 and m['actual_cumulative_model_returns']==2272
 assert m['original_final_guards_available'] is False and m['parameter_versions_unchanged'] is None and m['model_hash_after'] is None
 analysis=read(a.run/'native_value_analysis.json');assert analysis['scope']['analyzer_sha256']==m['source_hashes']['analyze_gate_native_value.py']
 inventory=read(a.run/'numeric_inventory.json');files=sorted((a.run/'inputs').glob('*.npz'))
 assert len(files)==32 and set(inventory)=={p.relative_to(a.run).as_posix() for p in files}
 for p in files:
  assert p.is_file() and not p.is_symlink()
  assert inventory[p.relative_to(a.run).as_posix()]=={'bytes':p.stat().st_size,'sha256':sha(p)}
 assert analysis['scope']['source_hashes']=={p.name:inventory[p.relative_to(a.run).as_posix()]['sha256'] for p in files},'Analysis must bind every current raw shard exactly'
 manifest_hash=sha(a.run/'manifest.json');inventory_hash=sha(a.run/'numeric_inventory.json')
 numerical=read(a.numerical_audit)
 assert {'status','audited_manifest_sha256','analysis_sha256','raw_inventory_sha256','audit_source_sha256'}<=set(numerical)
 assert numerical['status']=='passed' and numerical['audited_manifest_sha256']==manifest_hash
 assert numerical['analysis_sha256']==sha(a.run/'native_value_analysis.json') and numerical['raw_inventory_sha256']==inventory_hash
 assert numerical['audit_source_sha256']==sha(a.numerical_source)
 readout=read(a.readout_audit)
 assert {'schema','status','run_manifest_sha256','inventory_sha256','snapshot_shard_sha256','source_sha256'}<=set(readout)
 assert readout['schema']=='T19_independent_saved_readout_audit_v2' and readout['status']=='passed'
 assert readout['run_manifest_sha256']==manifest_hash and readout['inventory_sha256']==inventory_hash
 assert readout['snapshot_shard_sha256']==inventory['inputs/wiki_validation__0__natural.npz']['sha256']
 assert readout['source_sha256']==sha(a.readout_source)
 # Keep the pre-weight header-key failure in the public execution history while
 # withholding its traceback, which contains private local file locators.
 history=read(a.readout_history);failure=read(a.readout_failure)
 assert history['preserved_all_attempts'] is True and history['model_forwards']==0 and history['gpu_computation'] is False and history['file_deletions']==0
 assert history['corrected_attempt']['source_sha256']==readout['source_sha256'] and history['corrected_attempt']['exit_code']==0
 assert failure['status']=='failed' and failure['auditor_sha256']==history['first_attempt']['source_sha256']
 failure_public={k:failure[k] for k in ('status','error_type','error','auditor_sha256','scope')}
 execution_review=read(a.readout_execution_review);key_review=read(a.readout_key_review)
 assert execution_review['status']=='passed_after_parent_narrow_key_review' and execution_review['corrected_source_sha256']==readout['source_sha256']
 assert execution_review['run_manifest_sha256']==manifest_hash and execution_review['audit_json_sha256']==sha(a.readout_audit)
 assert execution_review['parent_review_sha256']==sha(a.readout_key_review) and key_review['status']=='passed' and key_review['new_source_sha256']==readout['source_sha256']
 history={'initial_event_prior_state':history,'final_execution_review':execution_review,'key_compatibility_review':key_review}
 known_keys(history,set('initial_event_prior_state final_execution_review key_compatibility_review status first_attempt source source_sha256 failure failure_scope failure_capsule log corrected_attempt only_change thresholds_metric_guards_and_other_schema_unchanged exit_code numerical_outputs_not_inspected_at_note_creation preserved_all_attempts model_forwards gpu_computation file_deletions first_attempt_preserved first_attempt_scope corrected_source corrected_source_sha256 parent_review_file parent_review_sha256 chronology thresholds_unchanged run_manifest_sha256 audit_json_sha256 audit_markdown_sha256 checked_utc scope old_source_sha256 new_source_sha256 exact_changes old_checkpoint_key actual_checkpoint_key embedding_header dtype shape data_offsets checkpoint_file_sha256 local_transformers_gpt2_source_sha256 bias_free_tied_lm_head_source_verified pretrained_payload_read failure_preserved chronology_note'.split()))
 readout_public=public_readout(readout)
 numerical_public=public_numerical(numerical)
 for obj in (analysis,numerical_public,readout_public,history,failure_public):safe(obj)
 for source in (a.numerical_source,a.readout_source):
  assert source.is_file() and not source.is_symlink() and source.suffix=='.py'
  assert not re.search(r'/home/[A-Za-z0-9_.-]+/',source.read_text()),'Private locator in auditor source'
 return m,analysis,inventory,files,numerical_public,readout_public,history,failure_public

def numerical_export(a):
 import numpy as np
 import torch
 from gate_wiki_storage import read_npz
 m,analysis,inventory,files,numerical,readout,history,failure_public=numerical_prerequisites(a)
 fresh(a.out,256*1024**2);safe(analysis);put(a.out/'native_value_analysis.json',analysis)
 sm=safe_manifest(m);safe(sm);put(a.out/'manifest.json',sm)
 for name,obj in (('numerical_audit.json',numerical),('readout_audit.json',readout),('readout_execution_history.json',history),('readout_failure.json',failure_public)):put(a.out/'audits'/name,obj)
 copy_exact(a.numerical_source,a.out/'audits/numerical_audit.py',numerical['audit_source_sha256'])
 copy_exact(a.readout_source,a.out/'audits/readout_audit.py',readout['source_sha256'])
 index=[];array_count=0
 for p in files:
  entry=inventory[p.relative_to(a.run).as_posix()];assert entry=={'bytes':p.stat().st_size,'sha256':sha(p)}
  d=read_npz(p);arrays={}
  for group in ('metrics','suffix'):
   for mode,z in d[group].items():
    for key in METRICS:
     t=z[key];n=127 if group=='suffix' and key=='loss' else 128 if group=='suffix' else 511
     assert t.shape==(n,) and t.dtype==(torch.bool if key.startswith('agreement') else torch.float32) and torch.isfinite(t).all()
     arrays[group+'__'+mode+'__'+key]=t.numpy()
  for group in ('norms','suffix_norms'):
   for mode,layers in d[group].items():
    if mode=='native_repeat':assert layers=={};continue
    assert set(layers)=={10} and set(layers[10])==NORM_KEYS
    for key,t in layers[10].items():
     n=12 if key=='source_v_norm' else 127 if group=='suffix_norms' else 511
     assert t.shape==(n,) and t.dtype==(torch.float32 if key=='s' else torch.float64) and torch.isfinite(t).all()
     arrays[group+'__'+mode+'__10__'+key]=t.numpy()
  ledger=controls({k:d[k] for k in ('metadata','checks','operations','suffix_operations')},arrays);safe(ledger)
  put(a.out/'inputs'/(p.stem+'.json'),ledger)
  assert all(t.dtype in (np.dtype('float32'),np.dtype('float64'),np.dtype('bool')) for t in arrays.values())
  dest=a.out/'inputs'/p.name
  with dest.open('xb') as f:np.savez_compressed(f,**arrays)
  with np.load(dest,allow_pickle=False) as stored:assert set(stored.files)==set(arrays) and all(np.array_equal(stored[k],v) for k,v in arrays.items())
  index.append({'file':p.name,'raw_sha256':sha(p),'export_sha256':sha(dest),'metadata':d['metadata'],'arrays':sorted(arrays)});array_count+=len(arrays)
 provenance=a.run/'execution_provenance';public=a.out/'execution_provenance'
 for label in ('original','segment_A','segment_B'):
  name='last_checkpoint_manifest.json' if label=='original' else 'manifest.json';v=safe_manifest(read(provenance/label/name));safe(v);put(public/label/name,v)
  # Hash complete private journals; never copy private locator or raw-data payload.
  put(public/label/'journal_commitment.json',{'sha256':sha(provenance/label/'journal.jsonl'),'bytes':(provenance/label/'journal.jsonl').stat().st_size})
 f=read(provenance/'original_interruption.json');f={k:f[k] for k in ('status','cause','journal_counts','journal_final_sha256','committed_shards','completed_group_forwards','model_returns_after_committed_groups','missing_guards')}
 safe(f);put(public/'original_interruption.json',f)
 gate=read(provenance/'continuity_gate.json');gate={k:v for k,v in gate.items() if k not in ('reference_actual','replay_actual')};safe(gate);put(public/'continuity_gate.json',gate)
 origins=read(provenance/'shard_origins.json');origins={n:{k:v[k] for k in ('origin','sha256','bytes')} for n,v in origins.items()};safe(origins);put(public/'shard_origins.json',origins)
 put(a.out/'index.json',index)
 result={'schema':'native_value_safe_numeric_draft_v2','raw_composite_manifest_sha256':sha(a.run/'manifest.json'),
  'publication_audit_contract':'T19_numerical_and_saved_readout_v1','numerical_audit_sha256':sha(a.numerical_audit),'readout_audit_sha256':sha(a.readout_audit),
  'readout_execution_history_sha256':sha(a.readout_history),'readout_failure_private_capsule_sha256':sha(a.readout_failure),
  'readout_execution_review_sha256':sha(a.readout_execution_review),'readout_key_review_sha256':sha(a.readout_key_review),
  'input_shards':32,'retained_main_records':1096,'retained_suffix_records':1088,'numeric_arrays':array_count,'raw_inventory':inventory,
  'omitted':'Input/prediction IDs, source corpus text, credentials/config paths, weights, raw gates, QKV, mixtures/contexts, hidden states, full-vocabulary rows and failure arrays. Local raw evidence remains hash-bound.',
  'interruption_limitation':'18uncommitted original returns have no recoverable arrays. Original final parameter/version/precision guards are unavailable and cannot be retroactively claimed. Continuity gate is same-machine compatibility only.'}
 safe(result);put(a.out/'export.json',result);print('DRAFT_NATIVE_NUMERIC_EXPORTED',array_count,flush=True)

def main():
 global OWNED_OUT
 OWNED_OUT=None
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 e=sub.add_parser('export-metadata')
 for n in ('original','inputs','reference','out'):e.add_argument('--'+n,type=Path,required=True)
 e=sub.add_parser('rebuild')
 for n in ('config','package','out'):e.add_argument('--'+n,type=Path,required=True)
 e=sub.add_parser('stage-replay')
 for n in ('config','package','inputs','out'):e.add_argument('--'+n,type=Path,required=True)
 e=sub.add_parser('export-numeric')
 for n in ('run','numerical-audit','numerical-source','readout-audit','readout-source','readout-history','readout-failure','readout-execution-review','readout-key-review','out'):e.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 try:{'export-metadata':metadata_export,'rebuild':rebuild,'stage-replay':stage_replay,'export-numeric':numerical_export}[a.action](a)
 except BaseException as error:
  # No cleanup/overwrite. A prior output rejection never gains a failure file.
  if OWNED_OUT is not None and a.out.is_dir() and OWNED_OUT==(a.out.resolve(),a.out.stat().st_dev,a.out.stat().st_ino) and not (a.out/'draft_failure.json').exists():put(a.out/'draft_failure.json',{'status':'failed','action':a.action,'error_type':type(error).__name__,'error':str(error),'model_forwards':0,'pretrained_model_loaded':False})
  raise
if __name__=='__main__':main()
