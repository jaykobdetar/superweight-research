"""Stage or explicitly run T17 with safe metadata and an existing calibration.

Default mode is tokenizer-only preparation. --execute explicitly requests only
1,992 additive-study forwards, never an older model study or calibration run.
No Llama/OLMo tokenizer/model or historical research directory is required.
"""
import argparse,ast,copy,hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PARENT_SHA={'manifest.json':'60909cdc2af2240ba99bf103c3cbf62146bddffd43c4f8f6810d092632fb5515',
 'selection.json':'56bbb1076fc342ae1f504bf8bca0d72b521ff1b69269e51575375a144eba50ae'}
CAL_SHA={'calibration_inputs.json':'24b1bdf8c1a0744bbd70e0936d24cbf741835aab691bfe496bd9eb809100557e',
 'calibration_raw.pt':'d41d5a91c699d30d5414a2916f7350a9093eda4b0004d8bb7b2fb458ebb31935',
 'calibration_maps.pt':'ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098',
 'calibration_tokens.pt':'a99fa8403d600404b4725bdc8ab8e051e78e70a864a3483b614879d68ccc668c',
 'calibration_metrics.pt':'84ffcd7b4daf67c8f068dd258465721ae760cab71932d8c5891307b2463aa2d8',
 'calibration_operations.json':'a26e553b554fe9f155dd2b67cb1663177ba4a458a444919d4f787c3df75f191e',
 'calibration_spectra.json':'2910c3de477e1d137fb35dbb7f7a55fec583e795cb6c243332372b0c69bba138'}
REBUILD_OPERATIONS_SHA='e76e5efc9e097edabe67643c1159d6a22d4d585a28514396927cdde042389f2f'
MAP_DIGEST='fbf127a1a618e709b0d8e39625a156f11402c9703ac4b55ae3dcfd3711571c81'
MODEL_HASH='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527'
METADATA_SHA='13cca2b5ec8ef9993e01df36f0d86aecfc0bab96c4b598fd94a887fec0605a98'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def copy_exact(src,dst,expected):
 assert sha(src)==expected,('Source hash mismatch',str(src))
 dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists(),'Never overwrite a staged source'
 shutil.copyfile(src,dst);assert sha(dst)==expected

def bind_input_hashes(source,bindings):
 """Literal INPUT_HASHES substitution only; prove every other AST node identical."""
 tree=ast.parse(source);matches=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='INPUT_HASHES'for t in n.targets)]
 assert len(matches)==1 and len(matches[0].targets)==1,'Exactly one plain INPUT_HASHES assignment required'
 node=matches[0];assert ast.literal_eval(node.value)==bindings['original'],'Staged source is not the canonical runner'
 assert set(bindings['regenerated'])==set(bindings['original'])
 assert set(bindings['changed_metadata_files'])=={'preflight.json','data_manifest.json','exclusion_inventory.json'}
 for n in set(bindings['original'])-set(bindings['changed_metadata_files']):assert bindings['regenerated'][n]==bindings['original'][n],('Canonical tensor/history identity changed',n)
 lines=source.splitlines(keepends=True);changed=''.join(lines[:node.lineno-1])+"INPUT_HASHES = "+repr(bindings['regenerated'])+'\n'+''.join(lines[node.end_lineno:])
 def normalized(text):
  t=ast.parse(text)
  for n in t.body:
   if isinstance(n,ast.Assign)and any(isinstance(a,ast.Name)and a.id=='INPUT_HASHES'for a in n.targets):n.value=ast.Constant(value='METADATA_BINDING_ONLY')
  return ast.dump(t,include_attributes=False)
 assert normalized(source)==normalized(changed),'Nonmetadata AST changed'
 return changed

def calibration_valid(cal,files):
 m=read(cal/'manifest.json');assert m['status']=='completed'
 assert m['model_hash_before']==m['model_hash_after']==MODEL_HASH and m['parameter_versions_unchanged'] is True
 assert m['calibration_maps_tensor_digest']==MAP_DIGEST and m['calibration_inputs_sha256']==CAL_SHA['calibration_inputs.json']
 assert m['revision']=='75e09b43581151bd1d9ef6700faa605df408979f'
 actual={n:sha(cal/n)for n in CAL_SHA}
 for n,h in CAL_SHA.items():
  if n!='calibration_operations.json':assert actual[n]==h,('Frozen calibration changed',n)
 if m.get('kind')=='calibration_only_rebuild':
  assert actual['calibration_operations.json']==REBUILD_OPERATIONS_SHA
  assert m['expected_forwards']==m['forward_passes']==m['calibration_forwards']==8 and m['evaluation_forwards']==0
  assert m['calibration_raw_sha256']==CAL_SHA['calibration_raw.pt']and m['calibration_maps_sha256']==CAL_SHA['calibration_maps.pt']
 else:assert actual['calibration_operations.json']==CAL_SHA['calibration_operations.json']
 for n,h in m['source_hashes'].items():assert files.get(n)==h,('Calibration source dependency missing/changed',n)
 return m,actual

def inventory_valid(source_dir,files):
 required={'gpt2_gate_additive.py','analyze_gate_additive.py','gate_additive_failure.py','gate_additive_readout.py',
 'gate_additive_inputs.py','gate_localization_inputs.py','prepare_additive_replay_inputs.py','reproduce_additive.py'}
 assert required<=set(files)
 for n,h in files.items():
  assert Path(n).name==n and Path(n).suffix in('.py','.md')and len(h)==64
  assert sha(source_dir/n)==h,('Source inventory mismatch',n)
 tree=ast.parse((source_dir/'gpt2_gate_additive.py').read_text())
 owns=[ast.literal_eval(n.value)for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='own'for t in n.targets)]
 assert len(owns)==1 and set(owns[0])<=set(files),'Runner own-source hash scope incomplete'
 for n in files:
  if not n.endswith('.py'):continue
  tree=ast.parse((source_dir/n).read_text())
  for node in ast.walk(tree):
   names=[node.module.split('.')[0]]if isinstance(node,ast.ImportFrom)and node.module else[a.name.split('.')[0]for a in node.names]if isinstance(node,ast.Import)else[]
   for local in names:
    if(source_dir/(local+'.py')).is_file():assert local+'.py'in files,('Unstaged local import',n,local)

def child_outcome(out):
 """Read recorded child facts after an uncertain/failed subprocess return."""
 result={'model_forwards':None,'forward_attempts':None,'calibration_forwards':None,
         'model_loaded':None,'gpu_used':None,'child_manifest_status':'unavailable'}
 path=out/'manifest.json'
 if not path.is_file():return result
 try:m=read(path)
 except (OSError,ValueError)as error:
  result['child_manifest_status']='unreadable';result['child_manifest_read_error']=str(error);return result
 result['child_manifest_status']=m.get('status','unknown');result['child_manifest_sha256']=sha(path)
 for source,target in(('forward_passes','model_forwards'),('forward_attempts','forward_attempts'),('calibration_forwards','calibration_forwards')):
  value=m.get(source)
  if isinstance(value,int)and not isinstance(value,bool)and value>=0:result[target]=value
 if isinstance(m.get('model_loaded'),bool):result['model_loaded']=m['model_loaded']
 elif isinstance(result['model_forwards'],int)and result['model_forwards']>0:result['model_loaded']=True
 memory=[m[k]for k in('peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')if isinstance(m.get(k),int)and m[k]>=0]
 if any(memory):result['gpu_used']=True
 elif len(memory)==2 and m.get('status')in('completed','failed'):result['gpu_used']=False
 return result

def verify_completed_child(out):
 m=read(out/'manifest.json');audit=read(out/'audit.json')
 assert m['status']==m['phase']=='completed'and m['forward_passes']==m['forward_attempts']==1992
 assert m['records']==1000 and m['suffix_forwards']==992 and m['calibration_forwards']==0
 assert m['model_hash_before']==m['model_hash_after']==MODEL_HASH and m['parameter_versions_unchanged'] is True
 assert m['calibration_maps_unchanged']is True and m['calibration_maps_tensor_digest']==MAP_DIGEST
 assert audit['status']=='passed'and audit['forwards']==1992 and audit['records']==1000
 return {'status':'completed',**child_outcome(out)}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for n in('config','metadata','calibration-reference','parent-metadata','source-dir','source-manifest','work'):ap.add_argument('--'+n,type=Path,required=True)
 group=ap.add_mutually_exclusive_group();group.add_argument('--prepare-only',action='store_true');group.add_argument('--execute',action='store_true')
 a=ap.parse_args();started=time.time();work=a.work.resolve()
 assert not work.exists()or not any(work.iterdir()),'Use a fresh workspace; no overwrite/automatic resume'
 work.mkdir(parents=True,exist_ok=True);dest=work/'pulse_transfer';dest.mkdir()
 out=dest/'gpt2_gate_additive_run'
 state={'status':'checking','model_forwards':0,'calibration_forwards':0,'model_loaded':False,'gpu_used':False,'requested_execute':a.execute,'wrapper_sha256':sha(__file__)}
 save(work/'additive_preparation.json',state)
 try:
  assert shutil.disk_usage(work).free>=2*1024**3+64*1024**2,'Preparation needs2GiB reserve plus64MiB staging headroom'
  files=read(a.source_manifest)['files'];inventory_valid(a.source_dir,files)
  assert files['reproduce_additive.py']==sha(__file__),'Use the wrapper from the approved source set'
  calmeta,cal_hashes=calibration_valid(a.calibration_reference,files)
  assert sha(a.metadata/'identity.json')==METADATA_SHA
  mi=read(a.metadata/'identity.json');assert mi['canonical_calibration_maps_sha256']==CAL_SHA['calibration_maps.pt']
  parent={}
  for n,h in PARENT_SHA.items():
   assert sha(a.parent_metadata/n)==h;parent[n]=read(a.parent_metadata/n)
  for n,h in files.items():copy_exact(a.source_dir/n,dest/n,h)
  for n,h in PARENT_SHA.items():copy_exact(a.parent_metadata/n,dest/'gpt2_run'/n,h)
  cal=dest/'additive_calibration'
  for n,h in {**cal_hashes,'manifest.json':sha(a.calibration_reference/'manifest.json')}.items():copy_exact(a.calibration_reference/n,cal/n,h)
  metadata=dest/'additive_metadata'
  for n,h in {**mi['metadata_sha256'],'identity.json':METADATA_SHA}.items():
   assert Path(n).name==n;copy_exact(a.metadata/n,metadata/n,h)
  supplied=read(a.config);cfg={'models':{'gpt2':copy.deepcopy(supplied['models']['gpt2'])},'sources':{n:supplied['sources'][n]for n in('wiki_validation','mill','wollstonecraft','adagrad','rmsprop','lbfgs')}}
  cfg['models']['gpt2']['path']=str(Path(cfg['models']['gpt2']['path']).expanduser().resolve())
  cfg['sources']={n:str(Path(p).expanduser().resolve())for n,p in cfg['sources'].items()}
  config=dest/'inputs.local.json';save(config,cfg);inputs=dest/'additive_inputs'
  env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  subprocess.run([sys.executable,str(dest/'prepare_additive_replay_inputs.py'),'rebuild','--config',str(config),'--metadata',str(metadata),'--calibration',str(cal),'--out',str(inputs)],cwd=dest,env={**env,'CUDA_VISIBLE_DEVICES':''},check=True)
  rebuilt=read(inputs/'reconstruction.json');assert rebuilt['status']=='passed'and rebuilt['model_forwards']==0 and rebuilt['model_loaded'] is False
  bindings=read(inputs/'input_bindings.json');canonical=(dest/'gpt2_gate_additive.py').read_text();portable=bind_input_hashes(canonical,bindings)
  original=dest/'staging_source'/'gpt2_gate_additive.canonical.py';original.parent.mkdir();original.write_text(canonical)
  assert sha(original)==files['gpt2_gate_additive.py'];(dest/'gpt2_gate_additive.py').write_text(portable)
  staged={n:sha(dest/n)for n in files};assert {n for n in files if staged[n]!=files[n]}=={'gpt2_gate_additive.py'}
  state.update(status='prepared',source_manifest_sha256=sha(a.source_manifest),canonical_source_sha256=files,staged_source_sha256=staged,
   metadata_identity_sha256=METADATA_SHA,calibration_files_sha256=cal_hashes,parent_metadata_sha256=PARENT_SHA,
   calibration_reference_manifest_sha256=sha(a.calibration_reference/'manifest.json'),
   calibration_reference_kind=calmeta.get('kind',calmeta.get('study','frozen_reference')),
   calibration_operations_schema='reviewed_eight_forward_rebuild'if calmeta.get('kind')=='calibration_only_rebuild'else'original_study_calibration',
   input_reconstruction_sha256=sha(inputs/'reconstruction.json'),input_bindings_sha256=sha(inputs/'input_bindings.json'),
   ast_only_input_hashes_changed=True,staged_bytes=sum(p.stat().st_size for p in work.rglob('*')if p.is_file()),
   scope='GPT2 tokenizer only; supplied frozen calibration copied/verified; no Llama/OLMo files or old research outputs read. Original private metadata hashes retained as provenance; exact tensors/directions/start identities preserved. INPUT_HASHES-only staged adaptation is explicit and reviewable.')
  assert state['staged_bytes']<=64*1024**2
  save(work/'additive_preparation.json',state)
  if not a.execute:print('PREPARED_ADDITIVE_REPLAY',dest,'zero model forwards');return
  # Only an explicit invocation of --execute enters this branch. The canonical
  # runner's own space/RSS, source/hash and failure guards remain unchanged.
  for n,d in parent['manifest.json']['checkpoint_files'].items():
   p=Path(cfg['models']['gpt2']['path'])/n;assert p.stat().st_size==d['bytes']and sha(p)==d['sha256']
  out=dest/'gpt2_gate_additive_run';state.update(status='execution_requested',model_forwards=None,model_loaded=None,gpu_used=None);save(work/'additive_preparation.json',state)
  subprocess.run([sys.executable,str(dest/'gpt2_gate_additive.py'),'--config',str(config),'--inputs',str(inputs),'--calibration-reference',str(cal),'--out',str(out)],cwd=dest,env=env,check=True)
  subprocess.run([sys.executable,str(dest/'analyze_gate_additive.py'),str(out)],cwd=dest,env={**env,'CUDA_VISIBLE_DEVICES':''},check=True)
  state.update(verify_completed_child(out))
  save(work/'additive_reproduction.json',{'status':'completed','new_model_forwards':1992,'new_calibration_forwards':0,
   'study_manifest_sha256':sha(out/'manifest.json'),'study_audit_sha256':sha(out/'audit.json'),
   'scope':'Successful local reproduction is not a numeric reference comparison or an external independent replication.'})
 except BaseException as error:
  if state.get('requested_execute')and state.get('model_forwards')is None:state.update(child_outcome(out))
  state.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  state.update(wall_seconds=time.time()-started,peak_wrapper_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  save(work/'additive_preparation.json',state)
if __name__=='__main__':main()
