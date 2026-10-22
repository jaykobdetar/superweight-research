"""T20 safe numerical release and tokenizer-only portable reproduction adapter.

Reuses reviewed T19 membership/locator/codec machinery. Private source text,
IDs, activations and full history files never enter the public package. Replays
are separately labelled adaptations; only INPUT_HASHES of the scientific runner
changes. No pretrained model forward, network, GPU or cleanup operation here.
"""
import argparse,ast,copy,hashlib,importlib,json,os,re,resource,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
BASE_VALIDATOR_SHA='46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4'
assert hashlib.sha256((ROOT/'native_value_portable_draft.py').read_bytes()).hexdigest()==BASE_VALIDATOR_SHA
import native_value_portable_draft as base
assert Path(base.__file__).resolve()==(ROOT/'native_value_portable_draft.py').resolve()
sha,read,put,copy_exact,outside_sources,value_sha,clean,safe=base.sha,base.read,base.put,base.copy_exact,base.outside_sources,base.value_sha,base.clean,base.safe
PACKAGE_INVENTORY=base.PACKAGE_INVENTORY
PHYSICAL=base.PHYSICAL
REFERENCE=base.REFERENCE
METADATA_NAMES={'data_manifest.template.json','exclusion_inventory.json','start_identity.template.json','history_attestation.json','preparation.template.json'}
METRICS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')
GROUPS=('metrics','suffix','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms')
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched','orth_write_matched')
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
CANONICAL={'manifest.json':'bb46a4f94e7276e79563a9962b1977298fd915667b4b7dbce64ef402a2279164','post_write_norm_analysis.json':'3fa2df289866b3e6e1914c21372f70a1da9c7ae99f213c4039935b0dd45c31eb','numeric_inventory.json':'6e03399366029c1f12491262a2402e0d00aeedebda7c39a88e4239a69c23d087'}
AUDITS={'numerical':{'file':'t20_numerical_audit.json','sha256':'8c3f3b2ec20f5adafcfe1aee1fc0b76fc3b2e192cca612b75626a41904738e10','schema':'T20_independent_numerical_audit_v1','source':'post_write_norm_numerical_audit.py'},'projection':{'file':'t20_projection_audit.json','sha256':'203b5bd1d158eb21ac6a6d70f2f39384ec7e769a75a2464cda9326c7b5bf23f3','schema':'T20_offline_projection_audit_v2','source':'post_write_norm_projection_audit.py'},'readout':{'file':'t20_readout_audit.json','sha256':'a205c4c89a9216b7372330f741f0ae0f26f3324d29eb39667593e22c32d2a061','schema':'T20_saved_readout_audit_v1','source':'post_write_norm_readout_audit.py'}}
OWNED=None

def fresh(out,allowance=128*1024**2):
 global OWNED
 base.fresh(out,allowance);st=out.stat();OWNED=(out.resolve(),st.st_dev,st.st_ino)
def no_personal_source(path):assert not re.search(r'/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/',Path(path).read_text()),('Personal host locator in source',Path(path).name)
def literal_source_hashes():return FROZEN_SOURCE_HASHES
# This table is filled once from the completed, independently audited run.
FROZEN_SOURCE_HASHES={'GPT2_POST_WRITE_NORM_MATCHING_INPUT_ADDENDUM.md': '2f458ad30f0d92bfd1a1b031e0c0f8e5bcd775a90208d1333ebad2b81216ce1c', 'GPT2_POST_WRITE_NORM_MATCHING_PROPOSAL.md': 'e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff', 'GPT2_POST_WRITE_NORM_MATCHING_RESOURCE_ADDENDUM.md': 'adc8aadae813549fb2a8c808df8f8c84a4635fde4eca6b120dd25cd18658c213', 'analyze_post_write_norm.py': 'e3d36eb3584c7785d702bfa857481ae0c62bd0ac7c986e6e99ff2c42e790f45d', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gate_write_norm.py': '62e8b4c67a522043a6cb315b97c7cee34f84dc853cfcb4121679344b4b06445f', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_post_write_norm.py': '43afc0af75d722fed2586866bc0ffc86d3e3998522ba66da63f3ef6e08a927cf', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'merge_post_write_norm.py': '1dec15447d5917541fdbcc7ac77b6f87db5d947445d3eac1c5c9df62c11ed33a', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'native_value_portable_draft.py': '46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4', 'post_write_norm_inputs.py': 'e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48', 'post_write_norm_selector.py': '34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}

def canonical_run(run,*,shards=False):
 for n,h in CANONICAL.items():assert sha(run/n)==h,('Canonical T20 identity',n)
 m=read(run/'manifest.json');assert m['status']=='completed' and m['execution_type']=='planned_sequential_segments'
 assert m['forward_attempts']==m['forward_passes']==m['fully_processed_forwards']==776 and m['evaluation_forwards']==m['suffix_forwards']==384 and m['anchor_forwards']==8 and m['calibration_forwards']==0
 assert m['source_hashes']==FROZEN_SOURCE_HASHES and m['all_final_guards_passed'] is True
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True
 assert m['precision_settings_before']==m['precision_settings_after']
 for n,h in m['source_hashes'].items():assert Path(n).name==n and sha(run/'frozen_sources'/n)==h
 for n,h in m['input_dependencies'].items():assert sha(run/n)==h
 inventory=read(run/'numeric_inventory.json');expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c in CORPORA for w in range(4) for i in ('natural','bos') for b in ('000','111')};assert set(inventory)==expected
 if shards:
  assert {p.relative_to(run).as_posix() for p in (run/'inputs').iterdir()}==expected
  for n,v in inventory.items():assert v=={'sha256':sha(run/n),'bytes':(run/n).stat().st_size}
 return m,inventory

def metadata_objects(run,m):
 dm=read(run/'data_manifest.json');inv=read(run/'exclusion_inventory.json');pre=read(run/'preflight.json');assert pre['status']=='passed' and pre['history_hash_count']==713
 template=clean(dm);template.pop('literals');safe(template)
 # Keep every interval, full8192Fbuffer, all713hashes and T19natural/BOS
 # attestations. Omit private1370-file review universe/aliases, retain hashes.
 exclusion={k:clean(inv[k]) for k in ('sources','exclusions','retained_calibration_buffers','history_token_hashes','T19_used_spans','T19_identity_attestations','freshness_scope')}
 assert len(set(exclusion['history_token_hashes']))==713 and len(exclusion['retained_calibration_buffers'])==4 and len(exclusion['T19_identity_attestations'])==32 and len(exclusion['T19_used_spans'])==16
 for b in exclusion['retained_calibration_buffers']:
  assert b['char_end']-b['char_start']==8192 and any(r['char_start']<=b['char_start']<b['char_end']<=r['char_end'] for r in exclusion['exclusions'][b['corpus']])
 history={'schema':'T20_portable_inherited_history_v1','historical_exposure_is_inherited':True,'canonical_private_scan_sha256':m['input_dependencies']['historical_manifest_scan.json'],'canonical_private_review_sha256':m['input_dependencies']['historical_manifest_scan_review.json'],'canonical_private_exclusion_inventory_sha256':m['input_dependencies']['exclusion_inventory.json'],'independent_actual_input_audit_sha256':'fdb713b178812589f1e95b40410ccb32f06433d7e03730db34a48d615e06ba43','history_hash_count':713,'retained_F_buffers':4,'T19_used_spans':16,'T19_start_attestations':32,'scope':'Portable users inherit the reviewed history closure. No local1370-file scan, older model runs, Llama loading, historical token tensors or private fixture tree is required. Current lawful six sources/tokenizer and every candidate/selected identity are rechecked. Pretraining overlap remains unknown.'}
 preparation={k:pre[k] for k in ('evaluation_passages','evaluation_inputs','history_hash_count','accepted_identity_hash_count','historical_first_token_membership','proposal_sha256','input_addendum_sha256','script_sha256','selector_sha256')};preparation['canonical_private_preflight_sha256']=m['input_dependencies']['preflight.json']
 objects={'data_manifest.template.json':template,'exclusion_inventory.json':exclusion,'start_identity.template.json':clean(read(run/'start_identity_summary.json')),'history_attestation.json':history,'preparation.template.json':preparation}
 for v in objects.values():safe(v)
 return objects,value_sha(dm['literals'])

def package_members(identity):
 assert identity['schema']=='T20_portable_metadata_v1';safe(identity)
 assert identity['frozen_source_sha256']==FROZEN_SOURCE_HASHES and set(identity['metadata_sha256'])==METADATA_NAMES
 expected={'identity.json':None}
 for n,h in {**identity['frozen_source_sha256'],**identity['portable_source_sha256']}.items():
  assert Path(n).name==n and Path(n).suffix in ('.py','.md');expected['experiments/pulse_transfer/'+n]=h
 assert set(identity['portable_source_sha256'])=={'post_write_norm_portable.py','audit_post_write_norm_public.py'}
 for n,h in identity['metadata_sha256'].items():expected['metadata/'+n]=h
 assert set(identity['reference_audit']['files'])==set(REFERENCE)
 for n,v in identity['reference_audit']['files'].items():expected['calibration_reference/'+n]=v['sha256']
 assert set(identity['metadata_dependencies'])==set(base.METADATA_DEPENDENCY_HASHES)
 for n,v in identity['metadata_dependencies'].items():
  assert v=={'file':n.removeprefix('../').replace('/','__'),'sha256':base.METADATA_DEPENDENCY_HASHES[n]};expected['metadata_dependencies/'+v['file']]=v['sha256']
 for n,h in expected.items():base.relative_member(n);assert h is None or base.hash_value(h)
 return expected

def validate_package(package):
 dirs={'metadata','calibration_reference','metadata_dependencies','experiments','experiments/pulse_transfer'};actual=base.scan_package(package,dirs)
 assert {'identity.json',PACKAGE_INVENTORY}<=actual;identity=read(package/'identity.json');expected=package_members(identity);assert actual==set(expected)|{PACKAGE_INVENTORY},'Exact declared package membership required'
 inventory=read(package/PACKAGE_INVENTORY);safe(inventory);assert set(inventory)=={'schema','inventory_member','files','repository_mutated','outcomes_exported'} and inventory['schema']=='T20_metadata_inventory_v1' and inventory['inventory_member']==PACKAGE_INVENTORY and inventory['repository_mutated'] is False and inventory['outcomes_exported'] is False
 assert set(inventory['files'])==set(expected)
 for n,h in expected.items():
  entry=inventory['files'][n];assert entry=={'sha256':sha(package/n),'bytes':(package/n).stat().st_size} and (h is None or entry['sha256']==h)
 for n in METADATA_NAMES:safe(read(package/'metadata'/n))
 _,original=base.assigned((package/'experiments/pulse_transfer/gpt2_post_write_norm.py').read_text(),'INPUT_HASHES');assert original==identity['canonical_private_input_sha256']
 return identity

def export_metadata(a):
 outside_sources(a.out,a.run);m,_=canonical_run(a.run);objects,literal_commitment=metadata_objects(a.run,m);ref=base.reference_audit(a.run/'calibration_reference',read(a.run/'data_manifest.json'))
 for name,h in m['source_hashes'].items():no_personal_source(a.run/'frozen_sources'/name)
 for source in (Path(__file__),ROOT/'audit_post_write_norm_public.py'):no_personal_source(source)
 fresh(a.out)
 for n,v in objects.items():put(a.out/'metadata'/n,v)
 for n,h in m['source_hashes'].items():copy_exact(a.run/'frozen_sources'/n,a.out/'experiments/pulse_transfer'/n,h)
 portable={n:sha(ROOT/n) for n in ('post_write_norm_portable.py','audit_post_write_norm_public.py')}
 for n,h in portable.items():copy_exact(ROOT/n,a.out/'experiments/pulse_transfer'/n,h)
 for n in REFERENCE:copy_exact(a.run/'calibration_reference'/n,a.out/'calibration_reference'/n,ref['files'][n]['sha256'])
 deps={}
 for n,h in m['metadata_dependencies'].items():
  filename=n.removeprefix('../').replace('/','__');safe(read(a.run/'metadata_dependencies'/filename));copy_exact(a.run/'metadata_dependencies'/filename,a.out/'metadata_dependencies'/filename,h);deps[n]={'file':filename,'sha256':h}
 identity={'schema':'T20_portable_metadata_v1','canonical_run_manifest_sha256':CANONICAL['manifest.json'],'canonical_private_input_sha256':m['input_dependencies'],'frozen_source_sha256':m['source_hashes'],'portable_source_sha256':portable,'metadata_sha256':{n:sha(a.out/'metadata'/n) for n in objects},'metadata_dependencies':deps,'reference_audit':ref,'private_literals_commitment':literal_commitment,'physical_sources':list(PHYSICAL),'token_artifact_must_remain_byte_exact':True,'metadata_binding_scope':'Regenerated portable history/provenance metadata explicitly differs from private original bytes; only INPUT_HASHES changes in the staged scientific runner. Full776modelreplay is separate and unperformed by this adapter.','model_forwards':0,'calibration_forwards':0}
 safe(identity);put(a.out/'identity.json',identity);expected=package_members(identity);actual=base.scan_package(a.out,base.expected_dirs(expected));assert actual==set(expected)
 put(a.out/PACKAGE_INVENTORY,{'schema':'T20_metadata_inventory_v1','inventory_member':PACKAGE_INVENTORY,'files':{n:{'sha256':sha(a.out/n),'bytes':(a.out/n).stat().st_size} for n in sorted(actual)},'repository_mutated':False,'outcomes_exported':False});validate_package(a.out)

def portable_input_objects(package,d,private_dm):
 md=package/'metadata';history=read(md/'history_attestation.json');scan={'schema':'T20_portable_inherited_history_attestation_v1','historical_exposure_is_inherited':True,'attestation':history,'safe_exclusion_inventory_sha256':sha(md/'exclusion_inventory.json')};review={'status':'passed','kind':'portable_inherited_history_not_current_private_rescan','attestation_sha256':sha(md/'history_attestation.json'),'canonical_private_review_sha256':history['canonical_private_review_sha256']}
 private_dm=copy.deepcopy(private_dm);private_dm['exclusion_inventory_sha256']=sha(md/'exclusion_inventory.json')
 pre={'status':'passed','phase':'completed','kind':'T20_portable_tokenizer_only_reconstruction_v1','model_forwards':0,'pretrained_loaded':False,'gpu_used':False,'calibration_fitted':False,'evaluation_passages':16,'evaluation_inputs':32,'history_hash_count':713,'historical_exposure_is_inherited':True,'metadata_identity_sha256':sha(package/'identity.json'),'canonical_private_preflight_sha256':d['canonical_private_input_sha256']['preflight.json'],'current_sources_checked':list(PHYSICAL),'all_candidates_rechecked':True}
 return {'data_manifest.json':private_dm,'historical_manifest_scan.json':scan,'historical_manifest_scan_review.json':review,'preflight.json':pre,'start_identity_summary.json':read(md/'start_identity.template.json')}

def tokenizer(package,config):
 from transformers import AutoTokenizer
 dm=read(package/'metadata/data_manifest.template.json');cfg=read(config);spec=read(package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2'];gc=cfg['models']['gpt2'];assert gc['model_id']==spec['model_id'] and gc['revision']==spec['revision']
 for n,h in dm['tokenizer_sha256'].items():assert sha(Path(gc['path'])/n)==h
 tok=AutoTokenizer.from_pretrained(gc['path'],local_files_only=True);assert tok.is_fast and type(tok.bos_token_id)is int
 literals={name:{'literal':text,'ids':tok.encode(text,add_special_tokens=False)} for name,text in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]};literals['start_special']={'label':'bos','id':tok.bos_token_id}
 return tok,literals,cfg

def rebuild(a):
 outside_sources(a.out,a.package);d=validate_package(a.package);os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 import torch
 from datasets import Dataset
 code=a.package/'experiments/pulse_transfer';sys.path.insert(0,str(code));selector=importlib.import_module('post_write_norm_selector');assert Path(selector.__file__).resolve()==(code/'post_write_norm_selector.py').resolve()
 torch.set_num_threads(1);dm=read(a.package/'metadata/data_manifest.template.json');inv=read(a.package/'metadata/exclusion_inventory.json');tok,literals,cfg=tokenizer(a.package,a.config);assert value_sha(literals)==d['private_literals_commitment'];texts={}
 for c in PHYSICAL:
  p=Path(cfg['sources'][c]);spec=dm['sources'][c];assert sha(p)==spec['file_sha256']
  if c=='wiki_validation':text='\n\n'.join(Dataset.from_file(str(p))['text'])
  elif c in ('utilitarianism','ethics'):
   raw=p.read_text();lo=raw.find('*** START OF');hi=raw.rfind('*** END OF');assert lo>=0 and hi>lo;text=raw[raw.find('\n',lo)+1:hi]
  else:text=p.read_text()
  assert len(text)==spec['characters'] and selector.textsha(text)==spec['processed_text_sha256'];texts[c]=text
 texts['optimizers_new']='\n\n'.join(texts[c] for c in ('adadelta','adamax','asgd'));assert selector.textsha(texts['optimizers_new'])==dm['sources']['optimizers_new']['processed_text_sha256'];components=dm['sources']['optimizers_new']['components'];cursor=0
 for c,component in zip(('adadelta','adamax','asgd'),components):assert component=={'source':c,'char_start':cursor,'char_end':cursor+len(texts[c]),'file_sha256':dm['sources'][c]['file_sha256']};cursor+=len(texts[c])+2
 progress={};history=set(inv['history_token_hashes']);assert len(history)==713;selector.select_panel(CORPORA,texts,tok,inv['exclusions'],history,components,False,progress,bos_id=tok.bos_token_id)
 assert len(progress['items'])==16 and clean(progress['selected'])==dm['passages'] and progress['rejections']==dm['rejections'];assert sum(map(len,progress['rejections'].values()))==428
 private_dm=copy.deepcopy(dm);private_dm['passages']=progress['selected'];private_dm['literals']=literals;objects=portable_input_objects(a.package,d,private_dm)
 assert validate_package(a.package)==d
 for c in PHYSICAL:assert sha(Path(cfg['sources'][c]))==dm['sources'][c]['file_sha256']
 fresh(a.out,32*1024**2);torch.save(progress['items'],a.out/'tokens.pt');assert sha(a.out/'tokens.pt')==d['canonical_private_input_sha256']['tokens.pt']
 copy_exact(a.package/'metadata/exclusion_inventory.json',a.out/'exclusion_inventory.json')
 for n,v in objects.items():put(a.out/n,v)
 put(a.out/'input_bindings.json',{'schema':'T20_portable_input_binding_v1','original':d['canonical_private_input_sha256'],'regenerated':{n:sha(a.out/n) for n in d['canonical_private_input_sha256']},'metadata_identity_sha256':sha(a.package/'identity.json'),'tokens_byte_exact':True,'historical_exposure_is_inherited':True,'model_forwards':0})


def validate_private_inputs(a,d):
 import torch
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 expected=set(d['canonical_private_input_sha256'])|{'input_bindings.json'};assert base.scan_package(a.inputs,set())==expected
 binding=read(a.inputs/'input_bindings.json');assert set(binding)=={'schema','original','regenerated','metadata_identity_sha256','tokens_byte_exact','historical_exposure_is_inherited','model_forwards'} and binding['schema']=='T20_portable_input_binding_v1' and binding['original']==d['canonical_private_input_sha256'] and binding['metadata_identity_sha256']==sha(a.package/'identity.json') and binding['tokens_byte_exact'] is True and binding['historical_exposure_is_inherited'] is True and binding['model_forwards']==0
 assert set(binding['regenerated'])==set(d['canonical_private_input_sha256'])
 for n,h in binding['regenerated'].items():assert sha(a.inputs/n)==h
 assert sha(a.inputs/'tokens.pt')==d['canonical_private_input_sha256']['tokens.pt'] and sha(a.inputs/'exclusion_inventory.json')==sha(a.package/'metadata/exclusion_inventory.json')
 tok,literals,_=tokenizer(a.package,a.config);assert value_sha(literals)==d['private_literals_commitment'];items=torch.load(a.inputs/'tokens.pt',weights_only=True,map_location='cpu');dm=read(a.package/'metadata/data_manifest.template.json');assert len(items)==len(dm['passages'])==16
 for item,record in zip(items,dm['passages']):
  assert set(item)=={'corpus','window','tokens'} and (item['corpus'],item['window'])==(record['corpus'],record['window']);t=item['tokens'];assert t.dtype==torch.int64 and t.shape==(512,) and hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()==record['window_token_sha256'];bos=t.clone();bos[0]=tok.bos_token_id;assert hashlib.sha256(bos.numpy().tobytes()).hexdigest()==record['bos_window_token_sha256'];record['first_token_id']=int(t[0])
 dm['literals']=literals;objects=portable_input_objects(a.package,d,dm)
 for n,v in objects.items():assert read(a.inputs/n)==v,('Portable scientific metadata altered',n)
 return binding

def bind_runner(source,binding):
 node,expected=base.assigned(source,'INPUT_HASHES');assert expected==binding['original'] and set(expected)==set(binding['regenerated']) and binding['regenerated']['tokens.pt']==expected['tokens.pt']
 lines=source.splitlines(keepends=True);changed=''.join(lines[:node.lineno-1])+'INPUT_HASHES='+repr(binding['regenerated'])+'\n'+''.join(lines[node.end_lineno:])
 def normalized(text):
  tree=ast.parse(text)
  for n in tree.body:
   if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INPUT_HASHES' for t in n.targets):n.value=ast.Constant('ONLY_PORTABLE_METADATA_INPUT_HASHES')
  return ast.dump(tree,include_attributes=False)
 assert normalized(source)==normalized(changed);return changed

def stage_replay(a):
 outside_sources(a.out,a.package,a.inputs);d=validate_package(a.package);binding=validate_private_inputs(a,d);fresh(a.out,64*1024**2);dest=a.out/'pulse_transfer';dest.mkdir()
 for n,h in d['frozen_source_sha256'].items():
  if n=='gpt2_post_write_norm.py':(dest/n).open('x').write(bind_runner((a.package/'experiments/pulse_transfer'/n).read_text(),binding))
  else:copy_exact(a.package/'experiments/pulse_transfer'/n,dest/n,h)
 for n,v in d['metadata_dependencies'].items():copy_exact(a.package/'metadata_dependencies'/v['file'],dest/n,v['sha256'])
 for n in REFERENCE:copy_exact(a.package/'calibration_reference'/n,a.out/'calibration_reference'/n,d['reference_audit']['files'][n]['sha256'])
 prefix=[sys.executable,'-B',str((dest/'gpt2_post_write_norm.py').resolve()),'--config',str(a.config.resolve()),'--inputs',str(a.inputs.resolve()),'--calibration-reference',str((a.out/'calibration_reference').resolve())]
 commands={'A':prefix+['--segment','A','--out',str((a.out/'segment_A').resolve())],'B':prefix+['--segment','B','--previous-segment',str((a.out/'segment_A').resolve()),'--out',str((a.out/'segment_B').resolve())],'merge':[sys.executable,'-B',str((dest/'merge_post_write_norm.py').resolve()),'--segment-a',str((a.out/'segment_A').resolve()),'--segment-b',str((a.out/'segment_B').resolve()),'--out',str((a.out/'composite').resolve())]}
 put(a.out/'replay_preparation.json',{'status':'prepared_only','scheduled_model_forwards':776,'model_forwards':0,'calibration_forwards':0,'canonical_runner_sha256':d['frozen_source_sha256']['gpt2_post_write_norm.py'],'adapted_runner_sha256':sha(dest/'gpt2_post_write_norm.py'),'adaptation':'Only seven-entry INPUT_HASHES assignment; original intervention/schedule/scoring/all other AST exact. Regenerated portable history metadata is inherited and labelled, not original private metadata bytes.','commands':commands})


def known_keys(v,allowed):
 if isinstance(v,dict):
  assert set(v)<=set(allowed),('Unknown audit fields',set(v)-set(allowed))
  for w in v.values():known_keys(w,allowed)
 elif isinstance(v,list):
  for w in v:known_keys(w,allowed)
AUDIT_KEYS={'numerical': ['A', 'B', 'actual_vocabulary_readout_rows_retained', 'analysis_sha256', 'anchor_rows', 'arrays_verified', 'auditor_sha256', 'causal_limits', 'coefficient_reconstruction', 'committed', 'comparison_counts', 'comparisons', 'contrast_definitions', 'contrasts', 'controls', 'counts', 'current', 'disk_checks', 'exact', 'execution_review_sha256', 'file_deletions', 'final_sha256', 'forward_attempt', 'forward_completed', 'forward_intent', 'full_snapshot_context_norm_reconstruction', 'full_snapshot_context_reconstruction', 'full_snapshot_gate_logits', 'full_snapshot_gate_mass', 'full_snapshot_interpolation_recipe', 'full_snapshot_stable_mixture', 'full_snapshot_write_norm_reconstruction', 'full_snapshots_reconstructed', 'gate_norm_reconstruction', 'gate_probability_reconstruction', 'gpu_used', 'input_audit_sha256', 'journals', 'matching_RMS_diagnostics', 'matching_gap', 'maximum_absolute_error', 'maximum_symmetric_relative_coordinate_error', 'model_forwards', 'model_returned', 'native_identity', 'native_repeat', 'norm_aggregates', 'ordinary_shards', 'paired_bootstrap', 'paired_starts', 'peak_rss_bytes', 'policy_sha256', 'ppl exponentiated intervals', 'ppl exponentiated paired mean', 'pretrained_checkpoint_access', 'raw_reconstruction', 'raw_shards', 'relative_gap_quantiles', 'resource_supplement_sha256', 'run_manifest_sha256', 'schema', 'scope', 'source_and_copied_metadata_exact', 'source_offsets_and_history', 'source_q0', 'statistics', 'status', 'suffix', 'summaries', 'versioned_checkpoint_and_resource_evidence', 'versioned_checkpoints', 'wall_seconds', 'window_metrics', 'window_rows'], 'projection': ['000', '000__baseline', '000__current', '000__orth_full', '000__orth_write_matched', '000__remove_full', '000__remove_write_matched', '111', '111__baseline', '111__current', '111__orth_full', '111__orth_write_matched', '111__remove_full', '111__remove_write_matched', 'D', 'N', 'actual_context__baseline_write', 'arithmetic_scope', 'assumptions', 'auditor_sha256', 'bias_sha256', 'checkpoint_sha256', 'context_sha256', 'contract_sha256', 'coordinates', 'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'cudnn_benchmark', 'cudnn_deterministic', 'cudnn_sdp_enabled', 'cudnn_tf32_condition', 'decoder_sha256', 'delivered_context__proposed_full_write', 'deterministic_algorithms', 'elapsed_seconds', 'error_convention', 'execution_review_sha256', 'failed_coordinates', 'file_deletions', 'flash_sdp_enabled', 'float32_matmul_precision', 'gamma32_upper', 'gamma64_E_upper', 'gamma64_upper', 'gpu_computation', 'header_sha256', 'inventory_sha256', 'k', 'k_E', 'math_sdp_enabled', 'max_abs_error', 'max_bound', 'max_error_to_bound_ratio', 'max_input_flush_budget_upper', 'memory_efficient_sdp_enabled', 'min_bound', 'model_forwards', 'numeric_constants', 'observed_checks', 'orientation', 'passed', 'peak_rss_bytes', 'precision_settings_before', 'projection_comparisons', 'protocol_sha256', 'rms_error', 'run_manifest_sha256', 'schema', 'scope', 'segment_final_precision_guards_checked', 'shards', 'states', 'states_checked', 'status', 'u32', 'u64', 'weight_sha256', 'weight_tensors_loaded', 'write_sha256', 'zero_bound_coordinates', 'zero_bound_ratio'], 'readout': ['000__baseline', '000__current', '000__orth_full', '000__orth_write_matched', '000__remove_full', '000__remove_write_matched', '111__baseline', '111__current', '111__orth_full', '111__orth_write_matched', '111__remove_full', '111__remove_write_matched', 'argmax_mismatches', 'auditor_sha256', 'checkpoint_sha256', 'compatible_with_observed_drift', 'counts', 'dot_product_bound', 'elapsed_seconds', 'embedding_sha256', 'execution_review_sha256', 'file_deletions', 'full_vocabulary_rows', 'full_vocabulary_rows_per_state', 'gamma_2n', 'gpu_computation', 'head', 'inventory_sha256', 'kl_boundary', 'kl_boundary_to_trial', 'kl_c', 'kl_c_to_trial', 'left_id', 'left_margin', 'loss', 'margin', 'max_abs', 'max_abs_bound', 'max_error_to_bound_ratio', 'model_forwards', 'nll', 'peak_rss_bytes', 'prediction', 'protocol_sha256', 'query', 'query_rows', 'readout_vs_cpu_fp64', 'right_id', 'right_margin', 'rms', 'row_max_abs_drift', 'run_manifest_sha256', 'saved_metric_errors', 'saved_metric_max_abs', 'schema', 'scope', 'source_hidden_byte_exact', 'states', 'status', 'symmetric_relative_l2', 'target_positions', 'weight_tensors_loaded']}
AUDIT_ROOT_KEYS={'numerical': ['status', 'schema', 'auditor_sha256', 'policy_sha256', 'resource_supplement_sha256', 'execution_review_sha256', 'input_audit_sha256', 'run_manifest_sha256', 'analysis_sha256', 'statistics', 'arrays_verified', 'versioned_checkpoint_and_resource_evidence', 'full_snapshots_reconstructed', 'actual_vocabulary_readout_rows_retained', 'comparison_counts', 'controls', 'journals', 'scope', 'model_forwards', 'pretrained_checkpoint_access', 'gpu_used', 'file_deletions', 'peak_rss_bytes', 'wall_seconds'], 'projection': ['schema', 'status', 'auditor_sha256', 'contract_sha256', 'decoder_sha256', 'protocol_sha256', 'execution_review_sha256', 'run_manifest_sha256', 'inventory_sha256', 'checkpoint_sha256', 'header_sha256', 'weight_tensors_loaded', 'weight_sha256', 'bias_sha256', 'numeric_constants', 'orientation', 'precision_settings_before', 'segment_final_precision_guards_checked', 'arithmetic_scope', 'model_forwards', 'gpu_computation', 'file_deletions', 'states', 'shards', 'scope', 'states_checked', 'projection_comparisons', 'peak_rss_bytes', 'elapsed_seconds'], 'readout': ['schema', 'status', 'auditor_sha256', 'protocol_sha256', 'run_manifest_sha256', 'execution_review_sha256', 'inventory_sha256', 'checkpoint_sha256', 'embedding_sha256', 'weight_tensors_loaded', 'head', 'query_rows', 'target_positions', 'counts', 'saved_metric_max_abs', 'states', 'scope', 'model_forwards', 'gpu_computation', 'file_deletions', 'peak_rss_bytes', 'elapsed_seconds']}
NUMERICAL_COMPARISONS=['coefficient_reconstruction', 'full_snapshot_context_norm_reconstruction', 'full_snapshot_context_reconstruction', 'full_snapshot_gate_logits', 'full_snapshot_gate_mass', 'full_snapshot_interpolation_recipe', 'full_snapshot_stable_mixture', 'full_snapshot_write_norm_reconstruction', 'gate_norm_reconstruction', 'gate_probability_reconstruction', 'matching_RMS_diagnostics', 'matching_gap', 'norm_aggregates', 'paired_bootstrap', 'ppl exponentiated intervals', 'ppl exponentiated paired mean', 'relative_gap_quantiles', 'window_metrics']

def numeric_fields(value,names):
 assert set(value)==set(names)
 assert all(type(v) in (int,float) and not isinstance(v,bool) and __import__('math').isfinite(v) for v in value.values())
 return copy.deepcopy(value)

def audit_public(report,kind):
 spec=AUDITS[kind];assert report['schema']==spec['schema'] and report['status']=='passed';assert set(report)==set(AUDIT_ROOT_KEYS[kind]);known_keys(report,AUDIT_KEYS[kind])
 result={k:copy.deepcopy(report[k]) for k in AUDIT_ROOT_KEYS[kind] if k!='states'}
 if kind=='numerical':
  numeric_fields(report['statistics'],('anchor_rows','contrast_definitions','contrasts','paired_starts','summaries','window_rows'))
  numeric_fields(report['controls'],('current','native_identity','native_repeat','source_q0','suffix'))
  assert set(report['comparison_counts'])==set(NUMERICAL_COMPARISONS)
  for record in report['comparison_counts'].values():numeric_fields(record,('comparisons','exact','maximum_absolute_error','maximum_symmetric_relative_coordinate_error'))
  assert set(report['journals'])==set(report['versioned_checkpoint_and_resource_evidence'])=={'A','B'}
  for label in ('A','B'):
   evidence=report['versioned_checkpoint_and_resource_evidence'][label];assert set(evidence)=={'disk_checks','raw_shards','source_and_copied_metadata_exact','versioned_checkpoints'} and evidence['source_and_copied_metadata_exact'] is True
   numeric_fields({k:v for k,v in evidence.items() if k!='source_and_copied_metadata_exact'},('disk_checks','raw_shards','versioned_checkpoints'))
   journal=report['journals'][label];assert set(journal)=={'counts','final_sha256'};base.hash_value(journal['final_sha256']);numeric_fields(journal['counts'],('committed','forward_attempt','forward_completed','forward_intent','model_returned'))
 else:
  assert set(report['states'])=={b+'__'+a for b in ('000','111') for a in ARMS};result['states']={}
  for state,values in report['states'].items():
   if kind=='projection':
    assert set(values)=={'actual_context__baseline_write','delivered_context__proposed_full_write'}
    result['states'][state]={}
    for comparison,record in values.items():
     assert set(record)=={'coordinates','failed_coordinates','max_abs_error','rms_error','max_bound','min_bound','max_input_flush_budget_upper','error_convention','max_error_to_bound_ratio','zero_bound_coordinates','zero_bound_ratio','passed','context_sha256','write_sha256'}
     numeric_fields({k:v for k,v in record.items() if k not in ('error_convention','zero_bound_ratio','passed','context_sha256','write_sha256')},('coordinates','failed_coordinates','max_abs_error','rms_error','max_bound','min_bound','max_input_flush_budget_upper','max_error_to_bound_ratio','zero_bound_coordinates'))
     assert record['passed'] is True;result['states'][state][comparison]=copy.deepcopy(record)
   else:
    assert set(values)=={'nll','prediction','margin','kl_c_to_trial','kl_boundary_to_trial','saved_metric_errors','readout_vs_cpu_fp64','dot_product_bound','argmax_mismatches','source_hidden_byte_exact'}
    assert len(values['prediction'])==4 and all(type(x)is int and 0<=x<50257 for x in values['prediction'])
    public={}
    for name in ('nll','margin','kl_c_to_trial','kl_boundary_to_trial'):
     assert len(values[name])==4 and all(type(x)in(int,float) and __import__('math').isfinite(x) for x in values[name]);public[name]=copy.deepcopy(values[name])
    assert set(values['saved_metric_errors'])=={'loss','kl_c','kl_boundary'}
    public['saved_metric_errors']={name:numeric_fields(record,('max_abs','rms','symmetric_relative_l2')) for name,record in values['saved_metric_errors'].items()}
    public['readout_vs_cpu_fp64']=numeric_fields(values['readout_vs_cpu_fp64'],('max_abs','rms','symmetric_relative_l2'));public['dot_product_bound']=numeric_fields(values['dot_product_bound'],('gamma_2n','max_abs_bound','max_error_to_bound_ratio'))
    public['argmax_mismatches']=[]
    for row in values['argmax_mismatches']:
     assert set(row)=={'query','left_id','right_id','left_margin','right_margin','row_max_abs_drift','compatible_with_observed_drift'} and row['query'] in (1,126,127,510) and type(row['left_id'])is type(row['right_id'])is int
     numeric_fields({k:row[k] for k in ('query','left_margin','right_margin','row_max_abs_drift')},('query','left_margin','right_margin','row_max_abs_drift'));assert type(row['compatible_with_observed_drift'])is bool
     public['argmax_mismatches'].append({k:copy.deepcopy(row[k]) for k in ('query','left_margin','right_margin','row_max_abs_drift','compatible_with_observed_drift')})
    assert values['source_hidden_byte_exact'] is True;public['source_hidden_byte_exact']=True;result['states'][state]=public
 safe(result);result['public_omissions']='All prediction and mismatch token IDs omitted through explicit typed field selection; full private audit commitment retained. No weight or activation payload included.'
 return result

def scalar_ledger(value):
 import numpy as np
 assert not isinstance(value,np.ndarray),'Control ledgers may contain no raw array'
 if isinstance(value,dict):
  assert not any(base.private_key(k) for k in value),'Private field in control ledger';return {k:scalar_ledger(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [scalar_ledger(v) for v in value]
 assert value is None or type(value) in (str,int,float,bool);return value

def numeric_prerequisites(a):
 m,inventory=canonical_run(a.run,shards=True);analysis=read(a.run/'post_write_norm_analysis.json');assert analysis['scope']['source_hashes']=={Path(n).name:v['sha256'] for n,v in inventory.items()} and analysis['scope']['raw_manifest_sha256']==CANONICAL['manifest.json'] and analysis['scope']['raw_inventory_sha256']==CANONICAL['numeric_inventory.json'] and analysis['scope']['analyzer_sha256']==m['source_hashes']['analyze_post_write_norm.py'];safe(analysis)
 public={}
 for kind,spec in AUDITS.items():
  path=a.audits/spec['file'];assert sha(path)==spec['sha256'];r=read(path);assert r['run_manifest_sha256']==CANONICAL['manifest.json'] and r['auditor_sha256']==sha(a.audits/spec['source']);no_personal_source(a.audits/spec['source'])
  if kind=='numerical':assert r['analysis_sha256']==CANONICAL['post_write_norm_analysis.json']
  else:assert r['inventory_sha256']==CANONICAL['numeric_inventory.json']
  public[kind]=audit_public(r,kind)
 return m,inventory,analysis,public

def export_numeric(a):
 import numpy as np
 from analyze_post_write_norm import load_npz,GATE_KEYS,WRITE_KEYS
 outside_sources(a.out,a.run);m,inventory,analysis,audits=numeric_prerequisites(a);fresh(a.out,256*1024**2)
 put(a.out/'post_write_norm_analysis.json',analysis);safe(m);put(a.out/'execution.json',m)
 for kind,obj in audits.items():put(a.out/'audits'/(kind+'.json'),obj);copy_exact(a.audits/AUDITS[kind]['source'],a.out/'audits'/(kind+'.py'),obj['auditor_sha256'])
 index=[];count=0
 for name,entry in sorted(inventory.items()):
  raw=load_npz(a.run/name);arrays={}
  for group in ('metrics','suffix'):
   for arm,z in raw[group].items():
    for key in METRICS:
     t=z[key];n=127 if group=='suffix' and key=='loss' else 128 if group=='suffix' else 511;assert t.shape==(n,) and t.dtype==(np.bool_ if key.startswith('agreement') else np.float32) and np.isfinite(t).all();arrays[group+'__'+arm+'__'+key]=t
  for group in ('gate_norms','suffix_gate_norms'):
   for arm,layers in raw[group].items():
    if arm=='native_repeat':assert layers=={};continue
    assert set(layers)=={10} and set(layers[10])==GATE_KEYS
    for key,t in layers[10].items():
     n=12 if key=='source_v_norm' else 127 if group=='suffix_gate_norms' else 511;assert t.shape==(n,) and t.dtype==(np.float32 if key=='s' else np.float64) and np.isfinite(t).all();arrays[group+'__'+arm+'__10__'+key]=t
  for group in ('write_norms','suffix_write_norms'):
   for arm,values in raw[group].items():
    if arm=='native_repeat':assert values=={};continue
    assert set(values)==WRITE_KEYS['baseline' if arm=='native_identity' else arm]
    for key,t in values.items():assert t.shape==(512,) and t.dtype==np.float64 and np.isfinite(t).all();arrays[group+'__'+arm+'__'+key]=t
  ledger=scalar_ledger({k:raw[k] for k in ('metadata','checks','operations','suffix_operations')});safe(ledger);dest=a.out/'inputs'/Path(name).name;dest.parent.mkdir(exist_ok=True)
  with dest.open('xb') as f:np.savez_compressed(f,**arrays)
  with np.load(dest,allow_pickle=False) as stored:assert set(stored.files)==set(arrays) and all(stored[k].dtype==v.dtype and stored[k].shape==v.shape and stored[k].tobytes()==v.tobytes() for k,v in arrays.items())
  put(dest.with_suffix('.json'),ledger);index.append({'file':dest.name,'export_sha256':sha(dest),'ledger_sha256':sha(dest.with_suffix('.json')),'raw_sha256':entry['sha256'],'metadata':raw['metadata'],'arrays':sorted(arrays)});count+=len(arrays)
  del raw,arrays
 put(a.out/'index.json',index)
 put(a.out/'export.json',{'schema':'T20_restricted_numeric_v1','canonical_run_sha256':CANONICAL['manifest.json'],'analysis_sha256':CANONICAL['post_write_norm_analysis.json'],'raw_inventory_sha256':CANONICAL['numeric_inventory.json'],'audit_private_commitments':{kind:spec['sha256'] for kind,spec in AUDITS.items()},'adapter_sha256':sha(__file__),'raw_shards':64,'main_records':392,'suffix_records':384,'numeric_arrays':count,'omitted':'Source text,input IDs,prediction/mismatch IDs,fullgates,QKV,contexts,hidden,fullvocabulary rows,weights,credentials,privatehostpaths,privatehistoryscan andrawfailures. Local originals remain immutable/hashbound.'})
 files={p.relative_to(a.out).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.rglob('*') if p.is_file()};put(a.out/'numeric_export_inventory.json',{'schema':'T20_restricted_numeric_inventory_v1','files':files})

def main():
 global OWNED
 OWNED=None;parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
 for action,names in (('export-metadata',('run','out')),('rebuild',('package','config','out')),('stage-replay',('package','inputs','config','out')),('export-numeric',('run','audits','out'))):
  cmd=sub.add_parser(action)
  for name in names:cmd.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args()
 try:{'export-metadata':export_metadata,'rebuild':rebuild,'stage-replay':stage_replay,'export-numeric':export_numeric}[a.action](a)
 except BaseException as error:
  if OWNED is not None and a.out.is_dir() and OWNED==(a.out.resolve(),a.out.stat().st_dev,a.out.stat().st_ino):put(a.out/'portable_failure.json',{'status':'failed','action':a.action,'error_type':type(error).__name__,'error':str(error),'model_forwards':0})
  raise
if __name__=='__main__':main()
