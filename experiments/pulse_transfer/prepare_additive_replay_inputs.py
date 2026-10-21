"""Export safe T17 metadata or reconstruct its exact tokens with GPT2 only.

No model class/weights, Llama/OLMo tokenizer or previous experiment is invoked.
Canonical token/direction hashes stay strict. Sanitized metadata has new hashes:
original absolute paths and historical starting-token IDs are never exported.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,copy,hashlib,json,resource,shutil,time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer
import gate_additive_inputs as selection
import gate_localization_inputs as localization

ROOT=Path(__file__).resolve().parent
MODEL_ID='openai-community/gpt2'
REVISION='75e09b43581151bd1d9ef6700faa605df408979f'
EXPECTED={
 'preflight.json':'d0fa737595f63ff7c48507925a4ddb500013f5301a245878f85ee72beabee66b',
 'tokens.pt':'e009ddfbcee00ec08e013ef2cddc9c745cc2f8f90ca042a2e83d5e2c89120510',
 'data_manifest.json':'931f8c698aefa46f2b595121ef9de27764b8e9afdf34a81ff55043279ee16ebf',
 'directions.pt':'4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3',
 'directions.json':'0a27dcc857b221a9fdbd1265be678aa341d59f914eb3030cff5eb44690ff852e',
 'exclusion_inventory.json':'5722677bc774efda1b0b3e7bc8a8d943d3c34a032d33a991854e33bfc3432686',
 'historical_manifest_scan.json':'41d8a119d94e875113bc5280fa20703f028488b34799ed6519c2c31cd10d616c',
 'start_identity_summary.json':'15c7b0994209c68189c63184fb1049d0990d64c720dcf5662d02cfcfb28c49b4'}
MAP_SHA='ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098'
CAL_TOKENS_SHA='a99fa8403d600404b4725bdc8ab8e051e78e70a864a3483b614879d68ccc668c'
DEPENDENCIES={'gate_additive_inputs.py':'436235e4daa0e8494d804814c3aaf1ea4cb5128589f7448143af3b45331fa90e',
 'gate_localization_inputs.py':'7bf15b79d0e9c2ba57736667adb11511996943f58f69f3763e7ee4fa20338a06'}
METADATA_IDENTITY_SHA256='13cca2b5ec8ef9993e01df36f0d86aecfc0bab96c4b598fd94a887fec0605a98'
FILES=('data_manifest.template.json','exclusion_inventory.json','historical_manifest_scan.json','historical_preflight.json','sanitization.json')
sha=selection.sha;read=selection.read;dump=selection.dump;th=selection.th;textsha=selection.textsha

def safe(value):
 """Reject private paths and explicit token payload keys in shareable metadata."""
 forbidden={'input_ids','tokens','first_token_id','ids','id','calibration_first_token_ids','evaluation_natural_first_token_ids','bos_id'}
 if isinstance(value,dict):
  assert not(forbidden&set(value)),('private field',forbidden&set(value))
  for k,v in value.items():
   safe(k)
   if k=='pointer':
    assert isinstance(v,str)and v.startswith(('/data_manifest/','/exclusion_inventory/'))and'..'not in v.split('/')
   else:safe(v)
 elif isinstance(value,list):
  for v in value:safe(v)
 elif isinstance(value,str):
  assert not value.startswith(('/','file://','~','\\')),('absolute path',value)
  assert not(len(value)>2 and value[1]==':'and value[2]in'/\\'),('absolute Windows path',value)

def clean(value,pointer='',removed=None):
 if isinstance(value,dict):
  out={}
  for k,v in value.items():
   p=pointer+'/'+k
   if k in('path','first_token_id','ids','id'):
    if removed is not None:removed.append({'pointer':p,'omitted_kind':'source_locator'if k=='path'else'token_identity'})
   else:out[k]=clean(v,p,removed)
  return out
 if isinstance(value,list):return [clean(v,pointer+'/'+str(i),removed)for i,v in enumerate(value)]
 return value

def empty_out(out):
 assert not out.exists()or not any(out.iterdir()),'Use a new or empty output; preserve previous attempts'
 out.mkdir(parents=True,exist_ok=True)

def export_metadata(inputs,out):
 empty_out(out)
 assert all(sha(inputs/n)==h for n,h in EXPECTED.items()),'Canonical input identity changed'
 assert sha(ROOT/'gate_additive_inputs.py')==DEPENDENCIES['gate_additive_inputs.py']
 assert sha(ROOT/'gate_localization_inputs.py')==DEPENDENCIES['gate_localization_inputs.py']
 dm=read(inputs/'data_manifest.json');inv=read(inputs/'exclusion_inventory.json');removed=[]
 template=clean(dm,'/data_manifest',removed);inventory=clean(inv,'/exclusion_inventory',removed)
 hist=read(inputs/'historical_manifest_scan.json');pf=read(inputs/'preflight.json')
 for d in(template,inventory,hist,pf):safe(d)
 assert hist['new_source_matches']==[] and len(template['passages'])==16
 dump(out/FILES[0],template);dump(out/FILES[1],inventory)
 shutil.copyfile(inputs/'historical_manifest_scan.json',out/FILES[2]);shutil.copyfile(inputs/'preflight.json',out/FILES[3])
 dump(out/FILES[4],{'removed_fields':removed,'rule':'Remove absolute source path fields and token ID fields; keep source/token hashes, exact exclusion intervals, original provenance hashes and historical selection decisions.',
  'not_byte_identical':['data_manifest.json','exclusion_inventory.json','preflight.json'],
  'history_scope':'Inherited recorded history; no claim that original history files were re-audited by this export.'})
 identity={'kind':'t17_safe_input_metadata_v1','model_id':MODEL_ID,'revision':REVISION,
  'canonical_input_sha256':EXPECTED,'canonical_calibration_maps_sha256':MAP_SHA,'canonical_calibration_tokens_sha256':CAL_TOKENS_SHA,
  'metadata_sha256':{n:sha(out/n)for n in FILES},'corpora':list(selection.CORPORA),'passages_per_corpus':{c:4 for c in selection.CORPORA},
  'code_sha256':{n:sha(ROOT/n)for n in DEPENDENCIES},
  'exact_rebuild_files':['tokens.pt','directions.pt','directions.json','start_identity_summary.json','historical_manifest_scan.json'],
  'sanitized_metadata_files':['data_manifest.json','exclusion_inventory.json','preflight.json'],
  'scope':'No source text, token arrays, first-token IDs, original absolute paths or model weights included. History is frozen attested metadata; exact new inputs/directions are rebuilt from user-supplied lawful local files.'}
 safe(identity);dump(out/'identity.json',identity)
 assert sha(out/'identity.json')==METADATA_IDENTITY_SHA256
 return identity

def validate_metadata(metadata):
 assert sha(metadata/'identity.json')==METADATA_IDENTITY_SHA256,'Safe metadata identity changed'
 identity=read(metadata/'identity.json');safe(identity)
 assert identity['kind']=='t17_safe_input_metadata_v1' and identity['canonical_input_sha256']==EXPECTED
 assert identity['model_id']==MODEL_ID and identity['revision']==REVISION
 assert identity['canonical_calibration_maps_sha256']==MAP_SHA and identity['canonical_calibration_tokens_sha256']==CAL_TOKENS_SHA
 assert set(identity['metadata_sha256'])==set(FILES)
 for n,h in identity['metadata_sha256'].items():assert sha(metadata/n)==h,('metadata hash',n);safe(read(metadata/n))
 assert identity['code_sha256']['gate_additive_inputs.py']==DEPENDENCIES['gate_additive_inputs.py']
 for n,h in identity['code_sha256'].items():assert n in DEPENDENCIES and sha(ROOT/n)==h,('helper dependency',n)
 assert set(identity['code_sha256'])==set(DEPENDENCIES)
 assert sha(metadata/'historical_preflight.json')==EXPECTED['preflight.json']
 assert sha(metadata/'historical_manifest_scan.json')==EXPECTED['historical_manifest_scan.json']
 return identity

def reconstruct(config,metadata,calibration,out,tokenizer_factory=AutoTokenizer.from_pretrained):
 """Private workspace output. The injectable tokenizer factory is for tiny CPU tests."""
 started=time.time();empty_out(out);torch.set_num_threads(1)
 m={'status':'checking','model_forwards':0,'model_loaded':False,'gpu_used':False,'source_sha256':sha(__file__)}
 dump(out/'reconstruction.json',m)
 try:
  identity=validate_metadata(metadata);cfg=read(config);g=cfg['models']['gpt2']
  assert g['model_id']==MODEL_ID and g['revision']==REVISION
  template=read(metadata/'data_manifest.template.json');inv=read(metadata/'exclusion_inventory.json');history=read(metadata/'historical_manifest_scan.json')
  assert history['new_source_matches']==[] and set(inv['exclusions'])==set(selection.CORPORA)
  assert sha(calibration/'calibration_maps.pt')==MAP_SHA and sha(calibration/'calibration_tokens.pt')==CAL_TOKENS_SHA
  tokpath=Path(g['path']);tokenfiles={}
  for n,h in template['tokenizer_sha256'].items():
   assert Path(n).name==n and sha(tokpath/n)==h,('tokenizer identity',n);tokenfiles[n]=h
  tok=tokenizer_factory(tokpath,local_files_only=True);assert tok.is_fast
  texts={};sources=copy.deepcopy(template['sources'])
  for c in('wiki_validation','mill','wollstonecraft','adagrad','rmsprop','lbfgs'):
   p=Path(cfg['sources'][c]);s=sources[c]
   assert p.name==s['filename'] and sha(p)==s['file_sha256'],('source bytes',c)
   text='\n\n'.join(Dataset.from_file(str(p))['text'])if c=='wiki_validation'else p.read_text()
   text=selection.strip_book(text)if c in('mill','wollstonecraft')else text
   assert textsha(text)==s['processed_text_sha256']and len(text)==s['characters'];texts[c]=text
  texts['optimizers_new']='\n\n'.join(texts[c]for c in('adagrad','rmsprop','lbfgs'))
  assert textsha(texts['optimizers_new'])==sources['optimizers_new']['processed_text_sha256']
  assert len(texts['optimizers_new'])==sources['optimizers_new']['characters']
  assert inv['sources']==sources and inv['code_components']==sources['optimizers_new']['components']
  assert inv['historical_scan_sha256']==EXPECTED['historical_manifest_scan.json']
  for c,ex in inv['exclusions'].items():
   for r in ex:assert 0<=r['char_start']<r['char_end']<=len(texts[c])
  # Original scan file precedes the explicit localization-hash augmentation.
  # Retain that augmentation exactly without requiring old token files.
  historical=set(history['historical_window_token_hashes'])|{r['window_token_sha256']for r in inv['all15_localization_passages']}
  items,records,rejections=selection.select_windows({c:texts[c]for c in selection.CORPORA},tok,inv['exclusions'],historical,inv['code_components'])
  assert [clean(r)for r in records]==template['passages'],'Frozen selected identities/offsets changed'
  assert rejections==template['rejections'],'Frozen selection/rejection path changed'
  assert len(items)==16 and len({(i['corpus'],i['window'])for i in items})==16
  torch.save(items,out/'tokens.pt');assert sha(out/'tokens.pt')==EXPECTED['tokens.pt']
  dm=copy.deepcopy(template);dm['passages']=records
  for name,literal in dm['literals'].items():
   if 'literal'in literal:literal['ids']=tok(literal['literal'],add_special_tokens=False)['input_ids']
   else:assert name=='start_special'and literal['label']=='bos';literal['id']=tok.bos_token_id
  # Canonical source/offset/history content is unchanged, only private fields omitted.
  assert clean(dm)==template
  shutil.copyfile(metadata/'exclusion_inventory.json',out/'exclusion_inventory.json')
  dm['exclusion_inventory_sha256']=sha(out/'exclusion_inventory.json');dump(out/'data_manifest.json',dm)
  shutil.copyfile(metadata/'historical_manifest_scan.json',out/'historical_manifest_scan.json')
  maps=torch.load(calibration/'calibration_maps.pt',weights_only=True,map_location='cpu');directions=localization.directions(maps)
  torch.save(directions,out/'directions.pt');dump(out/'directions.json',{l:{k:v.tolist()if isinstance(v,torch.Tensor)else v for k,v in d.items()}for l,d in directions.items()})
  cal=torch.load(calibration/'calibration_tokens.pt',weights_only=True,map_location='cpu');assert len(cal)==8
  starts={int(x['tokens'][0])for x in cal};bos=dm['literals']['start_special']['id']
  dump(out/'start_identity_summary.json',{'calibration_first_token_ids':sorted(starts),'evaluation_natural_first_token_ids':[int(x['tokens'][0])for x in items],
    'natural_seen_in_calibration':[int(x['tokens'][0])in starts for x in items],'bos_id':bos,'bos_seen_in_calibration':bos in starts})
  for n in identity['exact_rebuild_files']:assert sha(out/n)==EXPECTED[n],('canonical exact rebuild',n)
  pf={'status':'passed','kind':'metadata_only_exact_token_reconstruction','model_forwards':0,'model_loaded':False,'gpu_used':False,
      'passages':16,'model_inputs':32,'passages_per_corpus':{c:4 for c in selection.CORPORA},
      'canonical_preflight_sha256':EXPECTED['preflight.json'],'canonical_input_sha256':EXPECTED,
      'safe_metadata_identity_sha256':sha(metadata/'identity.json'),'inherited_history_only':True,
      'output_sha256':{n:sha(out/n)for n in EXPECTED if n!='preflight.json'}}
  dump(out/'preflight.json',pf)
  actual={n:sha(out/n)for n in EXPECTED}
  dump(out/'input_bindings.json',{'kind':'explicit_metadata_binding_only','original':EXPECTED,'regenerated':actual,
      'exact_canonical_files':identity['exact_rebuild_files'],'changed_metadata_files':identity['sanitized_metadata_files'],
      'reason':'Original absolute paths and prior first-token IDs are not redistributed; history identity is retained as hashes and sanitized fields. Interventions/input tensor bytes unchanged.'})
  m.update(status='passed',passages=16,selector_rejections=sum(len(v)for v in rejections.values()),source_files=6,
       tokenizer_sha256=tokenfiles,metadata_identity_sha256=sha(metadata/'identity.json'),
       inherited_history_manifests=len(history['files']),exclusion_intervals={c:len(v)for c,v in inv['exclusions'].items()},
       input_bindings_sha256=sha(out/'input_bindings.json'),output_sha256=actual,
       canonical_exact={n:actual[n]==EXPECTED[n]for n in identity['exact_rebuild_files']},
       scope='Original historical records are inherited attestations; original OLMo/Llama model outputs/tokenizers and old research directories are not read. Fresh GPT2-only reconstruction checks source bytes, all candidate selection decisions, offsets, exclusions, exact input tensors and directions. Metadata hashes intentionally differ where private fields were removed.')
 except BaseException as error:
  m.update(status='failed',error_type=type(error).__name__,error=str(error));raise
 finally:
  m.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  dump(out/'reconstruction.json',m)
 return m

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='mode',required=True)
 e=sub.add_parser('export-metadata');e.add_argument('--inputs',type=Path,required=True);e.add_argument('--out',type=Path,required=True)
 r=sub.add_parser('rebuild')
 for n in('config','metadata','calibration','out'):r.add_argument('--'+n,type=Path,required=True)
 a=ap.parse_args()
 if a.mode=='export-metadata':export_metadata(a.inputs,a.out);print('EXPORTED_SAFE_ADDITIVE_METADATA')
 else:result=reconstruct(a.config,a.metadata,a.calibration,a.out);print('REBUILT_ADDITIVE_INPUTS',result['status'],'zero model forwards')
if __name__=='__main__':main()
