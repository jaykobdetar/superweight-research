"""T22 known experimental history chain and tokenizer-only new panel preparation.

No filesystem rescan: the audited T21 known745-history closure plus its32
accepted natural/BOS hashes and16used512-token spans yields777. Every inherited
exclusion/fullF buffer remains. Selection is disabled pending independent
wrapper/history review and a root attestation of no unaccounted experiments.
"""
import argparse,copy,json,os,re,resource,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
from post_write_norm_inputs import sha,read,value_sha,put,copy_exact,fresh,owned,PHYSICAL,CORPORA,PREPARATION_LIMIT,RESERVE,ADDENDUM_SHA
ROOT=Path(__file__).resolve().parent
# The inherited fresh() reserves64MiB conservatively; actual T22 output stays32MiB.
PREPARATION_LIMIT=32*1024**2
PROPOSAL_SHA='6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159'
SELECTOR_SHA='34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4'
HELPER_SHA='e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48'
VALIDATOR_SHA='39d05580f3cf6e48e0a9ee324b55f11113d0189021ec3064b11384205bb307f1'
T21_PACKAGE_ID='bc58c7b4ee0d77e4d96363ba67678d9b7f154d328006a238daa210343dff8cfd'
T21_MANIFEST_SHA='c4de3d3473eb93556bcdeb1161d1fc7d88f8c4e20c9fb38ad10b7096058299a9'
T21_DATA_MANIFEST_SHA='f0ff4671117fb5b1f540dded7329a21fc640e524d9f08648c721b37156011dd8'
T21_HISTORY_CHAIN_SHA='67106b29dc1991dc86c5f7af4e03e4accc7d09c18fcb50f1415f0f9dd71d103e'
PREPARATION_APPROVED=True
APPROVED_HISTORY_CHAIN_SHA='2ac54ea99d9e991434d364e19ceb483440e8bfeb85b7213bf19061df9a70edef'
APPROVED_HISTORY_REVIEW_SHA='453d5caee74c3730c40f8d87616322e55b8bd658800f99fa7510cd3893acfaff'

def pins():
 assert sha(ROOT/'GPT2_L11_KEY_VALUE_DONOR_PROPOSAL_v2.md')==PROPOSAL_SHA
 assert sha(ROOT/'post_write_norm_selector.py')==SELECTOR_SHA
 assert sha(ROOT/'post_write_norm_inputs.py')==HELPER_SHA
 assert sha(ROOT/'query_write_timing_portable.py')==VALIDATOR_SHA

def base_history(package,completed_run):
 pins()
 from query_write_timing_portable import validate_package
 assert sha(package/'identity.json')==T21_PACKAGE_ID
 identity=validate_package(package)
 assert sha(completed_run/'manifest.json')==T21_MANIFEST_SHA==identity['canonical_run_manifest_sha256']
 m=read(completed_run/'manifest.json')
 assert m['status']=='completed' and m['all_final_guards_passed'] is True
 assert m['forward_attempts']==m['forward_passes']==m['fully_processed_forwards']==648
 assert m['calibration_forwards']==0 and m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True
 assert m['precision_settings_before']==m['precision_settings_after']
 for name in ('data_manifest.json','exclusion_inventory.json','known_history_chain.json'):
  assert sha(completed_run/name)==m['input_dependencies'][name]==identity['canonical_private_input_sha256'][name]
 assert sha(completed_run/'data_manifest.json')==T21_DATA_MANIFEST_SHA
 assert sha(completed_run/'known_history_chain.json')==T21_HISTORY_CHAIN_SHA
 dm=read(completed_run/'data_manifest.json');inv=read(completed_run/'exclusion_inventory.json');old_chain=read(completed_run/'known_history_chain.json')
 template=read(package/'metadata/data_manifest.template.json')
 assert len(inv['history_token_hashes'])==len(set(inv['history_token_hashes']))==745
 inherited=set(inv['history_token_hashes']);assert inherited==set(old_chain['history_token_hashes'])
 assert inherited==set(read(package/'metadata/exclusion_inventory.json')['history_token_hashes'])
 attested=[];spans=[];expected={(c,w) for c in CORPORA for w in range(4)}
 assert len(dm['passages'])==len(template['passages'])==16 and {(r['corpus'],r['window']) for r in dm['passages']}==expected
 for row in dm['passages']:
  c,w=row['corpus'],row['window'];safe=next(r for r in template['passages'] if (r['corpus'],r['window'])==(c,w))
  for key in ('char_start','used_char_end','window_token_sha256','bos_window_token_sha256'):assert safe[key]==row[key]
  assert 0<=row['char_start']<row['used_char_end']<=dm['sources'][c]['characters']
  for start,key in (('natural','window_token_sha256'),('bos','bos_window_token_sha256')):
   h=row[key];assert isinstance(h,str) and re.fullmatch('[a-f0-9]{64}',h)
   attested.append({'corpus':c,'window':w,'identity':start,'token_sha256':h})
  spans.append({'corpus':c,'window':w,'char_start':row['char_start'],'char_end':row['used_char_end'],'origin':'T21_completed_accepted512_used_span','origin_manifest_sha256':T21_MANIFEST_SHA,'token_sha256':row['window_token_sha256'],'BOS_token_sha256':row['bos_window_token_sha256']})
 accepted={r['token_sha256'] for r in attested}
 assert len(accepted)==32 and not inherited&accepted and len(inherited|accepted)==777
 assert len(inv['retained_calibration_buffers'])==4
 for buffer in inv['retained_calibration_buffers']:
  assert buffer['corpus']=='wiki_test' and buffer['char_end']-buffer['char_start']==8192
  assert any(r['char_start']<=buffer['char_start'] and r['char_end']>=buffer['char_end'] for r in inv['exclusions']['wiki_test'])
 return {'identity':identity,'data_manifest':dm,'exclusion_inventory':inv,'spans':spans,'identity_attestations':attested,'inherited_hashes':sorted(inherited|accepted),
  'inherited_known_chain':old_chain,'bindings':{'T21_package_identity_sha256':T21_PACKAGE_ID,'T21_completed_manifest_sha256':T21_MANIFEST_SHA,
   'T21_data_manifest_sha256':T21_DATA_MANIFEST_SHA,'T21_exclusion_inventory_sha256':sha(completed_run/'exclusion_inventory.json'),
   'T21_known_history_chain_sha256':T21_HISTORY_CHAIN_SHA,'T21_safe_history_attestation_sha256':sha(package/'metadata/history_attestation.json')}}

def chain_value(ancestry):
 return {'schema':'T22_known_experimental_history_v1','proposal_sha256':PROPOSAL_SHA,'inherited_bindings':ancestry['bindings'],
  'inherited_T21_history_count':745,'T21_added_hash_count':32,'history_token_hashes':ancestry['inherited_hashes'],
  'inherited_known_history_chain':ancestry['inherited_known_chain'],'T21_used_spans':ancestry['spans'],'T21_identity_attestations':ancestry['identity_attestations'],
  'retained_calibration_buffers':ancestry['exclusion_inventory']['retained_calibration_buffers'],
  'history_scope':'Known recorded experimental chain, not a fresh filesystem rescan; GPT2 pretraining overlap unknown.',
  'historical_first_token_membership':'unavailable_not_used','model_forwards':0,'tokenizer_loaded':False}

def make_chain(a):
 ancestry=base_history(a.t21_package,a.t21_run)
 identity=fresh(a.out,a.t21_package,a.t21_run)
 try:put(a.out/'known_history_chain.json',chain_value(ancestry))
 except BaseException as error:
  if owned(a.out,identity):put(a.out/'chain_failure.json',{'error':repr(error),'model_forwards':0})
  raise

def reviewed_history(a,ancestry):
 assert APPROVED_HISTORY_CHAIN_SHA and APPROVED_HISTORY_REVIEW_SHA,'Reviewed known chain and no-unaccounted-experiment attestation required'
 assert sha(a.history_chain)==APPROVED_HISTORY_CHAIN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
 chain=read(a.history_chain);review=read(a.history_review)
 assert chain==chain_value(ancestry),'Known chain changed'
 assert review['status']=='passed' and review['known_history_chain_sha256']==APPROVED_HISTORY_CHAIN_SHA and review['proposal_sha256']==PROPOSAL_SHA
 assert review['no_unaccounted_model_experiments'] is True
 exclusions=copy.deepcopy(ancestry['exclusion_inventory']['exclusions'])
 for row in ancestry['spans']:exclusions[row['corpus']].append(copy.deepcopy(row))
 return chain,exclusions,set(ancestry['inherited_hashes']),[]

def finalize_preflight(out,meta):
 # Final normal cap includes the exact final JSON bytes; failed/overcap
 # metadata is still preserved and never labelled a successful preparation.
 if meta['status']=='passed':
  encoded=(json.dumps(meta,indent=2,allow_nan=False)+'\n').encode()
  used=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
  try:
   assert used+len(encoded)<=PREPARATION_LIMIT,('Complete preparation cap',used,len(encoded),PREPARATION_LIMIT)
   assert shutil.disk_usage(out).free>=RESERVE+len(encoded),'Preparation final free reserve'
  except BaseException as error:
   meta.update(status='failed',failure_phase='final_preflight_budget',error_type=type(error).__name__,error=str(error))
   put(out/'preflight.json',meta)
   raise
 put(out/'preflight.json',meta)

def prepare(a):
 assert PREPARATION_APPROVED,'Independent complete input wrapper review required before tokenizer selection'
 os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 import torch
 from datasets import Dataset
 from transformers import AutoTokenizer
 from post_write_norm_selector import select_panel,th,textsha
 ancestry=base_history(a.t21_package,a.t21_run);scan,exclusions,hashes,resolutions=reviewed_history(a,ancestry)
 identity=None;progress={};meta={'status':'running','phase':'checking_sources','model_forwards':0,'pretrained_loaded':False,'gpu_used':False,'calibration_fitted':False};started=time.time()
 try:
  identity=fresh(a.out,a.t21_package,a.t21_run,a.history_chain.parent);torch.set_num_threads(1);dm=ancestry['data_manifest'];cfg=read(a.config);spec=read(a.t21_package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2'];gc=cfg['models']['gpt2']
  assert gc['model_id']==spec['model_id'] and gc['revision']==spec['revision'];tokenizer_path=Path(gc['path'])
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True);assert tok.is_fast and type(tok.bos_token_id)is int
  literals={name:{'literal':literal,'ids':tok.encode(literal,add_special_tokens=False)} for name,literal in (('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n'))};literals['start_special']={'label':'bos','id':tok.bos_token_id}
  assert type(tok.bos_token_id)is int and value_sha(literals)==ancestry['identity']['private_literals_commitment']
  texts={}
  for corpus in PHYSICAL:
   path=Path(cfg['sources'][corpus]);source=dm['sources'][corpus];assert sha(path)==source['file_sha256']
   if corpus=='wiki_validation':value='\n\n'.join(Dataset.from_file(str(path))['text'])
   elif corpus in ('utilitarianism','ethics'):
    raw=path.read_text();lo=raw.find('*** START OF');hi=raw.rfind('*** END OF');assert lo>=0 and hi>lo;value=raw[raw.find('\n',lo)+1:hi]
   else:value=path.read_text()
   assert len(value)==source['characters'] and textsha(value)==source['processed_text_sha256'];texts[corpus]=value
  texts['optimizers_new']='\n\n'.join(texts[c] for c in ('adadelta','adamax','asgd'));assert textsha(texts['optimizers_new'])==dm['sources']['optimizers_new']['processed_text_sha256']
  components=dm['sources']['optimizers_new']['components'];cursor=0
  for c,component in zip(('adadelta','adamax','asgd'),components):assert component=={'source':c,'char_start':cursor,'char_end':cursor+len(texts[c]),'file_sha256':dm['sources'][c]['file_sha256']};cursor+=len(texts[c])+2
  inherited=sorted(hashes)
  put(a.out/'exclusion_inventory.json',{'sources':dm['sources'],'exclusions':exclusions,'retained_calibration_buffers':ancestry['exclusion_inventory']['retained_calibration_buffers'],'history_token_hashes':inherited,'inherited_bindings':ancestry['bindings'],'known_history_chain_sha256':sha(a.history_chain),'known_history_review_sha256':sha(a.history_review),'T21_used_spans':ancestry['spans'],'T21_identity_attestations':ancestry['identity_attestations'],'freshness_scope':'Recorded research exposures only; GPT2 pretraining overlap unknown.'})
  copy_exact(a.history_chain,a.out/'known_history_chain.json');copy_exact(a.history_review,a.out/'known_history_review.json')
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT-2*1024**2
  meta['phase']='selection';select_panel(CORPORA,texts,tok,exclusions,hashes,components,False,progress,bos_id=tok.bos_token_id)
  assert len(progress['items'])==len(progress['selected'])==16;torch.save(progress['items'],a.out/'tokens.pt')
  put(a.out/'data_manifest.json',{'sources':dm['sources'],'passages':progress['selected'],'rejections':progress['rejections'],'literals':literals,'tokenizer_sha256':dm['tokenizer_sha256'],'physical_length':512,'passages_per_corpus':{c:4 for c in CORPORA},'exclusion_inventory_sha256':sha(a.out/'exclusion_inventory.json'),'calibration_reference_manifest_sha256':dm['calibration_reference_manifest_sha256'],'calibration_reference_files':dm['calibration_reference_files'],'calibration_fit':'Frozen T18 F; no new calibration','selection_rule':{'frozen_base_selector_sha256':'810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a','adapted_selector_sha256':SELECTOR_SHA,'input_addendum_sha256':ADDENDUM_SHA,'historical_hashes_frozen_before_selection':True,'natural_and_BOS_checked':True,'own_pair_identical_allowed':True}})
  put(a.out/'start_identity_summary.json',{'historical_first_token_membership':'unavailable_not_used','natural_BOS_pairing_preserved':True,'start_intervention':'Replace token0 only; retain physical512 and tokens1..511','identical_start_pairs':[{'corpus':r['corpus'],'window':r['window'],'identical':r['identical_start_pair']} for r in progress['selected']]})
  pins();assert sha(a.history_chain)==APPROVED_HISTORY_CHAIN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
  for corpus in PHYSICAL:assert sha(Path(cfg['sources'][corpus]))==dm['sources'][corpus]['file_sha256']
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  reviewed_history(a,ancestry)
  assert base_history(a.t21_package,a.t21_run)==ancestry,'T21 package/run ancestry changed during selection'
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT and shutil.disk_usage(a.out).free>=RESERVE
  meta.update(status='passed',phase='completed',evaluation_passages=16,evaluation_inputs=32,history_hash_count=len(inherited),accepted_identity_hash_count=len(hashes)-len(inherited),historical_first_token_membership='unavailable_not_used',proposal_sha256=PROPOSAL_SHA,input_addendum_sha256=ADDENDUM_SHA,script_sha256=sha(__file__),selector_sha256=SELECTOR_SHA,source_package_identity_sha256=T21_PACKAGE_ID,known_history_chain_sha256=sha(a.history_chain),known_history_review_sha256=sha(a.history_review))
 except BaseException as error:
  meta.update(status='failed',failure_phase=meta['phase'],error_type=type(error).__name__,error=str(error))
  if owned(a.out,identity):
   if progress.get('items'):torch.save(progress['items'],a.out/'partial_tokens.pt')
   put(a.out/'failure_selection.json',{k:v for k,v in progress.items() if k!='items'})
  raise
 finally:
  if owned(a.out,identity):
   meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file()});finalize_preflight(a.out,meta)

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
 for action in ('build-history','prepare'):
  cmd=sub.add_parser(action)
  for name in ('t21-package','t21-run','out'):cmd.add_argument('--'+name,type=Path,required=True)
  if action=='prepare':
   for name in ('config','history-chain','history-review'):cmd.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();make_chain(a) if a.action=='build-history' else prepare(a)
if __name__=='__main__':main()
