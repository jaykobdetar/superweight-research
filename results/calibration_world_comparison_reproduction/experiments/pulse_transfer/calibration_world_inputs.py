"""T23 fixed-design evaluation history and tokenizer-only input SOURCE HOLD.

No historical rescan. Calibration16 adds observations of four already excluded
contexts, not sixteen hashes. Neither command authorizes model execution.
"""
import argparse,copy,hashlib,json,os,re,resource,shutil,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
from post_write_norm_inputs import sha,read,value_sha,put,copy_exact,fresh,owned,PHYSICAL,CORPORA,PREPARATION_LIMIT,RESERVE,ADDENDUM_SHA

ROOT=Path(__file__).resolve().parent
PREPARATION_LIMIT=32*1024**2
PREPARATION_APPROVED=True
APPROVED_HISTORY_CHAIN_SHA='55fd60df741c3eadc81487cf77b4fe77e68a4ac59968433af8a577b451be3258'
APPROVED_HISTORY_REVIEW_SHA='ef971da48a0f3c8e344b27e668146248484d25140c002adb42bb7c3fb1aaf3a0'
PROPOSAL_SHA='a78e9678ffceefc85abcccc2a11cd750754efd1adb7738caa03d508f957c2e7d'
SELECTOR_SHA='34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4'
HELPER_SHA='e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48'
INHERITED_INPUT_SOURCE_SHA='26dad5cd0fde2c6ee4c2bfa5c984e825fb50e30a197da4ceb7098539954914f4'
EXPOSURE_CHAIN_SHA='649465b0a8adfaa4889348ae4192bb915e2af427bea11e0ccbdaecedfd92cfbb'
EXPOSURE_REVIEW_SHA='568474587876e885eb72370b9523534f0a5487a50c5f125df5e4ad982d685623'
T22_MANIFEST_SHA='602d85764807d4678e6b61e5aa2e42244b29680a5b457c6e2bf230f7adf1d288'
T22_DATA_SHA='eb172817f41d0c47d2b746c7d074b5331bcce96d49dd7aee2219cc3ac9222227'
T22_EXCLUSIONS_SHA='5fcb3ee8386fb2ea34141cbf8a9904375a00db0c39b6e4736ea1d0ee8cec9941'
CAL_MANIFEST_SHA='f71a3c3b8af737f240b9d559b0786c90501f89dae7ef3a2fba2efacc004f6b2a'
CAL_AUDIT_SHA='50f41e1e4f31c2ecc123ef9c8f71b119a3512fd1795d1deca54f03cb073eba47'
CAL_ACCEPTANCE_SHA='87fe3cf002327fd7cde8bbe30bd8ea32ce9316d815a66d8ff59e2d61b74f5300'
CAL_INPUT_METADATA_SHA='b58e0dfc03cc6125cf618c9d66bd38b375db102e418971bf0a48b4fe2a58df04'


def pins():
 assert sha(ROOT/'GPT2_CALIBRATION_WORLD_COMPARISON_PROPOSAL_draft_v2.md')==PROPOSAL_SHA
 assert sha(ROOT/'post_write_norm_selector.py')==SELECTOR_SHA
 assert sha(ROOT/'post_write_norm_inputs.py')==HELPER_SHA
 assert sha(ROOT/'l11_key_value_donor_inputs.py')==INHERITED_INPUT_SOURCE_SHA


def base_history(t22_run,calibration_run,exposure_chain,exposure_review,calibration_acceptance,calibration_audit):
 pins()
 for path,h in ((exposure_chain,EXPOSURE_CHAIN_SHA),(exposure_review,EXPOSURE_REVIEW_SHA),
                (t22_run/'manifest.json',T22_MANIFEST_SHA),(t22_run/'data_manifest.json',T22_DATA_SHA),
                (t22_run/'exclusion_inventory.json',T22_EXCLUSIONS_SHA),
                (calibration_run/'manifest.json',CAL_MANIFEST_SHA),
                (calibration_acceptance,CAL_ACCEPTANCE_SHA),(calibration_audit,CAL_AUDIT_SHA)):
  assert path.is_file() and not path.is_symlink() and sha(path)==h,('Pinned metadata changed',path.name)
 exposure=read(exposure_chain);exposure_gate=read(exposure_review)
 assert exposure_gate['status']=='passed_recorded_exposure_metadata_not_selection_GO'
 assert exposure_gate['candidate_sha256']==EXPOSURE_CHAIN_SHA
 assert exposure_gate['all_inherited_exclusions_and_F_buffers_exact'] is True
 canonical=exposure['canonical_history_token_hashes'];suffixes=exposure['T22_auxiliary_suffix_token_hashes']
 assert len(canonical)==len(set(canonical))==809 and len(suffixes)==len(set(suffixes))==32
 assert not set(canonical)&set(suffixes) and len(set(canonical)|set(suffixes))==841
 assert all(isinstance(h,str) and re.fullmatch('[a-f0-9]{64}',h) for h in canonical+suffixes)
 assert len(exposure['T22_used_spans'])==16
 m=read(t22_run/'manifest.json');dm=read(t22_run/'data_manifest.json');inv=read(t22_run/'exclusion_inventory.json')
 assert m['status']=='completed' and m['all_final_guards_passed'] is True
 assert m['forward_attempts']==m['forward_passes']==m['fully_processed_forwards']==776 and m['calibration_forwards']==0
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True
 assert m['precision_settings_before']==m['precision_settings_after']
 assert m['input_dependencies']['data_manifest.json']==T22_DATA_SHA and m['input_dependencies']['exclusion_inventory.json']==T22_EXCLUSIONS_SHA
 assert set(inv['history_token_hashes'])|{r[k] for r in dm['passages'] for k in ('window_token_sha256','bos_window_token_sha256')}==set(canonical)
 assert len(dm['passages'])==16 and {(r['corpus'],r['window']) for r in dm['passages']}=={(c,w) for c in CORPORA for w in range(4)}
 exclusions=copy.deepcopy(inv['exclusions'])
 for row in exposure['T22_used_spans']:
  original=next(r for r in dm['passages'] if (r['corpus'],r['window'])==(row['corpus'],row['window']))
  assert (row['char_start'],row['char_end'],row['token_sha256'],row['BOS_token_sha256'])==(original['char_start'],original['used_char_end'],original['window_token_sha256'],original['bos_window_token_sha256'])
  exclusions[row['corpus']].append(copy.deepcopy(row))
 assert exclusions==exposure['exclusions']
 buffers=exposure['retained_calibration_buffers']
 assert buffers==inv['retained_calibration_buffers'] and len(buffers)==4
 for b in buffers:
  assert b['corpus']=='wiki_test' and b['char_end']-b['char_start']==8192
  assert any(v['char_start']<=b['char_start']<b['char_end']<=v['char_end'] for v in exclusions['wiki_test'])
 cal=read(calibration_run/'manifest.json');audit=read(calibration_audit);acceptance=read(calibration_acceptance)
 assert cal['status']=='completed' and cal['all_final_guards_passed'] is True and cal['future_evaluation_authorized'] is False
 assert all(cal[k]==16 for k in ('forward_attempts','model_outputs_observed','forward_passes','fully_processed_forwards','calibration_forwards')) and cal['evaluation_forwards']==0
 assert cal['model_hash_before']==cal['model_hash_after']==m['model_hash_before']
 assert (cal['model_id'],cal['revision'])==(m['model_id'],m['revision'])
 assert cal['precision_settings_before']==cal['precision_settings_after']==m['precision_settings_before'] and cal['runtime_versions']==m['runtime_versions']
 assert audit['status']=='passed' and audit['audited_manifest_sha256']==CAL_MANIFEST_SHA and audit['new_calibration_calls']==16 and audit['PE_arrays_byte_exact']==64
 assert audit['source_hashes']==cal['source_hashes']
 assert acceptance['status']=='passed' and acceptance['result_sha256']==CAL_AUDIT_SHA and acceptance['run_manifest_sha256']==CAL_MANIFEST_SHA
 assert acceptance['actual_calibration_forwards']==16 and acceptance['heldout_evaluation_forwards']==0
 cal_inputs=calibration_run/'assets/calibration_wiki_inputs/calibration_inputs.json'
 assert sha(cal_inputs)==CAL_INPUT_METADATA_SHA
 cal_input_metadata=read(cal_inputs);passages=cal_input_metadata['passages'];assert len(passages)==4
 for key in ('file_sha256','processed_text_sha256','characters'):
  assert cal_input_metadata['sources']['wiki_test'][key]==dm['sources']['wiki_test'][key],('Same calibration-source coordinates',key)
 expected={}
 for w,row in enumerate(passages):
  assert row['corpus']=='wiki_test' and row['window']==w
  h=row['window_token_sha256'];assert h in set(canonical)
  assert any(b['char_start']==row['char_start'] and b['char_end']==row['extraction_char_end'] for b in buffers)
  expected[w]=h
 assert len(set(expected.values()))==4
 journal=calibration_run/'journal.jsonl';previous='0'*64;identities=[]
 for line in journal.read_bytes().splitlines():
  event=json.loads(line);assert event['previous_sha256']==previous
  assert json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode()==line
  previous=hashlib.sha256(line).hexdigest()
  if event['event']=='forward_intent':
   i=len(identities);world=('PE','CE','PN','CN')[i//4];w=i%4
   identity={'world':world,'window':w,'corpus':'wiki_test','length':512,'token_sha256':expected[w]}
   assert event['next_attempt']==i+1 and event['identity']==identity
   identities.append(identity)
 assert len(identities)==16 and previous==audit['journal']['chain_end_sha256']
 bindings={'T22_completed_manifest_sha256':T22_MANIFEST_SHA,'T22_data_manifest_sha256':T22_DATA_SHA,
  'T22_exclusion_inventory_sha256':T22_EXCLUSIONS_SHA,'accepted_exposure_candidate_sha256':EXPOSURE_CHAIN_SHA,
  'accepted_exposure_review_sha256':EXPOSURE_REVIEW_SHA,'T23_calibration_manifest_sha256':CAL_MANIFEST_SHA,
  'T23_calibration_audit_sha256':CAL_AUDIT_SHA,'T23_calibration_acceptance_sha256':CAL_ACCEPTANCE_SHA,
  'T23_calibration_journal_sha256':sha(journal),'T23_calibration_journal_chain_sha256':previous,
  'T23_calibration_inputs_metadata_sha256':CAL_INPUT_METADATA_SHA}
 return {'bindings':bindings,'data_manifest':dm,'exclusion_inventory':inv,'exclusions':exclusions,
  'inherited_known_chain':exposure,'canonical_hashes':list(canonical),'auxiliary_suffix_hashes':list(suffixes),
  'inherited_hashes':sorted(set(canonical)|set(suffixes)),
  'spans':copy.deepcopy(exposure['T22_used_spans']),'identity_attestations':copy.deepcopy(exposure['T22_identity_attestations']),
  'private_literals_commitment':value_sha(dm['literals']),
  'model_spec':{'model_id':m['model_id'],'revision':m['revision']},
  'calibration_exposure':{'manifest_sha256':CAL_MANIFEST_SHA,'observations':identities,
   'observed_calls':16,'distinct_reused_contexts':4,'new_canonical_hashes':[],
   'new_source_spans':[],'scope':'Four already excluded F contexts reused in all four worlds; no added independent input identities.'}}


def ancestry_args(a):
 return (a.t22_run,a.calibration_run,a.exposure_chain,a.exposure_review,a.calibration_acceptance,a.calibration_audit)


def chain_value(ancestry):
 return {'schema':'T23_evaluation_known_experimental_history_v1','proposal_sha256':PROPOSAL_SHA,
  'inherited_bindings':ancestry['bindings'],'canonical_history_count':809,'auxiliary_suffix_history_count':32,
  'combined_hash_guard_count':841,'history_token_hashes':ancestry['inherited_hashes'],
  'canonical_history_token_hashes':ancestry['canonical_hashes'],'auxiliary_suffix_token_hashes':ancestry['auxiliary_suffix_hashes'],
  'inherited_known_history_chain':ancestry['inherited_known_chain'],'T22_used_spans':ancestry['spans'],
  'T22_identity_attestations':ancestry['identity_attestations'],'calibration_exposure':ancestry['calibration_exposure'],
  'retained_calibration_buffers':ancestry['exclusion_inventory']['retained_calibration_buffers'],
  'history_scope':'Recorded experimental chain only; no new historical scan or GPT2 pretraining-disjointness claim.',
  'historical_first_token_membership':'unavailable_not_used','model_forwards':0,'tokenizer_loaded':False}


def make_chain(a):
 ancestry=base_history(*ancestry_args(a));identity=fresh(a.out,*ancestry_args(a))
 try:put(a.out/'known_history_chain.json',chain_value(ancestry))
 except BaseException as error:
  if owned(a.out,identity):put(a.out/'chain_failure.json',{'error':repr(error),'model_forwards':0})
  raise


def reviewed_history(a,ancestry):
 assert APPROVED_HISTORY_CHAIN_SHA and APPROVED_HISTORY_REVIEW_SHA,'Reviewed chain and scoped no-unaccounted-experiment attestation required'
 assert sha(a.history_chain)==APPROVED_HISTORY_CHAIN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
 chain=read(a.history_chain);review=read(a.history_review)
 assert chain==chain_value(ancestry),'Known chain changed'
 assert review['status']=='passed' and review['known_history_chain_sha256']==APPROVED_HISTORY_CHAIN_SHA and review['proposal_sha256']==PROPOSAL_SHA
 assert review['no_unaccounted_model_experiments'] is True
 return chain,copy.deepcopy(ancestry['exclusions']),set(ancestry['inherited_hashes']),[]


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
 ancestry=base_history(*ancestry_args(a));scan,exclusions,hashes,resolutions=reviewed_history(a,ancestry)
 identity=None;progress={};meta={'status':'running','phase':'checking_sources','model_forwards':0,'pretrained_loaded':False,'gpu_used':False,'calibration_fitted':False};started=time.time()
 try:
  identity=fresh(a.out,*ancestry_args(a),a.history_chain.parent);torch.set_num_threads(1);dm=ancestry['data_manifest'];cfg=read(a.config);spec=ancestry['model_spec'];gc=cfg['models']['gpt2']
  assert gc['model_id']==spec['model_id'] and gc['revision']==spec['revision'];tokenizer_path=Path(gc['path'])
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True);assert tok.is_fast and type(tok.bos_token_id)is int
  literals={name:{'literal':literal,'ids':tok.encode(literal,add_special_tokens=False)} for name,literal in (('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n'))};literals['start_special']={'label':'bos','id':tok.bos_token_id}
  assert type(tok.bos_token_id)is int and value_sha(literals)==ancestry['private_literals_commitment']
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
  put(a.out/'exclusion_inventory.json',{'sources':dm['sources'],'exclusions':exclusions,'retained_calibration_buffers':ancestry['exclusion_inventory']['retained_calibration_buffers'],'history_token_hashes':inherited,'inherited_bindings':ancestry['bindings'],'known_history_chain_sha256':sha(a.history_chain),'known_history_review_sha256':sha(a.history_review),'T22_used_spans':ancestry['spans'],'T22_identity_attestations':ancestry['identity_attestations'],'T23_calibration_exposure':ancestry['calibration_exposure'],'canonical_history_hash_count':809,'auxiliary_suffix_hash_count':32,'freshness_scope':'Recorded research exposures only; GPT2 pretraining overlap unknown.'})
  copy_exact(a.history_chain,a.out/'known_history_chain.json');copy_exact(a.history_review,a.out/'known_history_review.json')
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT-2*1024**2
  meta['phase']='selection';select_panel(CORPORA,texts,tok,exclusions,hashes,components,False,progress,bos_id=tok.bos_token_id)
  assert len(progress['items'])==len(progress['selected'])==16;torch.save(progress['items'],a.out/'tokens.pt')
  put(a.out/'data_manifest.json',{'sources':dm['sources'],'passages':progress['selected'],'rejections':progress['rejections'],'literals':literals,'tokenizer_sha256':dm['tokenizer_sha256'],'physical_length':512,'passages_per_corpus':{c:4 for c in CORPORA},'exclusion_inventory_sha256':sha(a.out/'exclusion_inventory.json'),'calibration_reference_manifest_sha256':dm['calibration_reference_manifest_sha256'],'calibration_reference_files':dm['calibration_reference_files'],'calibration_fit':'Accepted T23 PE/CE fits; fixed proposal v2; PN/CN diagnostic only','T23_calibration_manifest_sha256':CAL_MANIFEST_SHA,'T23_calibration_acceptance_sha256':CAL_ACCEPTANCE_SHA,'selection_rule':{'frozen_base_selector_sha256':'810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a','adapted_selector_sha256':SELECTOR_SHA,'input_addendum_sha256':ADDENDUM_SHA,'historical_hashes_frozen_before_selection':True,'natural_and_BOS_checked':True,'own_pair_identical_allowed':True}})
  put(a.out/'start_identity_summary.json',{'historical_first_token_membership':'unavailable_not_used','natural_BOS_pairing_preserved':True,'start_intervention':'Replace token0 only; retain physical512 and tokens1..511','identical_start_pairs':[{'corpus':r['corpus'],'window':r['window'],'identical':r['identical_start_pair']} for r in progress['selected']]})
  pins();assert sha(a.history_chain)==APPROVED_HISTORY_CHAIN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
  for corpus in PHYSICAL:assert sha(Path(cfg['sources'][corpus]))==dm['sources'][corpus]['file_sha256']
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  reviewed_history(a,ancestry)
  assert base_history(*ancestry_args(a))==ancestry,'Recorded T22/calibration ancestry changed during selection'
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT and shutil.disk_usage(a.out).free>=RESERVE
  meta.update(status='passed',phase='completed',evaluation_passages=16,evaluation_inputs=32,history_hash_count=len(inherited),accepted_identity_hash_count=len(hashes)-len(inherited),historical_first_token_membership='unavailable_not_used',proposal_sha256=PROPOSAL_SHA,input_addendum_sha256=ADDENDUM_SHA,script_sha256=sha(__file__),selector_sha256=SELECTOR_SHA,canonical_history_hash_count=809,auxiliary_suffix_hash_count=32,T22_completed_manifest_sha256=T22_MANIFEST_SHA,T23_calibration_manifest_sha256=CAL_MANIFEST_SHA,known_history_chain_sha256=sha(a.history_chain),known_history_review_sha256=sha(a.history_review))
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
  for name in ('t22-run','calibration-run','exposure-chain','exposure-review','calibration-acceptance','calibration-audit','out'):
   cmd.add_argument('--'+name,type=Path,required=True)
  if action=='prepare':
   for name in ('config','history-chain','history-review'):cmd.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();make_chain(a) if a.action=='build-history' else prepare(a)
if __name__=='__main__':main()
