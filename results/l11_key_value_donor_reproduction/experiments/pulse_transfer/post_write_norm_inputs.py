"""T20 current-history discovery and tokenizer-only fresh panel preparation.

HOLD: independent wrapper/history review before discovery/real selection.
Discovery and selection are separate immutable artifacts. Neither loads weights.
Unknown exposure evidence needs explicit reviewed resolution, never a folder-name
allowlist. Historical first-token membership is unavailable/not used.
"""
import argparse,copy,hashlib,json,os,re,resource,shutil,stat,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
PROPOSAL_SHA='e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff'
ADDENDUM_SHA='2f458ad30f0d92bfd1a1b031e0c0f8e5bcd775a90208d1333ebad2b81216ce1c'
SELECTOR_SHA='34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4'
VALIDATOR_SHA='46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4'
T19_PACKAGE_ID='7613df0d8796d758e4c2f8b868d7062408a6d5d27b3445935771112fbca360bc'
T19_EXPORT_SHA='13b21a7c807447bd6b6a831dd75a8847f75ddef14ea35866d6fc2d0bb7b41e13'
T19_EXPORT_AUDIT_SHA='11925eb7912ad793a6c7b65bcc9a0a091a1979818a41a66281cc36014c870140'
T19_EXPORT_INVENTORY_SHA='fc2efc8abc3ab4016021f35ff36f6f6fd428b9c217fadab012ae1c000fcc0934'
PHYSICAL=('wiki_validation','utilitarianism','ethics','adadelta','adamax','asgd')
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
DISCOVERY_ROOTS=('pulse_transfer','pulse_continuation','pulse_causal_tranche','superweight-causal-circuits')
PREPARATION_APPROVED=True
APPROVED_HISTORY_SCAN_SHA='db372ba1f73902c496a5f49bc5360aef797f1e6ce58216e92105efb2a5968230'
APPROVED_HISTORY_REVIEW_SHA='db70e141b3ea6c663ae204e5ba433aec8b9c5d39c58949a38a24638bf55900cb'
PREPARATION_LIMIT=64*1024**2
RESERVE=2*1024**3

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def value_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def copy_exact(source,dest):
 source,dest=Path(source),Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as a,dest.open('xb') as b:shutil.copyfileobj(a,b)
 assert sha(source)==sha(dest)
def fresh(out,*sources):
 assert all(not out.resolve().is_relative_to(Path(s).resolve()) for s in sources),'Output inside immutable source artifact'
 assert not out.exists() and not out.is_symlink() and all(not p.is_symlink() for p in out.absolute().parents),'Fresh nonsymlink output required'
 assert shutil.disk_usage(out.parent).free>=RESERVE+PREPARATION_LIMIT
 out.mkdir(parents=True);st=out.stat();return (st.st_dev,st.st_ino)
def owned(out,identity):return identity is not None and out.is_dir() and not out.is_symlink() and (out.stat().st_dev,out.stat().st_ino)==identity
def pins():
 assert sha(ROOT/'GPT2_POST_WRITE_NORM_MATCHING_PROPOSAL.md')==PROPOSAL_SHA and sha(ROOT/'GPT2_POST_WRITE_NORM_MATCHING_INPUT_ADDENDUM.md')==ADDENDUM_SHA
 assert sha(ROOT/'post_write_norm_selector.py')==SELECTOR_SHA

def base_history(package,t19_export,t19_audit,t19_inventory):
 assert sha(ROOT/'native_value_portable_draft.py')==VALIDATOR_SHA,'Imported portable validator source changed'
 from native_value_portable_draft import validate_package,scan_package,expected_dirs,relative_member
 pins();assert sha(package/'identity.json')==T19_PACKAGE_ID;identity=validate_package(package)
 assert sha(t19_export/'export.json')==T19_EXPORT_SHA and sha(t19_audit)==T19_EXPORT_AUDIT_SHA
 audit=read(t19_audit);assert audit['status']=='passed' and audit['export_manifest_sha256']==T19_EXPORT_SHA
 assert sha(t19_inventory)==audit['external_export_inventory_sha256']==T19_EXPORT_INVENTORY_SHA
 exported=read(t19_inventory)['files'];assert len(exported)==83 and scan_package(t19_export,expected_dirs(exported))==set(exported)
 for name,entry in exported.items():
  relative_member(name);path=t19_export/name;assert entry=={'sha256':sha(path),'bytes':path.stat().st_size}
 inv=read(package/'metadata/exclusion_inventory.json');dm=read(package/'metadata/data_manifest.template.json');old_scan=read(package/'metadata/historical_manifest_scan.json')
 assert len(inv['retained_calibration_buffers'])==4
 for buffer in inv['retained_calibration_buffers']:
  assert buffer['corpus']=='wiki_test' and buffer['char_end']-buffer['char_start']==8192
  assert any(r['char_start']<=buffer['char_start'] and r['char_end']>=buffer['char_end'] for r in inv['exclusions']['wiki_test'])
 index=read(t19_export/'index.json');assert len(index)==32
 attested={}
 for row in index:
  m=row['metadata'];key=(m['corpus'],m['window'],m['identity'])
  assert key not in attested and m['length']==512 and m['corpus'] in CORPORA and m['window'] in range(4) and m['identity'] in ('natural','bos')
  assert re.fullmatch('[a-f0-9]{64}',m['token_sha256']);attested[key]=m['token_sha256']
 assert set(attested)=={(c,w,i) for c in CORPORA for w in range(4) for i in ('natural','bos')}
 spans=[]
 for record in dm['passages']:
  c,w=record['corpus'],record['window'];assert record['window_token_sha256']==attested[(c,w,'natural')]
  spans.append({'corpus':c,'window':w,'char_start':record['char_start'],'char_end':record['used_char_end'],
   'origin':'T19_audited_selected_used_span','origin_manifest_sha256':sha(package/'metadata/data_manifest.template.json'),
   'token_sha256':attested[(c,w,'natural')],'BOS_token_sha256':attested[(c,w,'bos')]})
 assert len(spans)==16
 hashes=set(inv['history_token_hashes'])|set(old_scan['historical_window_token_hashes'])|set(attested.values())
 return {'identity':identity,'data_manifest':dm,'exclusion_inventory':inv,'old_scan':old_scan,'spans':spans,'identity_attestations':[{'corpus':c,'window':w,'identity':i,'token_sha256':h} for (c,w,i),h in sorted(attested.items())],
  'inherited_hashes':sorted(hashes),'bindings':{'package_identity_sha256':T19_PACKAGE_ID,'exclusion_inventory_sha256':sha(package/'metadata/exclusion_inventory.json'),'old_scan_sha256':sha(package/'metadata/historical_manifest_scan.json'),'T19_export_sha256':T19_EXPORT_SHA,'T19_index_sha256':sha(t19_export/'index.json'),'T19_export_audit_sha256':T19_EXPORT_AUDIT_SHA,'T19_export_inventory_sha256':T19_EXPORT_INVENTORY_SHA}}

def leaves(value,pointer=''):
 if isinstance(value,dict):
  for key,item in value.items():yield from leaves(item,pointer+'/'+str(key).replace('~','~0').replace('/','~1'))
 elif isinstance(value,list):
  for i,item in enumerate(value):yield from leaves(item,pointer+'/'+str(i))
 else:yield pointer,value
HASH_FAMILIES=frozenset(('history_token_hashes','historical_window_token_hashes','evaluation_union','calibration_union','window_token_hashes','token_hashes','evaluation_token_hashes','calibration_token_hashes'))
def token_hash_pointer(pointer):
 parts=pointer.split('/')[1:]
 return bool(parts) and (parts[-1].endswith('token_sha256') or any(part in HASH_FAMILIES for part in parts))
def assert_start_literal(tokenizer,literals,identity):
 # The public T19 template omits token IDs. Its audited private-literal SHA
 # binds the reconstructed BOS and literal encodings without exposing IDs.
 assert type(tokenizer.bos_token_id)is int and literals['start_special']['id']==tokenizer.bos_token_id
 assert value_sha(literals)==identity['private_value_commitments']['literals'],'Pinned T19 BOS/literal identity changed'
def relevant_name(name):
 return name.endswith('.json') and ('manifest' in name or name[:-5] in HASH_FAMILIES or name in {'calibration_inputs.json','exclusion_inventory.json','input_bindings.json','identity.json','index.json','preflight.json','replay_preparation.json','recovery_context.json'})
def json_files(root,excluded):
 assert root.is_dir() and not root.is_symlink();pending=[root];excluded=excluded if isinstance(excluded,(list,tuple)) else (excluded,)
 while pending:
  directory=pending.pop()
  for entry in sorted(os.scandir(directory),key=lambda v:v.name):
   path=Path(entry.path)
   st=entry.stat(follow_symlinks=False)
   if stat.S_ISLNK(st.st_mode):
    yield path,'symlink_not_followed';continue
   if path.name=='.git' or any(path.resolve().is_relative_to(v.resolve()) for v in excluded):continue
   if stat.S_ISDIR(st.st_mode):pending.append(path)
   elif stat.S_ISREG(st.st_mode) and relevant_name(path.name):yield path,'regular'

def discover(a):
 identity=None;report={'status':'running','model_forwards':0,'tokenizer_loaded':False,'pretrained_loaded':False,'gpu_used':False,'file_deletions':0}
 try:
  ancestry=base_history(a.t19_package,a.t19_export,a.t19_audit,a.t19_inventory);identity=fresh(a.out,a.t19_package,a.t19_export)
  source_ids={c:{ancestry['data_manifest']['sources'][c][k] for k in ('file_sha256','processed_text_sha256')} for c in PHYSICAL}
  source_names={c:ancestry['data_manifest']['sources'][c]['filename'] for c in PHYSICAL}
  source_ids['optimizers_new']={ancestry['data_manifest']['sources']['optimizers_new']['processed_text_sha256']}
  known_hashes=set(ancestry['inherited_hashes']);old_files={r['file']:r['sha256'] for r in ancestry['old_scan']['files']}
  known_spans={c:[(r['char_start'],r['char_end']) for r in ancestry['exclusion_inventory']['exclusions'].get(c,[])] for c in ancestry['exclusion_inventory']['sources']}
  for r in ancestry['spans']:known_spans.setdefault(r['corpus'],[]).append((r['char_start'],r['char_end']))
  files=[];unknown=[];aliases=[];ignored=[]
  for label in DISCOVERY_ROOTS:
   root=WORK/label;assert root.is_dir(),('Missing scoped history root',label)
   for path,kind in json_files(root,a.out):
    relative=path.relative_to(WORK).as_posix()
    if kind!='regular':ignored.append({'file':relative,'classification':kind});continue
    report['active_file']=relative;obj=read(path);h=sha(path);source_matches=[];windows=[];spans=[]
    for pointer,value in leaves(obj):
     if isinstance(value,str):
      for corpus,identities in source_ids.items():
       if value in identities:source_matches.append({'pointer':pointer,'source':corpus,'binding':'exact_source_or_processed_text_SHA256'})
       elif corpus in source_names and Path(value).name==source_names[corpus]:source_matches.append({'pointer':pointer,'source':corpus,'binding':'filename_availability_only'})
      if (token_hash_pointer(pointer) or path.stem in HASH_FAMILIES) and re.fullmatch('[a-f0-9]{64}',value):windows.append({'pointer':pointer,'token_sha256':value})
    def records(v,pointer='',inherited_corpus=None):
     if isinstance(v,dict):
      c=v.get('corpus',inherited_corpus);end=v.get('used_char_end',v.get('char_end'))
      if isinstance(c,str) and c in known_spans and type(v.get('char_start'))is int and type(end)is int:
       spans.append({'pointer':pointer,'corpus':c,'char_start':v['char_start'],'char_end':end})
      for key,item in v.items():
       exposure_container=pointer.rsplit('/',1)[-1] in ('exclusions','spans','intervals','passages','used_spans','selected_spans')
       records(item,pointer+'/'+str(key),key if key in known_spans and exposure_container else c)
     elif isinstance(v,list):
      for i,item in enumerate(v):records(item,pointer+'/'+str(i),inherited_corpus)
    records(obj)
    row={'file':relative,'sha256':h,'bytes':path.stat().st_size,'source_matches':source_matches,'window_attestations':windows,'spans':spans,'inherited_exact_file':old_files.get(relative)==h}
    files.append(row)
    if row['inherited_exact_file']:continue
    new_hashes=[v for v in windows if v['token_sha256'] not in known_hashes]
    new_spans=[v for v in spans if not any(lo<=v['char_start']<v['char_end']<=hi for lo,hi in known_spans[v['corpus']])]
    unresolved_source=bool(source_matches) and not (windows or spans)
    if new_hashes or new_spans or unresolved_source:
     evidence={'file':relative,'sha256':h,'source_matches':source_matches,'unclassified_hashes':new_hashes,'unclassified_spans':new_spans,'unresolved_source_linked_metadata':unresolved_source};evidence['evidence_id']=value_sha(evidence);unknown.append(evidence)
    elif source_matches or windows or spans:aliases.append({'file':relative,'sha256':h,'source_matches':source_matches,'window_attestations':windows,'spans':spans,'classification':'covered_membership_alias_provenance_retained'})
  report.pop('active_file',None)
  report.update(status='discovered_pending_independent_classification',proposal_sha256=PROPOSAL_SHA,input_addendum_sha256=ADDENDUM_SHA,
   inherited_bindings=ancestry['bindings'],inherited_history_hashes=ancestry['inherited_hashes'],T19_used_spans=ancestry['spans'],T19_identity_attestations=ancestry['identity_attestations'],
   roots=list(DISCOVERY_ROOTS),files=files,covered_aliases=aliases,unclassified_exposure_evidence=unknown,nonfollowed_links=ignored,
   historical_first_token_membership='unavailable_not_used',script_sha256=sha(__file__))
  put(a.out/'history_scan.json',report)
 except BaseException as error:
  if owned(a.out,identity):put(a.out/'discovery_failure.json',{**report,'status':'failed','error_type':type(error).__name__,'error':str(error)})
  raise
 return report

def reviewed_history(a,ancestry):
 assert APPROVED_HISTORY_SCAN_SHA and APPROVED_HISTORY_REVIEW_SHA,'History scan and independent classification not frozen'
 assert sha(a.history_scan)==APPROVED_HISTORY_SCAN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
 scan=read(a.history_scan);review=read(a.history_review)
 assert scan['inherited_bindings']==ancestry['bindings'] and scan['T19_used_spans']==ancestry['spans'] and scan['T19_identity_attestations']==ancestry['identity_attestations']
 assert scan['proposal_sha256']==PROPOSAL_SHA and scan['input_addendum_sha256']==ADDENDUM_SHA
 assert review['status']=='passed' and review['scan_sha256']==sha(a.history_scan) and review['proposal_sha256']==PROPOSAL_SHA and review['input_addendum_sha256']==ADDENDUM_SHA
 current=[];links=[]
 for label in DISCOVERY_ROOTS:
  for path,kind in json_files(WORK/label,(a.history_scan.parent,a.out)):
   if kind=='regular':current.append(path.relative_to(WORK).as_posix())
   else:links.append({'file':path.relative_to(WORK).as_posix(),'classification':kind})
 assert set(current)=={r['file'] for r in scan['files']},'Current history/availability file universe changed after discovery'
 assert sorted(links,key=lambda r:r['file'])==sorted(scan['nonfollowed_links'],key=lambda r:r['file']),'History symlink inventory changed'
 # Recheck every discovered file now, including source/availability aliases.
 for record in scan['files']:
  path=WORK/record['file'];assert path.resolve().is_relative_to(WORK.resolve()) and not path.is_symlink() and sha(path)==record['sha256']
 unknown={r['evidence_id']:r for r in scan['unclassified_exposure_evidence']};assert set(review['resolutions'])==set(unknown)
 exclusions=copy.deepcopy(ancestry['exclusion_inventory']['exclusions']);hashes=set(ancestry['inherited_hashes']);resolutions=[]
 for r in ancestry['spans']:exclusions[r['corpus']].append(copy.deepcopy(r))
 for key,record in unknown.items():
  resolution=review['resolutions'][key];assert resolution['file_sha256']==record['sha256'] and isinstance(resolution['reason'],str) and resolution['reason']
  assert resolution['classification'] in ('not_observed_source','include_source_exposure')
  if resolution['classification']=='include_source_exposure':
   assert {r['token_sha256'] for r in record['unclassified_hashes']}<=set(resolution['token_hashes'])
   assert all(any(r['corpus']==new['corpus'] and r['char_start']<=new['char_start']<new['char_end']<=r['char_end'] for r in resolution['spans']) for new in record['unclassified_spans'])
   for r in resolution['spans']:
    c=r['corpus'];assert c in exclusions and r['processed_text_sha256']==ancestry['data_manifest']['sources'][c]['processed_text_sha256']
    assert 0<=r['char_start']<r['char_end']<=ancestry['data_manifest']['sources'][c]['characters'];exclusions[c].append({**r,'origin':record['file'],'origin_sha256':record['sha256'],'reviewed_exposure_resolution':key})
   for h in resolution['token_hashes']:assert re.fullmatch('[a-f0-9]{64}',h);hashes.add(h)
  resolutions.append({'evidence_id':key,**resolution})
 return scan,exclusions,hashes,resolutions

def prepare(a):
 assert PREPARATION_APPROVED,'Independent complete input wrapper review required before tokenizer selection'
 os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
 import torch
 from datasets import Dataset
 from transformers import AutoTokenizer
 from post_write_norm_selector import select_panel,th,textsha
 ancestry=base_history(a.t19_package,a.t19_export,a.t19_audit,a.t19_inventory);scan,exclusions,hashes,resolutions=reviewed_history(a,ancestry)
 identity=None;progress={};meta={'status':'running','phase':'checking_sources','model_forwards':0,'pretrained_loaded':False,'gpu_used':False,'calibration_fitted':False};started=time.time()
 try:
  identity=fresh(a.out,a.t19_package,a.t19_export,a.history_scan.parent);torch.set_num_threads(1);dm=ancestry['data_manifest'];cfg=read(a.config);spec=read(a.t19_package/'metadata_dependencies/superweight-causal-circuits__manifests__transfer_models.json')['gpt2'];gc=cfg['models']['gpt2']
  assert gc['model_id']==spec['model_id'] and gc['revision']==spec['revision'];tokenizer_path=Path(gc['path'])
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True);assert tok.is_fast and type(tok.bos_token_id)is int
  literals={name:{'literal':literal,'ids':tok.encode(literal,add_special_tokens=False)} for name,literal in (('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n'))};literals['start_special']={'label':'bos','id':tok.bos_token_id}
  assert_start_literal(tok,literals,ancestry['identity'])
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
  put(a.out/'exclusion_inventory.json',{'sources':dm['sources'],'exclusions':exclusions,'retained_calibration_buffers':ancestry['exclusion_inventory']['retained_calibration_buffers'],'history_token_hashes':inherited,'inherited_bindings':ancestry['bindings'],'current_history_scan_sha256':sha(a.history_scan),'current_history_review_sha256':sha(a.history_review),'reviewed_resolutions':resolutions,'T19_used_spans':ancestry['spans'],'T19_identity_attestations':ancestry['identity_attestations'],'freshness_scope':'Recorded research exposures only; GPT2 pretraining overlap unknown.'})
  copy_exact(a.history_scan,a.out/'historical_manifest_scan.json');copy_exact(a.history_review,a.out/'historical_manifest_scan_review.json')
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT-2*1024**2
  meta['phase']='selection';select_panel(CORPORA,texts,tok,exclusions,hashes,components,False,progress,bos_id=tok.bos_token_id)
  assert len(progress['items'])==len(progress['selected'])==16;torch.save(progress['items'],a.out/'tokens.pt')
  put(a.out/'data_manifest.json',{'sources':dm['sources'],'passages':progress['selected'],'rejections':progress['rejections'],'literals':literals,'tokenizer_sha256':dm['tokenizer_sha256'],'physical_length':512,'passages_per_corpus':{c:4 for c in CORPORA},'exclusion_inventory_sha256':sha(a.out/'exclusion_inventory.json'),'calibration_reference_manifest_sha256':dm['calibration_reference_manifest_sha256'],'calibration_reference_files':dm['calibration_reference_files'],'calibration_fit':'Frozen T18 F; no new calibration','selection_rule':{'frozen_base_selector_sha256':'810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a','adapted_selector_sha256':SELECTOR_SHA,'input_addendum_sha256':ADDENDUM_SHA,'historical_hashes_frozen_before_selection':True,'natural_and_BOS_checked':True,'own_pair_identical_allowed':True}})
  put(a.out/'start_identity_summary.json',{'historical_first_token_membership':'unavailable_not_used','natural_BOS_pairing_preserved':True,'start_intervention':'Replace token0 only; retain physical512 and tokens1..511','identical_start_pairs':[{'corpus':r['corpus'],'window':r['window'],'identical':r['identical_start_pair']} for r in progress['selected']]})
  pins();assert sha(a.history_scan)==APPROVED_HISTORY_SCAN_SHA and sha(a.history_review)==APPROVED_HISTORY_REVIEW_SHA
  for corpus in PHYSICAL:assert sha(Path(cfg['sources'][corpus]))==dm['sources'][corpus]['file_sha256']
  for name,h in dm['tokenizer_sha256'].items():assert sha(tokenizer_path/name)==h
  reviewed_history(a,ancestry)
  assert base_history(a.t19_package,a.t19_export,a.t19_audit,a.t19_inventory)==ancestry,'T19 package/export ancestry changed during selection'
  assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<PREPARATION_LIMIT and shutil.disk_usage(a.out).free>=RESERVE
  meta.update(status='passed',phase='completed',evaluation_passages=16,evaluation_inputs=32,history_hash_count=len(inherited),accepted_identity_hash_count=len(hashes)-len(inherited),historical_first_token_membership='unavailable_not_used',proposal_sha256=PROPOSAL_SHA,input_addendum_sha256=ADDENDUM_SHA,script_sha256=sha(__file__),selector_sha256=SELECTOR_SHA,source_package_identity_sha256=T19_PACKAGE_ID,current_history_scan_sha256=sha(a.history_scan),current_history_review_sha256=sha(a.history_review))
 except BaseException as error:
  meta.update(status='failed',failure_phase=meta['phase'],error_type=type(error).__name__,error=str(error))
  if owned(a.out,identity):
   if progress.get('items'):torch.save(progress['items'],a.out/'partial_tokens.pt')
   put(a.out/'failure_selection.json',{k:v for k,v in progress.items() if k!='items'})
  raise
 finally:
  if owned(a.out,identity):
   meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,output_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file()});put(a.out/'preflight.json',meta)

def main():
 ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='action',required=True)
 for action in ('scan-history','prepare'):
  cmd=sub.add_parser(action)
  for name in ('t19-package','t19-export','t19-audit','t19-inventory','out'):cmd.add_argument('--'+name,type=Path,required=True)
  if action=='prepare':
   for name in ('config','history-scan','history-review'):cmd.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();discover(a) if a.action=='scan-history' else prepare(a)
if __name__=='__main__':main()
