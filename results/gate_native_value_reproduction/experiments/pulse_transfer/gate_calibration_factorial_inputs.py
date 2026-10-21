"""T18 tokenizer-only preflight. HOLD: parent static review before execution.

Uses existing local tokenizers/text only. Reconstructs OLMo Wiki-test/Hume
exposures from saved natural windows and their matching original token stream.
No model class, weights, model output metrics, calibration fit or GPU is loaded.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,hashlib,json,resource,shutil,time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
CALIBRATION=('wiki_test','hume')
EVALUATION=('wiki_validation','utilitarianism','ethics','optimizers_new')
PREFERRED=(125,375,625,875)
HISTORY_SHA='6ca2877e812a645367ab1ce5eab4c4b5bcb55e97998b8435d59c6b64233d69cd'
PINNED_SOURCES={
 'utilitarianism':('pg11224-utilitarianism.txt','2b2e413b8df7d42790546a6c636802f10da3ee472cc94a85d3e12204b6385241'),
 'ethics':('pg3800-ethics.txt','c549e5146cc820899acbed30f9c4e28e7bb8cd26046a98a977800ae62f856214'),
 'adadelta':('adadelta.py','1bba18c22d56a377c636cae2d9f83840e4318f199131f82404712e059dd662e3'),
 'adamax':('adamax.py','1f15c2e8f2bba4b6847d8242c8ead071b9d7df1ee5add1d4412f2a7b4bc986ff'),
 'asgd':('asgd.py','6a1e26565bea0b8448ef7fd6a1637b4930b18abafc0832d41387247e53a4eef1')}
DISK_FLOOR=2*1024**3
PREPARATION_ALLOWANCE=16*1024**2

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def th(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def overlap(a,b,c,d):return a<d and b>c
def strip_book(s):
 a=s.find('*** START OF');b=s.rfind('*** END OF')
 assert a>=0 and b>a,'Missing explicit Gutenberg delimiters'
 return s[s.find('\n',a)+1:b]

class Provenance:
 def __init__(self,original):
  p=ROOT/'gate_calibration_factorial_history_sources.json';assert sha(p)==HISTORY_SHA
  self.pins=read(p)['files'];self.original=original;self.checked={}
 def path(self,label):
  if label.startswith('original_reports/'):return self.original/label.removeprefix('original_reports/')
  p=(WORK/label).resolve();assert p.is_relative_to(WORK.resolve());return p
 def check(self,label):
  assert label in self.pins,('Unpinned historical dependency',label)
  p=self.path(label);assert sha(p)==self.pins[label],('Historical dependency changed',label)
  self.checked[label]=self.pins[label];return p
 def data(self,label):return read(self.check(label))
 def tensors(self,label):return load(self.check(label))
 def check_all(self):
  for label in self.pins:self.check(label)

def token_vectors(value,pointer='',corpus=None):
 """Explicit saved-token schema, including labeled development triples."""
 if isinstance(value,torch.Tensor):
  if corpus in ('wiki','hume'):
   assert value.dtype==torch.long and value.ndim==1 and 2<=len(value)<=4096,(pointer,'Unexpected source-window shape')
   yield corpus,pointer,value
 elif isinstance(value,dict):
  for k,v in value.items():yield from token_vectors(v,pointer+'/'+str(k),k if k in ('wiki','hume')else corpus)
 elif isinstance(value,(tuple,list)):
  if len(value)==3 and isinstance(value[0],str)and value[0]in('wiki','hume')and isinstance(value[2],torch.Tensor):
   yield from token_vectors(value[2],pointer+'/2',value[0])
  else:
   for i,v in enumerate(value):yield from token_vectors(v,pointer+'/'+str(i),corpus)

def unique_token_start(stream_bytes,x):
 """Match full int64 window bytes, then verify alignment and uniqueness."""
 needle=x.contiguous().numpy().tobytes();hits=[];at=0
 while True:
  at=stream_bytes.find(needle,at)
  if at<0:break
  if at%8==0:hits.append(at//8)
  at+=1
 assert len(hits)==1,('Saved window has unresolved original source occurrence',len(x),th(x),hits)
 return hits[0]

def map_olmo_calibration_sources(prov,texts,sources,exclusions,old_hashes,progress):
 """Validate every saved relevant natural source window; no guessed offsets."""
 manifest=prov.data('pulse_causal_tranche/run/manifest.json')
 assert manifest['code_sha256']['run_tranche.py']==prov.pins['pulse_causal_tranche/run_tranche.py']
 dm=prov.data('pulse_causal_tranche/run/data_manifest.json')
 original=prov.tensors('pulse_causal_tranche/run/tokens.pt')
 assert original['starts']==dm['starts']and original['start_tokens']==dm['start_tokens']
 tokenizer_path=Path(manifest['model']);assert tokenizer_path.name==manifest['revision']
 prior=prov.data('pulse_transfer/localization_inputs_v2/data_manifest.json')
 tokenizer_hashes={}
 for n,h in prior['olmo_tokenizer_sha256'].items():
  assert sha(tokenizer_path/n)==h;tokenizer_hashes[n]=h
 tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True);assert tok.is_fast
 streams={};offsets={};stream_bytes={};mapped=progress.setdefault('mapped_ranges',[]);coverage=progress.setdefault('coverage',[])
 for oldname,c in (('wiki','wiki_test'),('hume','hume')):
  assert dm['source_sha256'][oldname]==sources[c]['file_sha256']and dm['text_sha256'][oldname]==textsha(texts[c])
  e=tok(texts[c],add_special_tokens=False,return_offsets_mapping=True)
  x=torch.tensor(e['input_ids'],dtype=torch.long)
  assert len(x)==dm['token_count'][oldname]and torch.equal(x,original['corpora'][oldname])
  streams[oldname]=x;offsets[oldname]=e['offset_mapping'];stream_bytes[oldname]=x.numpy().tobytes()
 def add(c,s,x,origin,pointer,kind='saved_actual_source_window'):
  n=len(x);assert 0<=s<s+n<=len(streams[c])and torch.equal(streams[c][s:s+n],x)
  pairs=offsets[c][s:s+n];a=min(v[0]for v in pairs);b=max(v[1]for v in pairs)
  assert 0<=a<b<=len(texts['wiki_test'if c=='wiki'else c])
  rec={'corpus':'wiki_test'if c=='wiki'else c,'char_start':a,'char_end':b,'token_start':s,'token_length':n,
   'token_sha256':th(x),'family':kind,'origin':origin,'pointer':pointer,
   'verification':'exact original pinned token stream and offset slice; saved-window calls additionally require unique full-vector match'}
  exclusions[rec['corpus']].append(rec);mapped.append(rec);old_hashes.add(th(x))
 # Exact original tranche exposure: calibration128; development Wiki512;
 # first8 evaluation starts reach1024, and remaining4 Wiki causal starts512.
 for c in ('wiki','hume'):
  for w,s in enumerate(dm['starts'][c]['calibration']):add(c,s,streams[c][s:s+128],'pulse_causal_tranche/run_tranche.py',f'calibration/{c}/{w}','source_code_and_manifest_original128')
  if c=='wiki':
   for w,s in enumerate(dm['starts'][c]['development']):add(c,s,streams[c][s:s+512],'pulse_causal_tranche/run_tranche.py',f'development/{c}/{w}','source_code_and_manifest_original512')
  for w,s in enumerate(dm['starts'][c]['evaluation'][:8]):add(c,s,streams[c][s:s+1024],'pulse_causal_tranche/run_tranche.py',f'evaluation/{c}/{w}','source_code_and_manifest_original1024')
  if c=='wiki':
   for w,s in enumerate(dm['starts'][c]['evaluation'][8:],8):add(c,s,streams[c][s:s+512],'pulse_causal_tranche/run_tranche.py',f'evaluation/{c}/{w}','source_code_and_manifest_original512')
 # Preserve larger predeclared original1024 exclusion buffers from the old
 # factorial manifest, explicitly marked conservative rather than actual.
 fd=prov.data('pulse_continuation/factorial_run/data_manifest.json')
 for c in ('wiki','hume'):
  expected=[[s,s+1024]for ss in dm['starts'][c].values()for s in ss]
  assert fd[c]['excluded_tranche1_spans']==expected
  for i,(s,e)in enumerate(expected):add(c,s,streams[c][s:e],'pulse_continuation/factorial_run/data_manifest.json',f'{c}/excluded/{i}','historically_declared_conservative1024_buffer')
 # Every direct canonical saved-token file is pinned. Derived copies are
 # independently exact-matched too. Missing source occurrence is a hard stop.
 actual_files={p.relative_to(WORK).as_posix()for p in(WORK/'pulse_continuation').glob('*_run/*tokens*.pt')}
 expected_files={k for k in prov.pins if k.startswith('pulse_continuation/')and k.endswith('.pt')and 'tokens'in Path(k).name}
 assert actual_files==expected_files,('Unreviewed historical token-file universe',actual_files^expected_files)
 for label in sorted(expected_files):
  progress['active_file']=label
  data=prov.tensors(label);implicit='wiki'if label.endswith('calibration_tokens_wiki.pt')else'hume'if label.endswith('calibration_tokens_hume.pt')else None
  vectors=list(token_vectors(data,corpus=implicit))
  if label=='pulse_continuation/upstream_run/tokens.pt':vectors+=list(token_vectors(data['development'],'/development','wiki'))
  for c,pointer,x in vectors:
   progress['active_vector']={'corpus':c,'pointer':pointer,'length':len(x),'sha256':th(x)}
   s=unique_token_start(stream_bytes[c],x);add(c,s,x,label,pointer)
  coverage.append({'file':label,'sha256':prov.pins[label],'matched_wiki_hume_vectors':len(vectors),
   'other_sources':'Other corpus branches are distinct sources and are not offset-transferred.'})
 # Four runs omit their own input file, but their frozen code/manifest points
 # to an already checked saved source panel. Generation suffixes are generated
 # outputs, not additional observed-source text; its saved prompts are checked.
 aliases={'identity_run':('template_run','tokens_sha256'),'mediation_run':('gate_run','tokens_sha256'),
  'normalization_run':('gate_run','tokens_sha256'),'generation_ties_run':('generation_run',None)}
 for run,(parent,key)in aliases.items():
  m=prov.data(f'pulse_continuation/{run}/manifest.json');target=f'pulse_continuation/{parent}/tokens.pt'
  if key:assert m[key]==prov.pins[target]
  coverage.append({'run':run,'input_parent':target,'parent_sha256':prov.pins[target],
   'basis':'Pinned run implementation and manifest; no new observed-source suffix beyond checked parent.'})
 # Verify all run implementation bindings; do not silently accept a new run.
 runs={p.parent.name for p in(WORK/'pulse_continuation').glob('*_run/manifest.json')}
 pinned_runs={Path(k).parent.name for k in prov.pins if k.startswith('pulse_continuation/')and k.endswith('/manifest.json')}
 assert runs==pinned_runs
 for run in sorted(runs):
  m=prov.data(f'pulse_continuation/{run}/manifest.json');name='cache_bypass.py'if run=='cache_run'else run.removesuffix('_run')+'.py'
  assert m['script_sha256']==prov.pins['pulse_continuation/'+name],('Changed historical implementation',run)
 del tok,streams,offsets,stream_bytes,original
 progress.update(tokenizer_sha256=tokenizer_hashes,
  scope='All pinned canonical observed-source OLMo Wiki-test/Hume windows, original conservative buffers and saved prefixes/cropped contexts; generated output continuations are not source windows.')
 progress.pop('active_file',None);progress.pop('active_vector',None)
 return progress

def leaves(x,pointer=''):
 if isinstance(x,dict):
  for k,v in x.items():yield pointer+'/'+str(k)+'/key',str(k);yield from leaves(v,pointer+'/'+str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from leaves(v,pointer+'/'+str(i))
 elif isinstance(x,str):yield pointer,x

def scan_new_sources(out,sources):
 files=[];matches=[];old_hashes=set()
 for base in(ROOT,WORK/'pulse_continuation',WORK/'pulse_causal_tranche'):
  paths=set(base.rglob('*manifest*.json'))|set(base.rglob('calibration_inputs.json'))
  for p in sorted(paths):
   if not p.is_file()or p.resolve().is_relative_to(out.resolve()):continue
   d=read(p);label=p.relative_to(WORK).as_posix();files.append({'file':label,'sha256':sha(p)})
   for pointer,value in leaves(d):
    if pointer.rsplit('/',1)[-1].endswith('token_sha256')and len(value)==64:old_hashes.add(value)
    for name in PINNED_SOURCES:
     s=sources[name]
     if value in(s['file_sha256'],s['processed_text_sha256'])or Path(value).name==s['filename']:
      matches.append({'manifest':label,'pointer':pointer,'source':name})
 return {'files':files,'new_source_matches':matches,'historical_window_token_hashes':sorted(old_hashes),
  'scope':'Canonical super-weight manifests; availability inventory is not inference exposure. No unrelated UIT or personal search.'}

def select_panel(corpora,texts,tokenizer,exclusions,old_hashes,components,full_buffers,progress):
 items=progress.setdefault('items',[]);records=progress.setdefault('selected',[]);rejections=progress.setdefault('rejections',{})
 for c in corpora:
  fresh=[];rejections[c]=[];assert len(texts[c])>=8192
  for w,target in enumerate(PREFERRED):
   tried=set();accepted=False
   for j in sorted(range(1001),key=lambda n:(abs(n-target),n)):
    a=j*(len(texts[c])-8192)//1000
    if a in tried:continue
    tried.add(a);buffer=texts[c][a:a+8192];e=tokenizer(buffer,add_special_tokens=False,return_offsets_mapping=True)
    reason=None;b=None;hits=[];x=None;component=None
    if len(e['input_ids'])<512:reason='fewer_than512_tokens'
    else:
     x=torch.tensor(e['input_ids'][:512],dtype=torch.long);pairs=e['offset_mapping'][:512]
     assert all(0<=lo<=hi<=8192 for lo,hi in pairs)
     b=a+max(hi for lo,hi in pairs);end=a+8192 if full_buffers else b
     hits=[r for r in exclusions[c]if overlap(a,end,r['char_start'],r['char_end'])]
     if hits:reason='historical_span_overlap'
     elif any(overlap(a,end,lo,hi)for lo,hi in fresh):reason='new_panel_overlap'
     elif th(x)in old_hashes:reason='historical_or_new_exact_token_window'
     elif c=='optimizers_new':
      valid=[r for r in components if a>=r['char_start']and b<=r['char_end']]
      if len(valid)!=1:reason='used_span_crosses_component_boundary'
      else:component=valid[0]['source']
    if reason:
     rejections[c].append({'window':w,'preferred_numerator':target,'numerator':j,'char_start':a,'used_char_end':b,
      'tested_char_end':a+8192 if full_buffers else b,'reason':reason,'overlap_origins':sorted(set(r['origin']for r in hits))});continue
    fresh.append((a,a+8192 if full_buffers else b));old_hashes.add(th(x));items.append({'corpus':c,'window':w,'tokens':x})
    records.append({'corpus':c,'window':w,'fraction_numerator':j,'fraction_denominator':1000,'preferred_numerator':target,
     'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,'passage_sha256':textsha(buffer),
     'window_token_sha256':th(x),'first_token_id':int(x[0]),'code_component':component,'exclusion_extent':'full8192_buffer'if full_buffers else'used512_token_span'})
    accepted=True;break
   if not accepted:raise RuntimeError(f'No eligible fixed-grid passage for {c} window{w}; unchanged exclusions; no automatic fallback')
 return items,records,rejections

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--original-reports',type=Path,default=Path('/tmp/superweight_experiment'))
 for name in PINNED_SOURCES:ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();out=a.out;assert not out.exists()or not any(out.iterdir()),'Use a new directory; preserve failures'
 out.mkdir(parents=True,exist_ok=True);started=time.time();torch.set_num_threads(1)
 meta={'status':'checking','phase':'provenance','model_forwards':0,'model_loaded':False,'gpu_used':False,
  'script_sha256':sha(__file__),'proposal_sha256':sha(ROOT/'GPT2_GATE_CALIBRATION_FACTORIAL_PROPOSAL.md'),
  'history_sources_sha256':HISTORY_SHA,'calibration_passages_requested':8,'evaluation_passages_requested':16,
  'preferred_numerators':list(PREFERRED),'grid_denominator':1000,'physical_length':512}
 dump(out/'preflight.json',meta);cal={};ev={};sources={};exclusions={c:[]for c in CALIBRATION+EVALUATION};prov=None;history={};mapping={}
 try:
  free=shutil.disk_usage(out).free;assert free>=DISK_FLOOR+PREPARATION_ALLOWANCE,('Insufficient preflight headroom',free)
  meta['disk_preflight']={'free_bytes':free,'reserve_bytes':DISK_FLOOR,'allowance_bytes':PREPARATION_ALLOWANCE}
  prov=Provenance(a.original_reports);prov.check_all();cfg=read(a.config)
  old=prov.data('pulse_transfer/gpt2_run/data_manifest.json');model=prov.data('superweight-causal-circuits/manifests/transfer_models.json')['gpt2']
  gc=cfg['models']['gpt2'];assert gc['model_id']==model['model_id']and gc['revision']==model['revision'];gtpath=Path(gc['path']);tokenizer_hashes={}
  for n in('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
   assert sha(gtpath/n)==model['checkpoint_files'][n]['sha256'];tokenizer_hashes[n]=sha(gtpath/n)
  texts={}
  for c in CALIBRATION+('wiki_validation',):
   p=Path(cfg['sources'][c]);spec=old['sources'][c];assert sha(p)==spec['sha256']
   s='\n\n'.join(Dataset.from_file(str(p))['text'])if c.startswith('wiki_')else strip_book(p.read_text())
   assert textsha(s)==spec['processed_text_sha256'];texts[c]=s
   sources[c]={'filename':p.name,'path':str(p),'file_sha256':sha(p),'processed_text_sha256':textsha(s),'characters':len(s)}
  inventory=prov.data('pulse_transfer/review/future_calibration_resource_inventory.json');available={r['name']:r for r in inventory['candidates']}
  for name,(filename,h)in PINNED_SOURCES.items():
   p=getattr(a,name);assert p.name==filename and sha(p)==h==available[filename]['sha256']
   s=strip_book(p.read_text())if name in('utilitarianism','ethics')else p.read_text();texts[name]=s
   sources[name]={'filename':filename,'path':str(p),'file_sha256':h,'processed_text_sha256':textsha(s),'characters':len(s)}
  components=[];offset=0
  for name in('adadelta','adamax','asgd'):
   components.append({'source':name,'char_start':offset,'char_end':offset+len(texts[name]),'file_sha256':sources[name]['file_sha256']});offset+=len(texts[name])+2
  texts['optimizers_new']='\n\n'.join(texts[n]for n in('adadelta','adamax','asgd'))
  sources['optimizers_new']={'processed_text_sha256':textsha(texts['optimizers_new']),'characters':len(texts['optimizers_new']),'components':components,'separator':'two newlines'}
  inherited=prov.data('pulse_transfer/localization_inputs_v2/exclusion_inventory.json')
  for label,h in inherited['input_hashes'].items():
   p=(WORK/label).resolve();assert p.is_relative_to(WORK.resolve())and sha(p)==h
   prov.checked[label]=h
  exclusions['wiki_validation'].extend(inherited['evaluation_and_olmo_calibration']['wiki_validation'])
  for r in inherited['full_transfer_calibration_development_buffers']:
   c=r['corpus'];assert c in CALIBRATION and r['source_identity']['processed_text_sha256']==sources[c]['processed_text_sha256']
   exclusions[c].append({**r,'family':'inherited_full8192_calibration_or_development_buffer'})
  # Add all T16/T17 Wiki used spans and exact token hashes. Verify extraction
  # and saved input arrays; the same tokenizer verifies their end offsets below.
  gt=AutoTokenizer.from_pretrained(gtpath,local_files_only=True);assert gt.is_fast;old_hashes=set();prior_eval_starts=set()
  for folder in('localization_inputs_v2','additive_inputs'):
   d=prov.data(f'pulse_transfer/{folder}/data_manifest.json');xs=prov.tensors(f'pulse_transfer/{folder}/tokens.pt')
   for rec in d['passages']:
    x=next(v['tokens']for v in xs if v['corpus']==rec['corpus']and v['window']==rec['window']);assert th(x)==rec['window_token_sha256'];old_hashes.add(th(x));prior_eval_starts.add(int(x[0]))
    if rec['corpus']!='wiki_validation':continue
    s=rec['char_start'];buf=texts['wiki_validation'][s:s+8192];assert textsha(buf)==rec['passage_sha256']
    e=gt(buf,add_special_tokens=False,return_offsets_mapping=True);assert torch.equal(torch.tensor(e['input_ids'][:512]),x)
    b=s+max(hi for lo,hi in e['offset_mapping'][:512]);assert b==rec['used_char_end']
    exclusions['wiki_validation'].append({'char_start':s,'char_end':b,'origin':f'pulse_transfer/{folder}/data_manifest.json','family':folder,'window':rec['window'],'token_sha256':th(x)})
  # The independently executed narrow audit supplements the previously omitted
  # original report ranges. Its source bindings are rechecked without rerunning it.
  audit=prov.data('pulse_transfer/review/original_report_span_audit.json');assert audit['status']=='passed'and audit['source_text_sha256']==sources['wiki_validation']['processed_text_sha256']
  for path,h in audit['source_sha256'].items():assert sha(Path(path))==h
  assert audit['script_sha256']==prov.pins['pulse_transfer/review/original_report_span_audit.py']
  for r in audit['ranges']:
   exclusions['wiki_validation'].append({**r,'origin':f"original_reports/{r['run']}/tokens.pt",'family':'original_report_exact_tokenizer_span'});old_hashes.add(r['token_sha256'])
  dump(out/'original_report_prior_overlap_check.json',{'audit_sha256':prov.pins['pulse_transfer/review/original_report_span_audit.json'],'overlaps':audit['overlaps'],'checked_passages':audit['checked_passages'],'scope':'Missing inherited inventory coverage is distinct from actual overlap.'})
  meta['phase']='historical_olmo_mapping';map_olmo_calibration_sources(prov,texts,sources,exclusions,old_hashes,mapping)
  dump(out/'olmo_source_mapping.json',mapping)
  history=scan_new_sources(out,sources);dump(out/'historical_manifest_scan.json',history)
  assert not history['new_source_matches'],('Unresolved new-source historical usage',history['new_source_matches'])
  old_hashes.update(history['historical_window_token_hashes'])
  for c,rs in exclusions.items():
   for r in rs:assert 0<=r['char_start']<r['char_end']<=len(texts[c]),(c,r)
  dump(out/'exclusion_inventory.json',{'sources':sources,'exclusions':exclusions,'provenance_hashes':prov.checked,
   'olmo_mapping_sha256':sha(out/'olmo_source_mapping.json'),'historical_scan_sha256':sha(out/'historical_manifest_scan.json'),
   'original_report_audit_sha256':prov.pins['pulse_transfer/review/original_report_span_audit.json'],
   'freshness_scope':'Recorded research source exposure, with declared conservative historical buffers; not pretraining exclusion or universal prior exposure.'})
  meta['phase']='calibration_selection';select_panel(CALIBRATION,texts,gt,exclusions,old_hashes,components,True,cal)
  assert len(cal['items'])==len(cal['selected'])==8;torch.save(cal['items'],out/'calibration_tokens.pt')
  dump(out/'calibration_inputs.json',{'sources':{c:sources[c]for c in CALIBRATION},'passages':cal['selected'],'rejections':cal['rejections'],
   'vectors_per_layer':4080,'query_positions':[1,510],'calibration_world':'P_source_V0_zero_L4_to_L11','metrics_reference':'self/no-native-C','new_full_buffers_disjoint':True})
  meta['phase']='evaluation_selection';select_panel(EVALUATION,texts,gt,exclusions,old_hashes,components,False,ev)
  assert len(ev['items'])==len(ev['selected'])==16;torch.save(ev['items'],out/'tokens.pt')
  dump(out/'data_manifest.json',{'sources':sources,'passages':ev['selected'],'rejections':ev['rejections'],'literals':old['literals'],
   'tokenizer_sha256':tokenizer_hashes,'original_maps_sha256':prov.pins['pulse_transfer/gpt2_gate_fixed512_run/calibration_maps.pt'],
   'original_directions_sha256':prov.pins['pulse_transfer/localization_inputs_v2/directions.pt'],
   'exclusion_inventory_sha256':sha(out/'exclusion_inventory.json'),'selection_rule':{'grid_denominator':1000,'preferred_numerators':list(PREFERRED),
    'order':'distance then numerator; four passages required per corpus','calibration':'full8192 extraction buffers','evaluation':'used512-token span','code_used_span_inside_single_component':True}})
  for n in('directions.pt','directions.json'):shutil.copyfile(prov.check('pulse_transfer/localization_inputs_v2/'+n),out/('original_'+n))
  oldcal=prov.tensors('pulse_transfer/gpt2_gate_fixed512_run/calibration_tokens.pt');ostarts={int(x['tokens'][0])for x in oldcal};fstarts={int(x['tokens'][0])for x in cal['items']};bos=old['literals']['start_special']['id']
  original_eval=prov.tensors('pulse_transfer/gpt2_gate_fixed512_run/tokens.pt');original_eval_starts={int(x['tokens'][0])for x in original_eval}
  dump(out/'start_identity_summary.json',{'original_calibration_first_token_ids':sorted(ostarts),'fresh_calibration_first_token_ids':sorted(fstarts),
   'original_evaluation_reference':'gpt2_gate_fixed512_run','original_evaluation_first_token_ids':sorted(original_eval_starts),
   'prior_T16_T17_evaluation_first_token_ids':sorted(prior_eval_starts),
   'fresh_calibration_seen_in_original':[int(x['tokens'][0])in ostarts for x in cal['items']],
   'fresh_calibration_seen_in_original_evaluation':[int(x['tokens'][0])in original_eval_starts for x in cal['items']],
   'fresh_calibration_seen_in_T16_T17_evaluation':[int(x['tokens'][0])in prior_eval_starts for x in cal['items']],
   'evaluation_first_token_ids':[int(x['tokens'][0])for x in ev['items']],
   'evaluation_seen_in_original':[int(x['tokens'][0])in ostarts for x in ev['items']],
   'evaluation_seen_in_fresh':[int(x['tokens'][0])in fstarts for x in ev['items']],
   'bos_id':bos,'bos_seen_in_original':bos in ostarts,'bos_seen_in_fresh':bos in fstarts})
  total=sum(p.stat().st_size for p in out.iterdir()if p.is_file());assert total<PREPARATION_ALLOWANCE and shutil.disk_usage(out).free>=DISK_FLOOR
  meta.update(status='passed',phase='completed',calibration_passages=8,evaluation_passages=16,evaluation_inputs=32,
   calibration_passages_per_corpus={c:4 for c in CALIBRATION},evaluation_passages_per_corpus={c:4 for c in EVALUATION},
   source_scan_matches=0,preparation_bytes=total,calibration_full_buffer_exclusion=True)
 except BaseException as error:
  meta.update(status='failed',failure_phase=meta['phase'],error_type=type(error).__name__,error=str(error))
  if cal.get('items'):torch.save(cal['items'],out/'partial_calibration_tokens.pt')
  if ev.get('items'):torch.save(ev['items'],out/'partial_evaluation_tokens.pt')
  dump(out/'failure_selection.json',{'calibration':{k:v for k,v in cal.items()if k!='items'},'evaluation':{k:v for k,v in ev.items()if k!='items'},
   'sources':sources,'exclusions':exclusions,'historical_scan':history,'olmo_mapping':mapping,'provenance':{}if prov is None else prov.checked})
  raise
 finally:
  meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=time.strftime('%FT%TZ',time.gmtime()))
  meta['output_sha256']={p.name:sha(p)for p in sorted(out.iterdir())if p.is_file()and p.name!='preflight.json'}
  dump(out/'preflight.json',meta)
 print('CALIBRATION_FACTORIAL_PREFLIGHT',meta['status'],'0 model forwards')
if __name__=='__main__':main()
