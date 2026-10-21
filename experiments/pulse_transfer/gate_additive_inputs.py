"""DRAFT tokenizer-only additive-panel preflight. Do not execute before design review.

No model class/weights or experiment outcomes are loaded. Historical exclusion
verification is inherited explicitly, while new-source usage is scanned anew.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
CORPORA=('wiki_validation','mill','wollstonecraft','optimizers_new')
PREFERRED=(125,375,625,875)
SOURCE_HASHES={
 'mill':('pg34901-on-liberty.txt','fad90451253e40190995748ed5d5b60b48b0881cd1a37fa705e471bc628b36ca'),
 'wollstonecraft':('pg3420-a-vindication-of-the-rights-of-woman-with-strictures-on-political-and-moral-subjects.txt','7d79652f43541759d8ff0ce467372fa4a534d3f9dc32013e01ca99ea04b286a8'),
 'adagrad':('adagrad.py','f159c004adcfd2cc12041fb9c548f6aab383a301e04c1aa7f66e91f484c99c1c'),
 'rmsprop':('rmsprop.py','11344fad2c05f2998cb2465581039306972228e9ad97cc70a6eef2cb129992b9'),
 'lbfgs':('lbfgs.py','dcc45934c2407d4b186db383ba8d329961fb8237c30cb1ec56ffff6d96c24651')}
INHERITED_HASHES={
 'localization_inputs_v2/exclusion_inventory.json':'cbc58d05d3eb00903140e788be80d500773693e2d3bae82bb8fe6c76c35f509d',
 'localization_inputs_v2/data_manifest.json':'387d214aa87c8533f59171ab0e8179ffb4b3adf3ab2494dcacec39a732671fc7',
 'localization_inputs_v2/tokens.pt':'cac494daff1f70b1787430e80a32a35af6e6ac8939d94f8acafe5d423e306a84',
 'localization_inputs_v2/directions.pt':'4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3',
 'localization_inputs_v2/directions.json':'0a27dcc857b221a9fdbd1265be678aa341d59f914eb3030cff5eb44690ff852e',
 'gpt2_gate_fixed512_run/calibration_tokens.pt':'a99fa8403d600404b4725bdc8ab8e051e78e70a864a3483b614879d68ccc668c',
 'gpt2_gate_fixed512_run/calibration_maps.pt':'ed62f089168abeec63434e5f6dfa83ff8e7e139bc362e952456ea34859b5a098'}
HISTORY_ROOTS=(ROOT,ROOT.parent/'pulse_continuation',ROOT.parent/'pulse_causal_tranche')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def overlap(a,b,c,d):return a<d and b>c

def strip_book(s):
 a=s.find('*** START OF');b=s.rfind('*** END OF')
 assert a>=0 and b>a,'Expected explicit Gutenberg delimiters'
 return s[s.find('\n',a)+1:b]

def leaves(value,pointer=''):
 if isinstance(value,dict):
  for k,v in value.items():
   yield pointer+'/'+str(k)+'/key',str(k)
   yield from leaves(v,pointer+'/'+str(k))
 elif isinstance(value,list):
  for i,v in enumerate(value):yield from leaves(v,pointer+'/'+str(i))
 elif isinstance(value,str):yield pointer,value

def scan_history(out,sources):
 """Inventory only canonical research manifests; no unrelated personal/UIT scan."""
 paths=set()
 for base in HISTORY_ROOTS:
  assert base.is_dir()
  for pattern in ('*manifest*.json','calibration_inputs.json'):
   paths.update(p for p in base.rglob(pattern)if p.is_file()and not p.resolve().is_relative_to(out.resolve()))
 files=[];matches=[];tokens=set()
 for p in sorted(paths):
  d=read(p);label=str(p.relative_to(ROOT.parent));files.append({'path':label,'sha256':sha(p)})
  for pointer,value in leaves(d):
   if pointer.rsplit('/',1)[-1].endswith('token_sha256')and len(value)==64:tokens.add(value)
   for name,source in sources.items():
    if name=='wiki_validation':continue
    if value in(source['file_sha256'],source['processed_text_sha256'])or Path(value).name==source['filename']:
     matches.append({'manifest':label,'pointer':pointer,'source':name,
      'matched_by':'hash'if value in(source['file_sha256'],source['processed_text_sha256'])else'filename'})
 return {'roots':[str(p.relative_to(ROOT.parent))for p in HISTORY_ROOTS],
  'files':files,'new_source_matches':matches,'historical_window_token_hashes':sorted(tokens),
  'scope':'Canonical super-weight manifests only; no unrelated UIT/private files. Availability inventory is not a historical manifest.'}

def select_windows(texts,tokenizer,exclusions,old_token_hashes,components,progress=None):
 """Outcome-free deterministic grid search; retain rejection data if exhausted."""
 progress={}if progress is None else progress
 items=progress.setdefault('items',[]);records=progress.setdefault('selected',[]);rejections=progress.setdefault('rejections',{})
 for c in CORPORA:rejections.setdefault(c,[])
 selected={c:[]for c in CORPORA};seen_hashes=set(old_token_hashes)
 for c in CORPORA:
  assert len(texts[c])>=8192,(c,'source too short')
  for w,target in enumerate(PREFERRED):
   tried=set();accepted=False
   for j in sorted(range(1001),key=lambda j:(abs(j-target),j)):
    a=j*(len(texts[c])-8192)//1000
    if a in tried:continue
    tried.add(a);buffer=texts[c][a:a+8192]
    e=tokenizer(buffer,add_special_tokens=False,return_offsets_mapping=True)
    reason=None;b=None;x=None;component=None
    if len(e['input_ids'])<512:reason='fewer_than512_tokens'
    else:
     x=torch.tensor(e['input_ids'][:512],dtype=torch.long)
     offsets=e['offset_mapping'][:512];assert all(0<=lo<=hi<=8192 for lo,hi in offsets)
     b=a+max(hi for lo,hi in offsets);assert a<b<=a+8192
     hit=[r for r in exclusions[c]if overlap(a,b,r['char_start'],r['char_end'])]
     if hit:reason='historical_span_overlap'
     elif any(overlap(a,b,aa,bb)for aa,bb in selected[c]):reason='new_panel_overlap'
     elif th(x)in seen_hashes:reason='previous_exact_window_token_hash'
     elif c=='optimizers_new':
      matches=[r for r in components if a>=r['char_start']and b<=r['char_end']]
      if len(matches)!=1:reason='used_span_crosses_code_component_boundary'
      else:component=matches[0]['source']
    if reason:
     rejections[c].append({'window':w,'preferred_numerator':target,'numerator':j,'char_start':a,'used_char_end':b,'reason':reason,
       'overlaps':hit if reason=='historical_span_overlap'else[]})
     continue
    selected[c].append((a,b));seen_hashes.add(th(x));items.append({'corpus':c,'window':w,'tokens':x})
    records.append({'corpus':c,'window':w,'fraction_numerator':j,'fraction_denominator':1000,'preferred_numerator':target,
     'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,'passage_sha256':textsha(buffer),
     'window_token_sha256':th(x),'first_token_id':int(x[0]),'code_component':component})
    accepted=True;break
   if not accepted:
    error=RuntimeError(f'No eligible fourth-grid-policy passage for {c}, window {w}; no policy fallback')
    error.partial={'items':items,'selected':records,'rejections':rejections};raise error
 return items,records,rejections

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
 for name in SOURCE_HASHES:ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
 assert not out.exists()or not any(out.iterdir()),'Use a new output directory; preserve prior failures'
 out.mkdir(parents=True,exist_ok=True);started=time.time();torch.set_num_threads(1)
 meta={'status':'checking','draft_policy_version':1,'model_forwards':0,'model_loaded':False,'gpu_used':False,
  'script_sha256':sha(__file__),'proposal_sha256':sha(ROOT/'GPT2_GATE_ADDITIVE_PROPOSAL.md'),
  'corpora':list(CORPORA),'target_passages_per_corpus':4,'candidate_denominator':1000,
  'preferred_numerators':list(PREFERRED),'eligibility_before_outputs':True}
 dump(out/'preflight.json',meta)
 provenance={};records=[];rejections={};items=[];exclusions={c:[]for c in CORPORA};history={};sources={}
 def remember(p):
  p=Path(p);provenance[str(p.relative_to(ROOT.parent))]=sha(p);return read(p)
 try:
  for rel,h in INHERITED_HASHES.items():assert sha(ROOT/rel)==h,(rel,'inherited hash mismatch')
  inv=remember(ROOT/'localization_inputs_v2/exclusion_inventory.json')
  for rel,h in inv['input_hashes'].items():
   p=(ROOT.parent/rel).resolve();assert p.is_relative_to(ROOT.parent.resolve())
   assert sha(p)==h,(rel,'inherited provenance changed');provenance[rel]=h
  cfg=read(args.config);old=remember(ROOT/'gpt2_run/data_manifest.json')
  tm=remember(ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json')['gpt2'];gcfg=cfg['models']['gpt2']
  assert gcfg['revision']==tm['revision']and gcfg['model_id']==tm['model_id']
  gtpath=Path(gcfg['path']);tokhash={}
  for name in('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
   p=gtpath/name;spec=tm['checkpoint_files'][name]
   assert sha(p)==spec['sha256']and p.stat().st_size==spec['bytes'];tokhash[name]=sha(p)
  inventory=remember(ROOT/'review/future_calibration_resource_inventory.json')
  available={x['name']:x for x in inventory['candidates']}
  wp=Path(cfg['sources']['wiki_validation']);assert sha(wp)==old['sources']['wiki_validation']['sha256']
  texts={'wiki_validation':'\n\n'.join(Dataset.from_file(str(wp))['text'])}
  assert textsha(texts['wiki_validation'])==old['sources']['wiki_validation']['processed_text_sha256']
  sources['wiki_validation']={'filename':wp.name,'file_sha256':sha(wp),'processed_text_sha256':textsha(texts['wiki_validation']),
   'characters':len(texts['wiki_validation']),'path':str(wp)}
  for name,(filename,h)in SOURCE_HASHES.items():
   p=getattr(args,name);assert p.name==filename and sha(p)==h==available[filename]['sha256']
   s=p.read_text();processed=strip_book(s)if name in('mill','wollstonecraft')else s
   sources[name]={'filename':filename,'path':str(p),'file_sha256':h,'processed_text_sha256':textsha(processed),'characters':len(processed)}
   texts[name]=processed
  components=[];offset=0
  for name in('adagrad','rmsprop','lbfgs'):
   components.append({'source':name,'char_start':offset,'char_end':offset+len(texts[name]),'file_sha256':sources[name]['file_sha256']})
   offset+=len(texts[name])+2
  texts['optimizers_new']='\n\n'.join(texts[n]for n in('adagrad','rmsprop','lbfgs'))
  sources['optimizers_new']={'processed_text_sha256':textsha(texts['optimizers_new']),'characters':len(texts['optimizers_new']),
   'components':components,'separator':'two newline characters'}
  # Scan individual files plus new books, never treat an availability listing as prior inference.
  history=scan_history(out,{k:v for k,v in sources.items()if k!='optimizers_new'})
  dump(out/'historical_manifest_scan.json',history)
  assert not history['new_source_matches'],('Unresolved new-source historical usage; exact spans required',history['new_source_matches'])
  exclusions['wiki_validation'].extend(inv['evaluation_and_olmo_calibration']['wiki_validation'])
  for r in inv['full_transfer_calibration_development_buffers']:
   if r['corpus']=='wiki_validation':
    assert r['source_identity']['processed_text_sha256']==sources['wiki_validation']['processed_text_sha256']
    exclusions['wiki_validation'].append({**r,'family':'inherited_full_calibration_buffer'})
  loc=remember(ROOT/'localization_inputs_v2/data_manifest.json');locitems=load(ROOT/'localization_inputs_v2/tokens.pt')
  actual=remember(ROOT/'gpt2_gate_localization_run/manifest.json')
  assert actual['status']=='completed'and actual['forward_passes']==1868
  assert sha(ROOT/'gpt2_gate_localization_run/data_manifest.json')==INHERITED_HASHES['localization_inputs_v2/data_manifest.json']
  assert sha(ROOT/'gpt2_gate_localization_run/tokens.pt')==INHERITED_HASHES['localization_inputs_v2/tokens.pt']
  assert len(loc['passages'])==len(locitems)==15
  for r in loc['passages']:
   x=next(v['tokens']for v in locitems if v['corpus']==r['corpus']and v['window']==r['window'])
   assert x.shape==(512,)and th(x)==r['window_token_sha256']
   history['historical_window_token_hashes'].append(th(x))
   if r['corpus']=='wiki_validation':
    assert loc['sources']['wiki_validation']['processed_text_sha256']==sources['wiki_validation']['processed_text_sha256']
    assert textsha(texts['wiki_validation'][r['char_start']:r['extraction_char_end']])==r['passage_sha256']
    exclusions['wiki_validation'].append({'char_start':r['char_start'],'char_end':r['used_char_end'],
     'family':'completed_localization','origin':'pulse_transfer/localization_inputs_v2/data_manifest.json','window':r['window']})
  for r in exclusions['wiki_validation']:assert 0<=r['char_start']<r['char_end']<=len(texts['wiki_validation'])
  evidence={'inherited_inventory_sha256':INHERITED_HASHES['localization_inputs_v2/exclusion_inventory.json'],
   'inherited_provenance_hashes':provenance,'historical_scan_sha256':sha(out/'historical_manifest_scan.json'),
   'exclusions':exclusions,'all15_localization_passages':loc['passages'],'sources':sources,
   'code_components':components,'freshness_scope':'Verified recorded research source spans plus exact historical token hashes; no training-text claim.',
   'inherited_olmo_offsets_recomputed':False}
  dump(out/'exclusion_inventory.json',evidence)
  tokenizer=AutoTokenizer.from_pretrained(gtpath,local_files_only=True);assert tokenizer.is_fast
  items,records,rejections=select_windows({c:texts[c]for c in CORPORA},tokenizer,exclusions,
    set(history['historical_window_token_hashes']),components,{'items':items,'selected':records,'rejections':rejections})
  assert len(items)==len(records)==16 and all(sum(r['corpus']==c for r in records)==4 for c in CORPORA)
  torch.save(items,out/'tokens.pt')
  for n in('directions.pt','directions.json'):shutil.copyfile(ROOT/'localization_inputs_v2'/n,out/n)
  assert sha(out/'directions.pt')==INHERITED_HASHES['localization_inputs_v2/directions.pt']
  dump(out/'data_manifest.json',{'sources':sources,'passages':records,'literals':old['literals'],
   'tokenizer_sha256':tokhash,'maps_sha256':INHERITED_HASHES['gpt2_gate_fixed512_run/calibration_maps.pt'],
   'directions_sha256':sha(out/'directions.pt'),'exclusion_inventory_sha256':sha(out/'exclusion_inventory.json'),
   'selection_rule':{'grid_denominator':1000,'preferred_numerators':PREFERRED,'order':'absolute distance then numerator',
    'four_per_corpus_required':True,'used_code_span_inside_one_component':True},'rejections':rejections})
  cal=load(ROOT/'gpt2_gate_fixed512_run/calibration_tokens.pt');starts={int(x['tokens'][0])for x in cal}
  bos=old['literals']['start_special']['id']
  dump(out/'start_identity_summary.json',{'calibration_first_token_ids':sorted(starts),
   'evaluation_natural_first_token_ids':[int(x['tokens'][0])for x in items],
   'natural_seen_in_calibration':[int(x['tokens'][0])in starts for x in items],'bos_id':bos,'bos_seen_in_calibration':bos in starts})
  meta.update(status='passed',passages=16,model_inputs=32,passages_per_corpus={c:4 for c in CORPORA},source_scan_matches=0,scan_scope=history['scope'])
 except BaseException as error:
  if hasattr(error,'partial'):
   records=error.partial['selected'];rejections=error.partial['rejections'];items=error.partial['items']
  meta.update(status='failed',error_type=type(error).__name__,error=str(error))
  if items:torch.save(items,out/'partial_tokens.pt')
  dump(out/'failure_selection.json',{'selected':records,'rejections':rejections,'sources':sources,
   'exclusions':exclusions,'historical_scan':history,'provenance':provenance})
  raise
 finally:
  meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   finished_utc=time.strftime('%FT%TZ',time.gmtime()))
  meta['output_sha256']={p.name:sha(p)for p in sorted(out.iterdir())if p.is_file()and p.name!='preflight.json'}
  dump(out/'preflight.json',meta)
 print('ADDITIVE_INPUT_PREFLIGHT',meta['status'],len(records),'zero model forwards')

if __name__=='__main__':main()
