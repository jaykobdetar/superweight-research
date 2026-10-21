"""Independent tokenizer-only replay of selected spans and nearest-grid eligibility."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import json,hashlib,resource
import torch
from datasets import Dataset
from transformers import AutoTokenizer
P=Path(__file__).resolve().parents[1];D=P/'additive_inputs';torch.set_num_threads(1)
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def overlaps(a,b,c,d):return a<d and c<b
m=read(D/'preflight.json');dm=read(D/'data_manifest.json');ex=read(D/'exclusion_inventory.json');scan=read(D/'historical_manifest_scan.json');rows=dm['passages'];items=torch.load(D/'tokens.pt',weights_only=True,map_location='cpu')
assert m['status']=='passed'and m['model_forwards']==0 and m['model_loaded']is False and m['gpu_used']is False
assert len(rows)==len(items)==16 and m['model_inputs']==32
for n,h in m['output_sha256'].items():assert sha(D/n)==h
for name,h in ex['inherited_provenance_hashes'].items():assert sha(P.parent/name)==h
for f in scan['files']:assert sha(P.parent/f['path'])==f['sha256']
assert len(scan['files'])==84 and not scan['new_source_matches']
assert sha(P/'localization_inputs_v2/exclusion_inventory.json')==ex['inherited_inventory_sha256']
for n in ('directions.pt','directions.json'):assert sha(D/n)==sha(P/'localization_inputs_v2'/n)
texts={};sourcechecks=0
for name,s in dm['sources'].items():
 if name=='optimizers_new':continue
 p=Path(s['path']);assert sha(p)==s['file_sha256'];sourcechecks+=1
 if name=='wiki_validation':t='\n\n'.join(Dataset.from_file(str(p))['text'])
 else:
  t=p.read_text()
  if name in ('mill','wollstonecraft'):
   start=t.index('*** START OF');end=t.rindex('*** END OF');t=t[t.index('\n',start)+1:end]
 assert textsha(t)==s['processed_text_sha256']and len(t)==s['characters'];texts[name]=t
texts['optimizers_new']='\n\n'.join(texts[k]for k in ('adagrad','rmsprop','lbfgs'));assert textsha(texts['optimizers_new'])==dm['sources']['optimizers_new']['processed_text_sha256']
cfg=read(P/'inputs.local.json');tokpath=Path(cfg['models']['gpt2']['path'])
for n,h in dm['tokenizer_sha256'].items():assert sha(tokpath/n)==h
T=AutoTokenizer.from_pretrained(tokpath,local_files_only=True);assert T.is_fast
seen=set(scan['historical_window_token_hashes']);selected={c:[]for c in m['corpora']};replayed_candidates=0;grid={};startmatches=[]
oldloc=read(P/'localization_inputs_v2/data_manifest.json');assert len(ex['all15_localization_passages'])==15
assert ex['all15_localization_passages']==oldloc['passages']
for r in oldloc['passages']:seen.add(r['window_token_sha256'])
for r,item in zip(rows,items):
 c=r['corpus'];w=r['window'];assert item['corpus']==c and item['window']==w and item['tokens'].shape==(512,)
 text=texts[c];target=(125,375,625,875)[w];assert r['preferred_numerator']==target
 priorstarts=set();found=None
 for j in sorted(range(1001),key=lambda j:(abs(j-target),j)):
  a=j*(len(text)-8192)//1000
  if a in priorstarts:continue
  priorstarts.add(a);buffer=text[a:a+8192];enc=T(buffer,add_special_tokens=False,return_offsets_mapping=True);replayed_candidates+=1
  if len(enc['input_ids'])<512:continue
  xx=torch.tensor(enc['input_ids'][:512],dtype=torch.long);b=a+max(hi for lo,hi in enc['offset_mapping'][:512]);hh=th(xx)
  if any(overlaps(a,b,e['char_start'],e['char_end'])for e in ex['exclusions'][c]):continue
  if any(overlaps(a,b,aa,bb)for aa,bb in selected[c]):continue
  if hh in seen:continue
  if c=='optimizers_new'and not any(a>=e['char_start']and b<=e['char_end']for e in ex['code_components']):continue
  found=(j,a,b,hh,xx,buffer);break
 assert found is not None
 j,a,b,hh,xx,buffer=found
 assert (j,a,b,hh)==(r['fraction_numerator'],r['char_start'],r['used_char_end'],r['window_token_sha256'])
 assert torch.equal(xx,item['tokens'])and textsha(buffer)==r['passage_sha256']and r['extraction_char_end']==a+8192
 selected[c].append((a,b));seen.add(hh);grid.setdefault(c,[]).append(j)
 assert not any(overlaps(a,b,e['char_start'],e['used_char_end'])for e in oldloc['passages']if e['corpus']==c)
cal=torch.load(P/'gpt2_gate_fixed512_run/calibration_tokens.pt',weights_only=True,map_location='cpu');ids={int(i['tokens'][0])for i in cal};summary=read(D/'start_identity_summary.json')
for i in items:
 if int(i['tokens'][0])in ids:startmatches.append({'corpus':i['corpus'],'window':i['window'],'token_id':int(i['tokens'][0])})
assert summary['natural_seen_in_calibration']==[int(i['tokens'][0])in ids for i in items]
assert summary['bos_seen_in_calibration']==(summary['bos_id']in ids)
out={'status':'passed','preflight_sha256':sha(D/'preflight.json'),'source_files_verified':sourcechecks,'historical_manifests':len(scan['files']),'inherited_provenance_files':len(ex['inherited_provenance_hashes']),'Wiki_exclusions':len(ex['exclusions']['wiki_validation']),'independently_retokenized_selected_windows':16,'independently_replayed_grid_candidates':replayed_candidates,'grid':grid,'all15_localization_records_verified':True,'natural_starts_seen_in_calibration':startmatches,'BOS_seen_in_calibration':summary['bos_seen_in_calibration'],'direction_files_exact':True,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'tokenizer_only':True,'model_loaded':False,'gpu_used':False,'scope':'Fresh relative to inventoried research spans and token hashes, not unseen in pretraining or all machine activity.'}
assert out['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
