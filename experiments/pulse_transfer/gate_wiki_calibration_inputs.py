"""T18 Wiki-only revision: metadata/tokenizer preparation, held for static review.

Reuse exactly the four preserved Wiki calibration passages from the failed
two-source preflight. Select only the originally proposed evaluation panel.
No model, model weights, calibration fitting, OLMo/Llama tokenizer, or GPU.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
V1_SHA='810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a'
V1_PROPOSAL_SHA='d43136b704705ba1a6a3f95f7a72ca83ccc481ff13a44fb7af7072ed4de19680'
FAILED_FILES={
 'preflight.json':'0106ad5125408c21dc6d37941afd77f3de80c74ebc13013a18b6f98e1738201a',
 'failure_selection.json':'180adfbd6d556e7fdeca6a86e90a5796472547e43816e2b2664eaeaa63f1599e',
 'exclusion_inventory.json':'ce87628cbc3507e51d785b0eb39ae496886f1166683c283e7f3a92aa5f22f41c',
 'historical_manifest_scan.json':'cf7ec1ca442d3f0aaad4dd89f82da6661fca9b044252099e9bfc1fa76618d7e7',
 'olmo_source_mapping.json':'7b5f39d5be210bf4997c684ba01249b711561878385997642f7db341fa787515',
 'original_report_prior_overlap_check.json':'a5ff8065c9ee1bfe89a30e3379ecf21962237d99913460bf4a18a9d526d26f1c',
 'partial_calibration_tokens.pt':'dcdc07d717bac319889491f637e0c98fd8f4c05dc663fb58aef994a2ca7faae5'}
AUDIT_SHA='ba7efb46473e2f99d76b230abfebe3056274d0c7613f77a97a1af470fd8fb632'
AUDIT_SCRIPT_SHA='60fb677cc4f620543c93f6363989edb67e5c44f8d96829ea2276feafcdcf2a07'
DISK_FLOOR=2*1024**3
PREPARATION_ALLOWANCE=16*1024**2

def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()

# Reuse reviewed definitions, not main(), with the exact dependency checked
# before import. No mutable constants or assertion substitutions are used.
assert file_sha(ROOT/'gate_calibration_factorial_inputs.py')==V1_SHA
import gate_calibration_factorial_inputs as old
import torch
from datasets import Dataset
from transformers import AutoTokenizer

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--config',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--failed-inputs',type=Path,default=ROOT/'calibration_factorial_inputs')
 ap.add_argument('--original-reports',type=Path,default=Path('/tmp/superweight_experiment'))
 a=ap.parse_args();out=a.out
 assert not out.exists()or not any(out.iterdir()),'Use a new directory; preserve all failed/successful preparations'
 out.mkdir(parents=True,exist_ok=True);started=time.time();torch.set_num_threads(1)
 meta={'status':'checking','phase':'provenance','revision':'same_source_W_old4_W_fresh4_with_M8_anchor',
  'model_forwards':0,'model_loaded':False,'gpu_used':False,'calibration_fitted':False,
  'script_sha256':file_sha(__file__),'proposal_sha256':file_sha(ROOT/'GPT2_GATE_WIKI_CALIBRATION_REVISION.md'),
  'original_helper_sha256':V1_SHA,'original_proposal_sha256':V1_PROPOSAL_SHA,
  'failed_input_sha256':FAILED_FILES,'capacity_audit_sha256':AUDIT_SHA,'capacity_audit_script_sha256':AUDIT_SCRIPT_SHA,
  'calibration_passages_requested':4,'evaluation_passages_requested':16,'evaluation_inputs_requested':32,
  'calibration_selection':'Reuse exact four partial Wiki passages; no new calibration search',
  'evaluation_selection':'Unchanged v1 fixed-grid and historical-exclusion rules',
  'physical_length':512,'preferred_numerators':list(old.PREFERRED),'grid_denominator':1000}
 old.dump(out/'preflight.json',meta)
 ev={};cal=[];sources={};exclusions={};provenance={};scan={};records=[]
 try:
  free=shutil.disk_usage(out).free
  assert free>=DISK_FLOOR+PREPARATION_ALLOWANCE,('Insufficient input-preparation headroom',free)
  meta['disk_preflight']={'free_bytes':free,'reserve_bytes':DISK_FLOOR,'allowance_bytes':PREPARATION_ALLOWANCE}
  assert file_sha(ROOT/'GPT2_GATE_CALIBRATION_FACTORIAL_PROPOSAL.md')==V1_PROPOSAL_SHA
  for name,h in FAILED_FILES.items():assert file_sha(a.failed_inputs/name)==h,('Changed failed-preflight evidence',name)
  assert file_sha(ROOT/'review/gate_calibration_factorial_failure_audit.json')==AUDIT_SHA
  assert file_sha(ROOT/'review/gate_calibration_factorial_failure_audit.py')==AUDIT_SCRIPT_SHA
  audit=old.read(ROOT/'review/gate_calibration_factorial_failure_audit.json')
  assert audit['status']=='passed'and audit['capacity']['hume']['all_integer_8192_starts']==0
  fm=old.read(a.failed_inputs/'preflight.json');failed=old.read(a.failed_inputs/'failure_selection.json')
  assert fm['status']=='failed'and fm['failure_phase']=='calibration_selection'and fm['model_forwards']==0
  assert fm['model_loaded']is False and fm['gpu_used']is False and not failed['evaluation']
  assert fm['script_sha256']==V1_SHA and fm['proposal_sha256']==V1_PROPOSAL_SHA
  prov=old.Provenance(a.original_reports);prov.check_all()
  for label,h in failed['provenance'].items():
   assert file_sha(prov.path(label))==h,('Changed inherited provenance',label)
   provenance[label]=h
  assert old.read(a.failed_inputs/'exclusion_inventory.json')['provenance_hashes']==provenance
  assert old.read(a.failed_inputs/'exclusion_inventory.json')['exclusions']==failed['exclusions']
  sources=failed['sources'];exclusions=failed['exclusions'];records=failed['calibration']['selected']
  assert len(records)==4 and [(r['corpus'],r['window'])for r in records]==[('wiki_test',i)for i in range(4)]
  assert [r['fraction_numerator']for r in records]==[128,375,625,874]
  assert records==audit['wiki_selected_verified']
  cfg=old.read(a.config);model=prov.data('superweight-causal-circuits/manifests/transfer_models.json')['gpt2']
  gc=cfg['models']['gpt2'];assert gc['model_id']==model['model_id']and gc['revision']==model['revision']
  gtpath=Path(gc['path']);tokenizer_hashes={}
  for name in('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
   h=file_sha(gtpath/name);assert h==model['checkpoint_files'][name]['sha256'];tokenizer_hashes[name]=h
  texts={}
  for c,spec in sources.items():
   if c=='optimizers_new':continue
   p=Path(spec['path']);assert file_sha(p)==spec['file_sha256']
   if c.startswith('wiki_'):s='\n\n'.join(Dataset.from_file(str(p))['text'])
   elif c in('hume','utilitarianism','ethics'):s=old.strip_book(p.read_text())
   else:s=p.read_text()
   assert len(s)==spec['characters']and old.textsha(s)==spec['processed_text_sha256'];texts[c]=s
  texts['optimizers_new']='\n\n'.join(texts[n]for n in('adadelta','adamax','asgd'))
  assert old.textsha(texts['optimizers_new'])==sources['optimizers_new']['processed_text_sha256']
  components=sources['optimizers_new']['components'];cursor=0
  for component,name in zip(components,('adadelta','adamax','asgd')):
   assert component=={'source':name,'char_start':cursor,'char_end':cursor+len(texts[name]),'file_sha256':sources[name]['file_sha256']}
   cursor+=len(texts[name])+2
  assert len(components)==3
  gt=AutoTokenizer.from_pretrained(gtpath,local_files_only=True);assert gt.is_fast
  cal=old.load(a.failed_inputs/'partial_calibration_tokens.pt');assert len(cal)==4
  buffers=[]
  for rec,item in zip(records,cal):
   assert item['corpus']==rec['corpus']and item['window']==rec['window']
   start=rec['char_start'];end=rec['extraction_char_end'];assert end-start==8192
   text=texts['wiki_test'][start:end];assert old.textsha(text)==rec['passage_sha256']
   e=gt(text,add_special_tokens=False,return_offsets_mapping=True);x=torch.tensor(e['input_ids'][:512],dtype=torch.long)
   assert torch.equal(x,item['tokens'])and old.th(x)==rec['window_token_sha256']and int(x[0])==rec['first_token_id']
   assert start+max(hi for lo,hi in e['offset_mapping'][:512])==rec['used_char_end']
   assert not any(old.overlap(start,end,r['char_start'],r['char_end'])for r in exclusions['wiki_test'])
   assert not any(old.overlap(start,end,lo,hi)for lo,hi in buffers);buffers.append((start,end))
  # Byte-for-byte partial input reuse; no repacking under a changed filename.
  shutil.copyfile(a.failed_inputs/'partial_calibration_tokens.pt',out/'calibration_tokens.pt')
  assert file_sha(out/'calibration_tokens.pt')==FAILED_FILES['partial_calibration_tokens.pt']
  old.dump(out/'calibration_inputs.json',{'sources':{'wiki_test':sources['wiki_test']},'passages':records,
   'rejections':failed['calibration']['rejections']['wiki_test'],'calibration_name':'W_fresh4',
   'vectors_per_layer':2040,'query_positions':[1,510],'calibration_world':'P_source_V0_zero_L4_to_L11',
   'metrics_reference':'self/no-native-C','selection_reused_from_failed_preflight':True,
   'source_failed_preflight_sha256':FAILED_FILES['preflight.json'],'full8192_buffers_disjoint':True})
  # Keep historical sources and the failed record pinned, without repeating the
  # original OLMo tokenizer mapping or original/model calibration forwards.
  for name in('preflight.json','exclusion_inventory.json','olmo_source_mapping.json','historical_manifest_scan.json'):
   shutil.copyfile(a.failed_inputs/name,out/('prior_'+name))
  old.dump(out/'exclusion_inventory.json',{'sources':sources,'exclusions':exclusions,'provenance_hashes':provenance,
   'inherited_failed_inventory_sha256':FAILED_FILES['exclusion_inventory.json'],'capacity_audit_sha256':AUDIT_SHA,
   'retained_calibration_buffers':[{'corpus':'wiki_test','char_start':a,'char_end':b}for a,b in buffers],
   'historical_exclusions_unchanged':True,'freshness_scope':'Recorded source exposure plus declared conservative buffers; not pretraining exclusion or universal prior exposure.'})
  meta['phase']='evaluation_history_scan'
  scan=old.scan_new_sources(out,sources);old.dump(out/'historical_manifest_scan.json',scan)
  assert not scan['new_source_matches'],('Unresolved new-source historical usage',scan['new_source_matches'])
  old_hashes=set(scan['historical_window_token_hashes'])|set(failed['historical_scan']['historical_window_token_hashes'])
  old_hashes.update(r['token_sha256']for r in failed['olmo_mapping']['mapped_ranges'])
  old_hashes.update(r['token_sha256']for r in old.read(ROOT/'review/original_report_span_audit.json')['ranges'])
  prior_starts=set()
  for folder in('localization_inputs_v2','additive_inputs'):
   xs=prov.tensors(f'pulse_transfer/{folder}/tokens.pt')
   old_hashes.update(old.th(v['tokens'])for v in xs);prior_starts.update(int(v['tokens'][0])for v in xs)
  old_hashes.update(old.th(v['tokens'])for v in cal)
  meta['phase']='evaluation_selection'
  old.select_panel(old.EVALUATION,texts,gt,exclusions,old_hashes,components,False,ev)
  assert len(ev['items'])==len(ev['selected'])==16
  assert {c:sum(r['corpus']==c for r in ev['selected'])for c in old.EVALUATION}=={c:4 for c in old.EVALUATION}
  torch.save(ev['items'],out/'tokens.pt');baseline=prov.data('pulse_transfer/gpt2_run/data_manifest.json')
  old.dump(out/'data_manifest.json',{'sources':sources,'passages':ev['selected'],'rejections':ev['rejections'],
   'literals':baseline['literals'],'tokenizer_sha256':tokenizer_hashes,'physical_length':512,
   'passages_per_corpus':{c:4 for c in old.EVALUATION},'calibration_name':'W_fresh4',
   'original_mixture_maps_sha256':prov.pins['pulse_transfer/gpt2_gate_fixed512_run/calibration_maps.pt'],
   'original_mixture_raw_sha256':prov.pins['pulse_transfer/gpt2_gate_fixed512_run/calibration_raw.pt'],
   'original_mixture_tokens_sha256':prov.pins['pulse_transfer/gpt2_gate_fixed512_run/calibration_tokens.pt'],
   'original_mixture_directions_sha256':prov.pins['pulse_transfer/localization_inputs_v2/directions.pt'],
   'original_wiki_filter':{'corpus':'wiki_test','windows':[0,1,2,3],'vectors_per_layer':2040},
   'exclusion_inventory_sha256':file_sha(out/'exclusion_inventory.json'),
   'selection_rule':{'grid_denominator':1000,'preferred_numerators':list(old.PREFERRED),
    'order':'distance then numerator; four passages required per corpus','calibration':'exact retained full8192 buffers',
    'evaluation':'used512-token span','code_used_span_inside_single_component':True}})
  for name in('directions.pt','directions.json'):
   shutil.copyfile(prov.check('pulse_transfer/localization_inputs_v2/'+name),out/('mixture_'+name))
  original=prov.tensors('pulse_transfer/gpt2_gate_fixed512_run/calibration_tokens.pt')
  wiki_old=[v for v in original if v['corpus']=='wiki_test'];assert len(original)==8 and len(wiki_old)==4
  old_starts={int(v['tokens'][0])for v in wiki_old};mix_starts={int(v['tokens'][0])for v in original};new_starts={int(v['tokens'][0])for v in cal};bos=baseline['literals']['start_special']['id']
  old.dump(out/'start_identity_summary.json',{'W_old4_first_token_ids':sorted(old_starts),'W_fresh4_first_token_ids':sorted(new_starts),
   'M8_first_token_ids':sorted(mix_starts),'prior_T16_T17_evaluation_first_token_ids':sorted(prior_starts),
   'fresh_seen_in_old_wiki':[int(v['tokens'][0])in old_starts for v in cal],
   'fresh_seen_in_M8':[int(v['tokens'][0])in mix_starts for v in cal],
   'fresh_seen_in_T16_T17_evaluation':[int(v['tokens'][0])in prior_starts for v in cal],
   'evaluation_first_token_ids':[int(v['tokens'][0])for v in ev['items']],
   'evaluation_seen_in_old_wiki':[int(v['tokens'][0])in old_starts for v in ev['items']],
   'evaluation_seen_in_fresh':[int(v['tokens'][0])in new_starts for v in ev['items']],
   'evaluation_seen_in_M8':[int(v['tokens'][0])in mix_starts for v in ev['items']],
   'bos_id':bos,'bos_seen_in_old_wiki':bos in old_starts,'bos_seen_in_fresh':bos in new_starts,'bos_seen_in_M8':bos in mix_starts})
  for name,h in FAILED_FILES.items():assert file_sha(a.failed_inputs/name)==h
  size=sum(p.stat().st_size for p in out.iterdir()if p.is_file());assert size<PREPARATION_ALLOWANCE and shutil.disk_usage(out).free>=DISK_FLOOR
  meta.update(status='passed',phase='completed',calibration_passages=4,evaluation_passages=16,evaluation_inputs=32,
   calibration_passages_per_corpus={'wiki_test':4},evaluation_passages_per_corpus={c:4 for c in old.EVALUATION},
   reused_calibration_token_file_sha256=file_sha(out/'calibration_tokens.pt'),calibration_records_exact=True,
   original_failed_artifacts_unchanged=True,source_scan_matches=0,preparation_bytes=size)
 except BaseException as error:
  meta.update(status='failed',failure_phase=meta['phase'],error_type=type(error).__name__,error=str(error))
  if ev.get('items'):torch.save(ev['items'],out/'partial_evaluation_tokens.pt')
  old.dump(out/'failure_selection.json',{'retained_calibration':records,'evaluation':{k:v for k,v in ev.items()if k!='items'},
   'sources':sources,'exclusions':exclusions,'historical_scan':scan,'provenance':provenance})
  raise
 finally:
  meta.update(wall_seconds=time.time()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=time.strftime('%FT%TZ',time.gmtime()))
  meta['output_sha256']={p.name:file_sha(p)for p in sorted(out.iterdir())if p.is_file()and p.name!='preflight.json'}
  old.dump(out/'preflight.json',meta)
 print('WIKI_CALIBRATION_PREFLIGHT',meta['status'],'0 model forwards')

if __name__=='__main__':main()
