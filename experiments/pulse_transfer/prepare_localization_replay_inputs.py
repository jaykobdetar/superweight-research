"""Rebuild frozen localization tokens/directions with a tokenizer only.

Historical selection/exclusion evidence is copied as historical evidence. New
reconstruction checks source bytes, offsets, exclusion intervals and token/map
hashes. No old OLMo/Llama/GPT2 model experiment is rerun.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,json,resource,shutil,time
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from gate_localization_inputs import directions,sha,th,textsha,strip_book,overlap,dump
from gpt2_gate_localization import INPUT_HASHES

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 started=time.time();torch.set_num_threads(1);out=args.out
 assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(args.config.read_text());ref=args.metadata
 template=json.loads((ref/'data_manifest.template.json').read_text());preflight=json.loads((ref/'preflight.json').read_text())
 assert preflight['status']=='passed'and preflight['model_forwards']==0
 assert sha(ref/'preflight.json')==INPUT_HASHES['preflight.json']
 assert sha(ref/'exclusion_inventory.json')==INPUT_HASHES['exclusion_inventory.json']
 assert sha(args.calibration/'calibration_maps.pt')==template['original_maps_sha256']
 tokpath=Path(cfg['models']['gpt2']['path'])
 for n,h in template['tokenizer_sha256'].items():assert sha(tokpath/n)==h,('tokenizer',n)
 tok=AutoTokenizer.from_pretrained(tokpath,local_files_only=True);assert tok.is_fast
 texts={}
 for c,p in cfg['sources'].items():
  p=Path(p);identity=template['sources'][c];assert sha(p)==identity['file_sha256'],('source',c)
  text='\n\n'.join(Dataset.from_file(str(p))['text'])if c.startswith('wiki_')else p.read_text()
  text=strip_book(text)if c in('hume','dubois','hobbes')else text
  assert textsha(text)==identity['processed_text_sha256']and len(text)==identity['characters'];texts[c]=text
 texts['optimizers']='\n\n'.join(texts[c]for c in('sgd','adam','adamw'))
 inv=json.loads((ref/'exclusion_inventory.json').read_text());items=[];spans={};overlap_checks=0
 canonical_corpora={c:c for c in ('wiki_validation','dubois','hobbes','optimizers')}
 for record in template['passages']:
  c,w,a=canonical_corpora[record['corpus']],record['window'],record['char_start'];text=texts[c][a:a+8192]
  assert len(text)==8192 and textsha(text)==record['passage_sha256']
  assert a==record['fraction_numerator']*(len(texts[c])-8192)//record['fraction_denominator']
  e=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert len(e['input_ids'])>=512
  x=torch.tensor(e['input_ids'][:512],dtype=torch.long);b=a+max(t[1]for t in e['offset_mapping'][:512])
  assert th(x)==record['window_token_sha256']and b==record['used_char_end']and a+8192==record['extraction_char_end']
  assert record['first_token_id']=='REBUILD_FROM_LOCAL_TOKENIZER';record['first_token_id']=int(x[0])
  for r in inv['evaluation_and_olmo_calibration'][c]:
   assert not overlap(a,b,r['char_start'],r['char_end']);overlap_checks+=1
  for lo,hi in spans.setdefault(c,[]):assert not overlap(a,b,lo,hi)
  spans[c].append((a,b));items.append({'corpus':c,'window':w,'tokens':x})
 assert len(items)==15
 torch.save(items,out/'tokens.pt');dump(out/'data_manifest.json',template)
 maps=torch.load(args.calibration/'calibration_maps.pt',weights_only=True,map_location='cpu');ds=directions(maps)
 torch.save(ds,out/'directions.pt');dump(out/'directions.json',{l:{k:v.tolist()if isinstance(v,torch.Tensor)else v for k,v in d.items()}for l,d in ds.items()})
 cal=torch.load(args.calibration/'calibration_tokens.pt',weights_only=True,map_location='cpu');starts={int(i['tokens'][0])for i in cal};bos=template['literals']['start_special']['id']
 dump(out/'start_identity_summary.json',{'calibration_first_token_ids':sorted(starts),'evaluation_natural_first_token_ids':[int(i['tokens'][0])for i in items],'natural_seen_in_calibration':[int(i['tokens'][0])in starts for i in items],'bos_id':bos,'bos_seen_in_calibration':bos in starts})
 for n in('preflight.json','exclusion_inventory.json'):shutil.copyfile(ref/n,out/n)
 for n,h in preflight['output_sha256'].items():assert sha(out/n)==h,('rebuilt artifact',n)
 assert all(sha(out/n)==h for n,h in INPUT_HASHES.items())
 dump(out/'reconstruction.json',{'status':'passed','model_forwards':0,'model_loaded':False,'gpu_used':False,'passages':15,'historical_interval_checks':overlap_checks,'source_sha256':sha(__file__),'new_output_sha256':{n:sha(out/n)for n in preflight['output_sha256']},'historical_preflight_sha256':sha(out/'preflight.json'),'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Historical exclusion provenance is inherited, not regenerated from old model outputs. New reconstruction verifies matching source/tokenizer bytes, offsets, supplied exclusion coordinates and exact frozen artifacts. Historical preflight timestamps/resources are not attributed to this new invocation.'})
 print('RECONSTRUCTED_LOCALIZATION_INPUTS',len(items),'zero model forwards')
if __name__=='__main__':main()
