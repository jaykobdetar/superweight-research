"""Prepare private local input tensors from separately obtained, hash-checked files."""
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
REPO=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);args=ap.parse_args();cfg=json.loads(args.config.read_text());work=args.work.resolve();assert not work.exists(),'Choose a new work directory; no existing work is overwritten.'
 manifest=json.loads((REPO/'manifests/input_sources.json').read_text());paths={k:Path(v).expanduser().resolve() for k,v in cfg['sources'].items()}
 for k,p in paths.items():assert sha(p)==manifest['sources'][k]['sha256'],f'Input hash mismatch: {k}. Do not call changed-source output an exact replication.'
 assert set(paths)==set(manifest['sources'])
 tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True);texts={}
 for k,p in paths.items():texts[k]='\n\n'.join(Dataset.from_file(str(p))['text']) if k.startswith('wiki_') else p.read_text()
 h=texts['hume'];a=h.find('*** START OF');b=h.rfind('*** END OF');texts['hume']=h[h.find('\n',a)+1:b if b>=0 else None] if a>=0 else h
 texts['optimizers']='\n\n'.join(texts[k] for k in ('sgd','adam','adamw'))
 ids={k:torch.tensor(tok.encode(texts[k],add_special_tokens=False)) for k in ('wiki_test','hume','wiki_validation','dubois','hobbes','optimizers')}
 for k,x in ids.items():assert th(x)==manifest['corpora'][k]['token_sha256'],f'Tokenizer/content mismatch: {k}'
 shutil.copytree(REPO/'experiments',work/'experiments');root=work/'experiments/pulse_continuation';prepared=work/'prepared';prepared.mkdir()
 old={'corpora':{'wiki':ids['wiki_test'],'hume':ids['hume']},'starts':manifest['old_starts'],'start_tokens':{'natural':None,'eos':50279,'the':510,'period':15}}
 p=work/'experiments/pulse_causal_tranche/run';p.mkdir();torch.save(old,p/'tokens.pt')
 first={};later={};head={};dm={}
 for c in ('wiki_validation','dubois','hobbes','optimizers'):
  d=manifest['confirmation'][c];x=ids[c];first[c]=[x[s:s+512].clone() for s in d['evaluation_starts'][:4]];ls=[s+768 for s in d['evaluation_starts'][:4]];later[c]=[x[s:s+512].clone() for s in ls];hs=[4096,5120,6144,7168] if c=='optimizers' else [16384,18432,20480,22528];head[c]=[x[s:s+512].clone() for s in hs];dm[c]={'calibration_starts':d['calibration_starts'],'first_starts':d['evaluation_starts'][:4],'later_starts':ls,'head_starts':hs}
 dev=[(c,i,old['corpora'][c][s:s+512].clone()) for c in ('wiki','hume') for i,s in enumerate(old['starts'][c]['development'])]
 p=root/'upstream_run';p.mkdir();torch.save({'development':[x[2] for x in dev[:8]],'evaluation':first},p/'tokens.pt')
 torch.save(later,prepared/'later.pt');torch.save({'development':dev,'evaluation':head},prepared/'heads.pt');dump(prepared/'data_manifest.json',dm)
 dump(work/'input_identity.json',{'config_sha256':sha(args.config),'model':str(Path(cfg['model']).resolve()),'input_file_hashes':{k:sha(v) for k,v in paths.items()},'token_hashes':{k:th(x) for k,x in ids.items()},'data_manifest':dm,'scripts_source':'repository experiments copied unchanged; prepare functions overridden with these checked tensors'})
 print('Prepared verified inputs in',work)
if __name__=='__main__':main()
