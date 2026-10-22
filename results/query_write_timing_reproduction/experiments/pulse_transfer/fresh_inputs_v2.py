"""Fresh fixed character passages, with actual token-span non-overlap checks."""
import hashlib,json,math
from pathlib import Path
import torch
from datasets import Dataset
from run_transfer import sha,th,strip_book,dump

def prepare_fresh(cfg,tok,parent,out,fractions=(.15,.35,.55,.75),additional_parents=()):
 old=json.loads((parent/'data_manifest.json').read_text());sources={k:Path(v) for k,v in cfg['sources'].items()};texts={};src={}
 for k,p in sources.items():
  assert sha(p)==old['sources'][k]['sha256'];s='\n\n'.join(Dataset.from_file(str(p))['text']) if k.startswith('wiki_') else p.read_text()
  if k in ('hume','dubois','hobbes'):s=strip_book(s)
  assert hashlib.sha256(s.encode()).hexdigest()==old['sources'][k]['processed_text_sha256'];texts[k]=s;src[k]={'name':p.name,'sha256':sha(p)}
 texts['optimizers']='\n\n'.join(texts[k] for k in ('sgd','adam','adamw'));items=[];meta={'sources':src,'fractions':list(fractions),'passages':[],'old_used_spans':{},'old_manifest_sha256':sha(parent/'data_manifest.json'),'literals':old['literals']}
 for c in ('wiki_validation','dubois','hobbes','optimizers'):
  s=texts[c];olds=[r for r in old['passages'] if r['split']=='evaluation' and r['corpus']==c];spans=[]
  for r in olds:
   a=r['char_start'];ss=s[a:a+8192];enc=tok(ss,add_special_tokens=False,return_offsets_mapping=True);x=torch.tensor(enc['input_ids'][:512]);assert th(x)==r['window_token_sha256'];spans.append([a,a+enc['offset_mapping'][511][1]])
  for prior in additional_parents:
   other=json.loads((prior/'data_manifest.json').read_text())
   for r in other['passages']:
    if r['corpus']!=c:continue
    aa=r['char_start'];enc=tok(s[aa:aa+8192],add_special_tokens=False,return_offsets_mapping=True);xx=torch.tensor(enc['input_ids'][:512]);assert th(xx)==r['window_token_sha256'];spans.append([aa,aa+enc['offset_mapping'][511][1]])
  meta['old_used_spans'][c]=spans;fresh=[]
  for i,f in enumerate(fractions):
   a=math.floor(f*(len(s)-8192));ss=s[a:a+8192];enc=tok(ss,add_special_tokens=False,return_offsets_mapping=True);assert len(enc['input_ids'])>=512;x=torch.tensor(enc['input_ids'][:512]);b=a+enc['offset_mapping'][511][1];assert all(b<=aa or a>=bb for aa,bb in spans+fresh),(c,a,b,spans);fresh.append([a,b]);items.append({'corpus':c,'window':i,'tokens':x});meta['passages'].append({'corpus':c,'window':i,'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,'passage_sha256':hashlib.sha256(ss.encode()).hexdigest(),'window_token_sha256':th(x)})
 dump(out/'data_manifest.json',meta);torch.save(items,out/'tokens.pt');return items,meta
