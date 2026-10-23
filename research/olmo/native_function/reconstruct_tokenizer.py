"""Minimal byte-level BPE replay, accepted only after exact corpus token hashes.

Implements the saved tokenizer's NFC + GPT-2 ByteLevel regex semantics using
Unicode categories and standard-library code. No model or training involved.
"""
import os
if os.environ.get('OLMO_ALLOW_LOCAL_TEXT_JOINS') != '1':
    raise SystemExit('Optional local-corpus step. Read README_REPRODUCTION.txt, supply the tokenizer/corpora, then set OLMO_ALLOW_LOCAL_TEXT_JOINS=1. Outputs may contain source text; do not redistribute without checking licenses.')

from pathlib import Path
import functools, hashlib, json, re, unicodedata
import numpy as np

TOKENIZER=Path(__file__).resolve().parent/'inputs/tokenizer.json'

def load_window(corpus,start,length=512):
    root=Path(__file__).resolve().parent/'inputs'
    filename=f'{corpus}_ids.npy'
    if corpus=='wiki_validation' and not (root/filename).exists():filename='wiki_validation_partial_ids.npy'
    if (root/filename).exists():return np.load(root/filename)[start:start+length]
    if not (root/'recovered_windows.npz').exists():raise FileNotFoundError('Supply locally reconstructed corpus arrays; no input tokens are distributed.')
    with np.load(root/'recovered_windows.npz') as a:
        x=a[f'{corpus}__{start}'];assert len(x)==length;return x.copy()

def byte_encoder():
    bs=list(range(ord('!'),ord('~')+1))+list(range(ord('¡'),ord('¬')+1))+list(range(ord('®'),ord('ÿ')+1))
    cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    return dict(zip(bs,map(chr,cs)))

def cat(c):
    if c.isspace():return 'S'
    k=unicodedata.category(c)[0]
    return k if k in ('L','N') else 'P'

def pretokens(s):
    i=0
    while i<len(s):
        if s[i]=="'":
            hit=next((x for x in ("'s","'t","'re","'ve","'m","'ll","'d") if s.startswith(x,i)),None)
            if hit:yield hit;i+=len(hit);continue
        start=i;j=i
        if s[j]==' ' and j+1<len(s) and cat(s[j+1])!='S':j+=1
        c=cat(s[j])
        if c!='S':
            j+=1
            while j<len(s) and cat(s[j])==c:j+=1
            yield s[start:j];i=j;continue
        j=i+1
        while j<len(s) and cat(s[j])=='S':j+=1
        if j<len(s) and j-i>1:j-=1
        yield s[i:j];i=j

class Tokenizer:
    def __init__(self):
        d=json.loads(TOKENIZER.read_text());self.vocab=d['model']['vocab'];self.inv={v:k for k,v in self.vocab.items()}
        self.ranks={tuple(s.split(' ')):i for i,s in enumerate(d['model']['merges'])}
        self.enc=byte_encoder();self.dec={v:k for k,v in self.enc.items()}
        self.added={x['content']:x['id'] for x in d['added_tokens']}
        self.special=re.compile('('+'|'.join(re.escape(s) for s in sorted(self.added,key=len,reverse=True))+')')
    @functools.lru_cache(maxsize=100000)
    def bpe(self,t):
        pieces=list(t)
        while len(pieces)>1:
            pairs=[(self.ranks.get((a,b),float('inf')),i) for i,(a,b) in enumerate(zip(pieces,pieces[1:]))]
            rank,i=min(pairs)
            if rank==float('inf'):break
            pair=(pieces[i],pieces[i+1]);new=[];j=0
            while j<len(pieces):
                if j+1<len(pieces) and (pieces[j],pieces[j+1])==pair:new.append(pieces[j]+pieces[j+1]);j+=2
                else:new.append(pieces[j]);j+=1
            pieces=new
        return tuple(self.vocab[p] for p in pieces)
    def encode(self,s):
        ids=[]
        for chunk in self.special.split(unicodedata.normalize('NFC',s)):
            if chunk in self.added:ids.append(self.added[chunk]);continue
            for t in pretokens(chunk):ids.extend(self.bpe(''.join(self.enc[b] for b in t.encode('utf-8'))))
        return np.array(ids,dtype='<i8')
    def decode(self,ids):
        out=bytearray()
        for i in ids:
            t=self.inv[int(i)]
            if t in self.added and (int(i)<2 or int(i)>=50254):out.extend(t.encode('utf-8'))
            else:out.extend(self.dec[c] for c in t)
        return out.decode('utf-8',errors='replace')

if __name__=='__main__':
    root=Path(__file__).resolve().parent;tok=Tokenizer();manifest=json.loads((root/'input_manifest.json').read_text());texts={}
    for name in ('dubois','hobbes','sgd','adam','adamw'):
        p=root/'inputs'/manifest['sources'][name]['filename']
        if p.exists():texts[name]=p.read_text()
    if all(k in texts for k in ('sgd','adam','adamw')):texts['optimizers']='\n\n'.join(texts[k] for k in ('sgd','adam','adamw'))
    p=root/'inputs/wiki_validation_rows.json'
    if p.exists():texts['wiki_validation']='\n\n'.join(json.loads(p.read_text()))
    receipt=[]
    for name,text in texts.items():
        if name not in manifest['corpora']:continue
        source=manifest['corpora'][name];textsha=hashlib.sha256(text.encode()).hexdigest()
        if textsha!=source['text_sha256']:
            print(name,'TEXT MISMATCH',textsha,flush=True);receipt.append({'corpus':name,'text_match':False,'text_sha256':textsha});continue
        ids=tok.encode(text);h=hashlib.sha256(ids.tobytes()).hexdigest();okay=h==source['token_sha256']
        print(name,'text matched','tokens',len(ids),'MATCH',okay,h,flush=True)
        receipt.append({'corpus':name,'text_match':True,'token_match':okay,'token_count':len(ids),'token_sha256':h})
        if okay:np.save(root/'inputs'/f'{name}_ids.npy',ids)
    (root/'inputs/reconstruction_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
