"""Prospective T20 natural/BOS duplicate adapter; synthetic tests before selection.

Only acceptance/hash-add and explicit metadata differ from the frozen selector.
Grid enumeration, precedence, tokenizer, offsets, spans/components are retained.
No model/tokenizer/checkpoint loading occurs by importing this module.
"""
import hashlib
import torch
PREFERRED=(125,375,625,875)
ORIGINAL_SOURCE_SHA256='810014f87d8179d8c7ac9c967dbfd92ccef9ffdeff331ace0d4cac104dd7ed1a'

def textsha(s):return hashlib.sha256(s.encode()).hexdigest()

def th(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

def overlap(a,b,c,d):return a<d and b>c

def bos_hash(x,bos_id):
 assert type(bos_id)is int and 0<=bos_id<50257
 y=x.clone();y[0]=bos_id
 return th(y)

def select_panel(corpora,texts,tokenizer,exclusions,old_hashes,components,full_buffers,progress,*,bos_id):
 items=progress.setdefault('items',[]);records=progress.setdefault('selected',[]);rejections=progress.setdefault('rejections',{})
 historical_hashes=frozenset(old_hashes)
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
     elif bos_hash(x,bos_id)in old_hashes:reason='historical_or_new_exact_BOS_token_window'
     elif c=='optimizers_new':
      valid=[r for r in components if a>=r['char_start']and b<=r['char_end']]
      if len(valid)!=1:reason='used_span_crosses_component_boundary'
      else:component=valid[0]['source']
    if reason:
     rejections[c].append({'window':w,'preferred_numerator':target,'numerator':j,'char_start':a,'used_char_end':b,
      'tested_char_end':a+8192 if full_buffers else b,'reason':reason,'overlap_origins':sorted(set(r['origin']for r in hits)),
      'BOS_duplicate_scope':('historical'if bos_hash(x,bos_id)in historical_hashes else'previously_selected')if reason=='historical_or_new_exact_BOS_token_window'else None});continue
    fresh.append((a,a+8192 if full_buffers else b));old_hashes.add(th(x));old_hashes.add(bos_hash(x,bos_id));items.append({'corpus':c,'window':w,'tokens':x})
    records.append({'corpus':c,'window':w,'fraction_numerator':j,'fraction_denominator':1000,'preferred_numerator':target,
     'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,'passage_sha256':textsha(buffer),
     'window_token_sha256':th(x),'first_token_id':int(x[0]),'bos_window_token_sha256':bos_hash(x,bos_id),'identical_start_pair':th(x)==bos_hash(x,bos_id),'code_component':component,'exclusion_extent':'full8192_buffer'if full_buffers else'used512_token_span'})
    accepted=True;break
   if not accepted:raise RuntimeError(f'No eligible fixed-grid passage for {c} window{w}; unchanged exclusions; no automatic fallback')
 return items,records,rejections
