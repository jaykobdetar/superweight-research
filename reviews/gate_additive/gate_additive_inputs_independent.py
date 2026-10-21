"""Synthetic tokenizer-free selection/history fixtures; no external input/model reads."""
from pathlib import Path
import sys,json,tempfile,hashlib,random,resource
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import gate_additive_inputs as h
class Characters:
 def __call__(self,s,**kw):return {'input_ids':[ord(c)for c in s],'offset_mapping':[(i,i+1)for i in range(len(s))]}
rng=random.Random(19871);texts={c:''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')for _ in range(60000))for c in h.CORPORA};components=[{'source':'one','char_start':0,'char_end':60000}];ex={c:[]for c in h.CORPORA};checks=[]
items,rows,rejections=h.select_windows(texts,Characters(),ex,set(),components)
assert len(rows)==16 and all(r['fraction_numerator']==r['preferred_numerator']for r in rows)
for c in h.CORPORA:
 rs=[r for r in rows if r['corpus']==c]
 assert all(not h.overlap(a['char_start'],a['used_char_end'],b['char_start'],b['used_char_end'])for i,a in enumerate(rs)for b in rs[i+1:])
checks.append('16 deterministic preferred windows, no span overlap')
# Exclude the exact first preferred window; separately derive nearest legal grid choice.
first=rows[0];excluded={c:[]for c in h.CORPORA};excluded['wiki_validation']=[{'char_start':first['char_start'],'char_end':first['used_char_end']}]
_,rr,rj=h.select_windows(texts,Characters(),excluded,set(),components)
legal=[j for j in range(1001)if not h.overlap(j*(60000-8192)//1000,j*(60000-8192)//1000+512,first['char_start'],first['used_char_end'])]
assert rr[0]['fraction_numerator']==min(legal,key=lambda j:(abs(j-125),j))
assert rj['wiki_validation'][0]['reason']=='historical_span_overlap';checks.append('Nearest eligible grid and lower-numerator tie ordering independently derived')
_,rr,rj=h.select_windows(texts,Characters(),ex,{first['window_token_sha256']},components)
assert rr[0]['window_token_sha256']!=first['window_token_sha256']and rj['wiki_validation'][0]['reason']=='previous_exact_window_token_hash';checks.append('Historical exact token-window duplicate rejected')
# Force an optimizer boundary through its first preferred used span.
a=125*(60000-8192)//1000;b=a+256;cp=[{'source':'one','char_start':0,'char_end':b},{'source':'two','char_start':b+2,'char_end':60000}]
_,rr,rj=h.select_windows(texts,Characters(),ex,set(),cp)
assert any(r['reason']=='used_span_crosses_code_component_boundary'for r in rj['optimizers_new'])
assert all(any(r['char_start']>=c['char_start']and r['used_char_end']<=c['char_end']for c in cp)for r in rr if r['corpus']=='optimizers_new');checks.append('Code used spans stay inside one component; crossing preferred candidate rejected')
oldcorpora=h.CORPORA;h.CORPORA=('wiki_validation',)
try:
 try:h.select_windows({'wiki_validation':texts['wiki_validation']},Characters(),{'wiki_validation':[{'char_start':0,'char_end':60000}]},set(),[])
 except RuntimeError as e:assert len(e.partial['selected'])==0 and len(e.partial['rejections']['wiki_validation'])==1001;checks.append('Exhaustion preserves complete rejection evidence and does not relax policy')
 else:raise AssertionError('Exhaustion accepted')
finally:h.CORPORA=oldcorpora
oldroot=h.ROOT;oldhist=h.HISTORY_ROOTS
try:
 with tempfile.TemporaryDirectory(prefix='additive-history-fixture-')as td:
  root=Path(td);h.ROOT=root/'pulse_transfer';h.HISTORY_ROOTS=(h.ROOT,root/'pulse_continuation',root/'pulse_causal_tranche')
  for p in h.HISTORY_ROOTS:p.mkdir()
  out=h.ROOT/'current';out.mkdir();(out/'data_manifest.json').write_text(json.dumps({'new':'new.txt'}))
  marker='a'*64;tok='b'*64
  (h.ROOT/'historical_manifest.json').write_text(json.dumps({'sources':{'book':{'file':'new.txt','sha256':marker}},'window_token_sha256':tok}))
  (h.ROOT/'resource_inventory.json').write_text(json.dumps({'source':'new.txt'}))
  scan=h.scan_history(out,{'newbook':{'file_sha256':marker,'processed_text_sha256':'c'*64,'filename':'new.txt'}})
  assert len(scan['files'])==1 and len(scan['new_source_matches'])==2 and scan['historical_window_token_hashes']==[tok]
  assert all('current'not in m['manifest']for m in scan['new_source_matches']);checks.append('Canonical manifest scan detects filename/hash collisions and token hashes while excluding own output/availability inventory')
finally:h.ROOT=oldroot;h.HISTORY_ROOTS=oldhist
out={'status':'passed_synthetic_helper_review','helper_sha256':hashlib.sha256((P/'gate_additive_inputs.py').read_bytes()).hexdigest(),'checks':checks,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'tokenizer_loaded':False,'model_loaded':False,'gpu_used':False,'limits':'Synthetic selection/history tests and static review; actual pinned-source tokenizer preflight remains separate.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
