from pathlib import Path
import json,re,unicodedata,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1];J=json.loads((ROOT/'sources/tokenizer.json').read_text());vocab=J['model']['vocab'];ranks={tuple(x.split()):i for i,x in enumerate(J['model']['merges'])}
bs=list(range(ord('!'),ord('~')+1))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
for b in range(256):
 if b not in bs:bs.append(b);cs.append(256+n);n+=1
enc=dict(zip(bs,map(chr,cs)));pattern=re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
def encode(t):
 assert t.isascii(),'ASCII-only reference tokenizer port';out=[]
 for piece in pattern.findall(unicodedata.normalize('NFC',t)):
  word=tuple(enc[b] for b in piece.encode())
  while len(word)>1:
   pair=min(zip(word,word[1:]),key=lambda z:ranks.get(z,float('inf')))
   if pair not in ranks:break
   new=[];i=0
   while i<len(word):
    if i+1<len(word) and (word[i],word[i+1])==pair:new.append(word[i]+word[i+1]);i+=2
    else:new.append(word[i]);i+=1
   word=tuple(new)
  out.extend(vocab[w] for w in word)
 return out
for s,expected in [('The',[510]),('.',[15]),('\n',[187]),('\n\n',[535]),(' ',[209])]:assert encode(s)==expected,(s,encode(s))
texts={
'water_cycle':'''The water in a small lake changes throughout the year as rain, evaporation, and streams alter its depth. A field team visits the same shore every week to measure temperature and record the weather. They place a ruler against a fixed stone rather than relying on the apparent position of the water from a photograph. In spring, snow melting in the hills adds water even on days without rain. During a warm summer, evaporation may remove more water than the streams supply. The team compares measurements from different seasons and keeps a separate note of unusual storms. These observations help them distinguish a brief change in the weather from a sustained change in the local water cycle. They also inspect the reeds, take photographs, and record whether the ground is dry.''',
'library':'''The public library keeps several old maps in a cabinet near the reading room. Before a map is moved, a member of staff checks its catalogue number and records the condition of the paper. Some sheets have small tears along their edges, while others have faded labels or repairs made many years ago. A conservator works on one map at a time and uses a clean support to prevent new folds. Readers can consult digital copies while the original sheets are being treated. The library plans to show a small selection in an exhibition about the town and its surrounding farms. The exhibition notes explain how roads and waterways changed, and they distinguish details recorded on the maps from later interpretations of the landscape.''',
'bread':'''The first batch of bread was ready shortly after sunrise. The baker had mixed flour and water the previous evening, leaving enough time for the dough to develop before adding the remaining ingredients. A cooler room slows the process, so the schedule is adjusted according to the season rather than followed exactly every day. After the dough has risen, it is divided into pieces of equal weight and shaped on a lightly floured surface. The loaves rest again before going into the oven. Steam helps keep their surfaces flexible during the first part of baking. When the crust has browned, the baker checks that the bread is cooked and lets it cool on a rack. Notes about temperature, timing, and texture make it easier to compare the next batch.''',
'programming':'''The program reads a list of measurements from a text file and checks that each row contains a date and a number. It reports missing values instead of silently replacing them with zero. A separate function groups valid measurements by month and calculates their average. The original rows are retained so that a reader can trace every result back to its source. Tests cover an empty file, a single row, repeated dates, and lines containing unexpected characters. The developer first runs these tests on a small example before processing the complete data set. If an error occurs, the program records the line number and a short explanation. The final summary includes the number of valid observations as well as the calculated averages, making the limits of the data easier to see.'''}
panels=[]
for name,text in texts.items():
 ids=encode(text);assert len(ids)>=129;ids=ids[:129];panels.append({'name':name,'text':text,'token_ids':ids,'token_ids_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
refs=json.loads((ROOT.parent/'sources/hf_refs.json').read_text())['branches'];steps=[2000,4000,8000,56000,80000,88000,120000]
selected=[next(r for r in refs if r['name']=='main')]+[next(r for r in refs if re.match(fr'step{s}-',r['name'])) for s in steps]
ids=sorted(set([15,187,209,510,535]+[i for p in panels for i in p['token_ids']]))
protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkpoints':selected,'panel_type':'Four new fixed ASCII prose probes composed for this study. Not a random corpus sample or training-held-out data. Selected before checkpoint inference.','panels':panels,'embedding_ids':ids,'neurons':[1710,4820,5755,8041],'q':128,'p':64,'worlds':['ordinary','newline_U','newline_R','newline_R_edge_H14','newline_R_edge_H0','newline_R_edge_all','newline_R_mlp0_q_zero','newline_U_mlp0_q_zero','empty_ordinary','empty_isolated','empty_patch_H12','empty_patch_H0','empty_mlp0_q_zero'],'edits':'Newline worlds replace token187/535 through128 with209, force source0=510, p64=209(U)/187(R), q128=187. Empty worlds use original prefix, force0=128=510. Ordinary forces only0=510. No suffix needed because all measured variables are causally determined through128.','numerics':'Primary trajectory float32 NumPy port of transformers4.57.6 OLMo forward formulas; separate FP16-storage/operation-rounding approximation for matching existing reference source pulses, not claimed bit-exact CUDA SDPA.','validation_gate':{'The_source_pulse':-258.62060546875,'period_source_pulse':-253.878173828125,'reference_max_relative_error':0.01,'self_only_source_relative_tolerance_float32':0.0001,'causal_prefix_relative_tolerance_float32':0.0001,'endpoint_qualitative':'Median newline U and R-H14-deletion magnitude >100; R and H0 control <1; H12 patch restores at least half of isolated-versus-ordinary empty-history pulse gap. Failures stop developmental interpretation.'},'resources':{'download_cap_bytes':2750000000,'process_rss_cap_bytes':4000000000,'cpu_threads':4,'one_checkpoint_at_a_time':True,'gpu':False,'training':False}}
path=ROOT/'protocol.json';assert not path.exists();path.write_text(json.dumps(protocol,indent=2));print('unique IDs',len(ids),'max ID',max(ids),'protocol sha256',hashlib.sha256(path.read_bytes()).hexdigest())
