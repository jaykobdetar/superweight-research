"""Independent saved-input and direction audit; CPU only, no model/tokenizer load."""
from pathlib import Path
import hashlib,json,resource,collections
import numpy as np
import torch
B=Path(__file__).resolve().parents[2];P=B/'pulse_transfer';D=P/'localization_inputs_v2'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
m=read(D/'preflight.json');assert m['status']=='passed'and m['model_forwards']==0 and m['model_loaded']is False and m['gpu_used']is False
assert sha(P/'gate_localization_inputs.py')==m['script_sha256'];assert sha(P/'GPT2_GATE_LOCALIZATION_PROPOSAL.md')==m['proposal_sha256']
for n,h in m['output_sha256'].items():assert sha(D/n)==h
inv=read(D/'exclusion_inventory.json');input_checked=0
for n,h in inv['input_hashes'].items():assert sha(B/n)==h;input_checked+=1
old=read(P/'gpt2_gate_localization_inputs/failure_selection.json')['selected'];dm=read(D/'data_manifest.json');items=load(D/'tokens.pt');assert len(items)==len(dm['passages'])==15
assert dm['excluded_inventory_sha256']==sha(D/'exclusion_inventory.json')
counts=collections.Counter();checked=0
for row,item in zip(dm['passages'],items):
 assert row==next(r for r in old if(r['corpus'],r['window'])==(row['corpus'],row['window']))
 assert row['corpus']==item['corpus']and row['window']==item['window'];x=item['tokens'];assert x.shape==(512,)and x.dtype==torch.int64 and th(x)==row['window_token_sha256']
 a,b=row['char_start'],row['used_char_end'];assert a<b<=row['extraction_char_end'];counts[row['corpus']]+=1
 chars=sum(dm['sources'][k]['characters']for k in ('sgd','adam','adamw'))+4 if row['corpus']=='optimizers'else dm['sources'][row['corpus']]['characters']
 assert a==row['fraction_numerator']*(chars-8192)//1000
 for ex in inv['evaluation_and_olmo_calibration'][row['corpus']]:
  assert not(a<ex['char_end']and b>ex['char_start']);checked+=1
 for ex in dm['passages']:
  if ex['corpus']==row['corpus']and ex['window']!=row['window']:assert not(a<ex['used_char_end']and b>ex['char_start'])
assert counts=={'wiki_validation':4,'dubois':4,'hobbes':4,'optimizers':3}
assert len(inv['full_transfer_calibration_development_buffers'])==48
for row in inv['full_transfer_calibration_development_buffers']:assert row['char_end']-row['char_start']==8192
shifts={r['corpus']:r['character_shift']for r in inv['olmo_coordinate_mapping']};assert shifts=={'wiki_validation':0,'dubois':824,'hobbes':817,'optimizers':0}
# Independent numpy implementation, not the helper's function.
maps=load(P/'gpt2_gate_fixed512_run/calibration_maps.pt');dirs=load(D/'directions.pt');errors={}
rng=np.random.default_rng(20261025)
for l in range(4,12):
 u0=maps[l]['basis_r'][:,0].numpy();u=u0/np.linalg.norm(u0)
 for j in range(12):
  q=maps[l]['basis_qr'][:,j].numpy();w=q-u*np.dot(u,q);wn=np.linalg.norm(w)
  if wn>=1e-6:break
 v=w/wn;pivot=np.argmax(abs(v));sign=-1 if v[pivot]<0 else 1;v*=sign
 assert dirs[l]['qr_column']==j and dirs[l]['sign_pivot']==int(pivot)and dirs[l]['sign_multiplier']==sign
 assert np.max(abs(u-dirs[l]['u'].numpy()))<1e-14 and np.max(abs(v-dirs[l]['v'].numpy()))<1e-14
 uf=dirs[l]['u'].float();vf=dirs[l]['v'].float();xs=torch.tensor(rng.normal(size=(511,12)),dtype=torch.float32);s=xs@uf
 n1=(s[:,None]*uf).norm(dim=-1);n2=(s[:,None]*vf).norm(dim=-1);er=float(((n1-n2).abs()/torch.maximum(torch.ones_like(n1),torch.maximum(n1,n2))).max())
 assert er<=1e-6;errors[l]={'independent_fp64_u_error':float(np.max(abs(u-dirs[l]['u'].numpy()))),'independent_fp64_v_error':float(np.max(abs(v-dirs[l]['v'].numpy()))),'fp32_511_query_norm_error':er}
# Exact counts and sign convention: comparison is matched local logit norm, not mass norm.
conditions=4+3*(8+1);main=30*conditions+8;suffix=30*conditions;assert(conditions,main,suffix,main+suffix)==(31,938,930,1868)
res={'status':'passed','proposal_sha256':sha(P/'GPT2_GATE_LOCALIZATION_PROPOSAL.md'),'input_helper_sha256':sha(P/'gate_localization_inputs.py'),'preflight_sha256':sha(D/'preflight.json'),'verified_input_provenance_files':input_checked,'selected_passages':15,'model_inputs':30,'passages_per_corpus':dict(counts),'prior_span_nonoverlap_comparisons':checked,'all15_records_match_original_failed_preflight':True,'all_selected_token_hashes_verified':True,'calibration_development_buffers':48,'olmo_character_shifts':shifts,'direction_checks':errors,'conditions':31,'main_records':938,'suffix_forwards':930,'total_forwards':1868,'model_loaded':False,'gpu_used':False,'tokenizer_loaded':False,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limitation':'Validates saved exclusion inventory and hashes; independent full tokenizer-offset reconstruction not repeated. Capacity3 statement reviewed as frozen-grid only; not all character starts.'}
Path(__file__).with_suffix('.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
