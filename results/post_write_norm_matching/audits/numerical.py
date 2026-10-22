"""Prospective independent T20 saved-array numerical audit; no analyzer imports.

Self-test uses in-memory synthetic evidence. Actual study access requires later
root source/policy/manifest review and a complete planned776-return composite.
No model loader, pretrained checkpoint/header access, GPU, or raw-study writes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,collections,copy,hashlib,io,itertools,json,math,resource,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
POLICY_SHA='a90bf7d5df077db32765c4096e2533ca684dc4e3344735875d2e30736bd48e0f'
ANALYZER_SHA='e3d36eb3584c7785d702bfa857481ae0c62bd0ac7c986e6e99ff2c42e790f45d'
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched','orth_write_matched')
BOUNDARIES=('000','111')
MODES=tuple(b+'__'+a for b in BOUNDARIES for a in ARMS)
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
FIELDS=('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
METRICS=('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')
COMMON={'actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'}
MATCH={'target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap'}
GATE_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}

def require(ok,message):
 if not ok:raise AssertionError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def put(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def fp(x):return {'sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'shape':list(x.shape),'dtype':'torch.'+x.dtype.name}
def nested_fp(x):
 if isinstance(x,np.ndarray):return fp(x)
 if isinstance(x,dict):return {k:nested_fp(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [nested_fp(v) for v in x]
 return x
def exact(a,b):
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and not a.dtype.hasobject and a.tobytes(order="C")==b.tobytes(order="C")
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
 return type(a)==type(b) and a==b
def load_npz(path):
 """Independent no-pickle decoder, validating every array identity and member."""
 with np.load(path,allow_pickle=False) as z:
  require(len(z.files)==len(set(z.files)),'no duplicate archive members')
  h=json.loads(z['metadata'].tobytes());require(h['schema']=='wiki_gate_lossless_npz_v1','lossless schema');seen=set()
  def decode(n):
   kind=n['type']
   if kind=='scalar':return n['value']
   if kind in ('tuple','list'):
    a=[decode(v) for v in n['items']];return tuple(a) if kind=='tuple' else a
   if kind=='dict':
    pairs=[(decode(k),decode(v)) for k,v in n['items']];out=dict(pairs);require(len(out)==len(pairs),'no duplicate keys');return out
   require(kind=='tensor' and n['array'] not in seen,'one physical array per tensor');seen.add(n['array']);a=z[n['array']]
   require(fp(a)=={k:n[k] for k in ('sha256','shape','dtype')},('tensor identity',n['array']));return a
  value=decode(h['object']);require(set(z.files)==seen|{'metadata'},'no omitted or extra array');return value,len(seen)
class Comparisons:
 def __init__(self,policy):self.policy=policy;self.counts=collections.Counter();self.exact=collections.Counter();self.maxima=collections.defaultdict(float);self.relative_maxima=collections.defaultdict(float)
 def check(self,a,b,category='statistics',atol=None,rtol=0):
  x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
  require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),('finite/shape',category,x.shape,y.shape))
  diff=np.abs(x-y);err=float(diff.max()) if diff.size else 0
  self.counts[category]+=int(x.size);self.exact[category]+=int(np.count_nonzero(x==y));self.maxima[category]=max(self.maxima[category],err)
  denominator=np.maximum(np.abs(x),np.abs(y));relative=np.divide(diff,denominator,out=np.zeros_like(diff),where=denominator!=0)
  self.relative_maxima[category]=max(self.relative_maxima[category],float(relative.max()) if relative.size else 0.)
  if atol is None:atol=self.policy['comparison_tolerances']['statistic_absolute']
  require(bool(np.all(diff<=atol+rtol*np.maximum(np.abs(x),np.abs(y)))),('numerical discrepancy',category,err,atol,rtol))
 def report(self):return {k:{'comparisons':n,'exact':self.exact[k],'maximum_absolute_error':self.maxima[k],'maximum_symmetric_relative_coordinate_error':self.relative_maxima[k]} for k,n in sorted(self.counts.items())}

def contrast_definitions():
 # Independently enumerated nine algebraic contrasts at each boundary, then
 # nine endpoint differences. No analyzer definitions or helper imports.
 names=('full_remove_effect','full_orth_effect','full_remove_minus_orth',
  'matched_remove_effect','matched_orth_effect','matched_remove_minus_orth',
  'matched_remove_minus_full','matched_orth_minus_full','matching_difference_of_differences')
 coefficients=((0,-1,1,0,0,0),(0,-1,0,1,0,0),(0,0,1,-1,0,0),
  (0,-1,0,0,1,0),(0,-1,0,0,0,1),(0,0,0,0,1,-1),
  (0,0,-1,0,1,0),(0,0,0,-1,0,1),(0,0,-1,1,1,-1))
 # Column order here deliberately differs from writer: current,baseline,Gu,Gv,Mu,Mv.
 columns=('current','baseline','remove_full','orth_full','remove_write_matched','orth_write_matched')
 specs={}
 for boundary in BOUNDARIES:
  for name,cs in zip(names,coefficients):
   terms={boundary+'__'+a:k for a,k in zip(columns,cs) if k}
   specs[boundary+'__'+name]={'terms':terms,'kind':'within_boundary',
    'primary':boundary=='000' and name=='matched_remove_minus_orth'}
 for name,cs in zip(names,coefficients):
  terms={b+'__'+a:sign*k for b,sign in (('111',1),('000',-1)) for a,k in zip(columns,cs) if k}
  specs['endpoint_111_minus_000__'+name]={'terms':terms,'kind':'endpoint_effect_difference','primary':False}
 require(len(specs)==27 and all(sum(x['terms'].values())==0 for x in specs.values()),'27 zero-sum contrasts')
 return specs

def indices():
 x=np.random.Generator(np.random.PCG64(20261021)).integers(0,4,(2000,4),dtype=np.int64)
 require(fp(x)['sha256']=='9e2a313244e72f2bfb4300e2497a4b8590cc189411863bdabed15b4922e2d4bc','shared bootstrap pin')
 return x

def bootstrap(values,ids):
 x=np.asarray(values,np.float64);require(x.shape==(4,) and np.isfinite(x).all(),'four finite clusters')
 return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[ids].mean(1),[.025,.975]).tolist(),'per_window':x.tolist()}

def roster(corpora):
 roles=[]
 for c,w,i,b in itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES):
  for arm in ARMS:
   roles.append((c,w,i,b,arm,'main'))
   if (w,i,b,arm)==(0,'natural','000','baseline'):
    roles.extend((c,w,i,b,a,'anchor') for a in ('native_repeat','native_identity'))
   roles.append((c,w,i,b,arm,'suffix'))
 return roles

def journal(path,expected):
 chain='0'*64;events=[];counts=collections.Counter();byrole={}
 for line in (path if isinstance(path,bytes) else Path(path).read_bytes()).splitlines(keepends=True):
  require(line.endswith(b'\n'),'complete journal line');event=json.loads(line);body=line[:-1]
  require(body==json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),'canonical journal')
  require(event['previous_sha256']==chain,'journal chain');chain=hashlib.sha256(body).hexdigest()
  counts[event['event']]+=1;events.append(event)
 kinds=('forward_intent','forward_attempt','model_returned','forward_completed')
 require(set(counts)<=set(kinds)|{'committed'},'journal known event roster')
 byattempt=collections.defaultdict(list)
 for kind in kinds:
  selected=[e for e in events if e['event']==kind];require(len(selected)==len(expected),'journal '+kind+' count')
  for j,e in enumerate(selected):
   role=tuple(e['identity'][k] for k in ('corpus','window','identity','boundary','arm','phase'))
   require(role==expected[j],('prescribed chronology',kind,j))
   require(e.get('attempt',e.get('next_attempt'))==j+1,'attempt counter')
   if kind in ('model_returned','forward_completed'):require(e['completed']==j+1,'return counter')
   if kind=='forward_completed':require(e['processed']==j+1,'processed counter');byrole[role]=e
   byattempt[j+1].append((events.index(e),kind))
 previous=-1
 for seq in byattempt.values():
  ordered=sorted(seq);require([kind for _,kind in ordered]==list(kinds),'intent/invoke/return/process chronology')
  require(ordered[0][0]>previous,'sequential calls never overlap journal stages');previous=ordered[-1][0]
 commits=[(j,e) for j,e in enumerate(events) if e['event']=='committed']
 expected_paths=[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(tuple(dict.fromkeys(r[0] for r in expected)),range(4),('natural','bos'),BOUNDARIES)]
 require([e['path'] for _,e in commits]==expected_paths,'32 ordered boundary commits')
 for at,event in commits:
  c,w,i,b=Path(event['path']).stem.split('__');w=int(w)
  complete=[(j,e) for j,e in enumerate(events) if e['event']=='forward_completed' and tuple(e['identity'][k] for k in ('corpus','window','identity','boundary'))==(c,w,i,b)]
  require(len(complete)==(14 if (w,i,b)==(0,'natural','000') else 12) and max(j for j,_ in complete)<at,'commit after actual second matched delivery and all suffix returns')
 return {'final_sha256':chain,'counts':dict(counts),'byrole':byrole,'events':events}

def validate_freeze_digest(manifest):
 # Exact five-key runner serialization; no added keys or alternate encoding.
 binding={'sources':manifest['source_hashes'],'inputs':manifest['input_dependencies'],
  'metadata':manifest['metadata_dependencies'],'runtime':manifest['runtime_versions'],
  'precision':manifest['precision_settings_before']}
 digest=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 require(digest==manifest['freeze_digest'],'independently reconstructed five-key freeze digest')
 return digest


def validate_native_gate_controls(operation,arm):
 gate=operation['native_gate_checks'][10]
 require(gate['source_context_exact'] is True,'native gate source context control')
 require(gate['unaffected_context_exact'] is True,'native gate unaffected context control')
 require(math.isfinite(gate['norm_match_error']) and gate['norm_match_error']<=1e-6,'frozen native gate norm control')
 # T20 uses original full context edits; it never enables legacy scaling.
 require(gate['legacy_errors']=={},'T20 has no legacy scaling path')
 unchanged=gate['unchanged_bypass']
 require(type(unchanged) is bool and type(gate['active']) is bool and gate['active'] is (not unchanged),'native gate activation/bypass consistency')
 if arm in ('baseline','current','native_identity'):require(unchanged,'unedited native context bypass')
 # The frozen hook's unaffected_context_exact means "not unchanged OR exact".
 # Only the actual unchanged branch implies whole-context equality.
 if unchanged:
  context=operation['context_records'][10]
  require(context['current']==context['delivered'],'unchanged native context exact fingerprint')


def provenance(out,review):
 m=read(out/'manifest.json');validate_freeze_digest(m);require(m['status']=='completed' and m['execution_type']=='planned_sequential_segments','completed planned study')
 require(review['run_manifest_sha256']==sha(out/'manifest.json'),'reviewed run manifest')
 for p in out.rglob('*'):require(p.name not in ('merge_failure.json','checkpoint_failure.json','failure_state.npz') and not p.name.endswith('__partial.npz'),'no failure/partial markers')
 for k,n in (('forward_attempts',776),('forward_passes',776),('fully_processed_forwards',776),('evaluation_forwards',384),('suffix_forwards',384),('anchor_forwards',8),('calibration_forwards',0)):
  require(m[k]==n,'complete count '+k)
 require(m['all_final_guards_passed'] is True and m['parameter_versions_unchanged'] is True,'final composite guards')
 require(m['model_hash_before']==m['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527','model parameter digest')
 require(m['precision_settings_before']==m['precision_settings_after'],'precision guard')
 require(m['precision_settings_before']['float32_matmul_precision']=='highest' and m['precision_settings_before']['cuda_matmul_allow_tf32'] is False,'full FP32 matmul scope')
 require(m['runtime_versions']=={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'},'pinned native runtime')
 require(m['model_id']=='openai-community/gpt2' and m['revision']=='75e09b43581151bd1d9ef6700faa605df408979f' and m['dtype']=='float32' and m['attention']=='sdpa','model/revision/precision/backend pin')
 require(m['physical_execution_lengths']==[512] and m['calibration_fit'] is False,'physical512 and zero fitting')
 require(m['source_hashes']['analyze_post_write_norm.py']==ANALYZER_SHA and m['calibration_map_digest']==F_DIGEST,'analyzer/F frozen')
 for name,h in m['source_hashes'].items():require(sha(out/'frozen_sources'/name)==h,'source '+name)
 for name,h in m['input_dependencies'].items():require(sha(out/name)==h,'input '+name)
 for name,h in m['metadata_dependencies'].items():require(sha(out/'metadata_dependencies'/name.removeprefix('../').replace('/','__'))==h,'metadata '+name)
 inventory=read(out/'numeric_inventory.json')
 expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(CORPORA,range(4),('natural','bos'),BOUNDARIES)}
 require(set(inventory)==expected=={p.relative_to(out).as_posix() for p in (out/'inputs').iterdir()},'exact64 shard universe')
 for name,v in inventory.items():
  p=out/name;require(not p.is_symlink() and v=={'sha256':sha(p),'bytes':p.stat().st_size},'shard inventory '+name)
 merger_chain='0'*64;copied=[]
 for line in (out/'journal.jsonl').read_bytes().splitlines():
  e=json.loads(line);require(line==json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),'canonical merger journal')
  require(e['previous_sha256']==merger_chain and e['event']=='copied_verified_boundary_shard','merger chain/copy-only chronology');merger_chain=hashlib.sha256(line).hexdigest()
  require(e['segment']==('A' if any(e['path'].startswith('inputs/'+c+'__') for c in CORPORA[:2]) else 'B'),'copy segment roster')
  require({k:e[k] for k in ('bytes','sha256')}==inventory[e['path']],'merger actual copy identity');copied.append(e['path'])
 require(copied==[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(CORPORA,range(4),('natural','bos'),BOUNDARIES)],'64ordered composite copies')
 reports={};require(set(m['original_segments'])=={'A','B'},'both original segments')
 for label,corpora in (('A',CORPORA[:2]),('B',CORPORA[2:])):
  base=out/'execution_provenance'/label;sm=read(base/'manifest.json');binding=m['original_segments'][label]
  for name,key in (('manifest.json','manifest_sha256'),('numeric_inventory.json','numeric_inventory_sha256'),('journal.jsonl','journal_sha256')):require(sha(base/name)==binding[key],'original evidence '+label+name)
  validate_freeze_digest(sm)
  require(sm['status']==sm['phase']=='completed' and sm['segment']==label and sm['execution_type']=='planned_segment','original complete segment')
  require(sm['all_final_guards_passed'] is True and sm['parameter_versions_unchanged'] is True,'segment final guards')
  require(sm['model_hash_before']==sm['model_hash_after']==m['model_hash_before'],'segment parameter guards')
  require(sm['precision_settings_before']==sm['precision_settings_after']==m['precision_settings_before'],'segment precision guards')
  for key in ('source_hashes','input_dependencies','metadata_dependencies','freeze_digest','runtime_versions','model_id','revision','dtype','attention','arithmetic_runtime','calibration_map_digest'):
   require(sm[key]==m[key],'shared segment '+key)
  for key,n in (('forward_attempts',388),('forward_passes',388),('fully_processed_forwards',388),('evaluation_forwards',192),('suffix_forwards',192),('anchor_forwards',4),('calibration_forwards',0)):require(sm[key]==n,'segment count '+key)
  si=read(base/'numeric_inventory.json');require(set(si)=={n for n in inventory if any(n.startswith('inputs/'+c+'__') for c in corpora)},'segment shard membership')
  for name,v in si.items():require(v==inventory[name],'original shard byte identity')
  j=journal(base/'journal.jsonl',roster(corpora));require(j['final_sha256']==binding['journal_chain_sha256'],'bound journal chain')
  for e in j['events']:
   if e['event']=='committed':require({k:e[k] for k in ('bytes','sha256')}==inventory[e['path']],'commit bytes identity')
  reports[label]=j
 require(read(out/'execution_provenance/B/manifest.json')['previous_segment_A']=={'manifest_sha256':sha(out/'execution_provenance/A/manifest.json'),'numeric_inventory_sha256':sha(out/'execution_provenance/A/numeric_inventory.json')},'B pins completed A')
 return m,inventory,reports
def norm(x):
 x=np.asarray(x,np.float64);return np.sqrt(np.sum(x*x,axis=-1,dtype=np.float64))

def match_arrays(base,left,right):
 for x in (base,left,right):require(x.shape==(512,768) and x.dtype==np.float32 and np.isfinite(x).all(),'finite FP32 write operands')
 require(exact(base[0],left[0]) and exact(base[0],right[0]),'source full candidate byte identity')
 b=base.astype(np.float64);ds=[a.astype(np.float64)-b for a in (left,right)];ns=[norm(d) for d in ds];target=np.minimum(*ns)
 k=(32*768*2.**-53)/(1-32*768*2.**-53);values=[];scales=[];budgets=[]
 for full,d,n in zip((left,right),ds,ns):
  scale=np.zeros(512,np.float64);nz=n!=0;scale[nz]=target[nz]/n[nz]
  require(np.isfinite(scale).all() and ((scale>=0)&(scale<=1)).all(),'scale finite unit interval')
  value=np.empty_like(base);zero=scale==0;one=scale==1;middle=~(zero|one)
  value[zero]=base[zero];value[one]=full[one]
  product=scale[middle,None]*d[middle];added=b[middle]+product;value[middle]=added.astype(np.float32)
  budget=2.**-24*norm(np.abs(b)+scale[:,None]*np.abs(d))+.5*np.sqrt(768)*2.**-149+k*(norm(b)+n+target)
  values.append(value);scales.append(scale);budgets.append(budget)
 return {'values':values,'scales':scales,'full_norms':ns,'target':target,'budgets':budgets}

def check_matching(norms,cmp):
 left,right=(norms[a] for a in ARMS[-2:]);target=left['target_norm'];a=left['actual_write_delta_norm'];b=right['actual_write_delta_norm'];gap=np.abs(a-b)
 require(exact(target,right['target_norm']),'common target bytes')
 for z,own,other in ((left,norms['remove_full'],norms['orth_full']),(right,norms['orth_full'],norms['remove_full'])):
  require(exact(z['own_full_norm'],own['actual_write_delta_norm']) and exact(z['other_full_norm'],other['actual_write_delta_norm']),'actual full-arm reference norms')
  require(exact(target,np.minimum(z['own_full_norm'],z['other_full_norm'])),'min full norm recipe')
  scale=np.zeros(512,np.float64);nz=z['own_full_norm']!=0;scale[nz]=target[nz]/z['own_full_norm'][nz]
  require(exact(scale,z['lambda']),'scale exact division/no epsilon')
  require(exact(z['signed_target_residual'],z['actual_write_delta_norm']-target),'actual signed residual')
  require(exact(z['absolute_pair_gap'],gap),'pair gap both delivered arms')
  require(np.all(np.abs(z['signed_target_residual'])<=z['rounding_budget']),'perquery target budget')
  require(np.count_nonzero(z['actual_write_delta_norm'][scale==0])==0,'exact zero endpoint norm')
  require(exact(z['actual_write_delta_norm'][scale==1],z['own_full_norm'][scale==1]),'exact unit endpoint norm')
  require(z['actual_write_delta_norm'][0]==target[0]==scale[0]==0,'source query identity')
 require(exact(left['rounding_budget'],right['other_rounding_budget']) and exact(right['rounding_budget'],left['other_rounding_budget']),'mirrored budgets')
 require(np.all(gap<=left['rounding_budget']+right['rounding_budget']),'perquery pair budget')
 domains={}
 for name,s in (('all_nonsource',slice(1,512)),('primary_scored',slice(1,511))):
  den=float(np.sqrt(np.mean(target[s]**2)));num=float(np.sqrt(np.mean(gap[s]**2)))
  require(np.count_nonzero(a[s])==np.count_nonzero(b[s])==0 if den==0 else num<=1e-5*den,'fixed RMS quality '+name)
  domains[name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None}
 return domains

def controls(shard,m,tokens,journal_events,counts):
 meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'))
 anchors=b=='000' and w==0 and i=='natural';expected=[]
 for arm in ARMS:
  if arm=='baseline' and anchors:expected.extend((a,None) for a in ('native_repeat','native_identity'))
  if arm!='baseline':expected.append(('source_q0',arm))
  if arm=='current':expected.append(('current',arm))
  expected.append(('suffix',arm))
 require([(x['kind'],x.get('arm')) for x in shard['checks']]==expected,'exact runtime control roster/order')
 for check in shard['checks']:
  flags=[v for k,v in check.items() if k.endswith(('_exact','_passed'))]
  require(flags and all(type(v) is bool and v for v in flags),'all hard/exact flags true');counts[check['kind']]+=1
 source=shard['source_reference'];require(set(source)==set(range(12)),'source references all12layers')
 for src in source.values():
  require(set(src)=={'q','k','v','hidden'} and all(x.shape==(1,768) and x.dtype==np.float32 and np.isfinite(x).all() for x in src.values()),'actual source reference arrays')
 expected_main=set(ARMS)|({'native_repeat','native_identity'} if anchors else set())
 require(set(shard['operations'])==expected_main and set(shard['suffix_operations'])==set(ARMS),'operation rosters')
 for variant,ops in (('main',shard['operations']),('suffix',shard['suffix_operations'])):
  rotated=tokens.copy();rotated[128:]=np.roll(tokens[128:],1);x=rotated if variant=='suffix' else tokens
  for arm,op in ops.items():
   if arm=='native_repeat':require(op=={},'unhooked repeat no hook evidence');continue
   role=(c,w,i,b,arm,'anchor' if arm=='native_identity' else variant);event=journal_events[role]
   require(event['hidden_fingerprint']['sha256']==shard['metrics' if variant=='main' else 'suffix'][arm]['final_hidden_sha256'],'journal hidden identity')
   require(event['hidden_fingerprint']['shape']==[1,512,768] and event['hidden_fingerprint']['dtype']=='torch.float32','journal full hidden shape')
   require(event['logits_fingerprint']['shape']==[512,50257] and event['logits_fingerprint']['dtype']=='torch.float32','journal retains full512 vocabulary logits fingerprint')
   require((op['projection_checks'],op['context_checks'],op['query_checks'])==(8,8,8),'all hooked layer counts')
   for field in ('projection_records','context_records','mask_records','native_diag'):require(set(op[field])==set(range(4,12)),'all8layer witnesses')
   require(exact(op['source_fingerprints'],nested_fp(source)),'source runtime fingerprint reference')
   for layer in range(4,12):
    p=op['projection_records'][layer];require(p['zeroed']==(b=='111'),'source V boundary')
    for name in ('q','k','nonsource_v'):require(p['before'][name]==p['after'][name],'Q/K/nonsourceV preserved')
    require(p['after']['source_v']==fp(source[layer]['v']),'actual sourceV fingerprint')
    if b=='111':require(exact(source[layer]['v'],np.zeros_like(source[layer]['v'])),'sourceV exact positivezero boundary')
    else:require(p['before']['source_v']==p['after']['source_v'],'native sourceV preserved')
    mask=op['mask_records'][layer];require(mask['model_argument_preserved'] is True and mask['kind'] in ('implicit_causal','preserved_explicit'),'native mask preserved')
    diag=op['native_diag'][layer]
    for key in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):require(diag[key]<=1e-4,'context hard threshold')
    require(diag['min_mass']>=0 and diag['max_mass']<=1+1e-6,'mass roundoff guard')
   validate_native_gate_controls(op,arm)
   wc=op['write_checks'];require(wc['source_write_bytes_exact'] is True and wc['native_mask_preserved'] is True and wc['proposed_context_and_actual_write_distinct'] is True,'write exact controls')
   key=op['write_cache_key'];require(all(key[k]==meta[k] for k in ('corpus','window','identity','boundary')),'cache input identity')
   require(key['variant']==variant and key['actual_token_sha256']==fp(x)['sha256'] and key['F_digest']==F_DIGEST and key['freeze_digest']==m['freeze_digest'],'variant/token/F/freeze cache identity')
   if arm in ARMS[-2:]:
    require(wc['paired_actual_delivery_complete']==(arm==ARMS[-1]),'paired flag only after second actual matched delivery')
    require(wc['counterpart_norm_scope']==('actual_delivered' if arm==ARMS[-1] else 'expected_CPU_rounded_not_yet_delivered'),'counterpart chronology scope')
   if variant=='suffix':
    main=shard['operations'][arm]
    for field in ('mask_records','source_fingerprints','prefix_context_fingerprints'):require(exact(op[field],main[field]),'suffix runtime prefix '+field)
   elif arm in ARMS:require(exact(op['gate_fingerprints'],nested_fp(shard['gates'][arm])),'main actual gate fingerprint')

def gate_reconstruction(shard,alpha,u,v,cmp):
 for arm,gates in shard['gates'].items():
  if arm=='native_repeat':require(gates=={},'native repeat gates empty');continue
  require(set(gates)=={10},'L10 gate only');g=gates[10];values=shard['gate_norms'][arm][10]
  require(set(values)==GATE_KEYS,'gate norm field roster')
  uf=u.astype(np.float32);vf=v.astype(np.float32)
  s=(g['b'].T.astype(np.float64)-np.log(np.arange(1,512))[:,None]-alpha.astype(np.float32))@uf.astype(np.float64)
  cmp.check(g['s'],s,'coefficient_reconstruction',1e-4)
  target=g['b'] if arm in ('baseline','current','native_identity') else (g['b'].T-g['s'][:,None]*(uf if arm.startswith('remove') else vf)).T
  require(exact(target,g['target_b']),'full edit exact FP32 elementwise gate proposal')
  sigmoid=np.exp(-np.logaddexp(0,-g['target_b'].astype(np.float64)))
  cmp.check(g['proposed_target_mass'],sigmoid,'gate_probability_reconstruction',3e-6)
  unchanged=arm in ('baseline','current','native_identity') or exact(g['target_b'],g['b'])
  require(exact(g['target_mass'],g['mass'] if unchanged else g['proposed_target_mass']),'unchanged native mass bypass')
  want={'s':g['s'],'hypothetical_pc1_norm':norm((g['s'][:,None]*uf).astype(np.float64)),
   'hypothetical_orthogonal_norm':norm((g['s'][:,None]*vf).astype(np.float64)),
   'logit_delta_norm':norm((g['target_b']-g['b']).T.astype(np.float64)),
   'mass_delta_norm':norm((g['target_mass']-g['mass']).T.astype(np.float64)),
   'source_v_norm':norm(shard['source_reference'][10]['v'].reshape(12,64))}
  want.update({name+'_norm':norm(g[name].T) for name in ('b','target_b','mass','target_mass')})
  for name,a in want.items():cmp.check(values[name],a,'gate_norm_reconstruction',1e-10,1e-12)

def snapshot(shard,cmp):
 meta=shard['metadata'];selected=(meta['corpus'],meta['window'],meta['identity'])==('wiki_validation',0,'natural')
 if not selected:require(shard['snapshots']==shard['readouts']=={},'no unselected full snapshots');return 0
 require(set(shard['snapshots'])==set(shard['readouts'])==set(ARMS),'six predesignated full snapshots')
 b=meta['boundary'];snaps=shard['snapshots'];base=snaps['baseline']['baseline_write']
 recipe=match_arrays(base[0],snaps['remove_full']['proposed_full_write'][0],snaps['orth_full']['proposed_full_write'][0])
 for arm,s in snaps.items():
  incoming=s['incoming_reference'];op=shard['operations'][arm];g=shard['gates'][arm][10]
  require(exact(incoming['gate_state'],g),'snapshot gate/full saved gate')
  for name in ('q','k','v'):require(incoming[name].shape==(1,12,512,64) and incoming[name].dtype==np.float32 and np.isfinite(incoming[name]).all(),'snapshot QKV')
  for name in ('baseline_write','proposed_full_write','returned_write'):require(s[name].shape==(1,512,768) and s[name].dtype==np.float32 and np.isfinite(s[name]).all(),'actual projected write shape')
  require(exact(s['baseline_write'],base),'actual baseline write common bytes')
  for name in ('q','k','v','mask','actual_context','stable_mixture'):require(exact(incoming[name],snaps['baseline']['incoming_reference'][name]),'same-state baseline incoming '+name)
  for name in ('q','k','v'):require(exact(incoming[name][0,:,0].reshape(1,768),shard['source_reference'][10][name]),'snapshot source reference')
  for name in ('q','k'):require(fp(incoming[name].transpose(0,2,1,3).reshape(1,512,768))==op['projection_records'][10]['after'][name],'snapshot full actual QK projection witness')
  require(fp(incoming['v'][:,:,1:].transpose(0,2,1,3).reshape(1,511,768))==op['projection_records'][10]['after']['nonsource_v'],'snapshot actual nonsourceV projection witness')
  require(fp(incoming['mask'])==op['mask_records'][10]['diagnostic_effective'],'snapshot native effective mask fingerprint')
  require(fp(incoming['actual_context'])==op['context_records'][10]['current'] and fp(incoming['delivered_context'])==op['context_records'][10]['delivered'],'snapshot actual/proposed context fingerprint')
  for name in ('actual_context','delivered_context'):require(incoming[name].shape==(1,512,768) and incoming[name].dtype==np.float32 and np.isfinite(incoming[name]).all(),'finite actual/proposed FP32 contexts')
  mask=incoming['mask'];require(mask.dtype==np.float32 and mask.shape==(1,1,512,512) and not np.isnan(mask).any() and not np.isposinf(mask).any(),'effective mask finite-or-negative-infinity')
  q,k,val=(incoming[name].astype(np.float64) for name in ('q','k','v'));scores=q@k.swapaxes(-1,-2)/8+mask.astype(np.float64)
  def softmax(z):
   z=z-z.max(-1,keepdims=True);e=np.exp(z);return e/e.sum(-1,keepdims=True)
  prob=softmax(scores);context=(prob@val).transpose(0,2,1,3).reshape(1,512,768)
  cmp.check(incoming['actual_context'],context,'full_snapshot_context_reconstruction',1e-4)
  stable=(softmax(scores[:,:,1:,1:])@val[:,:,1:])[0]
  mass=prob[0,:,1:,1:].sum(-1);non=scores[0,:,1:,1:];mx=non.max(-1)
  bg=mx+np.log(np.exp(non-mx[...,None]).sum(-1))-scores[0,:,1:,0]
  cmp.check(incoming['stable_mixture'],stable,'full_snapshot_stable_mixture',1e-4)
  cmp.check(g['mass'],mass,'full_snapshot_gate_mass',3e-6);cmp.check(g['b'],bg,'full_snapshot_gate_logits',1e-4)
  current=incoming['actual_context'].reshape(512,12,64).transpose(1,0,2)[:,1:]
  delivered=incoming['delivered_context'].reshape(512,12,64).transpose(1,0,2)[:,1:]
  unchanged=op['native_gate_checks'][10]['unchanged_bypass']
  predicted=current if unchanged else current+(g['target_mass']-g['mass'])[...,None]*(incoming['stable_mixture']-incoming['v'][0,:,:1])
  require(exact(predicted,delivered),'exact full context proposal FP32 formula')
  wc=op['write_checks']
  for name in ('baseline_write','proposed_full_write','returned_write'):require(wc[name]==fp(s[name]),'snapshot actual write fingerprint')
  require(exact(s['interpolation'],shard['write_norms'][arm]),'snapshot interpolation arrays')
  for name,a in (('actual_write_norm',norm(s['returned_write'][0])),('actual_write_delta_norm',norm(s['returned_write'][0].astype(np.float64)-base[0])),('proposed_full_write_delta_norm',norm(s['proposed_full_write'][0].astype(np.float64)-base[0])),('proposed_pre_context_delta_norm',norm((incoming['delivered_context'].astype(np.float64)-incoming['actual_context'].astype(np.float64))[0]))):
   # pre-context difference must use FP64 views (not FP32 subtract).
   if name=='proposed_pre_context_delta_norm':a=norm((incoming['delivered_context'].astype(np.float64)-incoming['actual_context'].astype(np.float64))[0])
   cmp.check(shard['write_norms'][arm][name],a,'full_snapshot_write_norm_reconstruction',1e-10,1e-12)
  if arm in ARMS[-2:]:
   j=0 if arm.startswith('remove') else 1;ref=snaps['remove_full' if j==0 else 'orth_full']
   require(exact(s['incoming_reference'],ref['incoming_reference']) and exact(s['proposed_full_write'],ref['proposed_full_write']),'cached actual same-state full candidate')
   require(exact(s['returned_write'][0],recipe['values'][j]),'CPU interpolation and returned device bytes')
   for name,a in (('target_norm',recipe['target']),('lambda',recipe['scales'][j]),('rounding_budget',recipe['budgets'][j])):cmp.check(shard['write_norms'][arm][name],a,'full_snapshot_interpolation_recipe',1e-10,1e-12)
  else:require(exact(s['returned_write'],s['proposed_full_write']),'unmatched actual full write delivery')
  for name,a in (('actual_context_norm',current.astype(np.float64)),('delivered_context_norm',delivered.astype(np.float64)),('context_delta_norm',delivered.astype(np.float64)-current.astype(np.float64)),('stable_mixture_norm',incoming['stable_mixture'].astype(np.float64))):
   cmp.check(shard['gate_norms'][arm][10][name],norm(a.transpose(1,0,2).reshape(511,768)),'full_snapshot_context_norm_reconstruction',1e-10,1e-12)
 return 6
def rows_from_shard(shard,native,rows,anchors,normrows,cmp):
 meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'));raw=shard['metrics'];baseline=raw['baseline']
 expected=set(ARMS)|({'native_repeat','native_identity'} if b=='000' and w==0 and i=='natural' else set())
 require(set(raw)==set(shard['gates'])==set(shard['gate_norms'])==set(shard['write_norms'])==expected,'main six arms plus only scheduled anchors')
 for key in ('suffix','suffix_gates','suffix_gate_norms','suffix_write_norms'):require(set(shard[key])==set(ARMS),'suffix arm roster')
 for arm,z in raw.items():
  for key in METRICS:
   a=z[key];dtype=np.int64 if key=='pred' else np.bool_ if key.startswith('agreement') else np.float32
   require(a.shape==(511,) and a.dtype==dtype and np.isfinite(a).all(),'metric shape/dtype/finite')
  require(((z['pred']>=0)&(z['pred']<50257)).all(),'prediction range')
  require(exact(z['agreement_c'],z['pred']==native['pred']) and exact(z['agreement_boundary'],z['pred']==baseline['pred']),'prediction agreements')
  if arm=='baseline':require(np.count_nonzero(z['kl_boundary'])==0,'baseline KL zero')
  if arm in ('current','native_repeat','native_identity'):
   require(all(exact(z[k],baseline[k]) for k in (*METRICS,'final_hidden_sha256')),'baseline/current/anchor actual metrics hidden')
  for view,(lo,hi) in VIEWS.items():
   vals={'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),
    'delta_nll_c':float((z['loss'][lo:hi].astype(np.float64)-native['loss'][lo:hi].astype(np.float64)).mean()),
    'delta_nll_boundary':float((z['loss'][lo:hi].astype(np.float64)-baseline['loss'][lo:hi].astype(np.float64)).mean()),
    **{key:float(z[key][lo:hi].astype(np.float64).mean()) for key in METRICS if key not in ('loss','pred')}}
   (rows if arm in ARMS else anchors)[c,i,w,view,b+'__'+arm]=vals
  if arm not in ARMS:continue
  suffix=shard['suffix'][arm]
  for key in METRICS:
   n=127 if key=='loss' else 128;require(exact(suffix[key],z[key][:n]),'suffix exact metric prefix')
  require(exact(suffix['target_indices'],np.arange(1,128,dtype=np.int64)) and exact(suffix['logit_query_indices'],np.arange(128,dtype=np.int64)),'physical target/query indexing')
  require(exact(shard['suffix_gates'][arm],{l:{k:a[:127] if a.ndim==1 else a[:,:127] for k,a in g.items()} for l,g in shard['gates'][arm].items()}),'suffix exact gate prefix')
  for family in ('gate','write'):
   vals=shard[family+'_norms'][arm];vals=vals[10] if family=='gate' else vals
   require(set(vals)==(GATE_KEYS if family=='gate' else COMMON|(MATCH if arm in ARMS[-2:] else set())),'norm key roster')
   for name,a in vals.items():
    shape=(12,) if name=='source_v_norm' else (511,) if family=='gate' else (512,)
    require(a.shape==shape and a.dtype==(np.float32 if name=='s' else np.float64) and np.isfinite(a).all(),'norm shape/dtype/finite')
    if name not in ('s','signed_target_residual'):require((a>=0).all(),'nonnegative norm/budget')
    suffix_vals=shard['suffix_'+family+'_norms'][arm];suffix_vals=suffix_vals[10] if family=='gate' else suffix_vals
    if family=='gate':require(exact(suffix_vals[name],a if name=='source_v_norm' else a[:127]),'suffix gate norm bytes')
    else:require(exact(suffix_vals[name][:128],a[:128]) and suffix_vals[name].shape==(512,) and suffix_vals[name].dtype==np.float64 and np.isfinite(suffix_vals[name]).all(),'suffix actual write norms prefix and full domain')
   if family=='write':
    require(vals['actual_write_delta_norm'][0]==vals['proposed_full_write_delta_norm'][0]==vals['proposed_pre_context_delta_norm'][0]==0,'source norm zeros')
    if arm in ('baseline','current'):require(all(np.count_nonzero(vals[k])==0 for k in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm')),'baseline/current zero norms')
   for view,(lo,hi) in VIEWS.items():
    if family=='gate':lo-=1;hi-=1
    selected={name:a if name=='source_v_norm' else a[lo:hi] for name,a in vals.items()}
    agg={name:{'scope':'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions','mean':float(a.astype(np.float64).mean()),'rms':float(np.sqrt(np.mean(a.astype(np.float64)**2))),'max_abs':float(np.abs(a).max())} for name,a in selected.items()}
    normrows[family][c,i,w,view,b+'__'+arm,10]=agg
 for variant,field in (('main','write_norms'),('suffix','suffix_write_norms')):check_matching(shard[field],cmp)

def statistics(analysis,rows,anchors,normrows,cmp):
 def index(items,keys):
  out={}
  for row in items:
   key=tuple(row[k] for k in keys);require(key not in out,'unique analysis row');out[key]=row
  return out
 keyfields=('corpus','identity','window','view','mode')
 for category,want in (('window_metrics',rows),('anchors',anchors)):
  found=index(analysis[category],keyfields);require(set(found)==set(want),'exact '+category+' membership')
  for key,vals in want.items():
   for field,a in vals.items():cmp.check(a,found[key][field],'window_metrics')
 for family in ('gate','write'):
  found=index(analysis[family+'_norm_window_summaries'],(*keyfields,'layer'));want=normrows[family]
  require(set(found)==set(want) and len(want)==1152,'norm aggregate rows')
  for key,vals in want.items():
   require(set(vals)==set(found[key]['values']),'norm summary fields')
   for name,a in vals.items():
    z=found[key]['values'][name];require(z['scope']==a['scope'],'norm domain scope')
    for field in ('mean','rms','max_abs'):cmp.check(a[field],z[field],'norm_aggregates',1e-10,1e-12)
 ids=indices();specs=contrast_definitions()
 summaries=index(analysis['summaries'],('corpus','identity','view','mode'))
 contrasts=index(analysis['contrasts'],('corpus','identity','view','name'))
 interactions=index(analysis['paired_start_interactions'],('corpus','view','name'))
 expected_s={(c,i,v,a) for c,i,v,a in itertools.product(CORPORA,('natural','bos'),VIEWS,MODES)}
 expected_c={(c,i,v,name) for c,i,v,name in itertools.product(CORPORA,('natural','bos'),VIEWS,specs)}
 expected_i={(c,v,'bos_minus_natural__'+name) for c,v,name in itertools.product(CORPORA,VIEWS,specs)}
 require(set(summaries)==expected_s and set(contrasts)==expected_c and set(interactions)==expected_i,'complete 288/648/324 unique summary domains')
 def compare(a,z):
  require(z['n']==4,'paired n4')
  for name in ('mean','ci95','per_window'):cmp.check(a[name],z[name],'paired_bootstrap')
 for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
  for mode in MODES:
   z=summaries[c,i,v,mode]['metrics'];require(set(z)==set(FIELDS)|{'perplexity','ppl_ratio_c','ppl_ratio_boundary'},'summary field roster')
   for field in FIELDS:
    a=bootstrap([rows[c,i,w,v,mode][field] for w in range(4)],ids);compare(a,z[field])
    if field in ('nll','delta_nll_c','delta_nll_boundary'):
     label={'nll':'perplexity','delta_nll_c':'ppl_ratio_c','delta_nll_boundary':'ppl_ratio_boundary'}[field]
     cmp.check(math.exp(a['mean']),z[label]['value'],'ppl exponentiated paired mean')
     cmp.check([math.exp(x) for x in a['ci95']],z[label]['ci95'],'ppl exponentiated intervals')
  for name,spec in specs.items():
   z=contrasts[c,i,v,name];require(all(z[k]==spec[k] for k in spec),'independent contrast definition')
   xs=[math.fsum(k*rows[c,i,w,v,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
   compare(bootstrap(xs,ids),z)
 for c,v in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   z=interactions[c,v,'bos_minus_natural__'+name]
   require(z['terms']==spec['terms'] and z['scope']=='paired_start_interaction_secondary','paired starts coefficient identity')
   xs=[math.fsum(k*(rows[c,'bos',w,v,mode]['nll']-rows[c,'natural',w,v,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
   compare(bootstrap(xs,ids),z)
 require((len(rows),len(anchors),len(summaries),len(contrasts),len(interactions))==(1152,24,288,648,324),'fixed full statistical counts')
 require(analysis['statistical_scope']['shared_indices_sha256']==fp(ids)['sha256'],'analysis shared bootstrap matrix')
 return {'window_rows':1152,'anchor_rows':24,'summaries':288,'contrasts':648,'paired_starts':324,'contrast_definitions':27}
def matching_analysis(shard,analysis_lookup,cmp):
 meta=shard['metadata'];key=tuple(meta[k] for k in ('corpus','window','identity','boundary'))
 for variant,field in (('main','write_norms'),('suffix','suffix_write_norms')):
  saved=analysis_lookup[(*key,variant)];vals=shard[field];domains=check_matching(vals,cmp)
  for name,z in domains.items():
   for metric,a in z.items():
    if a is None:require(saved['domains'][name][metric] is None,'undefined zero RMS ratio null')
    else:cmp.check(a,saved['domains'][name][metric],'matching_RMS_diagnostics',1e-10,1e-12)
  left,right=(vals[a] for a in ARMS[-2:]);target=left['target_norm'];gap=np.abs(left['actual_write_delta_norm']-right['actual_write_delta_norm'])
  pos=target[1:]>0;relative=gap[1:][pos]/target[1:][pos]
  cmp.check(float(gap.max()),saved['maximum_absolute_pair_gap'],'matching_gap',1e-10,1e-12)
  require(saved['zero_target_queries']==int(np.count_nonzero(target==0)),'zero target count')
  quantiles=np.quantile(relative,[0,.25,.5,.75,.95,.99,1]).tolist() if relative.size else None
  if quantiles is None:require(saved['positive_target_relative_gap_quantiles'] is None,'undefined relative quantiles')
  else:cmp.check(quantiles,saved['positive_target_relative_gap_quantiles'],'relative_gap_quantiles',1e-10,1e-12)
  require(saved['quantized_to_zero_positive_targets']==[int(np.count_nonzero((z['actual_write_delta_norm']==0)&(target>0))) for z in (left,right)],'quantized positive targets retained')

def audit(args):
 started=time.time();policy_path=ROOT/'review/post_write_norm_numerical_audit_policy.json';policy=read(policy_path)
 require(sha(policy_path)==POLICY_SHA,'frozen N policy')
 supplement=ROOT/'review/post_write_norm_numerical_audit_resource_supplement.json'
 require(sha(supplement)=='d2e52e5f74b43ed87ac4f5bbb7017c6344b68e1dcca29008b5ce5a81c0caee2b','resource-only supplement')
 require(read(supplement)['new_value']==384*1024**2 and read(supplement)['all_other_policy_fields_unchanged'] is True,'failure cap amendment only')
 review=read(args.execution_review)
 require(review['approved_numerical_execution'] is True and review['auditor_sha256']==sha(__file__) and review['policy_sha256']==POLICY_SHA,'later exact source/policy execution approval')
 # Explicit external input audit is a reviewed provenance dependency; N also
 # checks current token/start/suffix bytes and selection-manifest attestations.
 require(sha(args.input_audit)==review['input_audit_sha256'],'independent input/history audit binding')
 out=args.run;m,inventory,journals=provenance(out,review)
 for name,h in policy['frozen_bindings'].items():
  target=out/'frozen_sources'/name if name in m['source_hashes'] else ROOT/name
  require(sha(target)==h,'frozen policy dependency '+name)
 checkpoint_report=original_checkpoint_provenance(args,m,inventory);cmp=Comparisons(policy)
 analysis_path=out/'post_write_norm_analysis.json';analysis=read(analysis_path)
 require(analysis['scope']['analyzer_sha256']==ANALYZER_SHA and analysis['scope']['raw_manifest_sha256']==sha(out/'manifest.json') and analysis['scope']['raw_inventory_sha256']==sha(out/'numeric_inventory.json'),'frozen analysis exact study binding')
 require(analysis['scope']['source_hashes']=={Path(n).name:v['sha256'] for n,v in inventory.items()},'analysis64shard source binding')
 matching={tuple(z[k] for k in ('corpus','window','identity','boundary','variant')):z for z in analysis['matching_quality']}
 require(len(matching)==len(analysis['matching_quality'])==128,'matching quality128unique variants')
 maps,map_count=load_npz(out/'calibration_reference/calibration_maps.npz');directions,direction_count=load_npz(out/'calibration_reference/calibration_directions.npz')
 dm=read(out/'data_manifest.json');reference=out/'calibration_reference'
 require(sha(reference/'manifest.json')==dm['calibration_reference_manifest_sha256']=='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060','frozen T18 calibration manifest')
 for name in ('calibration_maps.npz','calibration_directions.npz'):require(sha(reference/name)==dm['calibration_reference_files'][name],'frozen F calibration file')
 digest=hashlib.sha256()
 for layer in sorted(maps['F']):
  for name,a in sorted(maps['F'][layer].items()):digest.update(f'{layer}:{name}:torch.{a.dtype.name}:{tuple(a.shape)}'.encode());digest.update(np.ascontiguousarray(a).tobytes())
 require(digest.hexdigest()==F_DIGEST,'independent frozen F map digest')
 alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
 require(all(a.dtype==np.float64 and a.shape==(12,) and np.isfinite(a).all() for a in (alpha,u,v)),'F directions finite FP64')
 require(abs(norm(u)-1)<=1e-12 and abs(norm(v)-1)<=1e-12 and abs(u@v)<=1e-12,'F orthonormality')
 # Only current-study token file; torch never loads a model/checkpoint here.
 import torch
 torch.set_num_threads(1);require(not torch.cuda.is_initialized(),'CPU-only audit')
 items=torch.load(out/'tokens.pt',weights_only=True,map_location='cpu')
 tokens={(item['corpus'],item['window']):item['tokens'].numpy().copy() for item in items}
 require(set(tokens)==set(itertools.product(CORPORA,range(4))),'16 current natural token records')
 passage_records={(r['corpus'],r['window']):r for r in dm['passages']};require(set(passage_records)==set(tokens),'selected passage provenance roster')
 native={};rows={};anchors={};normrows={'gate':{},'write':{}};counts=collections.Counter();arrays=map_count+direction_count;snapshots=0
 for name in sorted(inventory):
  shard,narrays=load_npz(out/name);arrays+=narrays;meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'))
  require(shard['schema']=='post_write_norm_explicit_v1' and meta['length']==512 and name==f'inputs/{c}__{w}__{i}__{b}.npz','canonical raw identity')
  x=tokens[c,w].copy();require(x.dtype==np.int64 and x.shape==(512,),'physical512 int64 tokens')
  r=passage_records[c,w];require(fp(x)['sha256']==r['window_token_sha256'],'natural selected hash')
  if i=='bos':x[0]=dm['literals']['start_special']['id'];require(fp(x)['sha256']==r['bos_window_token_sha256'],'BOS tokenzero replacement hash')
  require(fp(x)['sha256']==meta['token_sha256'],'shard token binding')
  j=journals['A' if c in CORPORA[:2] else 'B']
  for e in j['events']:
   if e['event']=='forward_intent' and all(e['identity'][k]==meta[k] for k in ('corpus','window','identity','boundary')):
    tx=x.copy()
    if e['identity']['phase']=='suffix':tx[128:]=np.roll(x[128:],1)
    require(e['token_fingerprint']==fp(tx),'intent actual natural/BOS/suffix bytes')
  if b=='000':native[c,w,i]=shard['metrics']['baseline']
  rows_from_shard(shard,native[c,w,i],rows,anchors,normrows,cmp)
  controls(shard,m,x,j['byrole'],counts);gate_reconstruction(shard,alpha,u,v,cmp)
  snapshots+=snapshot(shard,cmp);matching_analysis(shard,matching,cmp)
  # Actual whole512-logit fingerprints are independently compared for exact
  # controls; unsaved logits and context arrays are never reconstructed.
  base_event=j['byrole'][c,w,i,b,'baseline','main']
  for arm in ('current',*(() if not (b=='000' and w==0 and i=='natural') else ('native_repeat','native_identity'))):
   event=j['byrole'][c,w,i,b,arm,'anchor' if arm.startswith('native_') else 'main']
   require(event['hidden_fingerprint']==base_event['hidden_fingerprint'] and event['logits_fingerprint']==base_event['logits_fingerprint'],'full512 actual exact-control journal fingerprints')
 require(snapshots==12,'all12 full snapshots')
 result=statistics(analysis,rows,anchors,normrows,cmp)
 report={'status':'passed','schema':'T20_independent_numerical_audit_v1','auditor_sha256':sha(__file__),'policy_sha256':POLICY_SHA,
  'resource_supplement_sha256':sha(supplement),'execution_review_sha256':sha(args.execution_review),'input_audit_sha256':sha(args.input_audit),
  'run_manifest_sha256':sha(out/'manifest.json'),'analysis_sha256':sha(analysis_path),'statistics':result,'arrays_verified':arrays,
  'versioned_checkpoint_and_resource_evidence':checkpoint_report,'full_snapshots_reconstructed':snapshots,'actual_vocabulary_readout_rows_retained':48,'comparison_counts':cmp.report(),'controls':dict(counts),
  'journals':{label:{'final_sha256':j['final_sha256'],'counts':j['counts']} for label,j in journals.items()},
  'scope':{'raw_reconstruction':'12 selected full Q/K/V gates/context and actual CPU projected-write interpolation; projection and hidden-to-logit audit remain separate.',
   'ordinary_shards':'Saved gate/norm/source arrays independently checked. Unsaved ordinary contexts/hidden/logits use recorded runtime fingerprints and flags only, not full raw reconstruction.',
   'source_offsets_and_history':'Current selected records bound by source/input hashes and separately reviewed independent input/history audit; no historical restricted token IDs reopened.',
   'causal_limits':'Approximate Euclidean write norm matching; no direction, Jacobian, mediation, uniqueness, generation or fidelity claim.'},
  'model_forwards':0,'pretrained_checkpoint_access':False,'gpu_used':False,'file_deletions':0,
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-started}
 require(report['peak_rss_bytes']<=3*1024**3,'CPU RSS3GiB limit')
 return report
def expected_checkpoint_layout(corpora):
 # The frozen runner checkpoints twice before model loading, then after each
 # of32 boundary commits, then once more with final guards/canonical manifest.
 groups=[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES)]
 return [{'name':f'{j+1:06d}','committed':min(max(j-1,0),32),
   'forward_passes':sum(14 if '__0__natural__000.npz' in name else 12 for name in groups[:min(max(j-1,0),32)])}
  for j in range(35)]


def original_checkpoint_provenance(args,manifest,inventory):
 report={}
 for label,base,corpora in (('A',args.segment_a,CORPORA[:2]),('B',args.segment_b,CORPORA[2:])):
  require(base.is_dir() and not base.is_symlink(),'original segment directory')
  copied=args.run/'execution_provenance'/label
  for path in base.rglob('*'):require(path.name not in ('failure_state.npz','checkpoint_failure.json','merge_failure.json') and not path.name.endswith('__partial.npz'),'original failure evidence blocks accepted completion')
  for name in ('manifest.json','numeric_inventory.json','journal.jsonl','disk_checks.json'):require(sha(base/name)==sha(copied/name),'original/copy immutable '+name)
  sm=read(base/'manifest.json');si=read(base/'numeric_inventory.json')
  require(sm['failure_limit_bytes']==384*1024**2 and sm['normal_limit_bytes']==512*1024**2 and sm['disk_reserve_bytes']==2*1024**3,'prospective amended resource caps')
  groups=[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES)]
  for name,v in si.items():require(sha(base/name)==v['sha256'] and (base/name).stat().st_size==v['bytes'] and v==inventory[name],'original shard preserved')
  layout=expected_checkpoint_layout(corpora)
  checkpoints=sorted((base/'checkpoints').iterdir());require([p.name for p in checkpoints]==[z['name'] for z in layout],'two preload32boundary+final versioned checkpoints')
  for j,cp in enumerate(checkpoints):
   ci=read(cp/'numeric_inventory.json');n=layout[j]['committed']
   require(set(ci)==set(groups[:n]),'versioned exact committed prefix')
   for name,v in ci.items():require(v==si[name],'checkpoint source bytes')
   cm=read(cp/'manifest.json');require(cm['status']!='failed','no failed checkpoint')
   progress=read(cp/'progress.json');require(progress['forward_attempts']==progress['forward_passes']==progress['fully_processed_forwards'],'checkpoint counts separated but allprocessed beforecommit')
   expected=layout[j]['forward_passes']
   require(progress['forward_passes']==expected,'checkpoint exact completed roster prefix')
  require(sha(checkpoints[-1]/'manifest.json')==sha(base/'manifest.json'),'canonical final versioned manifest exact')
  disks=read(base/'disk_checks.json')
  for d in disks:
   require(d['free_bytes']>=d['required_free_bytes'] and d['reserve_bytes']==2*1024**3,'recorded reserve gate')
   require(d['used_bytes']+d['projected_bytes']<=512*1024**2 and d['failure_write'] is False,'normal cap/no failure write')
   require(d['remaining_normal_cap']==max(0,512*1024**2-d['used_bytes']),'complete remaining cap reserve')
   require(d['required_free_bytes']==d['reserve_bytes']+d['outstanding_other_caps']+384*1024**2+d['remaining_normal_cap'],'all outstanding caps')
  require(sm['peak_process_rss_kib']*1024<=3*1024**3,'recorded segment CPU peak')
  report[label]={'versioned_checkpoints':35,'disk_checks':len(disks),'raw_shards':len(si),'source_and_copied_metadata_exact':True}
 checkpoints=sorted((args.run/'checkpoints').iterdir());require([p.name for p in checkpoints]==['000001','000002','000003'],'merger checkpoint sequence')
 for cp,n in zip(checkpoints,(32,64,64)):require(len(read(cp/'numeric_inventory.json'))==n,'merger committed copy prefix')
 return report
def self_test():
 started=time.time();tests=[];policy=read(ROOT/'review/post_write_norm_numerical_audit_policy.json');cmp=Comparisons(policy)
 require(sha(ROOT/'review/post_write_norm_numerical_audit_policy.json')==POLICY_SHA,'prospective policy identity')
 ids=indices();specs=contrast_definitions();require(len(roster(CORPORA))==776 and len(roster(CORPORA[:2]))==388,'planned counts')
 require(not exact(np.array([0.],np.float32),np.array([-0.],np.float32)),'signedzero distinct')
 tests.append('policy_bootstrap_contrast_and776_roster_pins')
 frozen={'source_hashes':{'a.py':'a'*64},'input_dependencies':{'tokens.pt':'b'*64},
  'metadata_dependencies':{},'runtime_versions':{'numpy':'1.26.4'},'precision_settings_before':{'float32_matmul_precision':'highest'}}
 binding={'sources':frozen['source_hashes'],'inputs':frozen['input_dependencies'],'metadata':{},
  'runtime':frozen['runtime_versions'],'precision':frozen['precision_settings_before']}
 frozen['freeze_digest']=hashlib.sha256(json.dumps(binding,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 validate_freeze_digest(frozen)
 bad=copy.deepcopy(frozen);bad['source_hashes']['a.py']='c'*64
 try:validate_freeze_digest(bad)
 except AssertionError:pass
 else:raise AssertionError('bad freeze digest accepted')
 gate={'source_context_exact':True,'unaffected_context_exact':True,'norm_match_error':0.,
  'legacy_errors':{},'unchanged_bypass':True,'active':False}
 op={'native_gate_checks':{10:gate},'context_records':{10:{'current':{'sha256':'d'*64},'delivered':{'sha256':'d'*64}}}}
 validate_native_gate_controls(op,'baseline')
 edited=copy.deepcopy(op);edited['native_gate_checks'][10].update(unchanged_bypass=False,active=True)
 edited['context_records'][10]['delivered']={'sha256':'e'*64}
 validate_native_gate_controls(edited,'remove_full')
 native_rejected=0
 for key,value in (('source_context_exact',False),('unaffected_context_exact',False),('norm_match_error',2e-6),('legacy_errors',{'conditional_error':0.})):
  bad=copy.deepcopy(op);bad['native_gate_checks'][10][key]=value
  try:validate_native_gate_controls(bad,'baseline')
  except AssertionError:native_rejected+=1
 require(native_rejected==4,'native gate hostile flags rejected')
 tests.append('five_key_freeze_digest_and_native_gate_controls_narrow_hostiles')
 for corpora in (CORPORA[:2],CORPORA[2:]):
  layout=expected_checkpoint_layout(corpora)
  require(len(layout)==35 and [z['name'] for z in layout]==[f'{j:06d}' for j in range(1,36)],'exact35checkpoint names')
  require([z['committed'] for z in layout]==[0,0,*range(1,33),32],'two preload checkpoints then32commits thenfinal')
  require([z['forward_passes'] for z in layout[:4]]==[0,0,14,26] and layout[-2]['forward_passes']==layout[-1]['forward_passes']==388,'checkpoint counter prefixes')
  counts=collections.Counter(r[:4] for r in roster(corpora))
  ordered=list(itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES))
  for z in layout:
   require(z['forward_passes']==sum(counts[key] for key in ordered[:z['committed']]),'checkpoint layout agrees independent full roster')
 tests.append('in_memory35checkpoint_schema_two_preloads32commits_final')
 # Independent closed-form coefficient outcomes for a fixed six-arm vector.
 value={'baseline':3,'current':3,'remove_full':5,'orth_full':7,'remove_write_matched':11,'orth_write_matched':13}
 expected=(2,4,-2,8,10,-2,6,6,0)
 for boundary in BOUNDARIES:
  for (name,spec),want in zip(((k,z) for k,z in specs.items() if k.startswith(boundary+'__')),expected):
   require(sum(k*value[mode.split('__')[1]] for mode,k in spec['terms'].items())==want,'closedform nine contrasts')
 tests.append('explicit_nine_algebraic_contrasts_closedform')
 # In-memory production-shaped writes; include signedzero/subnormals.
 rng=np.random.default_rng(20261021);base=rng.normal(0,.2,(512,768)).astype(np.float32)
 left=base.copy();right=base.copy();left[1:]+=np.float32(.1);right[1:]-=np.float32(.2)
 base[0,0]=left[0,0]=right[0,0]=np.float32(-0.)
 result=match_arrays(base,left,right);norms={}
 for arm,full in (('remove_full',left),('orth_full',right)):
  norms[arm]={'actual_write_delta_norm':norm(full.astype(np.float64)-base.astype(np.float64))}
 for j,arm in enumerate(ARMS[-2:]):
  actual=norm(result['values'][j].astype(np.float64)-base.astype(np.float64));other=1-j
  norms[arm]={'target_norm':result['target'],'lambda':result['scales'][j],
   'own_full_norm':result['full_norms'][j],'other_full_norm':result['full_norms'][other],
   'rounding_budget':result['budgets'][j],'other_rounding_budget':result['budgets'][other],
   'actual_write_delta_norm':actual,'signed_target_residual':actual-result['target']}
 gap=np.abs(norms[ARMS[-2]]['actual_write_delta_norm']-norms[ARMS[-1]]['actual_write_delta_norm'])
 for arm in ARMS[-2:]:norms[arm]['absolute_pair_gap']=gap
 check_matching(norms,cmp)
 for full in (base,left):
  r=match_arrays(base,full,full)
  require(exact(r['values'][0],full) and exact(r['values'][1],full),'wholezero/wholeone identity')
 require(exact(result['values'][0][0],base[0]),'sourcebytes')
 bad=copy.deepcopy(norms);bad[ARMS[-2]]['lambda'][1]=0
 try:check_matching(bad,cmp)
 except AssertionError:pass
 else:raise AssertionError('wronglambda accepted')
 tests.append('production_shape_interpolation_exact_endpoints_source_and_wronglambda_rejection')
 # Full ordered388call stream with32commits, constructed entirely in memory.
 chain='0'*64;lines=[];roles=roster(CORPORA[:2])
 for j,role in enumerate(roles):
  identity=dict(zip(('corpus','window','identity','boundary','arm','phase'),role))
  for kind in ('forward_intent','forward_attempt','model_returned','forward_completed'):
   e={'previous_sha256':chain,'event':kind,'identity':identity}
   e['next_attempt' if kind=='forward_intent' else 'attempt']=j+1
   if kind in ('model_returned','forward_completed'):e['completed']=j+1
   if kind=='forward_completed':e['processed']=j+1
   body=json.dumps(e,sort_keys=True,separators=(',',':')).encode();chain=hashlib.sha256(body).hexdigest();lines.append(body+b'\n')
  if role[-2:]==('orth_write_matched','suffix'):
   c,w,i,b=role[:4];e={'previous_sha256':chain,'event':'committed','path':f'inputs/{c}__{w}__{i}__{b}.npz','bytes':1,'sha256':'0'*64}
   body=json.dumps(e,sort_keys=True,separators=(',',':')).encode();chain=hashlib.sha256(body).hexdigest();lines.append(body+b'\n')
 j=journal(b''.join(lines),roles);require(j['counts']['committed']==32,'32commit journal')
 try:journal(b''.join(lines[:-1]),roles)
 except AssertionError:pass
 else:raise AssertionError('missing commit accepted')
 tests.append('in_memory388journal_canonicalchain_order32commits_and_missingcommit_rejection')
 # Codec roundtrip of signedzero/subnormal fixture without touching disk.
 sample=np.array([0.,-0.,2.**-149],np.float32);header={'schema':'wiki_gate_lossless_npz_v1','object':{'type':'tensor','array':'a','shape':[3],'dtype':'torch.float32','sha256':fp(sample)['sha256']}}
 stream=io.BytesIO();np.savez(stream,metadata=np.frombuffer(json.dumps(header).encode(),np.uint8),a=sample);stream.seek(0)
 actual,n=load_npz(stream);require(n==1 and exact(actual,sample),'independent codec byte identity')
 tests.append('header_free_npz_signedzero_subnormal_roundtrip')
 # Complete in-memory statistical wire shape:1152+24rows,288summaries,
 #648contrasts and324pairedstarts, all independently indexed and checked.
 rows={};anchors={};nr={'gate':{},'write':{}}
 for c,i,w,view,mode in itertools.product(CORPORA,('natural','bos'),range(4),VIEWS,MODES):
  b,arm=mode.split('__');nll=float(value[arm]+w/16+(i=='bos')/8+(b=='111')/4)
  z={'nll':nll,'delta_nll_c':float(value[arm]-3+(b=='111')/4),'delta_nll_boundary':float(value[arm]-3),
   'kl_c':float((value[arm]-3)**2/256),'kl_boundary':float((value[arm]-3)**2/256),
   'agreement_c':.5,'agreement_boundary':.5}
  key=(c,i,w,view,mode);rows[key]=z
  for family in nr:nr[family][(*key,10)]={'fixture':{'scope':'scored_query_positions','mean':1.,'rms':1.,'max_abs':1.}}
 for c,view,arm in itertools.product(CORPORA,VIEWS,('native_repeat','native_identity')):
  anchors[c,'natural',0,view,'000__'+arm]=dict(rows[c,'natural',0,view,'000__baseline'])
 wire={'window_metrics':[{**dict(zip(('corpus','identity','window','view','mode'),key)),**z} for key,z in rows.items()],
  'anchors':[{**dict(zip(('corpus','identity','window','view','mode'),key)),**z} for key,z in anchors.items()],
  'summaries':[],'contrasts':[],'paired_start_interactions':[],
  'statistical_scope':{'shared_indices_sha256':fp(ids)['sha256']}}
 for family in nr:wire[family+'_norm_window_summaries']=[{**dict(zip(('corpus','identity','window','view','mode','layer'),key)),'values':z} for key,z in nr[family].items()]
 for c,i,view in itertools.product(CORPORA,('natural','bos'),VIEWS):
  for mode in MODES:
   metrics={field:bootstrap([rows[c,i,w,view,mode][field] for w in range(4)],ids) for field in FIELDS}
   for field,label in (('nll','perplexity'),('delta_nll_c','ppl_ratio_c'),('delta_nll_boundary','ppl_ratio_boundary')):
    metrics[label]={'value':math.exp(metrics[field]['mean']),'ci95':[math.exp(q) for q in metrics[field]['ci95']]}
   wire['summaries'].append({'corpus':c,'identity':i,'view':view,'mode':mode,'metrics':metrics})
  for name,spec in specs.items():
   xs=[math.fsum(k*rows[c,i,w,view,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
   wire['contrasts'].append({'corpus':c,'identity':i,'view':view,'name':name,**spec,**bootstrap(xs,ids)})
 for c,view in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   xs=[math.fsum(k*(rows[c,'bos',w,view,mode]['nll']-rows[c,'natural',w,view,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
   wire['paired_start_interactions'].append({'corpus':c,'view':view,'name':'bos_minus_natural__'+name,
    'terms':spec['terms'],'scope':'paired_start_interaction_secondary',**bootstrap(xs,ids)})
 stats=statistics(wire,rows,anchors,nr,cmp)
 wire['contrasts'][0]['mean']+=.01
 try:statistics(wire,rows,anchors,nr,cmp)
 except AssertionError:pass
 else:raise AssertionError('corrupt contrast accepted')
 tests.append('complete_statistical_wire1152_24_288_648_324_and_corruptcontrast_rejection')
 return {'status':'passed_prospective_core_fixture','auditor_sha256':sha(__file__),'tests':tests,
  'statistical_wire_counts':stats,'contrast_definitions':27,'bootstrap_indices_sha256':fp(ids)['sha256'],'planned_roster':776,
  'model_forwards':0,'actual_T20_outputs_read':False,'checkpoint_or_header_access':False,'gpu_used':False,'file_deletions':0,
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-started}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('run','segment-a','segment-b','execution-review','input-audit','result'):parser.add_argument('--'+name,type=Path)
 parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
 if args.self_test:print(json.dumps(self_test()));return
 require(all((args.run,args.segment_a,args.segment_b,args.execution_review,args.input_audit,args.result)),'complete reviewed execution arguments')
 require(not args.result.exists(),'new exclusive result path')
 try:put(args.result,audit(args))
 except BaseException as error:
  put(args.result.with_name(args.result.stem+'_failure_'+str(time.time_ns())+'.json'),{'status':'failed','error':str(error),'error_type':type(error).__name__,'traceback':traceback.format_exc(),'auditor_sha256':sha(__file__),'raw_study_unchanged':True})
  raise

if __name__=='__main__':main()
