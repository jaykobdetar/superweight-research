"""Prospective independent T21 saved-array/timing auditor. Actual audit HOLD.

No scientific analyzer import. T20 independent numerical routines are copied
with unchanged arithmetic; new native-only/support/timing ABI is explicit.
Small self-tests use synthetic arrays/metadata only; no model/tokenizer/GPU.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,collections,copy,hashlib,io,itertools,json,math,re,resource,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
POLICY_SHA='fe306aac7106e94af060048d197afbcef8e9dcb60d1a8c302d7539b9c52bc89a'
ANALYZER_SHA='54415feb2fade4da1569f92ba78172285eb955c1a72bccfb43ffc60edc3f760c'
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
ACTUAL_AUDIT_APPROVED=True
APPROVED_SOURCE_HASHES={'GPT2_QUERY_WRITE_TIMING_PROPOSAL_v4.md': 'a17e3e7e8150935950d58df1d2b266dac2bf91a6c65c4a8539d7e28fcf3e0819', 'GPT2_QUERY_WRITE_TIMING_RESOURCE_ADDENDUM.md': '37ef260913814273a2e8187c79f6e8253c0ac3a484bcbbca778210716a59ef7e', 'analyze_post_write_norm.py': 'e3d36eb3584c7785d702bfa857481ae0c62bd0ac7c986e6e99ff2c42e790f45d', 'analyze_query_write_timing.py': '54415feb2fade4da1569f92ba78172285eb955c1a72bccfb43ffc60edc3f760c', 'fresh_inputs.py': '3bfd96e50e84a09640b6c199768312bec59ef146973f0aa8c1aff3046a130f0a', 'fresh_inputs_v2.py': '7a0d251b577c6af917d92ba5ccf6b3b6f57df6ff2d2cae78f539206baf330519', 'gate_inputs.py': '437f2ef8df2e6d6ac5cc55fc95bf163790200bace66a72ab5bf855da34626b6d', 'gate_native_value.py': 'b206ecb881bf0a74f9b86166e2d173153f7e26716fa6ebd00b41fd8ef2d1bcfd', 'gate_native_value_storage.py': 'b4b7f8932a93136fcccc2ad423bd2bc89fd57c84c052273f15ef36cbdd63d78e', 'gate_query_write_timing.py': 'f9f2eb64795b6b40ce19a2b6187e4dbb7263d838ff152bf2d081b06260a0f0ab', 'gate_wiki_storage.py': '83937417a830e700679169efb3e39a7e001e93484138b42376cde66fb419aa06', 'gate_write_norm.py': '62e8b4c67a522043a6cb315b97c7cee34f84dc853cfcb4121679344b4b06445f', 'gpt2_gate_native_value.py': '7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f', 'gpt2_gate_subspace.py': '6e2ff4492a1f39c9e598f4434fdc58b2ac9a68f8e976323039ec42a05297b76e', 'gpt2_layer_writes.py': 'b6c166e03d792401b25b30696f1f1a4413679f5741b26bd0d00824822c82be24', 'gpt2_query_source.py': 'c1deeaf4035ff74ea8d7f724b93feadfc593d4785272921cf201728fb802f4a5', 'gpt2_query_write_timing.py': '539565fb028eb9fd217d49ed43b03e5ba6869c8ecc0d3ed2a98c0eca66064c92', 'gpt2_regrowth_kv.py': 'b85e7bf18ccc04d96b121bebbef8272d435da9553153f9a003a1982dbba087b8', 'gpt2_writes.py': 'fc1a98cbeb9e3d6e779ef0f61cc7c934be2cd2d127da24dfea2f146015a00181', 'llama_source_kv.py': '84a96ee919c422f72df04c9a55765368095e0ca2c03265c74a1f1afe20f1a1e2', 'merge_query_write_timing.py': '9dec8ae282642cf529509f633312d7c525852b663b4c7906b28b495863c62295', 'native_prefix_diagnostic.py': '561c3342d4810ece74a9f183c83c3e7bd81ab952eabf4c21b1d977a81ad40d4d', 'native_value_portable_draft.py': '46516322c9841bc6acac67b034c6f1187ba3c07d205879525f403129aa7e6bd4', 'post_write_norm_inputs.py': 'e13f93e5a905ea10c4ec98fbb0f50a0df991d50899707b503a7806103fe46f48', 'post_write_norm_portable.py': '0eaaa298745dc653c2ad48761d8dbf3abbb1be466b29226a410c06da1c782bde', 'post_write_norm_selector.py': '34acd93d09f49b341e77bc5ae7da711706cdfd0bc5b6d1f52808acfa07eee5d4', 'post_write_norm_storage.py': 'f7d8068b85f4521c7d7409c88de0b1e9a2c6fc582238525db975c2df22ead0d3', 'query_write_timing_inputs.py': '153ddc6b4b4cd91c2dc1385c6397ddb9c743a0553ea9405e43fc05f47e8aced9', 'query_write_timing_storage.py': '3e91ac9e069b4172985b8944c2294fff9d7632764585b9acb0fea39b7a332f88', 'run_transfer.py': '16a0240a8c679de05c8875e50304b1bc9aabe90e2c4c376b2fc89eeaec0de044'}
RESOURCE_SUPPLEMENT_SHA='8d4082f52c12e1e2dd27e603a24b782bcb14b8c90f42b0b194434b1384d9b793'
EFFECTIVE_FAILURE_LIMIT_BYTES=1024**3
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched_early','remove_write_matched_later','remove_write_matched_both','orth_write_matched_early','orth_write_matched_later','orth_write_matched_both')
SUPPORTS=('early','later','both')
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')
TIMING_ROWS=(1,126,127,510,511)
READOUT_ROWS=(1,126,127,510)
BOUNDARIES=('000',)
MODES=tuple('000__'+a for a in ARMS)
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
FIELDS=('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
METRICS=('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')
COMMON={'actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'}
MATCH={'target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap'}
GATE_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
def is_matched(arm):return '_write_matched_' in arm
def support_mask(support):
 require(support in SUPPORTS,'known support');q=np.arange(512)
 return ((q>=1)&(q<127)) if support=='early' else (q>=127) if support=='later' else q>=1


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
  require(len(complete)==(22 if (w,i,b)==(0,'natural','000') else 20) and max(j for j,_ in complete)<at,'commit after actual second matched delivery and all suffix returns')
 return {'final_sha256':chain,'counts':dict(counts),'byrole':byrole,'events':events}

def provenance(out,review):
 m=read(out/'manifest.json');validate_freeze_digest(m);require(m['status']=='completed' and m['execution_type']=='planned_sequential_segments','completed planned study')
 require(review['run_manifest_sha256']==sha(out/'manifest.json'),'reviewed run manifest')
 require(m['input_dependencies']==review['input_dependencies'] and m['metadata_dependencies']==review['metadata_dependencies'],'independently accepted input and metadata dependency bindings')
 for p in out.rglob('*'):require(p.name not in ('merge_failure.json','checkpoint_failure.json','failure_state.npz') and not p.name.endswith('__partial.npz'),'no failure/partial markers')
 for k,n in (('forward_attempts',648),('forward_passes',648),('fully_processed_forwards',648),('evaluation_forwards',320),('suffix_forwards',320),('anchor_forwards',8),('calibration_forwards',0)):
  require(m[k]==n,'complete count '+k)
 require(m['all_final_guards_passed'] is True and m['parameter_versions_unchanged'] is True,'final composite guards')
 require(m['model_hash_before']==m['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527','model parameter digest')
 require(m['precision_settings_before']==m['precision_settings_after'],'precision guard')
 require(m['precision_settings_before']['float32_matmul_precision']=='highest' and m['precision_settings_before']['cuda_matmul_allow_tf32'] is False,'full FP32 matmul scope')
 require(m['runtime_versions']=={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'},'pinned native runtime')
 require(m['model_id']=='openai-community/gpt2' and m['revision']=='75e09b43581151bd1d9ef6700faa605df408979f' and m['dtype']=='float32' and m['attention']=='sdpa','model/revision/precision/backend pin')
 require(m['physical_execution_lengths']==[512] and m['calibration_fit'] is False,'physical512 and zero fitting')
 require(m['boundaries']==['000'] and m['arms']==list(ARMS) and m['score_views']=={'full':[2,511],'early':[2,127],'later':[128,511]},'exact native ten-arm score/exec domains')
 require(m['source_hashes']['analyze_query_write_timing.py']==ANALYZER_SHA and m['source_hashes']==APPROVED_SOURCE_HASHES and m['calibration_map_digest']==F_DIGEST,'analyzer/F frozen')
 for name,h in m['source_hashes'].items():require(sha(out/'frozen_sources'/name)==h,'source '+name)
 for name,h in m['input_dependencies'].items():require(sha(out/name)==h,'input '+name)
 for name,h in m['metadata_dependencies'].items():require(sha(out/'metadata_dependencies'/name.removeprefix('../').replace('/','__'))==h,'metadata '+name)
 inventory=read(out/'numeric_inventory.json')
 expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(CORPORA,range(4),('natural','bos'),BOUNDARIES)}
 require(set(inventory)==expected=={p.relative_to(out).as_posix() for p in (out/'inputs').iterdir()},'exact32 shard universe')
 for name,v in inventory.items():
  p=out/name;require(not p.is_symlink() and v=={'sha256':sha(p),'bytes':p.stat().st_size},'shard inventory '+name)
 merger_chain='0'*64;copied=[]
 for line in (out/'journal.jsonl').read_bytes().splitlines():
  e=json.loads(line);require(line==json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),'canonical merger journal')
  require(e['previous_sha256']==merger_chain and e['event']=='copied_verified_boundary_shard','merger chain/copy-only chronology');merger_chain=hashlib.sha256(line).hexdigest()
  require(e['segment']==('A' if any(e['path'].startswith('inputs/'+c+'__') for c in CORPORA[:2]) else 'B'),'copy segment roster')
  require({k:e[k] for k in ('bytes','sha256')}==inventory[e['path']],'merger actual copy identity');copied.append(e['path'])
 require(copied==[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(CORPORA,range(4),('natural','bos'),BOUNDARIES)],'32ordered composite copies')
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
  for key,n in (('forward_attempts',324),('forward_passes',324),('fully_processed_forwards',324),('evaluation_forwards',160),('suffix_forwards',160),('anchor_forwards',4),('calibration_forwards',0)):require(sm[key]==n,'segment count '+key)
  si=read(base/'numeric_inventory.json');require(set(si)=={n for n in inventory if any(n.startswith('inputs/'+c+'__') for c in corpora)},'segment shard membership')
  for name,v in si.items():require(v==inventory[name],'original shard byte identity')
  j=journal(base/'journal.jsonl',roster(corpora));require(j['final_sha256']==binding['journal_chain_sha256'],'bound journal chain')
  for e in j['events']:
   if e['event']=='committed':require({k:e[k] for k in ('bytes','sha256')}==inventory[e['path']],'commit bytes identity')
  reports[label]=j
 require(read(out/'execution_provenance/B/manifest.json')['previous_segment_A']=={'manifest_sha256':sha(out/'execution_provenance/A/manifest.json'),'numeric_inventory_sha256':sha(out/'execution_provenance/A/numeric_inventory.json')},'B pins completed A')
 return m,inventory,reports

def controls(shard,m,tokens,journal_events,counts):
 meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'))
 anchors=b=='000' and w==0 and i=='natural';expected=[]
 for arm in ARMS:
  if arm=='baseline' and anchors:expected.extend((a,None) for a in ('native_repeat','native_identity'))
  if arm!='baseline':expected.append(('source_q0',arm))
  if arm=='current':expected.append(('current',arm))
  if arm.endswith('_later'):expected.extend((('later_only_prefix',arm),('later_only_prefix',arm)))
  expected.append(('suffix',arm))
 require([(x['kind'],x.get('arm')) for x in shard['checks']]==expected,'exact runtime control roster/order')
 later_checks=[z for z in shard['checks'] if z['kind']=='later_only_prefix']
 require([(z['arm'],z['variant']) for z in later_checks]==[(a,v) for a in ARMS if a.endswith('_later') for v in ('main','suffix')],'later prefix main/suffix check order')
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
   if is_matched(arm):
    require(wc['paired_actual_delivery_complete']==arm.startswith('orth'),'paired flag only after second actual matched delivery')
    require(wc['counterpart_norm_scope']==('actual_delivered_same_support' if arm.startswith('orth') else 'expected_CPU_rounded_not_yet_delivered_same_support'),'counterpart chronology scope')
   if variant=='suffix':
    main=shard['operations'][arm]
    for field in ('mask_records','source_fingerprints','prefix_context_fingerprints'):require(exact(op[field],main[field]),'suffix runtime prefix '+field)
   elif arm in ARMS:require(exact(op['gate_fingerprints'],nested_fp(shard['gates'][arm])),'main actual gate fingerprint')

def rows_from_shard(shard,native,rows,anchors,normrows,cmp):
 meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'));raw=shard['metrics'];baseline=raw['baseline']
 expected=set(ARMS)|({'native_repeat','native_identity'} if b=='000' and w==0 and i=='natural' else set())
 require(set(raw)==set(shard['gates'])==set(shard['gate_norms'])==set(shard['write_norms'])==expected,'main ten arms plus only scheduled anchors')
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
   require(set(vals)==(GATE_KEYS if family=='gate' else COMMON|(MATCH if is_matched(arm) else set())),'norm key roster')
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
  require(set(found)==set(want) and len(want)==960,'norm aggregate rows')
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
 require(set(summaries)==expected_s and set(contrasts)==expected_c and set(interactions)==expected_i,'complete 240/408/204 unique summary domains')
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
   z=contrasts[c,i,v,name];require(all(z[k]==spec[k] for k in spec),'independent contrast definition');require(z['primary']==(spec['primary_view']==v),'primary later view only')
   xs=[math.fsum(k*rows[c,i,w,v,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
   compare(bootstrap(xs,ids),z)
 for c,v in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   z=interactions[c,v,'bos_minus_natural__'+name]
   require(z['terms']==spec['terms'] and z['scope']=='paired_start_interaction_secondary','paired starts coefficient identity')
   xs=[math.fsum(k*(rows[c,'bos',w,v,mode]['nll']-rows[c,'natural',w,v,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
   compare(bootstrap(xs,ids),z)
 require((len(rows),len(anchors),len(summaries),len(contrasts),len(interactions))==(960,24,240,408,204),'fixed full statistical counts')
 require(analysis['statistical_scope']['shared_indices_sha256']==fp(ids)['sha256'],'analysis shared bootstrap matrix')
 return {'window_rows':960,'anchor_rows':24,'summaries':240,'contrasts':408,'paired_starts':204,'contrast_definitions':17}

def snapshot(shard,cmp):
 meta=shard['metadata'];selected=(meta['corpus'],meta['window'],meta['identity'])==('wiki_validation',0,'natural')
 if not selected:require(shard['snapshots']==shard['readouts']=={},'no unselected full snapshots');return 0
 require(set(shard['snapshots'])==set(shard['readouts'])==set(ARMS),'ten predesignated full snapshots')
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
  if is_matched(arm):
   j=0 if arm.startswith('remove') else 1;ref=snaps['remove_full' if j==0 else 'orth_full']
   require(exact(s['incoming_reference'],ref['incoming_reference']) and exact(s['proposed_full_write'],ref['proposed_full_write']),'cached actual same-state full candidate')
   mask=support_mask(arm.rsplit('_',1)[-1]);want=base[0].copy();want[mask]=recipe['values'][j][mask]
   require(exact(s['support_active'],mask),'saved exact support mask')
   require(exact(s['returned_write'][0],want),'CPU fixed interpolation support splice and returned device bytes')
   require(wc['full_matched_array']==fp(recipe['values'][j]),'full fixed matched-array commitment')
   for name,a in (('target_norm',np.where(mask,recipe['target'],0.)),('lambda',np.where(mask,recipe['scales'][j],0.)),('rounding_budget',np.where(mask,recipe['budgets'][j],0.))):cmp.check(shard['write_norms'][arm][name],a,'full_snapshot_interpolation_recipe',1e-10,1e-12)
  else:require(exact(s['returned_write'],s['proposed_full_write']),'unmatched actual full write delivery')
  for name,a in (('actual_context_norm',current.astype(np.float64)),('delivered_context_norm',delivered.astype(np.float64)),('context_delta_norm',delivered.astype(np.float64)-current.astype(np.float64)),('stable_mixture_norm',incoming['stable_mixture'].astype(np.float64))):
   cmp.check(shard['gate_norms'][arm][10][name],norm(a.transpose(1,0,2).reshape(511,768)),'full_snapshot_context_norm_reconstruction',1e-10,1e-12)
 return 10

def original_checkpoint_provenance(args,manifest,inventory):
 report={}
 for label,base,corpora in (('A',args.segment_a,CORPORA[:2]),('B',args.segment_b,CORPORA[2:])):
  require(base.is_dir() and not base.is_symlink(),'original segment directory')
  copied=args.run/'execution_provenance'/label
  for path in base.rglob('*'):require(path.name not in ('failure_state.npz','checkpoint_failure.json','merge_failure.json') and not path.name.endswith('__partial.npz'),'original failure evidence blocks accepted completion')
  for name in ('manifest.json','numeric_inventory.json','journal.jsonl','disk_checks.json'):require(sha(base/name)==sha(copied/name),'original/copy immutable '+name)
  sm=read(base/'manifest.json');si=read(base/'numeric_inventory.json')
  require(sm['failure_limit_bytes']==EFFECTIVE_FAILURE_LIMIT_BYTES and sm['normal_limit_bytes']==512*1024**2 and sm['disk_reserve_bytes']==2*1024**3,'prospective amended resource caps')
  groups=[f'inputs/{c}__{w}__{i}__{b}.npz' for c,w,i,b in itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES)]
  for name,v in si.items():require(sha(base/name)==v['sha256'] and (base/name).stat().st_size==v['bytes'] and v==inventory[name],'original shard preserved')
  layout=expected_checkpoint_layout(corpora)
  checkpoints=sorted((base/'checkpoints').iterdir());require([p.name for p in checkpoints]==[z['name'] for z in layout],'two preload16input+final versioned checkpoints')
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
   require(d['required_free_bytes']==d['reserve_bytes']+d['outstanding_other_caps']+EFFECTIVE_FAILURE_LIMIT_BYTES+d['remaining_normal_cap'],'all outstanding caps')
  require(sm['peak_process_rss_kib']*1024<=3*1024**3,'recorded segment CPU peak')
  report[label]={'versioned_checkpoints':19,'disk_checks':len(disks),'raw_shards':len(si),'source_and_copied_metadata_exact':True}
 checkpoints=sorted((args.run/'checkpoints').iterdir());require([p.name for p in checkpoints]==['000001','000002','000003'],'merger checkpoint sequence')
 for cp,n in zip(checkpoints,(16,32,32)):require(len(read(cp/'numeric_inventory.json'))==n,'merger committed copy prefix')
 return report

def contrast_definitions():
 # Independent coefficient construction from the prospective factorial estimands.
 out={}
 def add(name,terms,primary=None):
  out['000__'+name]={'terms':{'000__'+a:k for a,k in terms.items()},'kind':'within_native_boundary','primary_view':primary}
 for d in ('remove','orth'):
  for s in SUPPORTS:add(d+'_'+s+'_effect',{d+'_write_matched_'+s:1,'baseline':-1},'later' if s=='early' else None)
  add(d+'_full_effect',{d+'_full':1,'baseline':-1})
  add(d+'_support_interaction',{d+'_write_matched_both':1,d+'_write_matched_early':-1,d+'_write_matched_later':-1,'baseline':1})
  add(d+'_both_minus_full',{d+'_write_matched_both':1,d+'_full':-1})
 for s in SUPPORTS:add(s+'_remove_minus_orth',{'remove_write_matched_'+s:1,'orth_write_matched_'+s:-1},'later' if s=='early' else None)
 add('support_interaction_remove_minus_orth',{'remove_write_matched_both':1,'remove_write_matched_early':-1,'remove_write_matched_later':-1,'orth_write_matched_both':-1,'orth_write_matched_early':1,'orth_write_matched_later':1})
 add('both_direction_difference_minus_full',{'remove_write_matched_both':1,'orth_write_matched_both':-1,'remove_full':-1,'orth_full':1})
 require(len(out)==17 and all(sum(x['terms'].values())==0 for x in out.values()),'17 zero-sum coefficient maps')
 require(sum(x['primary_view']=='later' for x in out.values())==3,'three primary coefficient maps')
 return out

def check_matching(norms,cmp):
 domains={};baseline=norms['baseline'];full=np.minimum(norms['remove_full']['actual_write_delta_norm'],norms['orth_full']['actual_write_delta_norm'])
 for support in SUPPORTS:
  mask=support_mask(support);target=np.where(mask,full,0.)
  pair=[norms[d+'_write_matched_'+support] for d in ('remove','orth')];a,b=[z['actual_write_delta_norm'] for z in pair];gap=np.abs(a-b)
  require(exact(pair[0]['rounding_budget'],pair[1]['other_rounding_budget']) and exact(pair[1]['rounding_budget'],pair[0]['other_rounding_budget']),'mirrored masked budgets')
  for d,other,z in zip(('remove','orth'),('orth','remove'),pair):
   own=norms[d+'_full'];opp=norms[other+'_full'];both=norms[d+'_write_matched_both']
   require(exact(z['own_full_norm'],own['actual_write_delta_norm']) and exact(z['other_full_norm'],opp['actual_write_delta_norm']),'full actual reference norms')
   require(exact(z['proposed_full_write_delta_norm'],own['actual_write_delta_norm']) and exact(z['proposed_pre_context_delta_norm'],own['proposed_pre_context_delta_norm']),'unmasked hypothetical proposal diagnostics')
   require(exact(z['target_norm'],target),'masked min target bytes')
   scale=np.zeros(512,np.float64);nz=z['own_full_norm']!=0;scale[nz]=full[nz]/z['own_full_norm'][nz];scale=np.where(mask,scale,0.)
   require(exact(scale,z['lambda']) and ((scale>=0)&(scale<=1)).all(),'masked exact division no epsilon')
   for name in ('target_norm','lambda','rounding_budget','other_rounding_budget','actual_write_delta_norm'):
    require(exact(z[name],np.where(mask,both[name],0.)),'same full recipe subset '+name)
   require(exact(z['actual_write_norm'][~mask],baseline['actual_write_norm'][~mask]),'inactive write norm baseline')
   require(exact(z['signed_target_residual'],z['actual_write_delta_norm']-target) and exact(z['absolute_pair_gap'],gap),'delivered residual and pair gap')
   require(np.all(np.abs(z['signed_target_residual'])<=z['rounding_budget']),'perquery target rounding budget')
   require(np.count_nonzero(z['actual_write_delta_norm'][scale==0])==0,'zero endpoint delta')
   require(exact(z['actual_write_delta_norm'][scale==1],z['own_full_norm'][scale==1]),'unit endpoint norm')
   require(z['actual_write_delta_norm'][0]==target[0]==scale[0]==0,'source query unchanged')
  require(np.all(gap<=pair[0]['rounding_budget']+pair[1]['rounding_budget']),'perquery actual pair budget')
  domains[support]={}
  for name,idx in (('all_nonsource',np.arange(1,512)),('primary_scored',np.arange(1,511))):
   den=float(np.sqrt(np.mean(target[idx]**2)));num=float(np.sqrt(np.mean(gap[idx]**2)))
   require(np.count_nonzero(a[idx])==np.count_nonzero(b[idx])==0 if den==0 else num<=1e-5*den,'fixed support RMS quality')
   selected=idx[mask[idx]]
   def agg(v):return {'scope':'active_supported_queries','mean':float(v.mean()),'rms':float(np.sqrt(np.mean(v*v))),'max_abs':float(np.abs(v).max())}
   domains[support][name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None,'active_count':len(selected),'active_only_target':agg(target[selected]),'active_only_gap':agg(gap[selected])}
 return domains

def runtime_quality(target,a,b):
 gap=np.abs(a-b);positive=target>0;domains={}
 for name,sl in (('all_nonsource',slice(1,512)),('primary_scored',slice(1,511))):
  den=float(np.sqrt(np.mean(target[sl]*target[sl])));num=float(np.sqrt(np.mean(gap[sl]*gap[sl])))
  domains[name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None}
 return {'domains':domains,'maximum_positive_target_relative_gap':float((gap[positive]/target[positive]).max()) if positive.any() else None,'quantized_to_zero_positive_targets':[int(((x==0)&positive).sum()) for x in (a,b)],'zero_target_queries':int((~positive).sum()),'maximum_absolute_pair_gap':float(gap.max()),'maximum_per_arm_target_error':max(float(np.abs(x-target).max()) for x in (a,b))}

def compare_tree(a,b,cmp,category):
 if isinstance(a,dict):
  require(isinstance(b,dict) and set(a)<=set(b),'matching diagnostics keys')
  for k,v in a.items():compare_tree(v,b[k],cmp,category)
 elif a is None or isinstance(a,(str,bool)) or isinstance(a,(int,np.integer)):
  require(a==b,'matching categorical/count diagnostic')
 else:cmp.check(a,b,category,1e-10,1e-12)

def matching_analysis(shard,lookup,cmp,counts):
 key=tuple(shard['metadata'][k] for k in ('corpus','window','identity','boundary'))
 for variant,field,opfield in (('main','write_norms','operations'),('suffix','suffix_write_norms','suffix_operations')):
  vals=shard[field];ops=shard[opfield];domains=check_matching(vals,cmp)
  fullquality=None
  for direction in ('remove','orth'):
   full=ops[direction+'_full']['write_checks'];base=ops['baseline']['write_checks'];reference=None
   for support in SUPPORTS:
    arm=direction+'_write_matched_'+support;wc=ops[arm]['write_checks']
    require(wc['paired_actual_delivery_complete'] is (direction=='orth'),'actual pair chronological completion')
    require(wc['counterpart_norm_scope']==('actual_delivered_same_support' if direction=='orth' else 'expected_CPU_rounded_not_yet_delivered_same_support'),'actual pair counterpart scope')
    require(wc['support']==support and wc['target_scope']=='same_frozen_full_recipe_target_on_active_queries_zero_elsewhere','same support target semantics')
    require(wc['active_write_bytes_exact'] is True and wc['inactive_write_bytes_exact'] is True,'recorded support byte flags')
    require(wc['baseline_write']==base['returned_write'] and wc['proposed_full_write']==full['returned_write'],'recorded actual full recipe origin')
    require(wc['full_matched_array']['shape']==[512,768] and wc['full_matched_array']['dtype']=='torch.float32','full matched array fingerprint ABI')
    if reference is None:reference=wc['full_matched_array']
    else:require(wc['full_matched_array']==reference,'same direction full matched array all three supports')
    if fullquality is None:fullquality=wc['full_recipe_matching_quality']
    else:require(exact(fullquality,wc['full_recipe_matching_quality']),'one full recipe quality across directions/supports')
    counts['matched_arm_variant_records']+=1
   counts['direction_recipe_fingerprints']+=1
  counts['full_recipe_origins']+=1
  for support in SUPPORTS:
   saved=lookup[(*key,variant,support)];mask=support_mask(support);left,right=(vals[d+'_write_matched_'+support] for d in ('remove','orth'));target=left['target_norm'];gap=np.abs(left['actual_write_delta_norm']-right['actual_write_delta_norm']);positive=target[1:]>0
   compare_tree(domains[support],saved['domains'],cmp,'matching_RMS_diagnostics')
   expected_quality=runtime_quality(target,left['actual_write_delta_norm'],right['actual_write_delta_norm'])
   for d in ('remove','orth'):compare_tree(expected_quality,ops[d+'_write_matched_'+support]['write_checks']['matching_quality'],cmp,'recorded_matching_quality')
   if support=='both':compare_tree(expected_quality,fullquality,cmp,'recorded_full_recipe_quality')
   compare_tree({'active_queries':int(mask.sum()),'active_scored_queries':int(mask[1:511].sum()),'maximum_absolute_pair_gap':float(gap.max()),'zero_target_queries':int((target==0).sum()),'active_zero_target_queries':int((mask&(target==0)).sum()),'relative_quantile_probabilities':[0,.25,.5,.75,.95,.99,1]},saved,cmp,'matching_diagnostics')
   rel=gap[1:][positive]/target[1:][positive];q=np.quantile(rel,[0,.25,.5,.75,.95,.99,1]).tolist() if rel.size else None
   compare_tree(q,saved['positive_target_relative_gap_quantiles'],cmp,'matching_relative_quantiles')
   require(saved['quantized_to_zero_positive_targets']==[int(((z['actual_write_delta_norm']==0)&(target>0)).sum()) for z in (left,right)],'quantized positive target counts')
   counts['support_pairs']+=1;counts['RMS_domain_checks']+=2

def fingerprint_abi(value,shape):
 require(set(value)=={'shape','dtype','sha256'} and value['shape']==list(shape) and value['dtype']=='torch.float32' and re.fullmatch('[0-9a-f]{64}',value['sha256']) is not None,'FP32 recorded fingerprint ABI')

def timing_provenance(shard,tokens,journal_events,counts):
 meta=shard['metadata'];key=tuple(meta[k] for k in ('corpus','window','identity','boundary'))
 for variant,field in (('main','operations'),('suffix','suffix_operations')):
  ops=shard[field];base=ops['baseline'];baseline_prefix=base['timing_checks']['observed_model_readout_prefix'];fingerprint_abi(baseline_prefix,(127,50257))
  for arm,op in ops.items():
   if arm=='native_repeat':continue
   tc=op['timing_checks'];fps=op['timing_fingerprints'];support=arm.rsplit('_',1)[-1] if is_matched(arm) else None
   require(tc['support']==support and tc['later_query_start']==127 and tc['same_full_matched_array_across_supports'] is (support is not None),'timing support identity')
   require(tc['observed_nodes']==list(TIMING_NODES) and set(fps)==set(TIMING_NODES),'actual four observers before finish')
   for node in TIMING_NODES:
    width=2304 if node=='l11_qkv' else 768;fingerprint_abi(fps[node],(1,512,width))
    if arm=='baseline':require(node not in tc,'baseline records reference rather than fabricated guard')
    else:
     guard=tc[node];sl=[127,None] if support=='early' else [None,127] if support=='later' else [None,None] if arm in ('current','native_identity') else [0,1]
     n=385 if support=='early' else 127 if support=='later' else 512 if arm in ('current','native_identity') else 1
     require(guard['query_slice']==sl and guard['byte_exact'] is True and guard['actual']==guard['baseline'],'protected timing partition bytes')
     fingerprint_abi(guard['actual'],(1,n,width))
     if arm in ('current','native_identity'):require(fps[node]==base['timing_fingerprints'][node]==guard['actual'],'whole node current/writer identity')
    counts['anchor_timing_node_fingerprints' if arm=='native_identity' else 'science_timing_node_fingerprints']+=1
   if support=='later':
    guard=tc['later_only_prefix_logits'];require(isinstance(guard,dict) and guard['query_slice']==[0,127] and guard['byte_exact'] is True,'later actual readout prefix before finish')
    require(guard['actual']==guard['baseline']==baseline_prefix==tc['observed_model_readout_prefix']==tc['baseline_model_readout_prefix'],'actual observed prefix/operand/cache binding')
    counts['later_readout_runtime_prefix_guards']+=1
   else:require(tc['later_only_prefix_logits']=='not_required','no pending prefix check')
   if arm in ('baseline','native_identity'):require(tc['observed_model_readout_prefix']==tc['baseline_model_readout_prefix']==baseline_prefix,'baseline observer prefix cache')
   counts['anchor_timing_calls' if arm=='native_identity' else 'science_timing_calls']+=1
 selected=key[:3]==('wiki_validation',0,'natural')
 if not selected:require(shard['timing_samples']=={},'no unselected raw timing samples');return
 require(set(shard['timing_samples'])==set(ARMS),'predesignated ten timing samples')
 samples=shard['timing_samples'];readouts=shard['readouts'];base_samples=samples['baseline'];base_readout=readouts['baseline']
 for arm in ARMS:
  sample=samples[arm];z=readouts[arm];op=shard['operations'][arm];support=arm.rsplit('_',1)[-1] if is_matched(arm) else None
  require(set(sample)==set(TIMING_NODES)|{'query_rows'} and exact(sample['query_rows'],np.array(TIMING_ROWS,np.int64)),'five predetermined sampled timing rows')
  protected=list(range(5)) if arm in ('baseline','current') else [2,3,4] if support=='early' else [0,1] if support=='later' else []
  for node in TIMING_NODES:
   a=sample[node];require(a.dtype==np.float32 and a.shape==(1,5,2304 if node=='l11_qkv' else 768) and np.isfinite(a).all(),'raw sampled node fixed shape')
   require(exact(a[:,protected],base_samples[node][:,protected]),'saved protected timing row bytes');counts['raw_timing_rows']+=5
  require(z['hidden'].dtype==np.float32 and z['hidden'].shape==(1,512,768) and np.isfinite(z['hidden']).all(),'full selected hidden')
  require(z['logits'].dtype==np.float32 and z['logits'].shape==(4,50257) and np.isfinite(z['logits']).all(),'four selected vocabulary rows')
  require(exact(z['query_rows'],np.array(READOUT_ROWS,np.int64)) and exact(z['targets'],tokens[np.array(READOUT_ROWS)+1]),'selected readout target indices')
  require(z['token_sha256']==fp(tokens)['sha256'] and z['hidden_fingerprint']==fp(z['hidden']),'selected readout identity')
  event=journal_events[(*key,arm,'main')];require(event['hidden_fingerprint']==fp(z['hidden']),'saved hidden journal fingerprint')
  if arm in ('baseline','current'):require(exact(z['hidden'],base_readout['hidden']) and exact(z['logits'],base_readout['logits']),'saved selected current exact')
  if support=='later':require(exact(z['hidden'][:,:127],base_readout['hidden'][:,:127]) and exact(z['logits'][:2],base_readout['logits'][:2]),'saved later-only prefix hidden and vocabulary rows')
  counts['raw_vocabulary_rows']+=4

def expected_checkpoint_layout(corpora):
 groups=[f'inputs/{c}__{w}__{i}__000.npz' for c,w,i in itertools.product(corpora,range(4),('natural','bos'))]
 return [{'name':f'{j+1:06d}','committed':min(max(j-1,0),16),'forward_passes':sum(22 if '__0__natural__000.npz' in name else 20 for name in groups[:min(max(j-1,0),16)])} for j in range(19)]


def audit(args):
 require(ACTUAL_AUDIT_APPROVED and APPROVED_SOURCE_HASHES and RESOURCE_SUPPLEMENT_SHA,'actual T21 audit remains disabled pending independent exact-source execution review')
 started=time.time();policy_path=ROOT/'review/query_write_timing_numerical_audit_policy.json';policy=read(policy_path)
 require(sha(policy_path)==POLICY_SHA,'frozen N policy')
 supplement=ROOT/'review/query_write_timing_numerical_audit_resource_supplement.json'
 require(sha(supplement)==RESOURCE_SUPPLEMENT_SHA and read(supplement)['original_policy_sha256']==POLICY_SHA,'resource-only supplement')
 require(read(supplement)['new_value']==EFFECTIVE_FAILURE_LIMIT_BYTES and read(supplement)['all_other_policy_fields_unchanged'] is True,'failure cap amendment only')
 review=read(args.execution_review)
 require(review['approved_numerical_execution'] is True and review['auditor_sha256']==sha(__file__) and review['policy_sha256']==POLICY_SHA,'later exact source/policy execution approval')
 # Explicit external input audit is a reviewed provenance dependency; N also
 # checks current token/start/suffix bytes and selection-manifest attestations.
 require(sha(args.input_audit)==review['input_audit_sha256'] and read(args.input_audit)['status']=='passed','independent passed input/history audit binding')
 out=args.run;m,inventory,journals=provenance(out,review)
 for name,h in policy['frozen_bindings'].items():
  target=out/'frozen_sources'/name if name in m['source_hashes'] else ROOT/name
  require(sha(target)==h,'frozen policy dependency '+name)
 checkpoint_report=original_checkpoint_provenance(args,m,inventory);cmp=Comparisons(policy)
 analysis_path=out/'query_write_timing_analysis.json';analysis=read(analysis_path)
 require(analysis['scope']['analyzer_sha256']==ANALYZER_SHA and analysis['scope']['raw_manifest_sha256']==sha(out/'manifest.json') and analysis['scope']['raw_inventory_sha256']==sha(out/'numeric_inventory.json'),'frozen analysis exact study binding')
 require(analysis['scope']['source_hashes']=={Path(n).name:v['sha256'] for n,v in inventory.items()},'analysis32shard source binding')
 matching={tuple(z[k] for k in ('corpus','window','identity','boundary','variant','support')):z for z in analysis['matching_quality']}
 require(len(matching)==len(analysis['matching_quality'])==192,'matching quality192unique support pairs')
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
 native={};rows={};anchors={};normrows={'gate':{},'write':{}};counts=collections.Counter();runtime_extrema=collections.defaultdict(float);arrays=map_count+direction_count;snapshots=0
 for name in sorted(inventory):
  shard,narrays=load_npz(out/name);arrays+=narrays;meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'))
  require(shard['schema']=='query_write_timing_explicit_v1' and meta['length']==512 and name==f'inputs/{c}__{w}__{i}__{b}.npz','canonical raw identity')
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
  snapshots+=snapshot(shard,cmp);matching_analysis(shard,matching,cmp,counts)
  timing_provenance(shard,x,j['byrole'],counts)
  for field in ('operations','suffix_operations'):
   for op in shard[field].values():
    if not op:continue
    for diag in op['native_diag'].values():
     for k in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):runtime_extrema[k]=max(runtime_extrema[k],diag[k])
    runtime_extrema['gate_norm_match_error']=max(runtime_extrema['gate_norm_match_error'],op['native_gate_checks'][10]['norm_match_error'])
  # Actual whole512-logit fingerprints are independently compared for exact
  # controls; unsaved logits and context arrays are never reconstructed.
  base_event=j['byrole'][c,w,i,b,'baseline','main']
  for arm in ('current',*(() if not (b=='000' and w==0 and i=='natural') else ('native_repeat','native_identity'))):
   event=j['byrole'][c,w,i,b,arm,'anchor' if arm.startswith('native_') else 'main']
   require(event['hidden_fingerprint']==base_event['hidden_fingerprint'] and event['logits_fingerprint']==base_event['logits_fingerprint'],'full512 actual exact-control journal fingerprints')
 require(snapshots==10,'all10 full snapshots')
 require(all(counts[k]==n for k,n in (('support_pairs',192),('RMS_domain_checks',384),('matched_arm_variant_records',384),('full_recipe_origins',64),('science_timing_calls',640),('science_timing_node_fingerprints',2560),('anchor_timing_calls',4),('anchor_timing_node_fingerprints',16),('later_readout_runtime_prefix_guards',128),('raw_timing_rows',200),('raw_vocabulary_rows',40))),'exact timing/matching raw and runtime evidence counts')
 result=statistics(analysis,rows,anchors,normrows,cmp)
 report={'status':'passed','schema':'T21_independent_numerical_timing_audit_v1','auditor_sha256':sha(__file__),'policy_sha256':POLICY_SHA,
  'resource_supplement_sha256':sha(supplement),'execution_review_sha256':sha(args.execution_review),'input_audit_sha256':sha(args.input_audit),
  'audited_manifest_sha256':sha(out/'manifest.json'),'run_manifest_sha256':sha(out/'manifest.json'),'analysis_sha256':sha(analysis_path),'statistics':result,'arrays_verified':arrays,
  'versioned_checkpoint_and_resource_evidence':checkpoint_report,'full_snapshots_reconstructed':snapshots,'actual_vocabulary_readout_rows_retained':40,'sampled_timing_row_vectors':200,'comparison_counts':cmp.report(),'controls':dict(counts),'runtime_extrema':dict(runtime_extrema),'exact_control_maxima':{'full512_current_anchor_logit_fingerprint_mismatches':0,'current_anchor_saved_loss_max_abs':0.0,'current_anchor_saved_passage_mean_abs':0.0},
  'journals':{label:{'final_sha256':j['final_sha256'],'counts':j['counts']} for label,j in journals.items()},
  'scope':{'raw_reconstruction':'10 selected full Q/K/V gates/context and actual CPU projected-write interpolation; projection and hidden-to-logit audit remain separate.',
   'ordinary_shards':'Saved gate/norm/source arrays independently checked. Unsaved ordinary contexts/hidden/logits use recorded runtime fingerprints and flags only, not full raw reconstruction.',
   'timing':'Full2560 science-node fingerprints and protected-partition flags are runtime provenance. Raw200 sampled node rows are independently checked, not a reconstruction of the whole-node SHA; ordinary full127-query vocab prefixes remain runtime evidence.',
   'source_offsets_and_history':'Current selected records bound by source/input hashes and separately reviewed independent input/history audit; no historical restricted token IDs reopened.',
   'causal_limits':'Approximate Euclidean write norm matching with unequal support counts/dose; no isolated timing-versus-dose, mediation fraction, equivalence, uniqueness, generation or fidelity claim.'},
  'model_forwards':0,'pretrained_checkpoint_access':False,'gpu_used':False,'file_deletions':0,
  'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-started}
 require(report['peak_rss_bytes']<=3*1024**3,'CPU RSS3GiB limit')
 return report

def synthetic_norms():
 # Actual fixed-shape CPU writes, independent of any model/checkpoint.
 rng=np.random.default_rng(20261021);base=rng.normal(0,.25,(512,768)).astype(np.float32)
 left=base.copy();right=base.copy();left[1:]+=np.float32(.125);right[1:]-=np.float32(.25)
 base[0,0]=left[0,0]=right[0,0]=np.float32(-0.)
 recipe=match_arrays(base,left,right);norms={}
 for arm,full in (('baseline',base),('current',base),('remove_full',left),('orth_full',right)):
  delta=norm(full.astype(np.float64)-base.astype(np.float64));norms[arm]={'actual_write_norm':norm(full),'actual_write_delta_norm':delta,'proposed_full_write_delta_norm':delta.copy(),'proposed_pre_context_delta_norm':delta.copy()}
 for d,j in (('remove',0),('orth',1)):
  for support in SUPPORTS:
   mask=support_mask(support);value=base.copy();value[mask]=recipe['values'][j][mask];delta=norm(value.astype(np.float64)-base.astype(np.float64));target=np.where(mask,recipe['target'],0.)
   norms[d+'_write_matched_'+support]={'actual_write_norm':norm(value),'actual_write_delta_norm':delta,'proposed_full_write_delta_norm':recipe['full_norms'][j].copy(),'proposed_pre_context_delta_norm':recipe['full_norms'][j].copy(),'target_norm':target,'lambda':np.where(mask,recipe['scales'][j],0.),'own_full_norm':recipe['full_norms'][j].copy(),'other_full_norm':recipe['full_norms'][1-j].copy(),'rounding_budget':np.where(mask,recipe['budgets'][j],0.),'other_rounding_budget':np.where(mask,recipe['budgets'][1-j],0.),'signed_target_residual':delta-target}
 for support in SUPPORTS:
  a,b=(norms[d+'_write_matched_'+support]['actual_write_delta_norm'] for d in ('remove','orth'));gap=np.abs(a-b)
  for d in ('remove','orth'):norms[d+'_write_matched_'+support]['absolute_pair_gap']=gap.copy()
 return norms,(base,left,right),recipe

def synthetic_matching(norms,operands,recipe):
 domains=check_matching(norms,None);meta={'corpus':'wiki_validation','window':0,'identity':'natural','boundary':'000'};shard={'metadata':meta};lookup={}
 fullquality=runtime_quality(norms['remove_write_matched_both']['target_norm'],norms['remove_write_matched_both']['actual_write_delta_norm'],norms['orth_write_matched_both']['actual_write_delta_norm'])
 for variant,field,opfield in (('main','write_norms','operations'),('suffix','suffix_write_norms','suffix_operations')):
  shard[field]=norms;ops={}
  for arm,a in zip(('baseline','remove_full','orth_full'),operands):ops[arm]={'write_checks':{'returned_write':fp(a[None])}}
  for support in SUPPORTS:
   mask=support_mask(support);a,b=(norms[d+'_write_matched_'+support]['actual_write_delta_norm'] for d in ('remove','orth'));target=norms['remove_write_matched_'+support]['target_norm'];gap=np.abs(a-b);positive=target[1:]>0;relative=gap[1:][positive]/target[1:][positive]
   for j,d in enumerate(('remove','orth')):
    ops[d+'_write_matched_'+support]={'write_checks':{'support':support,'target_scope':'same_frozen_full_recipe_target_on_active_queries_zero_elsewhere','active_write_bytes_exact':True,'inactive_write_bytes_exact':True,'baseline_write':fp(operands[0][None]),'proposed_full_write':fp(operands[j+1][None]),'full_matched_array':fp(recipe['values'][j]),'full_recipe_matching_quality':fullquality,'matching_quality':runtime_quality(target,a,b),'paired_actual_delivery_complete':d=='orth','counterpart_norm_scope':'actual_delivered_same_support' if d=='orth' else 'expected_CPU_rounded_not_yet_delivered_same_support'}}
   lookup[('wiki_validation',0,'natural','000',variant,support)]={'domains':domains[support],'active_queries':int(mask.sum()),'active_scored_queries':int(mask[1:511].sum()),'maximum_absolute_pair_gap':float(gap.max()),'zero_target_queries':int((target==0).sum()),'active_zero_target_queries':int((mask&(target==0)).sum()),'relative_quantile_probabilities':[0,.25,.5,.75,.95,.99,1],'positive_target_relative_gap_quantiles':np.quantile(relative,[0,.25,.5,.75,.95,.99,1]).tolist() if relative.size else None,'quantized_to_zero_positive_targets':[int(((z==0)&(target>0)).sum()) for z in (a,b)]}
  shard[opfield]=ops
 return shard,lookup

def synthetic_timing():
 # The observer fixture uses all real declared dimensions, shared zero arrays;
 # these are synthetic values, never outputs from a model or old study.
 meta={'corpus':'wiki_validation','window':0,'identity':'natural','boundary':'000'};tokens=np.arange(512,dtype=np.int64)
 hidden=np.zeros((1,512,768),np.float32);logits=np.zeros((4,50257),np.float32);prefix=fp(np.zeros((127,50257),np.float32));fullfps={n:fp(np.zeros((1,512,2304 if n=='l11_qkv' else 768),np.float32)) for n in TIMING_NODES}
 shard={'metadata':meta,'operations':{},'suffix_operations':{},'timing_samples':{},'readouts':{}};events={}
 for variant,field in (('main','operations'),('suffix','suffix_operations')):
  for arm in (*ARMS,*(('native_repeat','native_identity') if variant=='main' else ())):
   if arm=='native_repeat':shard[field][arm]={};continue
   support=arm.rsplit('_',1)[-1] if is_matched(arm) else None
   tc={'support':support,'later_query_start':127,'same_full_matched_array_across_supports':support is not None,'observed_nodes':list(TIMING_NODES),'later_only_prefix_logits':'not_required'}
   if arm!='baseline':
    sl=[127,None] if support=='early' else [None,127] if support=='later' else [None,None] if arm in ('current','native_identity') else [0,1];n=385 if support=='early' else 127 if support=='later' else 512 if arm in ('current','native_identity') else 1
    for node in TIMING_NODES:
     f=fp(np.zeros((1,n,2304 if node=='l11_qkv' else 768),np.float32));tc[node]={'query_slice':sl,'byte_exact':True,'actual':f,'baseline':copy.deepcopy(f)}
   if arm in ('baseline','native_identity') or support=='later':tc.update(observed_model_readout_prefix=prefix,baseline_model_readout_prefix=prefix)
   if support=='later':tc['later_only_prefix_logits']={'query_slice':[0,127],'byte_exact':True,'actual':prefix,'baseline':prefix}
   shard[field][arm]={'timing_checks':tc,'timing_fingerprints':copy.deepcopy(fullfps)}
 for arm in ARMS:
  shard['timing_samples'][arm]={'query_rows':np.array(TIMING_ROWS,np.int64),**{n:np.zeros((1,5,2304 if n=='l11_qkv' else 768),np.float32) for n in TIMING_NODES}}
  shard['readouts'][arm]={'hidden':hidden,'logits':logits,'query_rows':np.array(READOUT_ROWS,np.int64),'targets':tokens[np.array(READOUT_ROWS)+1],'token_sha256':fp(tokens)['sha256'],'hidden_fingerprint':fp(hidden)}
  events[('wiki_validation',0,'natural','000',arm,'main')]={'hidden_fingerprint':fp(hidden)}
 return shard,tokens,events

def synthetic_statistics(cmp):
 ids=indices();specs=contrast_definitions();value={'baseline':3.,'current':3.,'remove_full':13.,'orth_full':29.,'remove_write_matched_early':5.,'remove_write_matched_later':7.,'remove_write_matched_both':11.,'orth_write_matched_early':17.,'orth_write_matched_later':19.,'orth_write_matched_both':23.}
 closed={'remove_early_effect':2,'remove_later_effect':4,'remove_both_effect':8,'remove_full_effect':10,'remove_support_interaction':2,'remove_both_minus_full':-2,'orth_early_effect':14,'orth_later_effect':16,'orth_both_effect':20,'orth_full_effect':26,'orth_support_interaction':-10,'orth_both_minus_full':-6,'early_remove_minus_orth':-12,'later_remove_minus_orth':-12,'both_remove_minus_orth':-12,'support_interaction_remove_minus_orth':12,'both_direction_difference_minus_full':4}
 require(set(specs)=={'000__'+x for x in closed},'17 independent named closedform expectations')
 for name,z in specs.items():require(sum(k*value[a.split('__')[1]] for a,k in z['terms'].items())==closed[name.removeprefix('000__')],'independent closedform contrast algebra')
 rows={};anchors={};nr={'gate':{},'write':{}}
 for c,i,w,view,mode in itertools.product(CORPORA,('natural','bos'),range(4),VIEWS,MODES):
  arm=mode.split('__')[1];z={'nll':value[arm]+w/16+(i=='bos')/8,'delta_nll_c':value[arm]-3,'delta_nll_boundary':value[arm]-3,'kl_c':(value[arm]-3)**2/256,'kl_boundary':(value[arm]-3)**2/256,'agreement_c':.5,'agreement_boundary':.5};key=(c,i,w,view,mode);rows[key]=z
  for family in nr:nr[family][(*key,10)]={'fixture':{'scope':'scored_query_positions','mean':1.,'rms':1.,'max_abs':1.}}
 for c,view,arm in itertools.product(CORPORA,VIEWS,('native_repeat','native_identity')):anchors[c,'natural',0,view,'000__'+arm]=dict(rows[c,'natural',0,view,'000__baseline'])
 wire={'window_metrics':[{**dict(zip(('corpus','identity','window','view','mode'),key)),**z} for key,z in rows.items()],'anchors':[{**dict(zip(('corpus','identity','window','view','mode'),key)),**z} for key,z in anchors.items()],'summaries':[],'contrasts':[],'paired_start_interactions':[],'statistical_scope':{'shared_indices_sha256':fp(ids)['sha256']}}
 for family in nr:wire[family+'_norm_window_summaries']=[{**dict(zip(('corpus','identity','window','view','mode','layer'),key)),'values':z} for key,z in nr[family].items()]
 for c,i,view in itertools.product(CORPORA,('natural','bos'),VIEWS):
  for mode in MODES:
   metrics={field:bootstrap([rows[c,i,w,view,mode][field] for w in range(4)],ids) for field in FIELDS}
   for field,label in (('nll','perplexity'),('delta_nll_c','ppl_ratio_c'),('delta_nll_boundary','ppl_ratio_boundary')):metrics[label]={'value':math.exp(metrics[field]['mean']),'ci95':[math.exp(q) for q in metrics[field]['ci95']]}
   wire['summaries'].append({'corpus':c,'identity':i,'view':view,'mode':mode,'metrics':metrics})
  for name,spec in specs.items():
   xs=[math.fsum(k*rows[c,i,w,view,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
   wire['contrasts'].append({'corpus':c,'identity':i,'view':view,'name':name,**spec,'primary':spec['primary_view']==view,**bootstrap(xs,ids)})
 for c,view in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   xs=[math.fsum(k*(rows[c,'bos',w,view,mode]['nll']-rows[c,'natural',w,view,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
   wire['paired_start_interactions'].append({'corpus':c,'view':view,'name':'bos_minus_natural__'+name,'terms':spec['terms'],'scope':'paired_start_interaction_secondary',**bootstrap(xs,ids)})
 result=statistics(wire,rows,anchors,nr,cmp)
 return result,wire,rows,anchors,nr

def self_test():
 started=time.time();tests=[];policy=read(ROOT/'review/query_write_timing_numerical_audit_policy.json');require(sha(ROOT/'review/query_write_timing_numerical_audit_policy.json')==POLICY_SHA,'prospective policy SHA');cmp=Comparisons(policy)
 def rejects(label,call):
  try:call()
  except (AssertionError,KeyError,ValueError):tests.append(label);return
  raise AssertionError('hostile accepted: '+label)
 require(len(roster(CORPORA))==648 and len(roster(CORPORA[:2]))==324,'exact648/324 roster');tests.append('648_roster_armmajor_anchor_exception')
 for corpora in (CORPORA[:2],CORPORA[2:]):
  layout=expected_checkpoint_layout(corpora);require([z['committed'] for z in layout]==[0,0,*range(1,17),16] and [z['forward_passes'] for z in layout[:4]]==[0,0,22,42] and layout[-1]['forward_passes']==324,'19 checkpoint layout')
 tests.append('19checkpoint_prefix_counts')
 require(not exact(np.array([0.],np.float32),np.array([-0.],np.float32)),'signedzero bytes')
 norms,operands,recipe=synthetic_norms();check_matching(norms,cmp);tests.append('full512_recipe_and_all_three_support_pairs')
 for label,mutate in (('wrong_lambda',lambda z:z['remove_write_matched_early']['lambda'].__setitem__(1,0.)),('inactive_delta',lambda z:z['remove_write_matched_later']['actual_write_delta_norm'].__setitem__(1,1.)),('wrong_full_origin',lambda z:z['remove_write_matched_both']['own_full_norm'].__setitem__(1,9.))):
  bad=copy.deepcopy(norms);mutate(bad);rejects(label,lambda:check_matching(bad,cmp))
 for full in (operands[0],operands[1]):
  r=match_arrays(operands[0],full,full);require(exact(r['values'][0],full) and exact(r['values'][1],full),'zero/unit endpoint bytes')
 tests.append('signedzero_zero_and_unit_endpoints')
 matched,lookup=synthetic_matching(norms,operands,recipe);mc=collections.Counter();matching_analysis(matched,lookup,cmp,mc);require(mc['support_pairs']==6 and mc['full_recipe_origins']==2 and mc['matched_arm_variant_records']==12,'synthetic two variants three support pairs');tests.append('matching_metadata_fullrecipe_fingerprints_and_chronology')
 bad=copy.deepcopy(matched);bad['operations']['orth_write_matched_later']['write_checks']['full_matched_array']['sha256']='f'*64;rejects('changed_full_recipe_fingerprint',lambda:matching_analysis(bad,lookup,Comparisons(policy),collections.Counter()))
 bad=copy.deepcopy(matched);bad['operations']['remove_write_matched_early']['write_checks']['paired_actual_delivery_complete']=True;rejects('premature_actual_partner_flag',lambda:matching_analysis(bad,lookup,Comparisons(policy),collections.Counter()))
 shard,tokens,events=synthetic_timing();counts=collections.Counter();timing_provenance(shard,tokens,events,counts);require(counts['raw_timing_rows']==200 and counts['raw_vocabulary_rows']==40 and counts['science_timing_calls']==20,'timing sample ABI');tests.append('four_node_timing_guards_and200rows40readouts')
 bad=copy.deepcopy(shard);bad['operations']['remove_write_matched_later']['timing_checks']['later_only_prefix_logits']='pending';rejects('pending_readout_guard',lambda:timing_provenance(bad,tokens,events,collections.Counter()))
 bad=copy.deepcopy(shard);bad['timing_samples']['remove_write_matched_early']['post_attention'][0,2,0]=np.float32(-0.);rejects('protected_sample_signedzero_mutation',lambda:timing_provenance(bad,tokens,events,collections.Counter()))
 bad=copy.deepcopy(shard);bad['operations']['orth_write_matched_early']['timing_checks']['block_output']['query_slice']=[None,127];rejects('wrong_protected_partition',lambda:timing_provenance(bad,tokens,events,collections.Counter()))
 stats,wire,rows,anchors,nr=synthetic_statistics(cmp);tests.append('17closedform_maps_full960_24_240_408_204_wire')
 wire['contrasts'][0]['mean']+=.01;rejects('corrupt_contrast',lambda:statistics(wire,rows,anchors,nr,Comparisons(policy)))
 # Canonical journal assembled in memory, no fixture model calls or filesystem.
 chain='0'*64;lines=[];roles=roster(CORPORA[:2])
 for j,role in enumerate(roles):
  identity=dict(zip(('corpus','window','identity','boundary','arm','phase'),role))
  for kind in ('forward_intent','forward_attempt','model_returned','forward_completed'):
   e={'previous_sha256':chain,'event':kind,'identity':identity,'next_attempt' if kind=='forward_intent' else 'attempt':j+1}
   if kind in ('model_returned','forward_completed'):e['completed']=j+1
   if kind=='forward_completed':e['processed']=j+1
   body=json.dumps(e,sort_keys=True,separators=(',',':')).encode();chain=hashlib.sha256(body).hexdigest();lines.append(body+b'\n')
  if role[-2:]==('orth_write_matched_both','suffix'):
   c,w,i,b=role[:4];e={'previous_sha256':chain,'event':'committed','path':f'inputs/{c}__{w}__{i}__{b}.npz','bytes':1,'sha256':'0'*64};body=json.dumps(e,sort_keys=True,separators=(',',':')).encode();chain=hashlib.sha256(body).hexdigest();lines.append(body+b'\n')
 j=journal(b''.join(lines),roles);require(j['counts']['committed']==16,'16ordered commits');tests.append('324_call_journal16commits')
 rejects('missing_commit',lambda:journal(b''.join(lines[:-1]),roles))
 sample=np.array([0.,-0.,2.**-149],np.float32);header={'schema':'wiki_gate_lossless_npz_v1','object':{'type':'tensor','array':'a','shape':[3],'dtype':'torch.float32','sha256':fp(sample)['sha256']}};stream=io.BytesIO();np.savez(stream,metadata=np.frombuffer(json.dumps(header).encode(),np.uint8),a=sample);stream.seek(0);actual,n=load_npz(stream);require(n==1 and exact(actual,sample),'codec signedzero/subnormal bytes');tests.append('lossless_decoder_inmemory_signedzero_subnormal')
 rejects('actual_access_hold_before_any_path_read',lambda:audit(argparse.Namespace()))
 require('torch' not in globals(),'no torch/model import in selftest')
 return {'status':'passed_prospective_synthetic_fixture','auditor_sha256':sha(__file__),'policy_sha256':POLICY_SHA,'tests':tests,'statistical_wire_counts':stats,'planned_roster':648,'comparison_counts':cmp.report(),'model_forwards':0,'actual_T21_outputs_read':False,'tokenizer_access':False,'checkpoint_or_header_access':False,'gpu_used':False,'file_deletions':0,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'wall_seconds':time.time()-started}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('run','segment-a','segment-b','execution-review','input-audit','result'):parser.add_argument('--'+name,type=Path)
 parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
 if args.self_test:
  try:result=self_test()
  except BaseException as error:
   if args.result and not args.result.exists():put(args.result,{'status':'failed_prospective_fixture','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'auditor_sha256':sha(__file__)})
   raise
  if args.result:put(args.result,result)
  print(json.dumps(result));return
 require(all((args.run,args.segment_a,args.segment_b,args.execution_review,args.input_audit,args.result)),'complete reviewed execution arguments');require(not args.result.exists(),'new exclusive result path')
 try:put(args.result,audit(args))
 except BaseException as error:
  put(args.result.with_name(args.result.stem+'_failure_'+str(time.time_ns())+'.json'),{'status':'failed','error':str(error),'error_type':type(error).__name__,'traceback':traceback.format_exc(),'auditor_sha256':sha(__file__),'raw_study_unchanged':True});raise
if __name__=='__main__':main()
