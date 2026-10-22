"""Prospective independent T22 numerical donor auditor. Actual audit HOLD.

This source contains independent coefficient/statistical computation and exact
T21 decoder/arithmetic helpers. Producer ABI implementation is source-only;
final source/resource/input/model and execution gates remain separate.
No analyzer imports. No fixture or actual data access is authorized by this file.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
import argparse,collections,copy,hashlib,io,itertools,json,math,re,resource,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
POLICY_SHA='f6df4bae611ad18de5b7bf12596838d9bc41ef894847b7ffa1de90012e0a20ac'
ANCESTOR_SHA='fd4e8eeb9c5697d094c5209756be0c2e949895eb0288d1630e2528e877838eec'
ACTUAL_AUDIT_APPROVED=False
RAW_ABI_READY=False
APPROVED_SOURCE_HASHES={}
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
BOUNDARIES=('000',)
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched_early','remove_l11_kv','orth_write_matched_early','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
DONORS=('remove_write_matched_early','orth_write_matched_early')
RECIPIENTS=('remove_l11_kv','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
L10_SNAPSHOTS=('baseline','remove_full','orth_full',*DONORS)
L11_SNAPSHOTS=tuple(a for a in ARMS if a=='baseline' or a in DONORS or a in RECIPIENTS)
MODES=tuple('000__'+a for a in ARMS)
FIELDS=('nll','kl_baseline','agreement_baseline')
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
E=np.arange(1,127,dtype=np.int64)
L=np.arange(127,512,dtype=np.int64)
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
ANALYZER_SHA='84e789a723521b9866ccd9258ab1e6d3723e301aebbea8b8860decc3f4ed0ee2'
METRICS=('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')
COMMON={'actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'}
MATCH={'target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap'}
GATE_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
L11_NORM_KEYS={a+b for a in ('q','k','v') for b in ('_norm','_delta_baseline_norm','_delta_original_norm')}
TIMING_NODES=('post_attention','mlp_output','block_output','l11_qkv')
TIMING_ROWS=(1,126,127,510,511)
READOUT_ROWS=(1,126,127,510)
L11_FIELDS=('c_attn_input','original_projection','expected_qkv','delivered_qkv','attention_context','attention_write','block_hidden','final_hidden')
LAYOUT_VERSION='T22_c_attn_Q_K_V_contiguous_allheads_E1_126_v1'
EFFECTIVE_FAILURE_LIMIT_BYTES=1024**3

def is_matched(arm):return arm in DONORS

def support_mask(support):
 require(support=='early','only donor early support');return (np.arange(512)>=1)&(np.arange(512)<127)


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

def contrast_definitions():
 """Nineteen explicit finite maps, independently constructed from the protocol."""
 B='baseline';result={}
 donor={'PC':'remove_write_matched_early','orth':'orth_write_matched_early'}
 recipient={'PC':{'K':'remove_l11_k','V':'remove_l11_v','KV':'remove_l11_kv'},'orth':{'K':'orth_l11_k','V':'orth_l11_v','KV':'orth_l11_kv'}}
 def add(name,terms,primary=False):
  require('000__'+name not in result,'unique prospective map')
  result['000__'+name]={'terms':{'000__'+a:k for a,k in terms.items()},'primary_later_NLL_and_KL':primary}
 for d in ('PC','orth'):
  r=recipient[d];add(d+'_donor_effect',{donor[d]:1,B:-1})
  for p in ('K','V','KV'):add(d+'_'+p+'_effect',{r[p]:1,B:-1},p!='KV')
  add(d+'_K_minus_V',{r['K']:1,r['V']:-1},True)
  add(d+'_interaction',{r['KV']:1,r['K']:-1,r['V']:-1,B:1},True)
  add(d+'_KV_minus_donor',{r['KV']:1,donor[d]:-1})
 for p in ('K','V','KV'):add(p+'_PC_minus_orth',{recipient['PC'][p]:1,recipient['orth'][p]:-1})
 add('donor_PC_minus_orth',{donor['PC']:1,donor['orth']:-1})
 add('interaction_PC_minus_orth',{recipient['PC']['KV']:1,recipient['PC']['K']:-1,recipient['PC']['V']:-1,recipient['orth']['KV']:-1,recipient['orth']['K']:1,recipient['orth']['V']:1})
 require(len(result)==19 and all(sum(z['terms'].values())==0 for z in result.values()),'19 zero-sum definitions')
 require(sum(z['primary_later_NLL_and_KL'] for z in result.values())==8,'eight co-primary estimands')
 return result

def primary(spec,metric,view):
 return spec['primary_later_NLL_and_KL'] and metric in ('nll','kl_baseline') and view=='later'

def l10_kind(arm):
 # Recipients follow a native L10 trajectory despite names starting remove/orth.
 require(arm in (*ARMS,'native_repeat','native_identity'),'known arm')
 if arm in ('remove_full','remove_write_matched_early'):return 'remove'
 if arm in ('orth_full','orth_write_matched_early'):return 'orth'
 return 'current' if arm=='current' else 'baseline'

def decode_qkv(projected):
 require(projected.shape==(1,512,2304) and projected.dtype==np.float32 and np.isfinite(projected).all(),'actual original/delivered combined QKV ABI')
 # Conv1D layout is contiguous Q768,K768,V768 blocks, not interleaved heads.
 return tuple(projected[:,:,j*768:(j+1)*768].reshape(1,512,12,64).transpose(0,2,1,3) for j in range(3))

def transplant_expected(base,donor,path):
 require(base.shape==donor.shape==(1,512,2304) and base.dtype==donor.dtype==np.float32,'matched QKV operand ABI')
 require(path in ('K','V','KV'),'known finite transplant path')
 value=base.copy()
 if 'K' in path:value[:,1:127,768:1536]=donor[:,1:127,768:1536]
 if 'V' in path:value[:,1:127,1536:2304]=donor[:,1:127,1536:2304]
 require(exact(value[:,:,:768],base[:,:,:768]) and exact(value[:,0],base[:,0]) and exact(value[:,127:],base[:,127:]),'query/source/later protected bytes')
 return value

def per_query_l11_norms(original,delivered,baseline):
 qo,ko,vo=decode_qkv(original);qd,kd,vd=decode_qkv(delivered);qb,kb,vb=decode_qkv(baseline)
 result={}
 for key,own,now,ref in zip(('q','k','v'),(qo,ko,vo),(qd,kd,vd),(qb,kb,vb)):
  def rows(x):return x.transpose(0,2,1,3).reshape(512,768).astype(np.float64)
  x,y,z=rows(own),rows(now),rows(ref)
  result[key+'_norm']=norm(y);result[key+'_delta_baseline_norm']=norm(y-z);result[key+'_delta_original_norm']=norm(y-x)
 return result

def verify_recorded_splice(original,delivered,donor,path,fingerprints,record):
 """Validate actual raw splice and its separate natural/expected commitments."""
 decode_qkv(original);decode_qkv(delivered)
 expected=original if path is None else transplant_expected(original,donor,path)
 require(exact(delivered,expected) and fp(expected)==fingerprints['expected_qkv'],'independent exact K/V transplant and expected commitment')
 require(fp(original)==record['original_projection']==fingerprints['original_projection'],'natural projection commitment remains original')
 require(fp(delivered)==record['delivered_qkv']==fingerprints['delivered_qkv'],'actual delivered-QKV commitment')
 require(exact(delivered[:,:,:768],original[:,:,:768]) and exact(delivered[:,0],original[:,0]) and exact(delivered[:,127:],original[:,127:]),'all Q/source/later original projection bytes')
 return expected

def verify_selected_L11_mask(mask,panel,record,native_mask_record):
 require(mask.dtype==np.float32 and mask.shape==(1,1,512,512) and not np.isnan(mask).any() and not np.isposinf(mask).any(),'explicit actual L11 mask reference')
 require(panel['mask_reference_byte_exact'] is True and panel['mask_fingerprint']==record['effective_mask']==native_mask_record['diagnostic_effective']==fp(mask),'all selected actual L11 mask commitments')
 require(record['effective_mask_baseline_byte_exact'] is True,'actual mask own-baseline byte guard')

def raw_window_rows(metrics,metadata):
 """Independent NLL/KL/agreement reductions; retain signed tiny KL values."""
 expected=set(ARMS)|({'native_repeat','native_identity'} if (metadata['window'],metadata['identity'])==(0,'natural') else set())
 require(set(metrics)==expected,'exact arm/anchor metric roster')
 result={};anchors={}
 for arm,z in metrics.items():
  for key,dtype in (('loss',np.float32),('kl_boundary',np.float32),('agreement_boundary',np.bool_)):
   require(z[key].shape==(511,) and z[key].dtype==dtype and np.isfinite(z[key]).all(),'saved scientific field shape/dtype/finite')
  for view,(lo,hi) in VIEWS.items():
   row={name:float(z[key][lo:hi].astype(np.float64).mean()) for name,key in zip(FIELDS,('loss','kl_boundary','agreement_boundary'))}
   key=(metadata['corpus'],metadata['identity'],metadata['window'],view,'000__'+arm)
   (result if arm in ARMS else anchors)[key]=row
 return result,anchors

def independent_statistics(rows,anchors):
 """Prospective statistical value model; final external wire names pending ABI."""
 require(set(rows)==set(itertools.product(CORPORA,('natural','bos'),range(4),VIEWS,MODES)),'1152wide window membership')
 require(set(anchors)=={(c,'natural',0,v,'000__'+a) for c,v,a in itertools.product(CORPORA,VIEWS,('native_repeat','native_identity'))},'24wide anchors')
 ids=indices();specs=contrast_definitions();summaries=[];contrasts=[];starts=[]
 for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
  for mode in MODES:
   metrics={f:bootstrap([rows[c,i,w,v,mode][f] for w in range(4)],ids) for f in FIELDS}
   own=metrics['nll'];delta=bootstrap([rows[c,i,w,v,mode]['nll']-rows[c,i,w,v,'000__baseline']['nll'] for w in range(4)],ids)
   transforms={'perplexity':{'value':math.exp(own['mean']),'ci95':[math.exp(z) for z in own['ci95']]},'ppl_ratio_baseline':{'value':math.exp(delta['mean']),'ci95':[math.exp(z) for z in delta['ci95']],'delta_nll':delta}}
   summaries.append({'corpus':c,'identity':i,'view':v,'mode':mode,'metrics':metrics,'transforms':transforms})
  for name,spec in specs.items():
   for f in FIELDS:
    values=[math.fsum(k*rows[c,i,w,v,mode][f] for mode,k in spec['terms'].items()) for w in range(4)]
    contrasts.append({'corpus':c,'identity':i,'view':v,'name':name,'metric':f,'terms':spec['terms'],'kind':'within_native_boundary','primary_view':'later' if spec['primary_later_NLL_and_KL'] else None,'primary':primary(spec,f,v),**bootstrap(values,ids)})
 for c,v in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   for f in FIELDS:
    values=[math.fsum(k*(rows[c,'bos',w,v,mode][f]-rows[c,'natural',w,v,mode][f]) for mode,k in spec['terms'].items()) for w in range(4)]
    starts.append({'corpus':c,'view':v,'name':'bos_minus_natural__'+name,'metric':f,'terms':spec['terms'],'scope':'paired_start_interaction_secondary','primary':False,**bootstrap(values,ids)})
 require((len(summaries),len(contrasts),len(starts))==(288,1368,684),'expanded three-metric counts')
 require(sum(z['primary'] for z in contrasts)==128,'128nominal co-primary comparisons')
 return {'summaries':summaries,'contrasts':contrasts,'paired_start_interactions':starts,'shared_indices_sha256':fp(ids)['sha256']}

def actual_audit(args):
 # Mandatory BEFORE reading any run, current-input or actual-outcome file.
 require(ACTUAL_AUDIT_APPROVED and RAW_ABI_READY and APPROVED_SOURCE_HASHES,'T22 raw audit HOLD pending complete ABI/source/fixture/input/model/execution review')
 started=time.time();policy_path=ROOT/'review/l11_key_value_donor_numerical_audit_policy_v3.json';require(sha(policy_path)==POLICY_SHA,'frozen N policyv3');policy=read(policy_path)
 review=read(args.execution_review)
 require(review['approved_numerical_execution'] is True and review['auditor_sha256']==sha(__file__) and review['policy_sha256']==POLICY_SHA,'later exact source/policy execution approval')
 require(sha(args.input_audit)==review['input_audit_sha256'] and read(args.input_audit)['status']=='passed','independent actual input/history audit binding')
 require(review['resource_dependencies'],'complete separately reviewed production source/witness/resource dependencies required')
 for name,h in review['resource_dependencies'].items():
  p=ROOT/name;require(not Path(name).is_absolute() and p.resolve().is_relative_to(ROOT.resolve()) and (name.startswith('review/') or name.startswith('GPT2_L11_')),'bounded exact resource dependency path');require(sha(p)==h,'resource dependency '+name)
 out=args.run;m,inventory,journals=provenance(out,review)
 require(m['raw_panel_contract']=={'L10':list(L10_SNAPSHOTS),'L11':['baseline',*DONORS,*RECIPIENTS],'end':list(ARMS),'readout_rows':list(READOUT_ROWS),'timing_rows':list(TIMING_ROWS)},'exact5/9/12 raw panel contract')
 require(m['slice_contract']=={'earlier':[1,127],'later':[127,512],'QKV_width':768,'layout':'contiguous_Q_K_V'},'fixed all-head transplant domains/layout')
 for name,h in policy['frozen_bindings'].items():
  p=out/'frozen_sources'/name if name in m['source_hashes'] else ROOT/name;require(sha(p)==h,'frozen policy dependency '+name)
 checkpoint_report=original_checkpoint_provenance(args,m,inventory);cmp=Comparisons(policy)
 analysis_path=out/'l11_key_value_donor_analysis.json';require(sha(analysis_path)==review['analysis_sha256'],'execution-reviewed complete analysis');analysis=read(analysis_path)
 require(analysis['scope']['analyzer_sha256']==ANALYZER_SHA and analysis['scope']['raw_manifest_sha256']==sha(out/'manifest.json') and analysis['scope']['raw_inventory_sha256']==sha(out/'numeric_inventory.json'),'analysis source/complete raw study binding')
 require(analysis['scope']['source_hashes']=={Path(n).name:b['sha256'] for n,b in inventory.items()},'analysis exact32shard binding')
 matching={tuple(z[k] for k in ('corpus','window','identity','boundary','variant','support')):z for z in analysis['matching_quality']};require(len(matching)==len(analysis['matching_quality'])==64,'exact64 donor-pair semantic rows')
 require(analysis['statistical_scope']['shared_indices_sha256']==fp(indices())['sha256'],'analysis common2000 bootstrap indices')
 maps,nmaps=load_npz(out/'calibration_reference/calibration_maps.npz');directions,ndirections=load_npz(out/'calibration_reference/calibration_directions.npz')
 dm=read(out/'data_manifest.json');reference=out/'calibration_reference'
 require(sha(reference/'manifest.json')==dm['calibration_reference_manifest_sha256']=='f059c11882bd5fbee321b4b5ccb5ff213f44f2edeb6141d2971efbf84ab03060','frozen original calibration manifest')
 for name in ('calibration_maps.npz','calibration_directions.npz'):require(sha(reference/name)==dm['calibration_reference_files'][name],'frozen fitted calibration arrays')
 digest=hashlib.sha256()
 for layer in sorted(maps['F']):
  for name,a in sorted(maps['F'][layer].items()):digest.update(f'{layer}:{name}:torch.{a.dtype.name}:{tuple(a.shape)}'.encode());digest.update(np.ascontiguousarray(a).tobytes())
 require(digest.hexdigest()==F_DIGEST,'independently reconstructed F digest')
 alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
 require(all(a.dtype==np.float64 and a.shape==(12,) and np.isfinite(a).all() for a in (alpha,u,v)),'fixed fitted FP64 gate vectors')
 require(abs(norm(u)-1)<=1e-12 and abs(norm(v)-1)<=1e-12 and abs(u@v)<=1e-12,'F orthonormal directions')
 # Only current-study input tensors are read after all explicit execution gates.
 # No tokenizer, pretrained checkpoint/header, model or GPU is used.
 import torch
 torch.set_num_threads(1);require(not torch.cuda.is_initialized(),'audit CPU only')
 items=torch.load(out/'tokens.pt',weights_only=True,map_location='cpu')
 tokens={(z['corpus'],z['window']):z['tokens'].numpy().copy() for z in items};require(len(tokens)==len(items)==16 and set(tokens)==set(itertools.product(CORPORA,range(4))),'sixteen unique current natural inputs')
 passages={(z['corpus'],z['window']):z for z in dm['passages']};require(set(passages)==set(tokens) and len(dm['passages'])==16,'exact selected source/span attestation roster')
 rows={};anchors={};normrows={f:{} for f in ('gate','write','l11')};counts=collections.Counter();runtime_extrema=collections.defaultdict(float);arrays=nmaps+ndirections;quality=[]
 for name in sorted(inventory):
  shard,narrays=load_npz(out/name);arrays+=narrays;meta=shard['metadata'];c,w,i,b=(meta[k] for k in ('corpus','window','identity','boundary'));selected=(c,w,i)==('wiki_validation',0,'natural')
  tops={'schema','metadata','metrics','suffix','gates','suffix_gates','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms','operations','suffix_operations','source_reference','snapshots','readouts','checks','l11_norms','suffix_l11_norms','l11_panels','timing_samples'}|({'l11_mask_reference'} if selected else set())
  require(set(shard)==tops and shard['schema']=='l11_key_value_donor_explicit_v1' and meta['length']==512 and b=='000' and name==f'inputs/{c}__{w}__{i}__{b}.npz','closed exact raw shard ABI/identity')
  require(set(meta)=={'corpus','window','identity','length','token_sha256','boundary'},'canonical input metadata fields')
  x=tokens[c,w].copy();p=passages[c,w];require(x.dtype==np.int64 and x.shape==(512,) and fp(x)['sha256']==p['window_token_sha256'],'actual natural512 token commitment')
  if i=='bos':x[0]=dm['literals']['start_special']['id'];require(fp(x)['sha256']==p['bos_window_token_sha256'],'actual token0-only BOS replacement')
  require(meta['token_sha256']==fp(x)['sha256'],'shard actual start-variant input commitment')
  journal_record=journals['A' if c in CORPORA[:2] else 'B'];events=journal_record['byrole']
  for e in journal_record['events']:
   if e['event']=='forward_intent' and tuple(e['identity'][k] for k in ('corpus','window','identity','boundary'))==(c,w,i,b):
    actual=x.copy()
    if e['identity']['phase']=='suffix':actual[128:]=np.roll(actual[128:],1)
    require(e['token_fingerprint']==fp(actual),'journal actual MAIN/SUFFIX input bytes')
  admit=journal_record['admissions'][name]
  def array_bytes(value):
   if isinstance(value,np.ndarray):return value.nbytes
   if isinstance(value,dict):return sum(array_bytes(k)+array_bytes(v) for k,v in value.items())
   if isinstance(value,(list,tuple)):return sum(array_bytes(z) for z in value)
   return 0
  require(admit['array_occurrences']==narrays and admit['serialized_raw_array_bytes']==array_bytes(shard),'actual raw array universe matches pre-encode record')
  rawrows,rawanchors,rawnorms=raw_metrics_and_norms(shard,cmp)
  require(not set(rows)&set(rawrows) and not set(anchors)&set(rawanchors),'no duplicate per-window rows');rows.update(rawrows);anchors.update(rawanchors)
  for family,values in rawnorms.items():require(not set(normrows[family])&set(values),'unique norm rows');normrows[family].update(values)
  control_and_operation_provenance(shard,m,x,events,cmp,counts);gate_reconstruction(shard,alpha,u,v,cmp)
  counts['full_L10_snapshots']+=snapshot(shard,cmp);selected_l11_panels(shard,x,events,cmp,counts);quality.extend(donor_matching_analysis(shard,matching,cmp,counts))
  for field in ('operations','suffix_operations'):
   for op in shard[field].values():
    if not op:continue
    for diag in op['native_diag'].values():
     for k in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):runtime_extrema[k]=max(runtime_extrema[k],diag[k])
    runtime_extrema['gate_norm_match_error']=max(runtime_extrema['gate_norm_match_error'],op['native_gate_checks'][10]['norm_match_error'])
  del shard
 counts['anchors']=counts['native_repeat']+counts['native_identity']
 expected={'current':64,'suffix':384,'source_q0':704,'anchors':8,'donor_later':128,'recipient_upstream':384,'KV_closure':128,'support_pairs':64,'RMS_domain_checks':128,'matched_arm_variant_records':128,'direction_recipe_fingerprints':128,'full_recipe_origins':64,'science_timing_calls':768,'science_timing_node_fingerprints':3072,'anchor_timing_calls':4,'anchor_timing_node_fingerprints':16,'full_L10_snapshots':5,'full_L11_snapshots':9,'full_endstates':12,'raw_timing_rows':240,'raw_vocabulary_rows':48}
 require(all(counts[k]==n for k,n in expected.items()),('complete actual raw/control counts',dict(counts),expected))
 statistics_report=compare_statistics(analysis,rows,anchors,cmp);compare_norm_summaries(analysis,normrows,cmp)
 result={'schema':'T22_independent_saved_array_donor_audit_v1','status':'passed','auditor_sha256':sha(__file__),'policy_sha256':POLICY_SHA,'execution_review_sha256':sha(args.execution_review),'input_audit_sha256':sha(args.input_audit),'resource_dependencies':review['resource_dependencies'],'audited_manifest_sha256':sha(out/'manifest.json'),'run_manifest_sha256':sha(out/'manifest.json'),'analysis_sha256':sha(analysis_path),'raw_inventory_sha256':sha(out/'numeric_inventory.json'),'source_hashes':m['source_hashes'],'input_dependencies':m['input_dependencies'],'metadata_dependencies':m['metadata_dependencies'],'F_digest':F_DIGEST,'model_parameter_digest':m['model_hash_before'],'runtime_versions':m['runtime_versions'],'precision_settings':m['precision_settings_before'],'statistics':statistics_report,'arrays_verified':arrays,'controls':dict(counts),'comparison_counts':cmp.report(),'runtime_extrema':dict(runtime_extrema),'donor_matching_diagnostics':quality,'exact_control_maxima':{'full512_current_anchor_logit_fingerprint_mismatches':0,'current_anchor_saved_loss_max_abs':0.,'current_anchor_saved_passage_mean_abs':0.,'later_saved_KV_donor_metric_mismatches':0},'journals':{label:{'final_sha256':j['final_sha256'],'counts':j['counts']} for label,j in journals.items()},'versioned_checkpoint_and_resource_evidence':checkpoint_report,'scope':{'raw_reconstruction':'Five selected L10 full QKV/gate/context/write recipes and nine L11 actual delivered-QKV/context panels using the explicitly saved actual L11 mask; original natural c_attn output remains distinct from delivered transplant. Twelve complete endstates,240 timing row vectors,48 vocabulary rows retained. Projection/readout verification remains separate.','unsaved_states':'Full ordinary timing/QKV/readout arrays and full385-row KV closures use recorded runtime actual-output fingerprints/flags and source guards, not independent reconstruction. Selected rows do not establish a whole-panel precision bound.','statistics':'Independent19coefficient maps x3metrics,128nominal co-primary comparisons,shared2000paired passage-cluster bootstrap. Signed tiny/negative FP32 KL preserved; no outcome fitting or posthoc tolerance change.','causal_limits':'Finite donor/recipient interventions under fixed architecture/guards; unequal K/V doses. No mediation fraction,intrinsic path importance,unique head/circuit,exact amplitude-only exclusion,universal transfer,generation/fidelity or priority claim.'},'model_forwards':0,'GPU_used':False,'pretrained_checkpoint_access':False,'file_deletions':0,'wall_seconds':time.time()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 require(result['peak_rss_bytes']<=3*1024**3,'independent audit process3GiB limit');return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('run','segment-a','segment-b','input-audit','execution-review','result'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();require(not args.result.exists(),'exclusive new result path')
 try:
  result=actual_audit(args);put(args.result,result);print(json.dumps({'status':result['status'],'audited_manifest_sha256':result['audited_manifest_sha256'],'wall_seconds':result['wall_seconds'],'peak_rss_bytes':result['peak_rss_bytes']}))
 except BaseException as error:
  failure=args.result.with_name(args.result.stem+'_failure_'+str(time.time_ns())+'.json')
  put(failure,{'status':'failed','auditor_sha256':sha(__file__),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'model_forwards':0,'GPU_used':False,'source_execution_gate':ACTUAL_AUDIT_APPROVED,'raw_ABI_ready_gate':RAW_ABI_READY});raise

# Checkpoint schedule is semantic; final producer record field names pending.
def expected_checkpoint_layout(corpora):
 groups=list(itertools.product(corpora,range(4),('natural','bos'),BOUNDARIES))
 return [{'name':f'{j+1:06d}','committed':min(max(j-1,0),16),'forward_passes':sum(26 if (w,i)==(0,'natural') else 24 for c,w,i,b in groups[:min(max(j-1,0),16)])} for j in range(19)]

def reconstruct_l11_context(delivered_qkv,mask,actual_context,cmp):
 q,k,v=decode_qkv(delivered_qkv)
 require(mask.shape==(1,1,512,512) and mask.dtype==np.float32 and not np.isnan(mask).any() and not np.isposinf(mask).any(),'bound diagnostic effective mask')
 require(actual_context.shape==(1,512,768) and actual_context.dtype==np.float32 and np.isfinite(actual_context).all(),'finite actual L11 context')
 scores=q.astype(np.float64)@k.astype(np.float64).swapaxes(-1,-2)/8+mask.astype(np.float64)
 shifted=scores-scores.max(-1,keepdims=True);p=np.exp(shifted);p/=p.sum(-1,keepdims=True)
 value=(p@v.astype(np.float64)).transpose(0,2,1,3).reshape(1,512,768)
 cmp.check(value,actual_context,'L11_delivered_QKV_context_reconstruction',1e-4)
 return value

def check_donor_matching(norms,cmp):
 """One actual early-support pair/variant;64pairs and128domains total."""
 left,right=(norms[a] for a in DONORS);target_full=np.minimum(norms['remove_full']['actual_write_delta_norm'],norms['orth_full']['actual_write_delta_norm'])
 mask=np.zeros(512,np.bool_);mask[1:127]=True;target=np.where(mask,target_full,0.)
 a,b=(z['actual_write_delta_norm'] for z in (left,right));gap=np.abs(a-b)
 require(exact(left['rounding_budget'],right['other_rounding_budget']) and exact(right['rounding_budget'],left['other_rounding_budget']),'paired early budgets')
 for direction,z,other in (('remove',left,'orth'),('orth',right,'remove')):
  own=norms[direction+'_full'];opposite=norms[other+'_full']
  require(exact(z['own_full_norm'],own['actual_write_delta_norm']) and exact(z['other_full_norm'],opposite['actual_write_delta_norm']),'same-state actual full norm references')
  require(exact(z['proposed_full_write_delta_norm'],own['actual_write_delta_norm']) and exact(z['proposed_pre_context_delta_norm'],own['proposed_pre_context_delta_norm']),'unmasked hypothetical full proposal norms')
  require(exact(z['target_norm'],target),'min full norm then early mask,not re-fit')
  scale=np.zeros(512,np.float64);nz=z['own_full_norm']!=0;scale[nz]=target_full[nz]/z['own_full_norm'][nz];scale=np.where(mask,scale,0.)
  require(exact(z['lambda'],scale) and ((scale>=0)&(scale<=1)).all(),'fixed recipe scale,noepsilon')
  require(exact(z['actual_write_norm'][~mask],norms['baseline']['actual_write_norm'][~mask]),'inactive baseline write norms')
  require(exact(z['signed_target_residual'],z['actual_write_delta_norm']-target) and exact(z['absolute_pair_gap'],gap),'actual pair diagnostics')
  require(np.all(np.abs(z['signed_target_residual'])<=z['rounding_budget']),'perquery target analytical bound')
  require(np.count_nonzero(z['actual_write_delta_norm'][scale==0])==0,'zero endpoint norm')
  require(exact(z['actual_write_delta_norm'][scale==1],z['own_full_norm'][scale==1]),'unit endpoint norm')
  for name in ('target_norm','lambda','rounding_budget','other_rounding_budget','actual_write_delta_norm'):require(exact(z[name][~mask],np.zeros(int((~mask).sum()),np.float64)),'inactive/source positivezero bytes '+name)
 require(np.all(gap<=left['rounding_budget']+right['rounding_budget']),'actual pair analytical bound')
 domains={}
 for name,index in (('all_nonsource',slice(1,512)),('primary_scored',slice(1,511))):
  den=float(np.sqrt(np.mean(target[index]**2)));num=float(np.sqrt(np.mean(gap[index]**2)))
  require(np.count_nonzero(a[index])==np.count_nonzero(b[index])==0 if den==0 else num<=1e-5*den,'fixed early RMS norm quality')
  domains[name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None}
 return domains

def compare_statistics(analysis,rows,anchors,cmp):
 expected=independent_statistics(rows,anchors)
 def table(items,keys):
  result={tuple(z[k] for k in keys):z for z in items};require(len(result)==len(items),'unique statistical wire rows');return result
 for family,want,keys in (('window_metrics',rows,('corpus','identity','window','view','mode')),('anchors',anchors,('corpus','identity','window','view','mode'))):
  found=table(analysis[family],keys);require(set(found)==set(want),'complete raw-derived metric wire')
  for key,z in want.items():
   for f in FIELDS:cmp.check(found[key][f],z[f],'window_metrics')
 def paired_record(x,y):
  require(x['n']==y['n']==4,'four paired passage clusters')
  for f in ('mean','ci95','per_window'):cmp.check(x[f],y[f],'paired_bootstrap')
 for family,keys in (('summaries',('corpus','identity','view','mode')),('contrasts',('corpus','identity','view','metric','name')),('paired_start_interactions',('corpus','view','metric','name'))):
  want=table(expected[family],keys);found=table(analysis[family],keys);require(set(want)==set(found),'full independent statistical domains')
  for key,x in want.items():
   y=found[key]
   if family=='summaries':
    require(set(y['metrics'])==set(FIELDS),'three scalar metric summaries')
    for f in FIELDS:paired_record(x['metrics'][f],y['metrics'][f])
    require(set(y['transforms'])=={'perplexity','ppl_ratio_baseline'},'nonlinear transformed paired means')
    for f in ('perplexity','ppl_ratio_baseline'):
     for part in ('value','ci95'):cmp.check(x['transforms'][f][part],y['transforms'][f][part],'paired_exponentials')
    paired_record(x['transforms']['ppl_ratio_baseline']['delta_nll'],y['transforms']['ppl_ratio_baseline']['delta_nll'])
   else:
    require(x['terms']==y['terms'] and x['primary']==y['primary'],'independent coefficient/primary identity')
    for f in ('kind','primary_view') if family=='contrasts' else ('scope',):require(x[f]==y[f],'contrast categorical scope')
    paired_record(x,y)
 require(sum(z['primary'] for z in analysis['contrasts'])==128,'all128nominalprimary/nooutcomeselection')
 return {'window_rows':1152,'anchor_rows':24,'summaries':288,'maps':19,'metrics':3,'contrasts':1368,'paired_starts':684,'primary':128,'shared_indices_sha256':fp(indices())['sha256']}



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

def fingerprint_abi(value,shape):
 require(set(value)=={'shape','dtype','sha256'} and value['shape']==list(shape) and value['dtype']=='torch.float32' and re.fullmatch('[0-9a-f]{64}',value['sha256']) is not None,'FP32 recorded fingerprint ABI')

def validate_native_gate_controls(operation,arm):
 gate=operation['native_gate_checks'][10]
 require(gate['source_context_exact'] is True,'native gate source context control')
 require(gate['unaffected_context_exact'] is True,'native gate unaffected context control')
 require(math.isfinite(gate['norm_match_error']) and gate['norm_match_error']<=1e-6,'frozen native gate norm control')
 # T20 uses original full context edits; it never enables legacy scaling.
 require(gate['legacy_errors']=={},'T20 has no legacy scaling path')
 unchanged=gate['unchanged_bypass']
 require(type(unchanged) is bool and type(gate['active']) is bool and gate['active'] is (not unchanged),'native gate activation/bypass consistency')
 if l10_kind(arm) in ('baseline','current'):require(unchanged,'unedited native context bypass')
 # The frozen hook's unaffected_context_exact means "not unchanged OR exact".
 # Only the actual unchanged branch implies whole-context equality.
 if unchanged:
  context=operation['context_records'][10]
  require(context['current']==context['delivered'],'unchanged native context exact fingerprint')

def gate_reconstruction(shard,alpha,u,v,cmp):
 for arm,gates in shard['gates'].items():
  if arm=='native_repeat':require(gates=={},'native repeat gates empty');continue
  require(set(gates)=={10},'L10 gate only');g=gates[10];values=shard['gate_norms'][arm][10]
  require(set(values)==GATE_KEYS,'gate norm field roster')
  uf=u.astype(np.float32);vf=v.astype(np.float32)
  s=(g['b'].T.astype(np.float64)-np.log(np.arange(1,512))[:,None]-alpha.astype(np.float32))@uf.astype(np.float64)
  cmp.check(g['s'],s,'coefficient_reconstruction',1e-4)
  target=g['b'] if l10_kind(arm) in ('baseline','current') else (g['b'].T-g['s'][:,None]*(uf if l10_kind(arm)=='remove' else vf)).T
  require(exact(target,g['target_b']),'full edit exact FP32 elementwise gate proposal')
  sigmoid=np.exp(-np.logaddexp(0,-g['target_b'].astype(np.float64)))
  cmp.check(g['proposed_target_mass'],sigmoid,'gate_probability_reconstruction',3e-6)
  unchanged=l10_kind(arm) in ('baseline','current') or exact(g['target_b'],g['b'])
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
 require(set(shard['snapshots'])==set(L10_SNAPSHOTS) and set(shard['readouts'])==set(ARMS),'five L10 snapshots and twelve endstates')
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
 return 5

def aggregate(values,scope):
 x=np.asarray(values,np.float64);require(x.ndim==1 and x.size and np.isfinite(x).all(),'finite norm aggregate')
 return {'scope':scope,'mean':float(x.mean()),'rms':float(np.sqrt(np.mean(x*x))),'max_abs':float(np.abs(x).max())}

def raw_metrics_and_norms(shard,cmp):
 """Validate saved numeric ABI before independent per-passage reductions."""
 meta=shard['metadata'];anchors=(meta['window'],meta['identity'])==(0,'natural');expected=set(ARMS)|({'native_repeat','native_identity'} if anchors else set())
 for family in ('metrics','gates','gate_norms','write_norms','l11_norms','operations'):require(set(shard[family])==expected,'main arm/anchor roster '+family)
 for family in ('suffix','suffix_gates','suffix_gate_norms','suffix_write_norms','suffix_l11_norms','suffix_operations'):require(set(shard[family])==set(ARMS),'suffix arm roster '+family)
 baseline=shard['metrics']['baseline'];normrows={'gate':{},'write':{},'l11':{}}
 for arm,z in shard['metrics'].items():
  require(set(z)==set(METRICS)|{'final_hidden_sha256'},'six numeric metrics plus actual whole-hidden commitment')
  require(re.fullmatch('[0-9a-f]{64}',z['final_hidden_sha256']) is not None,'actual hidden SHA')
  for name in METRICS:
   dtype=np.int64 if name=='pred' else np.bool_ if name.startswith('agreement') else np.float32
   require(z[name].dtype==dtype and z[name].shape==(511,) and np.isfinite(z[name]).all(),'main metric shape/dtype/finite')
  require(((z['pred']>=0)&(z['pred']<50257)).all(),'prediction vocabulary range')
  require(exact(z['kl_c'],z['kl_boundary']) and exact(z['agreement_c'],z['agreement_boundary']),'native000 shared baseline families')
  require(exact(z['agreement_boundary'],z['pred']==baseline['pred']),'agreement from actual saved predictions')
  if arm in ('baseline','current','native_repeat','native_identity'):
   require(all(exact(z[n],baseline[n]) for n in (*METRICS,'final_hidden_sha256')),'current and actual anchors exactly baseline')
   require(np.count_nonzero(z['kl_boundary'])==0,'baseline/current/anchor KL zeros')
  if arm=='native_repeat':
   require(all(shard[f][arm]=={} for f in ('gates','gate_norms','write_norms','l11_norms','operations')),'truly unhooked repeat has no hook data');continue
  if arm=='native_identity':
   for f in ('gates','gate_norms','write_norms','l11_norms'):require(exact(shard[f][arm],shard[f]['baseline']),'identity anchor saved hook arrays baseline')
   continue
  require(set(shard['gates'][arm])==set(shard['suffix_gates'][arm])=={10},'only L10 gate arrays')
  for name,a in shard['gates'][arm][10].items():
   require(set(shard['gates'][arm][10])=={'b','mass','target_b','target_mass','proposed_target_mass','s'},'six full gate fields')
   require(a.shape==((511,) if name=='s' else (12,511)) and a.dtype==np.float32 and np.isfinite(a).all(),'actual gate shape/dtype/finite')
  ss=shard['suffix'][arm];require(set(ss)==set(METRICS)|{'final_hidden_sha256','target_indices','logit_query_indices'},'suffix metric/commitment/index roster')
  require(re.fullmatch('[0-9a-f]{64}',ss['final_hidden_sha256']) is not None,'suffix actual full hidden SHA')
  require(exact(ss['target_indices'],np.arange(1,128,dtype=np.int64)) and exact(ss['logit_query_indices'],np.arange(128,dtype=np.int64)),'suffix physical prefix indices')
  for name in METRICS:require(exact(ss[name],z[name][:(127 if name=='loss' else 128)]),'exact saved suffix metric prefix')
  require(exact(shard['suffix_gates'][arm],{l:{k:a[:127] if a.ndim==1 else a[:,:127] for k,a in g.items()} for l,g in shard['gates'][arm].items()}),'exact suffix gate prefix')
  if l10_kind(arm) in ('baseline','current'):
   require(exact(shard['gates'][arm],shard['gates']['baseline']),'recipient/current L10 gate remains baseline')
  families=(('gate',shard['gate_norms'][arm][10],shard['suffix_gate_norms'][arm][10],GATE_KEYS),('write',shard['write_norms'][arm],shard['suffix_write_norms'][arm],COMMON|(MATCH if arm in DONORS else set())),('l11',shard['l11_norms'][arm],shard['suffix_l11_norms'][arm],L11_NORM_KEYS))
  for family,vals,svals,keys in families:
   require(set(vals)==set(svals)==keys,'exact norm field roster '+family)
   for name,a in vals.items():
    shape=(12,) if name=='source_v_norm' else (511,) if family=='gate' else (512,)
    require(a.shape==shape and a.dtype==(np.float32 if name=='s' else np.float64) and np.isfinite(a).all(),'saved norm shape/dtype/finite')
    if name not in ('s','signed_target_residual'):require((a>=0).all(),'nonnegative norms/budgets')
    if family=='gate':require(exact(svals[name],a if name=='source_v_norm' else a[:127]),'suffix gate norm bytes')
    else:
     require(svals[name].shape==(512,) and svals[name].dtype==np.float64 and np.isfinite(svals[name]).all() and exact(svals[name][:128],a[:128]),'full suffix norm domain and exact prefix')
     if name not in ('signed_target_residual',):require((svals[name]>=0).all(),'suffix nonnegative norm')
    if name=='lambda':require((a<=1).all() and (svals[name]<=1).all(),'lambda unit interval')
   if family=='write':
    for name in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'):
     require(vals[name][0]==svals[name][0]==0,'source unchanged write norm')
     if l10_kind(arm) in ('baseline','current'):require(np.count_nonzero(vals[name])==np.count_nonzero(svals[name])==0,'native recipient/current zero L10 deltas')
   for view,(lo,hi) in VIEWS.items():
    if family=='gate':lo-=1;hi-=1
    values={name:aggregate(a if name=='source_v_norm' else a[lo:hi],'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in vals.items()}
    key=(meta['corpus'],meta['identity'],meta['window'],view,'000__'+arm,11 if family=='l11' else 10)
    normrows[family][key]={'values':values}
    if family=='l11':normrows[family][key]['active_early_values']={name:aggregate(a[1:127],'active_E_queries1..126') for name,a in vals.items()}
 rows,anchor_rows=raw_window_rows(shard['metrics'],meta)
 return rows,anchor_rows,normrows

def compare_norm_summaries(analysis,normrows,cmp):
 keys=('corpus','identity','window','view','mode','layer')
 for family,want in normrows.items():
  records=analysis[family+'_norm_window_summaries'];found={tuple(z[k] for k in keys):z for z in records}
  require(len(found)==len(records)==len(want)==1152 and set(found)==set(want),'all1152 norm summary semantic keys '+family)
  for key,expected in want.items():
   for scope,values in expected.items():
    require(set(found[key][scope])==set(values),'complete norm aggregate keysets')
    for name,a in values.items():
     b=found[key][scope][name];require(set(a)==set(b) and a['scope']==b['scope'],'exact norm domain labels')
     for number in ('mean','rms','max_abs'):cmp.check(a[number],b[number],'norm_aggregates',1e-10,1e-12)

def donor_matching_analysis(shard,lookup,cmp,counts):
 key=tuple(shard['metadata'][k] for k in ('corpus','window','identity','boundary'));diagnostics=[]
 for variant,field,opfield in (('main','write_norms','operations'),('suffix','suffix_write_norms','suffix_operations')):
  values=shard[field];ops=shard[opfield];domains=check_donor_matching(values,cmp)
  target=values[DONORS[0]]['target_norm'];a,b=(values[d]['actual_write_delta_norm'] for d in DONORS);gap=np.abs(a-b)
  for name,domain in domains.items():
   domain['active_count']=126;domain['active_only_target']=aggregate(target[1:127],'active_E');domain['active_only_gap']=aggregate(gap[1:127],'active_E')
  full_quality=None
  for direction,arm in zip(('remove','orth'),DONORS):
   wc=ops[arm]['write_checks'];base=ops['baseline']['write_checks'];full=ops[direction+'_full']['write_checks']
   require(wc['paired_actual_delivery_complete'] is (direction=='orth'),'actual pair chronological completion')
   require(wc['counterpart_norm_scope']==('actual_delivered_same_support' if direction=='orth' else 'expected_CPU_rounded_not_yet_delivered_same_support'),'counterpart provenance')
   require(wc['support']=='early' and wc['target_scope']=='same_frozen_full_recipe_target_on_active_queries_zero_elsewhere','fixed early target recipe')
   require(wc['active_write_bytes_exact'] is True and wc['inactive_write_bytes_exact'] is True,'actual support byte flags')
   require(wc['baseline_write']==base['returned_write'] and wc['proposed_full_write']==full['returned_write'],'actual same-state full origin fingerprints')
   fingerprint_abi(wc['full_matched_array'],(512,768))
   if full_quality is None:full_quality=wc['full_recipe_matching_quality']
   else:require(exact(full_quality,wc['full_recipe_matching_quality']),'one full recipe quality across directions')
   compare_tree(runtime_quality(target,a,b),wc['matching_quality'],cmp,'recorded_actual_matching_quality')
   counts['matched_arm_variant_records']+=1;counts['direction_recipe_fingerprints']+=1
  saved=lookup[(*key,variant,'early')]
  require(saved['support']=='early' and saved['active_queries']==saved['active_scored_queries']==126,'matching report fixed support')
  compare_tree(domains,saved['domains'],cmp,'matching_RMS_diagnostics')
  cmp.check(float(gap.max()),saved['maximum_absolute_pair_gap'],'matching_maximum',1e-10,1e-12)
  require(saved['scope']=='Actual rounded donor norms after both donors; full proposals hypothetical outside E. L11 K/V doses not matched.','L10 match is not L11 dose match')
  counts['support_pairs']+=1;counts['RMS_domain_checks']+=2;counts['full_recipe_origins']+=1
  positive=target[1:]>0;relative=gap[1:][positive]/target[1:][positive]
  diagnostics.append({'corpus':key[0],'window':key[1],'identity':key[2],'boundary':key[3],'variant':variant,'domains':domains,'maximum_absolute_pair_gap':float(gap.max()),'positive_target_relative_gap_domain':'q1..511','relative_quantile_probabilities':[0,.25,.5,.75,.95,.99,1],'positive_target_relative_gap_quantiles':np.quantile(relative,[0,.25,.5,.75,.95,.99,1]).tolist() if relative.size else None,'zero_target_queries':int((target==0).sum()),'active_zero_target_queries':int((target[1:127]==0).sum()),'quantized_to_zero_positive_targets':[int(((z==0)&(target>0)).sum()) for z in (a,b)]})
 return diagnostics

def selected_l11_panels(shard,tokens,journal_events,cmp,counts):
 meta=shard['metadata'];key=tuple(meta[k] for k in ('corpus','window','identity','boundary'));selected=key[:3]==('wiki_validation',0,'natural')
 if not selected:
  require(shard['l11_panels']==shard['readouts']==shard['timing_samples']=={} and 'l11_mask_reference' not in shard,'no unselected full L11/raw mask/endstates');return
 require(set(shard['l11_panels'])==set(L11_SNAPSHOTS) and set(shard['readouts'])==set(shard['timing_samples'])==set(ARMS),'predesignated5/9/12 panel roster')
 panels=shard['l11_panels'];base=panels['baseline'];mask=shard['l11_mask_reference']
 require(mask.dtype==np.float32 and mask.shape==(1,1,512,512) and not np.isnan(mask).any() and not np.isposinf(mask).any(),'explicit actual L11 mask reference')
 base_projection=base['original_projection'];raw_fields={'c_attn_input','original_projection','delivered_qkv','attention_context','attention_write','block_hidden'}
 for arm,panel in panels.items():
  require(set(panel)==raw_fields|{'mask_fingerprint','mask_reference_byte_exact'},'exact six raw arrays and mask metadata')
  op=shard['operations'][arm];record=op['l11_records'];fps=op['l11_fingerprints']
  verify_selected_L11_mask(mask,panel,record,op['mask_records'][11])
  for name in raw_fields:
   a=panel[name];shape=(1,512,2304 if name in ('original_projection','delivered_qkv') else 768)
   require(a.shape==shape and a.dtype==np.float32 and np.isfinite(a).all() and fp(a)==fps[name]==record['fields'][name],'actual selected field fingerprint '+name)
  original=panel['original_projection'];delivered=panel['delivered_qkv']
  if arm in RECIPIENTS:
   require(exact(panel['c_attn_input'],base['c_attn_input']) and exact(original,base_projection),'recipient native incoming/residual and natural projection baseline')
   direction=arm.split('_')[0];donor=panels[direction+'_write_matched_early']['original_projection'];path=arm.rsplit('_',1)[-1].upper()
  else:
   donor=None;path=None
   if arm in DONORS:
    for name in ('c_attn_input','original_projection'):require(exact(panel[name][:,127:],base[name][:,127:]) and exact(panel[name][:,:1],base[name][:,:1]),'donor later/source natural states baseline')
  verify_recorded_splice(original,delivered,donor,path,fps,record)
  q,k,v=decode_qkv(delivered)
  for name,a in (('q',q),('k',k)):
   require(fp(a.transpose(0,2,1,3).reshape(1,512,768))==op['projection_records'][11]['after'][name],'native observer actually sees delivered Q/K')
  require(fp(v[:,:,1:].transpose(0,2,1,3).reshape(1,511,768))==op['projection_records'][11]['after']['nonsource_v'],'native observer actual transplanted V')
  reconstructed=per_query_l11_norms(original,delivered,base_projection)
  for name,a in reconstructed.items():cmp.check(a,shard['l11_norms'][arm][name],'selected_L11_norm_reconstruction',1e-10,1e-12)
  reconstruct_l11_context(delivered,mask,panel['attention_context'],cmp)
  require(fp(panel['attention_context'])==op['context_records'][11]['current']==op['context_records'][11]['delivered'],'actual L11 context remains native to delivered QKV')
  counts['full_L11_snapshots']+=1
 for arm in ARMS:
  op=shard['operations'][arm];sample=shard['timing_samples'][arm];readout=shard['readouts'][arm]
  require(set(sample)==set(TIMING_NODES)|{'query_rows'} and exact(sample['query_rows'],np.array(TIMING_ROWS,np.int64)),'five fixed timing queries')
  for node in TIMING_NODES:
   a=sample[node];require(a.shape==(1,5,2304 if node=='l11_qkv' else 768) and a.dtype==np.float32 and np.isfinite(a).all(),'actual selected timing rows')
   rows=list(range(5)) if arm in ('baseline','current') or (arm in RECIPIENTS and node!='l11_qkv') else [2,3,4] if arm in DONORS else []
   require(exact(a[:,rows],shard['timing_samples']['baseline'][node][:,rows]),'selected protected timing rows bytes')
   if arm in panels and node=='l11_qkv':require(exact(a,panels[arm]['delivered_qkv'][:,TIMING_ROWS]),'timing sample delivered-QKV raw binding')
   counts['raw_timing_rows']+=5
  require(readout['hidden'].shape==(1,512,768) and readout['hidden'].dtype==np.float32 and np.isfinite(readout['hidden']).all(),'selected complete endstate')
  require(readout['logits'].shape==(4,50257) and readout['logits'].dtype==np.float32 and np.isfinite(readout['logits']).all(),'selected four actual vocabulary rows')
  require(exact(readout['query_rows'],np.array(READOUT_ROWS,np.int64)) and exact(readout['targets'],tokens[np.array(READOUT_ROWS)+1]),'fixed actual readout queries/targets')
  require(readout['token_sha256']==fp(tokens)['sha256'] and readout['hidden_fingerprint']==fp(readout['hidden'])==journal_events[(*key,arm,'main')]['hidden_fingerprint']==op['l11_fingerprints']['final_hidden'],'endstate source/journal binding')
  if arm in ('baseline','current'):
   require(exact(readout['hidden'],shard['readouts']['baseline']['hidden']) and exact(readout['logits'],shard['readouts']['baseline']['logits']),'selected current full hidden/four vocabulary controls')
  if arm.endswith('_l11_kv'):
   donor=arm.split('_')[0]+'_write_matched_early';receipt=op['l11_records']['combined_KV_closure']
   for name in ('attention_context','attention_write','block_hidden'):
    require(exact(panels[arm][name][:,127:],panels[donor][name][:,127:]),'raw selected later combined-KV closure '+name)
    require(fp(panels[arm][name][:,127:])==receipt['checks'][name]['actual']==receipt['checks'][name]['donor'],'raw closure fingerprint binding')
   require(exact(readout['hidden'][:,127:],shard['readouts'][donor]['hidden'][:,127:]) and exact(readout['logits'][2:],shard['readouts'][donor]['logits'][2:]),'raw later finalhidden and predetermined q127/q510 donor readout closure')
   require(fp(readout['hidden'][:,127:])==receipt['checks']['final_hidden']['actual'],'raw full later finalhidden closure commitment')
  counts['full_endstates']+=1;counts['raw_vocabulary_rows']+=4

def control_and_operation_provenance(shard,manifest,tokens,journal_events,cmp,counts):
 meta=shard['metadata'];key=tuple(meta[k] for k in ('corpus','window','identity','boundary'));anchors=key[1:3]==(0,'natural');expected=[]
 for arm in ARMS:
  if arm=='baseline' and anchors:expected.extend((a,None,None) for a in ('native_repeat','native_identity'))
  if arm!='baseline':expected.append(('source_q0',arm,'main'))
  if arm=='current':expected.append(('current',arm,'main'))
  if arm!='baseline':expected.append(('source_q0',arm,'suffix'))
  if arm=='current':expected.append(('current',arm,'suffix'))
  expected.append(('suffix',arm,None))
 require([(z['kind'],z.get('arm'),z.get('variant')) for z in shard['checks']]==expected,'exact source/current/suffix/anchor comparison chronology')
 for check in shard['checks']:
  flags=[v for k,v in check.items() if k.endswith(('_exact','_passed'))];require(flags and all(type(v)is bool and v for v in flags),'all actual exact comparison flags pass');counts[check['kind']]+=1
 source=shard['source_reference'];require(set(source)==set(range(12)),'all12 source references')
 for layer,fields in source.items():require(set(fields)=={'q','k','v','hidden'} and all(a.shape==(1,768) and a.dtype==np.float32 and np.isfinite(a).all() for a in fields.values()),'actual q0 source arrays')
 cumulative={}
 for arm in ARMS:
  for variant,field,metricfield in (('main','operations','metrics'),('suffix','suffix_operations','suffix')):
   op=shard[field][arm];actual=tokens.copy()
   if variant=='suffix':actual[128:]=np.roll(actual[128:],1)
   expectedkey={**meta,'variant':variant,'actual_token_sha256':fp(actual)['sha256'],'F_digest':F_DIGEST,'freeze_digest':manifest['freeze_digest']}
   require(op['write_cache_key']==expectedkey,'exact input/start/variant/token/F/freeze cache key')
   for f in ('projection_records','context_records','mask_records','native_diag'):require(set(op[f])==set(range(4,12)),'all8 native layer witnesses')
   require((op['projection_checks'],op['context_checks'],op['query_checks'])==(8,8,8),'native bridge counts')
   require(exact(op['source_fingerprints'],nested_fp(source)),'actual q0 fingerprint references')
   if variant=='main':require(op['gate_fingerprints']==nested_fp(shard['gates'][arm]),'main saved gate fingerprint')
   for layer in range(4,12):
    p=op['projection_records'][layer];require(p['zeroed'] is False,'native000 no source-V zeroing')
    require(all(p['before'][n]==p['after'][n] for n in ('q','k','nonsource_v','source_v')) and p['after']['source_v']==fp(source[layer]['v']),'native observer preserves actual delivered projection')
    maskrecord=op['mask_records'][layer];require(maskrecord['model_argument_preserved'] is True and maskrecord['kind'] in ('implicit_causal','preserved_explicit'),'native model mask preserved')
    diag=op['native_diag'][layer]
    for name in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):require(math.isfinite(diag[name]) and diag[name]<=1e-4,'native hard context bridge')
    require(diag['min_mass']>=0 and diag['max_mass']<=1+1e-6,'allocation roundoff guard')
   validate_native_gate_controls(op,arm)
   wc=op['write_checks'];require(wc['source_write_bytes_exact'] is True and wc['native_mask_preserved'] is True and wc['proposed_context_and_actual_write_distinct'] is True,'actual write controls')
   if arm in RECIPIENTS:require(wc['recipient_actual_L10_write_baseline_bytes_exact'] is True,'recipient native L10 delivered write')
   record=op['l11_records'];fps=op['l11_fingerprints'];tc=op['timing_checks'];base=shard[field]['baseline']
   require(record['cache_key']==expectedkey and record['arm']==arm and record['layout_version']==LAYOUT_VERSION,'actual L11 cache/layout identity')
   require(record['head_count']==12 and record['head_dim']==64 and record['earlier_query_slice']==[1,127] and record['later_query_slice']==[127,512],'fixed allhead temporal domains')
   for name in ('natural_projection_distinct_from_delivered','native_observer_sees_delivered','native_mask_preserved','expected_splice_bytes_exact','all_Q_preserved','source_projection_preserved','later_KV_preserved','effective_mask_baseline_byte_exact','actual_model_output_observed','actual_output_shape_dtype_finite_validated'):require(record[name] is True,'successful L11 actual record '+name)
   require(record['closure_required'] is arm.endswith('_l11_kv') and record['recipient_upstream_passed'] is (True if arm in RECIPIENTS else None),'path-specific required guards')
   require(record['fields']==fps and set(fps)==set(L11_FIELDS),'all8 actual L11/endstate fingerprints')
   for name,fingerprint in fps.items():fingerprint_abi(fingerprint,(1,512,2304 if name in ('original_projection','expected_qkv','delivered_qkv') else 768))
   require(record['original_projection']==fps['original_projection'] and record['delivered_qkv']==fps['delivered_qkv'],'original/actual projection commitments')
   require(record['effective_mask']==base['l11_records']['effective_mask']==op['mask_records'][11]['diagnostic_effective'],'actual own-variant L11 mask')
   event=journal_events[(*key,arm,variant)];require(event['hidden_fingerprint']==fps['final_hidden'] and event['hidden_fingerprint']['sha256']==shard[metricfield][arm]['final_hidden_sha256'],'actual returned/processed full hidden provenance')
   fingerprint_abi(event['logits_fingerprint'],(512,50257))
   protected=record.get('protected',{})
   names=('c_attn_input','original_projection','attention_context','attention_write','block_hidden','final_hidden')
   require(set(protected)==(set() if arm=='baseline' else set(names)),'exact L11 protected-field roster')
   for name,g in protected.items():
    sl=[None,None] if arm=='current' or (arm in RECIPIENTS and name in ('c_attn_input','original_projection')) else [127,None] if arm in DONORS and name in ('c_attn_input','original_projection') else [0,1]
    width=2304 if name=='original_projection' else 768;n=512 if sl==[None,None] else 385 if sl==[127,None] else 1
    require(g['query_slice']==sl and g['byte_exact'] is True and g['actual']==g['reference'],'actual L11 protected slice bytes');fingerprint_abi(g['actual'],(1,n,width))
    if n==512:require(fps[name]==g['actual']==base['l11_fingerprints'][name],'whole protected L11 field/raw fingerprint binding')
   if arm!='baseline':require(record['actual_readout_control']=={'byte_exact':True,'query_slice':[None,None] if arm=='current' else [0,1]},'actual model-end readout identity/source guard')
   require(tc['observed_nodes']==list(TIMING_NODES) and set(op['timing_fingerprints'])==set(TIMING_NODES),'four actual timing nodes before finish')
   require(tc['support']==('early' if arm in DONORS else None) and tc['later_query_start']==127 and tc['same_full_matched_array_across_supports'] is (arm in DONORS),'timing support semantics')
   for node,fingerprint in op['timing_fingerprints'].items():
    width=2304 if node=='l11_qkv' else 768;fingerprint_abi(fingerprint,(1,512,width))
    if arm=='baseline':require(node not in tc,'baseline reference is not fabricated guard')
    else:
     sl=[None,None] if arm=='current' or (arm in RECIPIENTS and node!='l11_qkv') else [127,None] if arm in DONORS else [0,1];g=tc[node];n=512 if sl==[None,None] else 385 if sl==[127,None] else 1
     require(g['query_slice']==sl and g['byte_exact'] is True and g['actual']==g['baseline'],'timing protected slice bytes');fingerprint_abi(g['actual'],(1,n,width))
     if n==512:require(fingerprint==g['actual']==base['timing_fingerprints'][node],'whole protected timing node/raw fingerprint binding')
    counts['science_timing_node_fingerprints']+=1
   require(op['timing_fingerprints']['l11_qkv']==fps['delivered_qkv'],'timing observes delivered,not natural original QKV')
   if arm in DONORS:require(tc['donor_later_passed'] is True,'all4 donor later state guards');counts['donor_later']+=1
   else:require(tc['donor_later_passed'] is None,'no fictional donor guard')
   if arm in RECIPIENTS:counts['recipient_upstream']+=1
   if arm.endswith('_l11_kv'):
    direction=arm.split('_')[0];receipt=record['combined_KV_closure'];closurekey=direction+'/'+variant;cumulative[closurekey]=copy.deepcopy(receipt)
    require(receipt==op['closure_checks'][closurekey] and receipt['direction']==direction and receipt['variant']==variant,'actual current combined-KV closure receipt')
    require(receipt['cache_key']=={**expectedkey,'direction':direction,'donor_arm':direction+'_write_matched_early','support':'early','layout_version':LAYOUT_VERSION},'exact actual donor closure cache provenance')
    require(receipt['query_slice']==[127,512] and all(receipt[k] is True for k in ('actual_model_output_observed','all_byte_exact','passed','references_retired_after_all_checks')),'complete actual closure before retirement')
    require(set(receipt['checks'])=={'attention_context','attention_write','block_hidden','final_hidden','logits'},'five actual closure operands')
    for name,g in receipt['checks'].items():
     require(g['byte_exact'] is True and g['actual']==g['donor'],'full actual combined closure byte equality');fingerprint_abi(g['actual'],(385,50257) if name=='logits' else (1,385,768))
    donor=direction+'_write_matched_early'
    for name in METRICS:require(exact(shard[metricfield][arm][name][127:],shard[metricfield][donor][name][127:]),'saved later actual KV/donor metric bytes')
    counts['KV_closure']+=1
   require(exact(op['closure_checks'],cumulative),'all cumulative receipts reached exactly once and preserved unchanged in chronology')
   require(all(r['passed'] is True for r in op['closure_checks'].values()),'no failed cumulative receipt')
   if arm in RECIPIENTS and not arm.endswith('_l11_kv'):require(len(cumulative)==4,'all four closures before any single K/V')
   active=record['active_summary'];require(active['query_slice']==[1,127] and active['query_count']==126 and set(active['norms'])==L11_NORM_KEYS,'active L11 norm domain')
   values=shard['l11_norms' if variant=='main' else 'suffix_l11_norms'][arm]
   for name,a in values.items():
    want={'mean':float(a[1:127].mean()),'RMS':float(np.sqrt(np.mean(a[1:127]**2))),'max':float(a[1:127].max())};compare_tree(want,active['norms'][name],cmp,'recorded_L11_active_norms')
   if variant=='suffix':
    main=shard['operations'][arm]
    for f in ('mask_records','source_fingerprints','prefix_context_fingerprints'):require(exact(op[f],main[f]),'suffix runtime prefix '+f)
   counts['science_timing_calls']+=1
 # Truly unhooked repeat carries no operations. Identity anchor has genuine
 # observers, but is not one of the24 advancing scientific arm/variant calls.
 if anchors:
  op=shard['operations']['native_identity'];baseline=shard['operations']['baseline'];e=journal_events[(*key,'native_identity','anchor')]
  require(op['l11_records']['arm']=='baseline' and op['closure_checks']=={} and op['l11_records']['actual_output_shape_dtype_finite_validated'] is True,'identity observe-only baseline route')
  require(op['l11_fingerprints']==baseline['l11_fingerprints'] and op['timing_fingerprints']==baseline['timing_fingerprints'] and e['hidden_fingerprint']==op['l11_fingerprints']['final_hidden'],'actual identity node/fullhidden equality')
  require(op['timing_checks']['observed_nodes']==list(TIMING_NODES),'identity observer nodes')
  validate_native_gate_controls(op,'native_identity')
  require(op['source_fingerprints']==baseline['source_fingerprints'] and op['mask_records']==baseline['mask_records'],'identity source/mask actual witnesses')
  for layer in range(4,12):
   for name in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):require(math.isfinite(op['native_diag'][layer][name]) and op['native_diag'][layer][name]<=1e-4,'identity native context bridge')
  for node in TIMING_NODES:
   g=op['timing_checks'][node];require(g['byte_exact'] is True and g['query_slice']==[None,None] and g['actual']==g['baseline']==baseline['timing_fingerprints'][node],'identity complete timing node guard');counts['anchor_timing_node_fingerprints']+=1
  counts['anchor_timing_calls']+=1
 for variant in ('main','suffix'):
  baseline=journal_events[(*key,'baseline',variant)];current=journal_events[(*key,'current',variant)]
  require(all(baseline[k]==current[k] for k in ('hidden_fingerprint','logits_fingerprint')),'actual whole512 current control fingerprints')
 if anchors:
  baseline=journal_events[(*key,'baseline','main')]
  for arm in ('native_repeat','native_identity'):require(all(journal_events[(*key,arm,'anchor')][k]==baseline[k] for k in ('hidden_fingerprint','logits_fingerprint')),'whole512 actual anchor fingerprints')

def provenance(out,review):
 m=read(out/'manifest.json');validate_freeze_digest(m);require(m['status']=='completed' and m['execution_type']=='planned_sequential_segments','completed planned study')
 require(review['run_manifest_sha256']==sha(out/'manifest.json'),'reviewed run manifest')
 require(m['input_dependencies']==review['input_dependencies'] and m['metadata_dependencies']==review['metadata_dependencies'],'independently accepted input and metadata dependency bindings')
 for p in out.rglob('*'):require(p.name not in ('merge_failure.json','checkpoint_failure.json','failure_state.npz','failure_stream_error.json') and not p.name.endswith('__partial.npz') and not p.name.startswith('journal_preservation_error_'),'no failure/partial markers')
 for k,n in (('forward_attempts',776),('forward_passes',776),('fully_processed_forwards',776),('evaluation_forwards',384),('suffix_forwards',384),('anchor_forwards',8),('calibration_forwards',0)):
  require(m[k]==n,'complete count '+k)
 require(m['model_outputs_observed']==776 and m['observation_origins']=={'hook_model_end':772,'runner_after_return':4},'actual total observation routes')
 require(m['control_counts']=={'current':64,'suffix':384,'donor_later':128,'recipient_upstream':384,'KV_closure':128,'source_q0':704,'anchors':8},'complete actual controls')
 require(m['all_final_guards_passed'] is True and m['parameter_versions_unchanged'] is True,'final composite guards')
 require(m['model_hash_before']==m['model_hash_after']=='2eaf003b11dbb54c7fe511397a4c77d8fc734983b8dbf66794eed8e546073527','model parameter digest')
 require(m['precision_settings_before']==m['precision_settings_after'],'precision guard')
 require(m['precision_settings_before']['float32_matmul_precision']=='highest' and m['precision_settings_before']['cuda_matmul_allow_tf32'] is False,'full FP32 matmul scope')
 require(m['runtime_versions']=={'python':'3.12.3','torch':'2.11.0+cu130','transformers':'4.57.6','numpy':'1.26.4','cuda_runtime':'13.0'},'pinned native runtime')
 require(m['model_id']=='openai-community/gpt2' and m['revision']=='75e09b43581151bd1d9ef6700faa605df408979f' and m['dtype']=='float32' and m['attention']=='sdpa','model/revision/precision/backend pin')
 require(m['physical_execution_lengths']==[512] and m['calibration_fit'] is False,'physical512 and zero fitting')
 require(m['boundaries']==['000'] and m['arms']==list(ARMS) and m['score_views']=={'full':[2,511],'early':[2,127],'later':[128,511]},'exact native twelve-arm score/exec domains')
 require(m['source_hashes']['analyze_l11_key_value_donor.py']==ANALYZER_SHA and m['source_hashes']==APPROVED_SOURCE_HASHES and m['calibration_map_digest']==F_DIGEST,'analyzer/F frozen')
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
  require(sm['model_outputs_observed']==388 and sm['observation_origins']=={'hook_model_end':386,'runner_after_return':2},'actual segment observation routes')
  require(sm['control_counts']=={'current':32,'suffix':192,'donor_later':64,'recipient_upstream':192,'KV_closure':64,'source_q0':352,'anchors':4},'segment controls')
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

def original_checkpoint_provenance(args,manifest,inventory):
 report={}
 for label,base,corpora in (('A',args.segment_a,CORPORA[:2]),('B',args.segment_b,CORPORA[2:])):
  require(base.is_dir() and not base.is_symlink(),'original segment directory')
  copied=args.run/'execution_provenance'/label
  for path in base.rglob('*'):require(path.name not in ('failure_state.npz','checkpoint_failure.json','merge_failure.json','failure_stream_error.json') and not path.name.endswith('__partial.npz') and not path.name.startswith('journal_preservation_error_'),'original failure evidence blocks accepted completion')
  for name in ('manifest.json','numeric_inventory.json','journal.jsonl','disk_checks.json'):require(sha(base/name)==sha(copied/name),'original/copy immutable '+name)
  sm=read(base/'manifest.json');si=read(base/'numeric_inventory.json')
  require(sm['failure_limit_bytes']==EFFECTIVE_FAILURE_LIMIT_BYTES and sm['normal_limit_bytes']==512*1024**2 and sm['disk_reserve_bytes']==2*1024**3,'prospective amended resource caps')
  require(sm['model_load_attempted'] is True and sm['model_object_returned'] is True and sm['model_loaded'] is True,'one reached complete segment model-load route')
  require(sm['host_mem_available_minimum_bytes']==6*1024**3 and sm['host_mem_available_before_model_kib']*1024>=6*1024**3,'recorded pre-load6GiB host headroom')
  header_events=[e for e in read_journal_events(base/'journal.jsonl') if e['event']=='known_header_admission']
  require(sm['header_admission_before_model']=={k:v for k,v in header_events[0].items() if k not in ('event','previous_sha256')} and sm['header_admission_after_model']=={k:v for k,v in header_events[1].items() if k not in ('event','previous_sha256')},'actual pre/post-load metadata references exact journal receipts')
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
   require(progress['model_outputs_observed']==expected and progress['observation_origins']=={'hook_model_end':expected-sum(1 for z in roster(corpora)[:expected] if z[4]=='native_repeat'),'runner_after_return':sum(1 for z in roster(corpora)[:expected] if z[4]=='native_repeat')},'checkpoint actual observation counts by route')
  require(sha(checkpoints[-1]/'manifest.json')==sha(base/'manifest.json'),'canonical final versioned manifest exact')
  disks=read(base/'disk_checks.json')
  for d in disks:
   require(d['free_bytes']>=d['required_free_bytes'] and d['reserve_bytes']==2*1024**3,'recorded reserve gate')
   require(d['used_bytes']+d['projected_bytes']<=512*1024**2 and d['failure_write'] is False,'normal cap/no failure write')
   require(d['remaining_normal_cap']==max(0,512*1024**2-d['used_bytes']),'complete remaining cap reserve')
   require(d['required_free_bytes']==d['reserve_bytes']+d['outstanding_other_caps']+EFFECTIVE_FAILURE_LIMIT_BYTES+d['remaining_normal_cap'],'all outstanding caps')
  require(sm['peak_process_rss_kib']*1024<=4*1024**3,'recorded primary-model segment CPU peak')
  report[label]={'versioned_checkpoints':19,'disk_checks':len(disks),'raw_shards':len(si),'source_and_copied_metadata_exact':True}
 checkpoints=sorted((args.run/'checkpoints').iterdir());require([p.name for p in checkpoints]==['000001','000002','000003'],'merger checkpoint sequence')
 for cp,n in zip(checkpoints,(16,32,32)):require(len(read(cp/'numeric_inventory.json'))==n,'merger committed copy prefix')
 return report

def read_journal_events(path):
 return [json.loads(line) for line in Path(path).read_bytes().splitlines()]

def journal(path,expected):
 """Exact actual output observation routes, not inferred return counts."""
 chain='0'*64;events=[];counts=collections.Counter();byrole={};byattempt=collections.defaultdict(list)
 for line in Path(path).read_bytes().splitlines(keepends=True):
  require(line.endswith(b'\n'),'complete journal line');event=json.loads(line);body=line[:-1]
  require(body==json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),'canonical journal bytes')
  require(event['previous_sha256']==chain,'actual hash chain');chain=hashlib.sha256(body).hexdigest();events.append(event);counts[event['event']]+=1
 forward=('forward_intent','forward_attempt','model_output_observed','model_returned','forward_completed')
 require(set(counts)==set(forward)|{'known_header_admission','pre_encode_admission','committed'},'exact successful journal event universe')
 for kind in forward:
  selected=[(at,e) for at,e in enumerate(events) if e['event']==kind];require(len(selected)==len(expected),'all actual '+kind)
  for j,(at,e) in enumerate(selected):
   role=tuple(e['identity'][k] for k in ('corpus','window','identity','boundary','arm','phase'))
   require(role==expected[j] and e.get('attempt',e.get('next_attempt'))==j+1,'exact role/attempt chronology '+kind)
   if kind in ('model_returned','forward_completed'):require(e['completed']==j+1,'actual returned count')
   if kind=='model_output_observed':
    origin='runner_after_return' if role[4]=='native_repeat' else 'hook_model_end'
    require(e['observed']==j+1 and e['observer_origin']==origin,'genuine unhooked versus end-hook observation')
    hook=e['hook_event'];require(hook['arm']==('baseline' if role[4].startswith('native_') else role[4]) and hook['variant']==('suffix' if role[5]=='suffix' else 'main'),'observer actual arm/variant')
    intent=next(v for v in events[:at] if v['event']=='forward_intent' and v['next_attempt']==j+1)
    require(hook['actual_token_sha256']==intent['token_fingerprint']['sha256'],'actual observed input commitment')
    if origin=='hook_model_end':require(hook['event']=='model_output_observed' and hook['observe_only'] is (role[4]=='native_identity'),'owned hook versus identity-anchor route')
    else:require('cache_key' not in hook and 'observe_only' not in hook,'unhooked anchor has no fabricated hook cache/flag')
   if kind=='forward_completed':
    require(e['processed']==j+1,'actual fully processed count');fingerprint_abi(e['hidden_fingerprint'],(1,512,768));fingerprint_abi(e['logits_fingerprint'],(512,50257));byrole[role]=e
   byattempt[j+1].append((at,kind))
 previous=-1
 for attempt,seq in byattempt.items():
  ordered=sorted(seq);names=[kind for _,kind in ordered];wanted=list(forward)
  if expected[attempt-1][4]=='native_repeat':wanted[2:4]=['model_returned','model_output_observed']
  require(names==wanted and ordered[0][0]>previous,'sequential invoke/observation/return/processing order');previous=ordered[-1][0]
 require(collections.Counter(e['observer_origin'] for e in events if e['event']=='model_output_observed')=={'hook_model_end':386,'runner_after_return':2},'segment actual386+2 observation routes')
 headers=[(at,e) for at,e in enumerate(events) if e['event']=='known_header_admission']
 require(len(headers)==390 and [e['stage'] for _,e in headers]==['before_model_load','after_model_load_before_first_forward']+['before_forward:'+str(j+1) for j in range(388)],'preload/afterload/every-forward exact header admission roster')
 first_intent=next(at for at,e in enumerate(events) if e['event']=='forward_intent');require(headers[1][0]<first_intent,'both model admission gates before first actual intent')
 for at,e in headers:
  require(e['passed'] is True and e['rss_limit_bytes']==4*1024**3 and e['non_evidence_limit_bytes']==2240*1024**2,'recorded prospective primary-model4GiB/R02GiB header gate')
  require(e['non_evidence_observed_bytes']==max(0,e['current_rss_bytes']-e['owned_CPU_backing_storage_bytes']) and e['non_evidence_observed_bytes']<=e['non_evidence_limit_bytes'],'header observed base calculation')
  require(e['known_failure_header_raw_upper_bytes']==760300838 and e['known_tensor_contiguous_retained_copy_upper_bytes']==e['known_failure_header_raw_upper_bytes'],'known original owners plus retained per-occurrence contiguous copies')
  require(e['known_tensor_identity_contiguous_temporary_upper_bytes']==e['known_identity_byte_string_upper_bytes']==e['known_largest_identity_temporary_bytes']==512*50257*4,'known independent identity contiguous conversion and byte-string temporaries')
  require(e['known_header_phase_upper_bytes']==e['non_evidence_observed_bytes']+e['known_failure_header_raw_upper_bytes']+e['known_tensor_contiguous_retained_copy_upper_bytes']+e['known_tensor_identity_contiguous_temporary_upper_bytes']+e['known_identity_byte_string_upper_bytes']+96*1024**2 and e['known_header_phase_upper_bytes']<=e['rss_limit_bytes'],'recorded known complete header upper arithmetic including retained and transient copies')
  if e['stage'].startswith('before_forward:'):
   n=int(e['stage'].split(':')[1]);intent_at=next(j for j,v in enumerate(events) if v['event']=='forward_intent' and v['next_attempt']==n)
   require(at<intent_at and (n==1 or at>max(j for j,v in enumerate(events) if v['event']=='forward_completed' and v['attempt']==n-1)),'admission between previous processed return and next intent')
 corpora=tuple(dict.fromkeys(role[0] for role in expected));paths=[f'inputs/{c}__{w}__{i}__000.npz' for c,w,i in itertools.product(corpora,range(4),('natural','bos'))]
 commits=[(at,e) for at,e in enumerate(events) if e['event']=='committed'];admissions=[(at,e) for at,e in enumerate(events) if e['event']=='pre_encode_admission']
 require([e['path'] for _,e in commits]==[e['path'] for _,e in admissions]==paths,'sixteen ordered complete normal admissions/commits')
 for (commit_at,commit),(encode_at,admit) in zip(commits,admissions):
  c,w,i,b=Path(commit['path']).stem.split('__');group=(c,int(w),i,b)
  complete=[at for at,e in enumerate(events) if e['event']=='forward_completed' and tuple(e['identity'][k] for k in ('corpus','window','identity','boundary'))==group]
  require(len(complete)==(26 if (int(w),i)==(0,'natural') else 24) and max(complete)<encode_at<commit_at,'complete24/26 calls before normal encoding then commit')
  require(admit['failure_write'] is False and all(admit[k] is True for k in ('buffered_admitted','non_evidence_budget_passed','normalization_budget_passed')),'successful complete buffered admission')
  require(admit['header_phase_upper_bytes']<=admit['rss_limit_bytes']==4*1024**3 and admit['buffered_phase_upper_bytes']<=admit['rss_limit_bytes'],'recorded normal pre-encode primary-model RSS upper')
 return {'final_sha256':chain,'counts':dict(counts),'byrole':byrole,'events':events,'admissions':{e['path']:e for _,e in admissions}}

if __name__=='__main__':main()
