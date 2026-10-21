"""Independent T19 saved-array audit. No runner/analyzer imports or model loading.

Policy is frozen separately before outcomes. All writes are new review artifacts;
raw arrays, journals, interrupted/recovery/composite directories remain untouched.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,collections,datetime,gc,hashlib,itertools,json,math,resource,time,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];REVIEW=ROOT/'review'
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
BITS=tuple(format(i,'03b') for i in range(8))
ORDER=('000','111','001','010','011','100','101','110')
MODES=tuple(b+'__'+op for b in BITS for op in ('baseline','current','remove','orth'))+('111__remove_scale','111__orth_scale')
RUN_MODES=tuple(b+'__'+op for b in ORDER for op in (('baseline','remove','remove_scale','orth','orth_scale','current') if b=='111' else ('baseline','current','remove','orth')))
SNAPS=('000__baseline','000__remove','000__orth','111__baseline','111__remove','111__orth','111__remove_scale','111__orth_scale')
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
NORM_VIEWS={'full':(0,510),'early':(0,126),'later':(126,510)}
FIELDS=('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
NORM_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
ORIGINAL_SHA='7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f'
ADAPTER_SHA='457434e7adaddb984cddbf4a09778e49a36f7b2fa3aa0107816c4c0d0c5507d0'
MERGER_SHA='3b37a5e815d0c9fe8b4c67e904742f6ba1a2a9c244b0ba5c068360786b56c818'
ANALYZER_SHA='22f6526207034cc89f06dadde1877d35088836d010c5630bcb498a6e35f328d3'
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'

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
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(exact(v,w) for v,w in zip(a,b))
 return type(a)==type(b) and a==b
def load_npz(path):
 """Independent no-pickle decoder, validating every array identity and member."""
 with np.load(path,allow_pickle=False) as z:
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
def compare_payload_bytes(a,b):
 total=0;members=0
 with zipfile.ZipFile(a) as x,zipfile.ZipFile(b) as y:
  names=x.namelist();require(names==y.namelist() and len(set(names))==len(names),'continuity ordered member roster')
  require('metadata.npy' in names,'continuity recursive metadata')
  for n in names:
   require(x.getinfo(n).file_size==y.getinfo(n).file_size,'continuity size '+n)
   with x.open(n) as xx,y.open(n) as yy:
    while True:
     aa,bb=xx.read(1024**2),yy.read(1024**2);require(aa==bb,'exact continuity payload '+n)
     if not aa:break
     total+=len(aa)
   members+=1
 return {'members':members,'uncompressed_bytes':total,'container_sha_equal':sha(a)==sha(b)}

class Comparisons:
 def __init__(self,policy):self.policy=policy;self.counts=collections.Counter();self.exact=collections.Counter();self.maxima=collections.defaultdict(float)
 def check(self,a,b,category='statistics',atol=None,rtol=0):
  x,y=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
  require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),('finite/shape',category,x.shape,y.shape))
  diff=np.abs(x-y);err=float(diff.max()) if diff.size else 0
  self.counts[category]+=int(x.size);self.exact[category]+=int(np.count_nonzero(x==y));self.maxima[category]=max(self.maxima[category],err)
  if atol is None:atol=self.policy['statistic_absolute_tolerance']
  require(bool(np.all(diff<=atol+rtol*np.maximum(np.abs(x),np.abs(y)))),('numerical discrepancy',category,err,atol,rtol))
 def report(self):return {k:{'comparisons':n,'exact':self.exact[k],'maximum_absolute_error':self.maxima[k]} for k,n in sorted(self.counts.items())}

def full_roster():
 out=[]
 for c in CORPORA:
  for w in range(4):
   for i in ('natural','bos'):
    for mode in RUN_MODES:
     out.append((c,w,i,mode,'evaluation'))
     if w==0 and i=='natural' and mode=='000__baseline':out.extend((c,w,i,m,'anchor') for m in ('native_repeat','native_identity'))
     out.append((c,w,i,mode,'suffix_control'))
 return out
def journal(path,expected,completed_count=None):
 chain='0'*64;events=[];counts=collections.Counter();byrole={};completed_count=len(expected) if completed_count is None else completed_count
 for line in Path(path).read_bytes().splitlines(keepends=True):
  require(line.endswith(b'\n'),'complete journal line');e=json.loads(line);body=line[:-1]
  require(body==json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),'canonical journal event')
  require(e['previous_sha256']==chain,'journal previous hash');chain=hashlib.sha256(body).hexdigest();events.append(e);counts[e['event']]+=1
 for kind in ('forward_intent','forward_attempt','model_returned','forward_completed'):
  seq=[e for e in events if e['event']==kind];n=completed_count if kind=='forward_completed' else len(expected)
  require(len(seq)==n,('journal count',kind,len(seq),n))
  for index,e in enumerate(seq):
   d=e['identity'];role=tuple(d[k] for k in ('corpus','window','identity','mode','phase'))
   require(role==expected[index],('exact prescribed journal chronology',kind,index,role))
   require(e.get('attempt',e.get('next_model_attempt'))==index+1,'attempt index')
   if kind in ('model_returned','forward_completed'):require(e['completed']==index+1,'return counter')
   if kind=='forward_completed':byrole[role]=e
 return {'events':len(events),'counts':dict(counts),'final_sha256':chain,'byrole':byrole,'all':events}

def provenance(out):
 m=read(out/'manifest.json');require(m['status']==m['phase']=='completed' and m['execution_type']=='verified_composite_after_interruption','complete explicit composite required')
 require((m['forward_passes'],m['forward_attempts'],m['retained_scheduled_forwards'],m['actual_cumulative_model_returns'],m['actual_cumulative_model_attempts'])==(2184,2184,2184,2272,2272),'retained versus actual counts')
 require((m['evaluation_forwards'],m['suffix_forwards'],m['anchor_forwards'])==(1088,1088,8),'retained categories')
 require(m['actual_return_category_totals']=={'main':1130,'suffix':1130,'anchors':12},'actual categories')
 require(m['original_final_guards_available'] is False and m['parameter_versions_unchanged'] is None and m['model_hash_after'] is None and m['precision_settings_after'] is None,'original missing afterguards remain unknown')
 require(m['continuity_validation_forwards']==70 and m['discarded_uncommitted_original_returns']==18 and m['original_fully_processed_forwards']==1109,'extra invocation bookkeeping')
 require(m['merger_sha256']==MERGER_SHA and m['source_hashes']['analyze_gate_native_value.py']==ANALYZER_SHA,'frozen merger/analyzer')
 require(m['source_hashes']['gpt2_gate_native_value.py']==ORIGINAL_SHA and m['calibration_map_digest']==F_DIGEST,'original algorithm/map')
 require(m['bridge_policy']=={'context_max_abs':1e-4,'target_loss_max_abs':1e-3,'passage_mean_abs':1e-5},'frozen scientific bridge policy')
 require(m['physical_execution_lengths']==[512] and m['calibration_forwards']==0 and m['calibration_fit'] is False,'no fitting or length change')
 require(m['wall_seconds'] is None and m['peak_process_rss_kib'] is None and m['cumulative_runtime_and_original_final_peak_unknown'] is True,'no invented aggregate resources')
 for n,h in m['source_hashes'].items():require(sha(out/'frozen_sources'/n)==h,('composite frozen source',n))
 for n,h in m['input_dependencies'].items():require(sha(out/n)==h,('composite input',n))
 for n,h in m['metadata_dependencies'].items():require(sha(out/'metadata_dependencies'/n.removeprefix('../').replace('/','__'))==h,('metadata pin',n))
 forensic=read(REVIEW/'gate_native_value_interruption_20261021.json');old=ROOT/'gpt2_gate_native_value_run'
 require(not (old/'manifest.json').exists(),'original not relabeled')
 require({p.relative_to(old).as_posix() for p in old.rglob('*') if p.is_file()}==set(forensic['files']),'unchanged original file universe')
 for n,v in forensic['files'].items():require(sha(old/n)==v['sha256'] and (old/n).stat().st_size==v['bytes'],('original immutable file',n))
 require(sha(ROOT/'gpt2_gate_native_value_run.log')==forensic['log_sha256'],'original immutable log')
 inventory=read(out/'numeric_inventory.json');origins=read(out/'execution_provenance/shard_origins.json')
 expected={f'inputs/{c}__{w}__{i}.npz' for c in CORPORA for w in range(4) for i in ('natural','bos')}
 require(set(inventory)==set(origins)==expected and {p.relative_to(out).as_posix() for p in (out/'inputs').glob('*.npz')}==expected,'exact32shard universe')
 bases={'original_committed':old}
 for label in ('segment_A_new','segment_B_new'):
  paths={v['source_directory'] for v in origins.values() if v['origin']==label};require(len(paths)==1,'one physical origin per segment');bases[label]=Path(paths.pop())
 reports={};roles=full_roster();require(len(roles)==2184,'planned exact chronology')
 oldcp=read(old/'checkpoints/000018/manifest.json');prefix=read(old/'checkpoints/000018/numeric_inventory.json')
 jr=journal(old/'journal.jsonl',roles[:1110],1109);require(jr['counts']==forensic['journal_counts'] and jr['final_sha256']==forensic['journal_final_sha256'],'original complete chain')
 require(jr['counts']['committed']==16 and len(prefix)==16,'original committed prefix')
 reports['original']=jr
 for seg,label,total,main_count,anchors in [('A','segment_A_new',616,306,4),('B','segment_B_new',546,272,2)]:
  base=bases[label];sm=read(base/'manifest.json');si=read(base/'numeric_inventory.json')
  require(sm['status']==sm['phase']=='completed' and sm['execution_type']=='bounded_recovery_segment' and sm['recovery_segment']==seg,'segment final status')
  require((sm['forward_attempts'],sm['forward_passes'],sm['evaluation_forwards'],sm['suffix_forwards'],sm['anchor_forwards'])==(total,total,main_count,main_count,anchors),'segment final counts')
  require(sm['parameter_versions_unchanged'] is True and sm['model_hash_before']==sm['model_hash_after']==m['model_hash_before'],'segment parameter endguard')
  require(sm['precision_settings_after']==sm['precision_settings_before']==m['precision_settings_before'],'segment precision endguard')
  for key in ('runtime_versions','model_id','revision','dtype','attention','metadata_dependencies','input_dependencies','calibration_map_digest','calibration_reference_sha256'):
   require(sm[key]==oldcp[key],('unchanged segment provenance',seg,key))
  require(sm['original_source_hashes']==oldcp['source_hashes'],'algorithmic closure unchanged')
  for n,h in sm['source_hashes'].items():require(sha(base/'frozen_sources'/n)==h,('actual executed closure',seg,n))
  require(sm['source_hashes']['gpt2_gate_native_value_recovery.py']==ADAPTER_SHA,'reviewed execution adapter')
  proof=read(base/'frozen_sources/recovery_ast_proof.json');ctx=read(base/'frozen_sources/recovery_context.json')
  require(proof['adapted_runner_sha256']==sm['source_hashes']['gpt2_gate_native_value.py'] and proof['original_runner_sha256']==ORIGINAL_SHA and proof['complete_boundary_condition_and_suffix_loop_ast_exact'] is True,'scientific AST proof')
  require(sm['adapted_runner_ast_proof_sha256']==sha(base/'frozen_sources/recovery_ast_proof.json') and ctx['segment']==seg and ctx['adapter_sha256']==ADAPTER_SHA,'bound orchestration')
  if seg=='B':require(ctx['segment_a_manifest_sha256']==sha(bases['segment_A_new']/'manifest.json'),'B pins successful A')
  selected=[r for r in roles if (r[:3]==('wiki_validation',0,'natural') or r[0]=='ethics') ] if seg=='A' else [r for r in roles if r[0]=='optimizers_new']
  sj=journal(base/'journal.jsonl',selected);require(sj['counts']['committed']==(9 if seg=='A' else 8),'segment committed groups');reports[seg]=sj
  for n,v in si.items():require(sha(base/n)==v['sha256'] and (base/n).stat().st_size==v['bytes'],'every segment shard digest')
  require(sha(base/'manifest.json')==m['segment_manifests'][seg],'segment manifest bind')
  require(m['executed_segment_source_hashes'][seg]==sm['source_hashes'],'actual source hashes disclosed')
  prov=out/'execution_provenance'/('segment_'+seg)
  for name in ('manifest.json','numeric_inventory.json','journal.jsonl'):
   require(sha(prov/name)==sha(base/name),'self-contained segment evidence '+name)
  disk=read(base/'disk_checks.json');require(all(d['free_bytes']-d['projected_bytes']>=2*1024**3+256*1024**2 and d['normal_bytes']+d['projected_bytes']<=512*1024**2 for d in disk),'successful segment disk checks')
 for n,v in inventory.items():
  rec=origins[n];base=bases[rec['origin']];require(sha(out/n)==v['sha256']==rec['sha256']==sha(base/n) and (out/n).stat().st_size==v['bytes']==rec['bytes'],'lossless32shard provenance')
  corpus=Path(n).stem.rsplit('__',2)[0];require(rec['origin']==('original_committed' if corpus in CORPORA[:2] else 'segment_A_new' if corpus=='ethics' else 'segment_B_new'),'correct retained scientific origin')
 require({n for n,v in origins.items() if v['origin']=='original_committed'}==set(prefix),'original16 reused exactly')
 A=bases['segment_A_new'];continuity=compare_payload_bytes(old/'inputs/wiki_validation__0__natural.npz',A/'inputs/wiki_validation__0__natural.npz')
 fingerprint_matches=[];fingerprint_mismatches=[]
 for role,event in jr['byrole'].items():
  if role[0]!='ethics':continue
  other=reports['A']['byrole'][role]
  check={'identity':list(role),**{key:event[key]==other[key] for key in ('hidden_fingerprint','logits_fingerprint')}}
  (fingerprint_matches if all(check[k] for k in ('hidden_fingerprint','logits_fingerprint')) else fingerprint_mismatches).append(check)
 require(len(fingerprint_matches)+len(fingerprint_mismatches)==17,'all17 available uncommitted completed-event fingerprints compared')
 require(not fingerprint_mismatches,('uncommitted-original replay fingerprint mismatch; pause interpretation',fingerprint_mismatches))
 gate=read(A/'continuity_gate.json');require(gate['passed'] is True and gate['forward_passes_at_gate']==70,'hard pre-new-evaluation gate')
 require(gate['reference_sha256']==sha(old/'inputs/wiki_validation__0__natural.npz') and gate['replay_sha256']==sha(A/'inputs/wiki_validation__0__natural.npz'),'gate actual payload identities')
 require(sha(out/'execution_provenance/continuity_gate.json')==sha(A/'continuity_gate.json'),'self-contained continuity verdict')
 for name,h in m['calibration_reference_artifact_files'].items():require(sha(out/'execution_provenance/calibration_reference'/name)==h,'T18 reference file')
 dm=read(out/'data_manifest.json');require(m['calibration_reference_sha256']==dm['calibration_reference_manifest_sha256'],'T18 manifest pin')
 for n in ('calibration_maps.npz','calibration_directions.npz'):require(sha(out/n)==dm['calibration_reference_files'][n],'frozen calibration array file')
 categories={}
 for label,j in reports.items():
  cnt=collections.Counter(e['identity']['phase'] for e in j['all'] if e['event']=='model_returned');categories[label]={'main':cnt['evaluation'],'suffix':cnt['suffix_control'],'anchors':cnt['anchor']}
 require({k:sum(v[k] for v in categories.values()) for k in ('main','suffix','anchors')}=={'main':1130,'suffix':1130,'anchors':12},'independent actual event categories')
 return m,inventory,origins,bases,reports,{'journals':{k:{n:j[n] for n in ('events','counts','final_sha256')} for k,j in reports.items()},'actual_categories':categories,'continuity':continuity,'uncommitted_fingerprint_check':{'matches':fingerprint_matches,'mismatches':fingerprint_mismatches,'complete_events':17,'individual_hidden_and_logit_exact_comparisons':34,'last_unprocessed_original_return':'No forward_completed fingerprints or actual arrays available; not recovered.'},'original_files_unchanged':len(forensic['files']),'retained_shards':32,'retained_bytes':sum(v['bytes'] for v in inventory.values()),'missing_original_final_guards':m['original_missing_final_guards']}

def contrast_definitions():
 """Independent finite-difference coefficients on the Boolean cube."""
 specs={}
 def effect(b,op,sgn=1):return {b+'__'+op:sgn,b+'__baseline':-sgn}
 def add(name,kind,op,factor,terms):
  summed=collections.Counter()
  for d in terms:
   for k,v in d.items():summed[k]+=v
  specs[name]={'kind':kind,'operation':op,'factor':factor,'terms':{k:v for k,v in summed.items() if v}}
 for b in BITS:
  for op in ('remove','orth'):add('effect_'+op+'_'+b,'within_boundary_effect',op,None,[effect(b,op)])
  add('matched_remove_minus_orth_'+b,'matched_control',None,None,[{b+'__remove':1,b+'__orth':-1}])
 for op in ('remove','orth'):
  add('endpoint_111_minus_000_'+op,'endpoint_effect_difference',op,None,[effect('111',op),effect('000',op,-1)])
  for j,label in enumerate('ULD'):
   for zero in BITS:
    if zero[j]!='0':continue
    one=zero[:j]+'1'+zero[j+1:]
    add('factor_'+label+'_'+one+'_minus_'+zero+'_'+op,'controlled_factor_interaction',op,label,[effect(one,op),effect(zero,op,-1)])
  for pair in ((0,1),(0,2),(1,2)):
   other=({0,1,2}-set(pair)).pop();factor=''.join('ULD'[j] for j in pair)
   for fixed in '01':
    cube=[b for b in BITS if b[other]==fixed]
    terms=[effect(b,op,math.prod(2*int(b[j])-1 for j in pair)) for b in cube]
    add(f'two_factor_{factor}_fixed_{other}_{fixed}_{op}','two_factor_interaction',op,factor,terms)
  add('three_factor_ULD_'+op,'three_factor_interaction',op,'ULD',[effect(b,op,math.prod(2*int(v)-1 for v in b)) for b in BITS])
 require(len(specs)==64,'64 independent contrast definitions');return specs

def bootstrap(values,ids):
 x=np.asarray(values,dtype=np.float64);require(x.shape==(4,) and np.isfinite(x).all(),'four finite passage clusters')
 return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[ids].mean(axis=1),[.025,.975]).tolist(),'per_window':x.tolist()}

def statistics(analysis,rows,anchors,normrows,cmp,ids):
 def indexed(items,keys):
  result={}
  for r in items:
   key=tuple(r[k] for k in keys);require(key not in result,'no duplicate analysis row');result[key]=r
  return result
 metric_keys=('corpus','identity','window','view','mode')
 for category,expected,keyfields in [('window_metrics',rows,metric_keys),('anchors',anchors,metric_keys)]:
  found=indexed(analysis[category],keyfields);require(set(found)==set(expected),'complete '+category)
  for key,row in expected.items():
   for field in FIELDS:cmp.check(row[field],found[key][field],'window_metrics')
 norms=indexed(analysis['norm_window_summaries'],(*metric_keys,'layer'));require(set(norms)==set(normrows),'complete norm summaries')
 for key,row in normrows.items():
  require(set(norms[key]['values'])==set(row),'norm keys')
  for name,z in row.items():
   target=norms[key]['values'][name];require(z['scope']==target['scope'],'norm view scope')
   for f in ('mean','rms','max_abs'):cmp.check(z[f],target[f],'norm_aggregates',cmp.policy['norm_absolute_tolerance'],cmp.policy['norm_relative_tolerance'])
 summaries=indexed(analysis['summaries'],('corpus','identity','view','mode'));specs=contrast_definitions()
 contrasts=indexed(analysis['contrasts'],('corpus','identity','view','name'));paired=indexed(analysis['paired_start_interactions'],('corpus','view','name'))
 def compare_paired(expected,target,category):
  require(target['n']==4,'n4 paired bootstrap')
  for name in ('mean','ci95','per_window'):cmp.check(expected[name],target[name],category)
 def values(c,i,v,mode,field='nll'):return [rows[c,i,w,v,mode][field] for w in range(4)]
 for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
  for mode in MODES:
   target=summaries[c,i,v,mode];require(set(target['metrics'])==set(FIELDS)|{'perplexity','ppl_ratio_c','ppl_ratio_boundary'},'summary fields')
   for field in FIELDS:
    expected=bootstrap(values(c,i,v,mode,field),ids);compare_paired(expected,target['metrics'][field],'summary_bootstrap')
    if field in ('nll','delta_nll_c','delta_nll_boundary'):
     label={'nll':'perplexity','delta_nll_c':'ppl_ratio_c','delta_nll_boundary':'ppl_ratio_boundary'}[field]
     cmp.check(math.exp(expected['mean']),target['metrics'][label]['value'],'ppl_transforms')
     cmp.check([math.exp(q) for q in expected['ci95']],target['metrics'][label]['ci95'],'ppl_transforms')
  for name,spec in specs.items():
   target=contrasts[c,i,v,name]
   require(all(target[k]==spec[k] for k in ('kind','operation','factor','terms')),'independent contrast coefficients '+name)
   # Match declared term iteration only after independently verifying coefficients;
   # also check an fsum reduction independently to expose cancellation/reordering.
   xs=[sum(coef*rows[c,i,w,v,mode]['nll'] for mode,coef in target['terms'].items()) for w in range(4)]
   precise=[math.fsum(coef*rows[c,i,w,v,mode]['nll'] for mode,coef in spec['terms'].items()) for w in range(4)]
   cmp.check(xs,precise,'contrast_independent_fsum');compare_paired(bootstrap(xs,ids),target,'contrast_bootstrap')
 for c,v in itertools.product(CORPORA,VIEWS):
  for name,spec in specs.items():
   target=paired[c,v,'bos_minus_natural_'+name];require(target['terms']==spec['terms'] and target['scope']=='paired_start_interaction_secondary','paired start definition')
   xs=[sum(coef*(rows[c,'bos',w,v,mode]['nll']-rows[c,'natural',w,v,mode]['nll']) for mode,coef in target['terms'].items()) for w in range(4)]
   precise=[math.fsum(coef*(rows[c,'bos',w,v,mode]['nll']-rows[c,'natural',w,v,mode]['nll']) for mode,coef in spec['terms'].items()) for w in range(4)]
   cmp.check(xs,precise,'paired_start_independent_fsum');compare_paired(bootstrap(xs,ids),target,'paired_start_bootstrap')
 require((len(summaries),len(contrasts),len(paired))==(816,1536,768),'full summary/contrast/paired-start roster')
 return {'window_rows':len(rows),'anchor_rows':len(anchors),'norm_rows':len(normrows),'summary_rows':len(summaries),'contrasts':len(contrasts),'paired_start_contrasts':len(paired),'independent_definitions':64,'bootstrap_repetitions':2000,'shared_bootstrap_index_shape':list(ids.shape),'shared_bootstrap_index_sha256':hashlib.sha256(ids.tobytes()).hexdigest()}

def errors(a,b):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);d=a-b;den=max(float(np.linalg.norm(a.ravel())),float(np.linalg.norm(b.ravel())))
 return {'max_abs':float(np.abs(d).max()),'rms':float(np.sqrt(np.mean(d*d))),'symmetric_relative_l2':float(np.linalg.norm(d.ravel()))/den if den else 0.}
def compare_errors(actual,record,cmp,category):
 for k,v in actual.items():cmp.check(v,record[k],category,cmp.policy['norm_absolute_tolerance'],cmp.policy['norm_relative_tolerance'])
def tmean(x):return float(torch.from_numpy(np.asarray(x)).double().mean())
def expected_control_roster(has_anchors):
 roster=[];incoming={};delivered={};direct={}
 for mode in RUN_MODES:
  b,op=mode.split('__');u,l,d=b
  if mode=='000__baseline' and has_anchors:roster.extend((k,None,None) for k in ('native_repeat','native_identity'))
  if op!='baseline':roster.append(('source_q0',mode,None))
  if op=='current':roster.append(('current',mode,None))
  key=(u,op)
  if key in incoming:roster.append(('L10_incoming_causal_order',mode,incoming[key]))
  else:incoming[key]=mode
  key=(u,l,op)
  if key in delivered:roster.append(('D_causal_order',mode,delivered[key]))
  else:delivered[key]=mode
  if not op.endswith('_scale'):
   key=(u,d,op)
   if key in direct and direct[key].split('__')[0][1]!=l:roster.append(('direct_local_delta_identity',mode,direct[key]))
   else:direct[key]=mode
  if op.endswith('_scale'):roster.append(('arithmetic_bridge',mode,None))
  roster.append(('suffix',mode,None))
 return roster
def runtime_controls(shard,cmp,counts,extrema):
 raw=shard['metrics'];operations=shard['operations'];suffix_ops=shard['suffix_operations'];gates=shard['gates'];base=shard['metadata']
 expected=expected_control_roster(base['identity']=='natural' and base['window']==0)
 require(len(shard['checks'])==len(expected),'complete control roster')
 def record_scalar(name,x):
  require(math.isfinite(x),('finite runtime diagnostic',name));d=extrema.setdefault(name,{'min':x,'max':x,'count':0});d['min']=min(d['min'],x);d['max']=max(d['max'],x);d['count']+=1
 def incoming(mode):
  op=operations[mode];gg=gates[mode][10]
  return {**{k:op['projection_records'][10]['before'][k] for k in ('q','k','nonsource_v','source_v')},**{k:fp(gg[k]) for k in ('b','mass','s','target_b','target_mass')}}
 for check,(kind,mode,ref) in zip(shard['checks'],expected):
  require((check['kind'],check.get('mode'),check.get('reference_mode'))==(kind,mode,ref),'exact control order/identity')
  flags=[v for k,v in check.items() if k.endswith('_exact') or k.endswith('_passed')];require(flags and all(type(v)==bool and v for v in flags),'all exact/hard control flags true');counts[kind]+=1
  if kind=='L10_incoming_causal_order':
   require(exact(check['reference_fingerprints'],incoming(ref)) and exact(check['trial_fingerprints'],incoming(mode)) and exact(incoming(ref),incoming(mode)),'incoming fingerprint causal-order witness')
  elif kind=='D_causal_order':
   a=operations[ref]['context_records'][10];b=operations[mode]['context_records'][10]
   require(exact(check['reference_fingerprints'],a) and exact(check['trial_fingerprints'],b) and exact(a,b),'D fingerprint causal-order witness')
  elif kind=='direct_local_delta_identity':
   require(check['fp64']['max_abs']<=1e-4,'direct V recorded identity threshold')
   for side,m in [('reference',ref),('trial',mode)]:
    want={k:operations[m]['gate_checks'][10][k] for k in ('native_reconstruction_error','explicit_source_reconstruction_error')}
    require(check[side+'_reconstruction_residuals']==want,'direct identity reconstruction witness')
   for k,v in check['fp64'].items():record_scalar('direct_identity.'+k,v)
  elif kind=='arithmetic_bridge':
   delta=mode.removesuffix('_scale');difference=raw[mode]['loss'][1:].astype(np.float64)-raw[delta]['loss'][1:].astype(np.float64)
   require(exact(difference,check['loss_difference']),'bridge all510 actual loss differences')
   maximum=float(np.abs(difference).max());mean=abs(float(difference.mean()))
   require(maximum<=1e-3 and mean<=1e-5 and check['delivered_context']['max_abs']<=1e-4,'unchanged bridge three hard thresholds')
   record_scalar('bridge.target_loss_max_abs',maximum);record_scalar('bridge.passage_mean_abs',mean)
   for k in ('delivered_context','hidden','logits'):
    for field,value in check[k].items():record_scalar('bridge.'+k+'.'+field,value)
   require(check['kl_delta_to_scale'].shape==check['kl_scale_to_delta'].shape==(511,) and np.isfinite(check['kl_delta_to_scale']).all() and np.isfinite(check['kl_scale_to_delta']).all(),'bridge saved full511 KL vectors')
   want=np.flatnonzero(raw[delta]['pred']!=raw[mode]['pred']).tolist();require([z['query'] for z in check['argmax_mismatches']]==want,'bridge all511 argmax mismatch roster')
   for z in check['argmax_mismatches']:
    q=z['query'];require(z['delta_id']==int(raw[delta]['pred'][q]) and z['scale_id']==int(raw[mode]['pred'][q]),'bridge mismatch identities')
    require(z['compatible_with_drift']==(z['delta_margin']<=2*z['row_max_abs_logit_drift']),'bridge tie/drift classification')
 for phase,ops in [('main',operations),('suffix',suffix_ops)]:
  for mode,op in ops.items():
   if mode=='native_repeat':require(op=={},'unhooked anchor has no hook captures');continue
   require((op['projection_checks'],op['context_checks'],op['query_checks'])==(8,8,8),'all hooked layers completed')
   boundary=mode.split('__')[0] if mode in MODES else '000'
   require(set(op['projection_records'])==set(op['context_records'])==set(op['diag'])==set(op['mask_records'])==set(range(4,12)),'all8 per-layer witnesses')
   require(op['source_fingerprints']==nested_fp(shard['source_reference'][boundary]),'source q0 saved actual reference and runtime trial fingerprints')
   if phase=='main':require(op['gate_fingerprints']==nested_fp(shard['gates'][mode]),'stored gate arrays/runtime fingerprints')
   for layer in range(4,12):
    p=op['projection_records'][layer];zeroed=(4<=layer<=9 and boundary[0]=='1') or (layer==10 and boundary[1]=='1') or (layer==11 and boundary[2]=='1')
    require(p['zeroed']==zeroed,'prescribed source V layer mask')
    for field in ('q','k','nonsource_v'):require(p['before'][field]==p['after'][field],'Q/K/non-source V exactly untouched')
    src=shard['source_reference'][boundary][layer]
    require(p['after']['source_v']==fp(src['v']),'source V actual reference fingerprint')
    if zeroed:require(np.count_nonzero(src['v'])==0,'zeroed source V exact saved bytes')
    else:require(p['before']['source_v']==p['after']['source_v'],'live source V exactly retained')
    mask=op['mask_records'][layer];require(mask['model_argument_preserved'] is True and mask['kind'] in ('implicit_causal','preserved_explicit'),'native mask retained')
    ctx=op['context_records'][layer]
    if layer!=10 or mode.endswith(('__baseline','__current')) or mode=='native_identity':require(ctx['current']==ctx['delivered'],'unaffected context exact runtime fingerprint')
    diag=op['diag'][layer]
    for k in ('context_max_abs_error','native_reconstruction_error','explicit_source_reconstruction_error'):
     require(diag[k]<=1e-4,'context hard guard');record_scalar(k,diag[k])
    for k in ('source_complement_max_abs','ideal_mass_max_abs','min_mass','max_mass','mass_zeros','mass_ones','mass_above_one_roundoff'):record_scalar(k,diag[k])
    require(diag['min_mass']>=0 and diag['max_mass']<=1+1e-6,'mass guard no clipping')
   g=op['gate_checks'][10];require(g['source_context_exact'] is True and g['unaffected_context_exact'] is True and g['norm_match_error']<=1e-6,'source context/norm gate guard')
   record_scalar('gate.norm_match_error',g['norm_match_error'])
   for field in ('mapped_gate_fp64','delta_formula_fp64'):
    for k,v in g[field].items():record_scalar('gate.'+field+'.'+k,v)
   for k,v in g['legacy_errors'].items():require(v<=1e-4,'legacy local scaling guard');record_scalar('legacy.'+k,v)
   require(bool(g['legacy_errors'])==mode.endswith('_scale'),'legacy guards apply only scaling bridges')
   if phase=='main':
    cache=shard['gates'][mode][10];source=shard['source_reference'][boundary][10]['v']
    require(cache['b'].shape==cache['mass'].shape==cache['target_b'].shape==cache['target_mass'].shape==cache['proposed_target_mass'].shape==(12,511) and cache['s'].shape==(511,),'full gate shapes')
    unchanged=mode.endswith(('__baseline','__current')) or mode=='native_identity' or exact(cache['target_b'],cache['b'])
    require(g['unchanged_bypass']==unchanged and g['active']==(not unchanged),'gate operation activation witness')
    require(exact(cache['target_mass'],cache['mass'] if unchanged else cache['proposed_target_mass']),'native diagnostic mass bypass or proposed mass')
   else:
    require(exact(op['mask_records'],operations[mode]['mask_records']),'fixed512 suffix mask exact')
    require(exact(op['source_fingerprints'],operations[mode]['source_fingerprints']),'fixed512 suffix source exact')
    require(exact(op['prefix_context_fingerprints'],operations[mode]['prefix_context_fingerprints']),'fixed512 suffix prefix context exact runtime witness')

def metric_and_norm_rows(shard,cmp,rows,anchors,normrows,alpha,u,v):
 base=shard['metadata'];c,i,w=base['corpus'],base['identity'],base['window'];raw=shard['metrics'];has_anchors=(i=='natural' and w==0)
 expected=set(MODES)|({'native_repeat','native_identity'} if has_anchors else set())
 require(shard['schema']=='native_value_explicit_v1' and base['length']==512 and set(raw)==set(shard['gates'])==set(shard['norms'])==set(shard['operations'])==expected,'full main payload roster')
 require(set(shard['suffix'])==set(shard['suffix_gates'])==set(shard['suffix_norms'])==set(shard['suffix_operations'])==set(MODES),'full suffix payload roster')
 require(set(shard['source_reference'])==set(BITS),'source actual references for all8boundaries')
 for b,src in shard['source_reference'].items():
  require(set(src)==set(range(12)),'all12 source layers')
  for layer,s in src.items():
   require(set(s)=={'q','k','v','hidden'} and all(a.shape==(1,768) and a.dtype==np.float32 and np.isfinite(a).all() for a in s.values()),'complete source q/k/v/block hidden arrays')
 for mode,z in raw.items():
  b=mode.split('__')[0] if mode in MODES else '000';reference=raw[b+'__baseline'];control=raw['000__baseline']
  for k,dtype in [('loss',np.float32),('kl_c',np.float32),('kl_boundary',np.float32),('pred',np.int64),('agreement_c',np.bool_),('agreement_boundary',np.bool_)]:
   require(z[k].shape==(511,) and z[k].dtype==dtype and np.isfinite(z[k]).all(),'raw metric dtype/shape/finite')
  require(((z['pred']>=0)&(z['pred']<50257)).all(),'vocabulary prediction range')
  require(np.array_equal(z['agreement_c'],z['pred']==control['pred']) and np.array_equal(z['agreement_boundary'],z['pred']==reference['pred']),'raw prediction agreement identities')
  if mode.endswith('__baseline'):require(np.count_nonzero(z['kl_boundary'])==0,'baseline exact KL zero')
  if mode.endswith('__current') or mode not in MODES:
   require(all(exact(z[k],reference[k]) for k in ('loss','kl_c','kl_boundary','pred','agreement_c','agreement_boundary','final_hidden_sha256')),'current/unhooked/identity exact actual metrics and hidden fingerprint')
   if mode!='native_repeat':require(exact(shard['gates'][mode],shard['gates'][b+'__baseline']) and exact(shard['norms'][mode],shard['norms'][b+'__baseline']),'current/identity exact captured gates and norms')
  for view,(lo,hi) in VIEWS.items():
   metrics={'nll':tmean(z['loss'][lo:hi]),'delta_nll_c':tmean(z['loss'][lo:hi].astype(np.float64)-control['loss'][lo:hi].astype(np.float64)),'delta_nll_boundary':tmean(z['loss'][lo:hi].astype(np.float64)-reference['loss'][lo:hi].astype(np.float64)),**{k:tmean(z[k][lo:hi]) for k in ('kl_c','kl_boundary','agreement_c','agreement_boundary')}}
   (rows if mode in MODES else anchors)[c,i,w,view,mode]=metrics
  target=(rows if mode in MODES else anchors)
  for field in FIELDS:cmp.check(target[c,i,w,'full',mode][field],(126*target[c,i,w,'early',mode][field]+384*target[c,i,w,'later',mode][field])/510,'full_early_later_partition')
  if mode in MODES:
   suffix=shard['suffix'][mode]
   for k in ('loss','kl_c','kl_boundary','pred','agreement_c','agreement_boundary'):
    n=127 if k=='loss' else 128;require(exact(suffix[k],z[k][:n]),'fixed512 suffix exact prefix metric bytes')
   require(exact(suffix['target_indices'],np.arange(1,128,dtype=np.int64)) and exact(suffix['logit_query_indices'],np.arange(128,dtype=np.int64)),'suffix exact indexing')
   require(exact(shard['suffix_gates'][mode],{layer:{k:a[:127] if a.ndim==1 else a[:,:127] for k,a in vals.items()} for layer,vals in shard['gates'][mode].items()}),'suffix exact first127 gate arrays')
   require(exact(shard['suffix_norms'][mode],{layer:{k:a[:127] for k,a in vals.items()} for layer,vals in shard['norms'][mode].items()}),'suffix exact norm prefix')
  norms=shard['norms'][mode]
  if mode=='native_repeat':require(norms=={} and shard['gates'][mode]=={},'native repeat completely unhooked');continue
  require(set(norms)=={10} and set(norms[10])==NORM_KEYS,'14 L10 norm fields')
  vals=norms[10];gg=shard['gates'][mode][10];uf=u.astype(np.float32);vf=v.astype(np.float32)
  want={'s':gg['s'],'hypothetical_pc1_norm':np.linalg.norm((gg['s'][:,None]*uf).astype(np.float64),axis=1),'hypothetical_orthogonal_norm':np.linalg.norm((gg['s'][:,None]*vf).astype(np.float64),axis=1),'logit_delta_norm':np.linalg.norm((gg['target_b']-gg['b']).T.astype(np.float64),axis=1),'mass_delta_norm':np.linalg.norm((gg['target_mass']-gg['mass']).T.astype(np.float64),axis=1),'source_v_norm':np.linalg.norm(shard['source_reference'][b][10]['v'].reshape(12,64).astype(np.float64),axis=1)}
  want.update({name+'_norm':np.linalg.norm(gg[name].T.astype(np.float64),axis=1) for name in ('b','target_b','mass','target_mass')})
  for name,expected_array in want.items():cmp.check(vals[name],expected_array,'saved_gate_norm_reconstruction',cmp.policy['norm_absolute_tolerance'],cmp.policy['norm_relative_tolerance'])
  mapped=(gg['b'].T.astype(np.float64)-np.log(np.arange(1,512))[:,None]-alpha.astype(np.float32))@uf.astype(np.float64)
  cmp.check(gg['s'],mapped,'ff_projection_reconstruction',cmp.policy['fp64_snapshot_vs_fp32_absolute_tolerance'])
  op=mode.split('__')[1] if mode in MODES else 'baseline'
  expected_b=gg['b'] if op in ('baseline','current') else (gg['b'].T-gg['s'][:,None]*(uf if op.startswith('remove') else vf)).T
  require(exact(gg['target_b'],expected_b),'stored FF target exact FP32 elementwise formula')
  proposed=1/(1+np.exp(-gg['target_b'].astype(np.float64)));cmp.check(gg['proposed_target_mass'],proposed,'sigmoid_reconstruction',cmp.policy['fp64_probability_vs_fp32_absolute_tolerance'])
  for name,array in vals.items():
   require(array.shape==((12,) if name=='source_v_norm' else (511,)) and array.dtype==(np.float32 if name=='s' else np.float64) and np.isfinite(array).all(),'norm exact shape/dtype/finite')
   if name!='s':require((array>=0).all(),'norm nonnegative')
  for view,(lo,hi) in NORM_VIEWS.items():
   selected={k:(a if k=='source_v_norm' else a[lo:hi]) for k,a in vals.items()};agg={}
   for k,a in selected.items():
    ta=torch.from_numpy(a).double();agg[k]={'scope':'source_heads_constant_across_query_views' if k=='source_v_norm' else 'scored_query_positions','mean':float(ta.mean()),'rms':float(ta.square().mean().sqrt()),'max_abs':float(ta.abs().max())}
   normrows[c,i,w,view,mode,10]=agg

def logsoftmax(x):
 x=np.asarray(x,np.float64);maximum=x.max(axis=-1,keepdims=True);return x-maximum-np.log(np.exp(x-maximum).sum(axis=-1,keepdims=True))
def snapshot_audit(shard,tokens,cmp,u,v,snapshot_counts):
 identity=tuple(shard['metadata'][k] for k in ('corpus','window','identity'))
 if identity!=('wiki_validation',0,'natural'):
  require(shard['snapshots']==shard['readouts']=={},'full snapshots only prospectively fixed first Wiki natural input');return
 require(set(shard['snapshots'])==set(shard['readouts'])==set(SNAPS),'all8prospective snapshots/readouts')
 softmax=lambda x:np.exp(logsoftmax(x))
 atol=cmp.policy['fp64_snapshot_vs_fp32_absolute_tolerance'];prob_atol=cmp.policy['fp64_probability_vs_fp32_absolute_tolerance']
 qrows=np.array([1,126,127,510],dtype=np.int64)
 for mode in SNAPS:
  snap=shard['snapshots'][mode];op=shard['operations'][mode];gate=shard['gates'][mode][10];norms=shard['norms'][mode][10];readout=shard['readouts'][mode]
  require(set(snap)=={'q','k','v','mask','gate_state','stable_mixture','actual_context','delivered_context'},'complete full snapshot')
  require(exact(snap['gate_state'],gate),'snapshot gate equals separately stored full gate')
  for k in ('q','k','v'):require(snap[k].shape==(1,12,512,64) and snap[k].dtype==np.float32 and np.isfinite(snap[k]).all(),'full QKV shape/dtype/finite')
  require(snap['mask'].shape==(1,1,512,512) and snap['mask'].dtype==np.float32 and not np.isnan(snap['mask']).any() and not np.isposinf(snap['mask']).any(),'full preserved diagnostic mask')
  require(snap['actual_context'].shape==snap['delivered_context'].shape==(1,512,768),'full actual/delivered contexts')
  require(fp(snap['actual_context'])==op['context_records'][10]['current'] and fp(snap['delivered_context'])==op['context_records'][10]['delivered'],'actual saved contexts bind recorded fingerprints')
  require(op['prefix_context_fingerprints'][10]=={'current':fp(snap['actual_context'][:,:128]),'delivered':fp(snap['delivered_context'][:,:128])},'saved full128query prefix context fingerprints')
  require(fp(snap['mask'])==op['mask_records'][10]['diagnostic_effective'],'full effective mask identity')
  boundary,operation=mode.split('__');source=shard['source_reference'][boundary][10]
  for k in ('q','k','v'):require(exact(snap[k][0,:,0].reshape(1,768),source[k]),'snapshot source QKV/reference exact')
  projection=op['projection_records'][10]['after']
  for k in ('q','k'):require(fp(snap[k].transpose(0,2,1,3).reshape(1,512,768))==projection[k],'full Q/K actual projection fingerprint')
  require(fp(snap['v'][:,:,1:].transpose(0,2,1,3).reshape(1,511,768))==projection['nonsource_v'],'full non-source V actual projection fingerprint')
  # Reconstruct diagnostic attention independently in FP64 from saved Q/K/V/mask.
  q,k,value=(snap[n].astype(np.float64) for n in ('q','k','v'));scores=q@k.swapaxes(-1,-2)/8+snap['mask'].astype(np.float64)
  probabilities=softmax(scores);expected=(probabilities@value).transpose(0,2,1,3).reshape(1,512,768)
  cmp.check(snap['actual_context'],expected,'snapshot_context_fp64',atol)
  stable=(softmax(scores[:,:,1:,1:])@value[:,:,1:])[0];mass=probabilities[0,:,1:,1:].sum(axis=-1)
  ns=scores[0,:,1:,1:];mx=ns.max(axis=-1);b=mx+np.log(np.exp(ns-mx[...,None]).sum(axis=-1))-scores[0,:,1:,0]
  cmp.check(snap['stable_mixture'],stable,'snapshot_stable_mixture_fp64',atol);cmp.check(gate['mass'],mass,'snapshot_mass_fp64',prob_atol);cmp.check(gate['b'],b,'snapshot_gate_logit_fp64',atol)
  c=snap['actual_context'].reshape(512,12,64).transpose(1,0,2)[:,1:];delivered=snap['delivered_context'].reshape(512,12,64).transpose(1,0,2)[:,1:];v0=snap['v'][0,:,:1]
  unchanged=op['gate_checks'][10]['unchanged_bypass']
  mixed=c if unchanged else c*(gate['target_mass']/gate['mass'])[...,None] if operation.endswith('_scale') else c+(gate['target_mass']-gate['mass'])[...,None]*(snap['stable_mixture']-v0)
  predicted=snap['actual_context'].copy();predicted[:,1:]=mixed.transpose(1,0,2).reshape(1,511,768)
  cmp.check(predicted,snap['delivered_context'],'snapshot_FP32_delivery_formula',cmp.policy['fp32_elementwise_delivery_absolute_tolerance'])
  require(exact(snap['actual_context'][:,0],snap['delivered_context'][:,0]),'saved source-query context exact')
  if unchanged:require(exact(snap['actual_context'],snap['delivered_context']),'saved no-op full context exact')
  delta=delivered.astype(np.float64)-c.astype(np.float64)
  prescribed=np.zeros_like(delta) if unchanged else (gate['target_mass'].astype(np.float64)-gate['mass'].astype(np.float64))[...,None]*(snap['stable_mixture'].astype(np.float64)-v0.astype(np.float64))
  compare_errors(errors(delta,prescribed),op['gate_checks'][10]['delta_formula_fp64'],cmp,'snapshot_delta_witness')
  for key,a in [('context_delta_norm',delta),('actual_context_norm',c.astype(np.float64)),('delivered_context_norm',delivered.astype(np.float64)),('stable_mixture_norm',snap['stable_mixture'].astype(np.float64))]:
   want=np.linalg.norm(a.transpose(1,0,2).reshape(511,768),axis=1)
   cmp.check(norms[key],want,'snapshot_context_norms',cmp.policy['norm_absolute_tolerance'],cmp.policy['norm_relative_tolerance'])
  require(exact(readout['query_rows'],qrows) and exact(readout['targets'],tokens[qrows+1]),'saved readout physical/scored query-target alignment')
  require(readout['hidden'].shape==(1,512,768) and readout['hidden'].dtype==np.float32 and np.isfinite(readout['hidden']).all(),'full final hidden saved')
  require(readout['hidden_fingerprint']==fp(readout['hidden']) and readout['hidden_fingerprint']['sha256']==shard['metrics'][mode]['final_hidden_sha256'],'actual hidden bytes/metric identity')
  require(readout['token_sha256']==fp(tokens)['sha256']==shard['metadata']['token_sha256'],'readout tokens bind exact512token input')
  logits=readout['logits'];require(logits.shape==(4,50257) and logits.dtype==np.float32 and np.isfinite(logits).all(),'four actual device logit rows all50257vocabulary')
  lp=logsoftmax(logits);loss=-lp[np.arange(4),readout['targets']]
  cmp.check(loss,shard['metrics'][mode]['loss'][qrows],'saved_readout_nll',cmp.policy['saved_logit_readout_loss_and_kl_absolute_tolerance'])
  require(exact(np.argmax(logits,axis=1),shard['metrics'][mode]['pred'][qrows]),'actual4row predictions exact')
  for key,reference in [('kl_c','000__baseline'),('kl_boundary',boundary+'__baseline')]:
   lr=logsoftmax(shard['readouts'][reference]['logits']);kl=np.sum(np.exp(lr)*(lr-lp),axis=1)
   cmp.check(kl,shard['metrics'][mode][key][qrows],'saved_readout_kl',cmp.policy['saved_logit_readout_loss_and_kl_absolute_tolerance'])
  snapshot_counts['full_snapshots']+=1;snapshot_counts['actual_saved_logit_rows']+=4
 for check in shard['checks']:
  if check['kind']!='arithmetic_bridge':continue
  scale=check['mode'];delta=scale.removesuffix('_scale');ds,ss=shard['snapshots'][delta],shard['snapshots'][scale]
  for k in ('q','k','v','mask','actual_context','stable_mixture'):require(exact(ds[k],ss[k]),'bridge exact shared incoming saved tensors')
  compare_errors(errors(ds['delivered_context'],ss['delivered_context']),check['delivered_context'],cmp,'snapshot_bridge_context')
  compare_errors(errors(shard['readouts'][delta]['hidden'],shard['readouts'][scale]['hidden']),check['hidden'],cmp,'snapshot_bridge_hidden')
  a,b=(shard['readouts'][m]['logits'].astype(np.float64) for m in (delta,scale));la,lb=logsoftmax(a),logsoftmax(b)
  cmp.check(np.sum(np.exp(la)*(la-lb),axis=1),check['kl_delta_to_scale'][qrows],'snapshot_bridge_kl',cmp.policy['saved_logit_readout_loss_and_kl_absolute_tolerance'])
  cmp.check(np.sum(np.exp(lb)*(lb-la),axis=1),check['kl_scale_to_delta'][qrows],'snapshot_bridge_kl',cmp.policy['saved_logit_readout_loss_and_kl_absolute_tolerance'])
  require(float(np.abs(a-b).max())<=check['logits']['max_abs']+1e-12,'saved4row logit drift bounded by recorded all511row maximum')
  for z in check['argmax_mismatches']:
   if z['query'] not in qrows:continue
   idx=list(qrows).index(z['query']);aa=np.sort(a[idx])[-2:];bb=np.sort(b[idx])[-2:]
   cmp.check([aa[1]-aa[0],bb[1]-bb[0],np.abs(a[idx]-b[idx]).max()],[z['delta_margin'],z['scale_margin'],z['row_max_abs_logit_drift']],'snapshot_bridge_tie_diagnostics')
  snapshot_counts['full_bridge_pairs_recomputed']+=1

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'gpt2_gate_native_value_composite_run');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
 started=time.time();policy_path=REVIEW/'gate_native_value_numerical_audit_policy.json';policy=read(policy_path)
 require(policy['status']=='prospective_policy_frozen_before_complete_panel_processing','prospective immutable tolerance policy')
 ids=np.random.Generator(np.random.PCG64(20261021)).integers(0,4,size=(2000,4),dtype=np.int64)
 require(ids.shape==(2000,4) and len(contrast_definitions())==64 and len(expected_control_roster(False))==126 and len(expected_control_roster(True))==128,'preparation algebra/control roster')
 if args.prepare_only:
  prepared={'status':'prepared_no_outcomes_read','audit_source_sha256':sha(__file__),'policy_sha256':sha(policy_path),'shared_bootstrap_index_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'contrast_definitions':64,'controls_per_group':126,'anchor_group_controls':128,'model_forwards':0,'gpu_used':False,'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  put(REVIEW/'gate_native_value_numerical_audit_preparation.json',prepared);print(json.dumps(prepared,indent=2));return
 # This complete-panel gate precedes every raw numerical array read.
 out=args.out;manifest=read(out/'manifest.json');require(manifest['status']=='completed' and manifest['execution_type']=='verified_composite_after_interruption','no partial-panel audit')
 analysis_path=out/'native_value_analysis.json';require(analysis_path.is_file(),'frozen analyzer result must exist before raw numerical audit')
 analysis=read(analysis_path);require(analysis['scope']['analyzer_sha256']==ANALYZER_SHA,'frozen analyzer output')
 global torch
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 require(not torch.cuda.is_initialized(),'CPU-only reviewer')
 m,inventory,origins,bases,journals,prov=provenance(out);cmp=Comparisons(policy)
 require(analysis['scope']['seed']==20261021 and analysis['scope']['bootstrap_repetitions']==2000,'shared bootstrap schedule')
 require(analysis['scope']['source_hashes']=={Path(n).name:v['sha256'] for n,v in inventory.items()},'analysis exact32raw-shard binding')
 maps,map_arrays=load_npz(out/'calibration_maps.npz');directions,direction_arrays=load_npz(out/'calibration_directions.npz')
 h=hashlib.sha256()
 for layer in sorted(maps['F']):
  for name,a in sorted(maps['F'][layer].items()):h.update(f'{layer}:{name}:torch.{a.dtype.name}:{tuple(a.shape)}'.encode());h.update(np.ascontiguousarray(a).tobytes())
 require(h.hexdigest()==F_DIGEST,'independently recomputed frozen F-map digest')
 alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
 require(alpha.dtype==u.dtype==v.dtype==np.float64 and alpha.shape==u.shape==v.shape==(12,),'frozen FP64 FF basis shape')
 require(abs(np.linalg.norm(u)-1)<=1e-12 and abs(np.linalg.norm(v)-1)<=1e-12 and abs(u@v)<=1e-12,'frozen FF orthonormality')
 dm=read(out/'data_manifest.json');input_items=torch.load(out/'tokens.pt',map_location='cpu',weights_only=True);tokens={(z['corpus'],z['window']):z['tokens'].numpy().copy() for z in input_items}
 require(len(tokens)==16,'all16fixed evaluation passages')
 rows={};anchors={};normrows={};counts=collections.Counter();extrema={};snapshot_counts=collections.Counter();array_count=map_arrays+direction_arrays;metric_records=0;suffix_records=0
 for name in sorted(inventory):
  shard,narrays=load_npz(out/name);array_count+=narrays;base=shard['metadata'];c,w,i=(base[k] for k in ('corpus','window','identity'));x=tokens[c,w].copy()
  if i=='bos':x[0]=dm['literals']['start_special']['id']
  require(fp(x)['sha256']==base['token_sha256'],'paired-start full512token identity')
  record=next(z for z in dm['passages'] if (z['corpus'],z['window'])==(c,w));require(fp(tokens[c,w])['sha256']==record['window_token_sha256'],'pinned passage token binding')
  label=origins[name]['origin'];j=journals['original' if label=='original_committed' else 'A' if label=='segment_A_new' else 'B']
  for mode,z in shard['metrics'].items():
   role=(c,w,i,mode,'anchor' if mode.startswith('native_') else 'evaluation');event=j['byrole'][role]
   require(event['hidden_fingerprint']['sha256']==z['final_hidden_sha256'] and event['hidden_fingerprint']['shape']==[1,512,768] and event['hidden_fingerprint']['dtype']=='torch.float32','saved metric and actual model-return hidden fingerprint link')
  # Intent/attempt tokens include suffix physical rotation while target/source identity stays fixed.
  for event in j['all']:
   if event['event'] not in ('forward_intent','forward_attempt'):continue
   ident=event['identity']
   if (ident['corpus'],ident['window'],ident['identity'])!=(c,w,i):continue
   actual=x.copy()
   if ident['phase']=='suffix_control':actual[128:]=np.roll(actual[128:],1)
   require(event['token_fingerprint']==fp(actual),'actual invoked fixed512 tokens / suffix rotation')
  metric_and_norm_rows(shard,cmp,rows,anchors,normrows,alpha,u,v);runtime_controls(shard,cmp,counts,extrema);snapshot_audit(shard,x,cmp,u,v,snapshot_counts)
  metric_records+=len(shard['metrics']);suffix_records+=len(shard['suffix']);del shard;gc.collect()
 require((metric_records,suffix_records)==(1096,1088) and sum(counts.values())==4040,'all2184retained forward metrics and4040controls')
 require(snapshot_counts=={'full_snapshots':8,'actual_saved_logit_rows':32,'full_bridge_pairs_recomputed':2},'prospective full saved snapshot scope')
 stat_counts=statistics(analysis,rows,anchors,normrows,cmp,ids)
 require((len(rows),len(anchors),len(normrows))==(3264,24,3276),'all statistical output scopes')
 require(not torch.cuda.is_initialized(),'review never initialized CUDA')
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;require(peak<3*1024**2,'review <=3GiB peak RSS')
 result={'status':'passed','audited_manifest_sha256':sha(out/'manifest.json'),'analysis_sha256':sha(analysis_path),'raw_inventory_sha256':sha(out/'numeric_inventory.json'),'audit_source_sha256':sha(__file__),'comparison_policy_sha256':sha(policy_path),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes':m['source_hashes'],'executed_segment_source_hashes':m['executed_segment_source_hashes'],'input_hashes':m['input_dependencies'],'counts':{'retained_scientific_forwards':2184,'actual_model_returns':2272,'main_metric_records_including_anchors':metric_records,'suffix_metric_records':suffix_records,'saved_array_identities_verified':array_count,'controls':dict(counts),**stat_counts},'provenance':prov,'bootstrap_contrast_exactness':{k:v for k,v in cmp.report().items() if 'bootstrap' in k or 'contrast' in k or 'paired_start' in k},'all_numerical_comparisons':cmp.report(),'runtime_extrema':extrema,'saved_snapshot_scope':dict(snapshot_counts),'evidence_scope':{'recomputed_from_actual_arrays':'All32shards metric statistics, suffix/gate/norm exact prefixes, actual source references, saved FF gate/axis algebra,64contrast definitions,1536contrasts,768paired-start contrasts, shared2000bootstrap indices, all norm aggregates; complete8L10 QKV/mask/context snapshots and32actual vocabulary-logit rows.','recorded_runtime_witnesses':'Other full contexts/hidden/logits and512direct-V causal-order comparisons are supported by runtime fingerprints, recorded exact-control flags, scalar errors and complete per-input control rosters; full ordinary arrays were not saved and cannot be independently reconstructed.','model_weights':'No model weights loaded/read; hidden→LM-head logits not recomputed. Actual saved logits support readout NLL/KL/prediction checks.','interruption':'Original missing final guards remain unavailable and cause unknown. Exact70forward continuity establishes replay compatibility only.','uncertainty':'Four passage clusters per corpus, paired starts, fixed T18 calibration; correlated contrasts and no multiplicity adjustment.'},'resources':{'wall_seconds':time.time()-started,'peak_rss_kib':peak,'model_forwards':0,'pretrained_weights_read':False,'CUDA_initialized':False}}
 put(REVIEW/'gate_native_value_numerical_audit.json',result)
 with (REVIEW/'gate_native_value_numerical_audit.md').open('x') as f:
  f.write('Independent T19 numerical audit: PASS\n\n')
  f.write('The complete composite manifest `'+result['audited_manifest_sha256']+'` and analyzer output `'+result['analysis_sha256']+'` pass the independent saved-array audit. All32raw shards retain exact origin bytes, all three execution journals match prescribed chronology, and70forward continuity compares every ordered payload byte. Counts remain2184retained scientific forwards and2272actual model returns, including70validation and18discarded uncommitted returns.\n\n')
  f.write('Recomputed3264window rows,24anchor rows,3276norm rows,816summaries,1536contrasts and768paired-start contrasts with the same2000passage bootstrap indices. All4040recorded controls and unchanged bridge hard thresholds pass. Exact comparison counts, maximum residuals, input/source bindings, runtime extrema and resource measurements are in the JSON report. The tolerance policy was frozen before complete-panel numerical processing.\n\n')
  f.write('Actual-array reconstruction covers all saved gate/norm/source data, eight completeL10snapshots and32saved vocabulary-logit rows. Other full contexts/hidden/logits and direct-V paired states remain recorded runtime fingerprint/control evidence; no claim of independently recomputing unsaved arrays. No model weights or GPU were used. The original interrupted process lacks final source/input/calibration, parameter/version and precision guards; exact continuity does not restore them. Interruption cause remains unknown.\n')
 print('INDEPENDENT_NATIVE_VALUE_NUMERICAL_AUDIT_PASSED',json.dumps({'counts':result['counts'],'resources':result['resources'],'audited_manifest_sha256':result['audited_manifest_sha256']},allow_nan=False))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  failure={'status':'failed','error_type':type(e).__name__,'error':str(e),'audit_source_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_forwards':0,'gpu_used':False}
  put(REVIEW/('gate_native_value_numerical_audit_failure_'+str(time.time_ns())+'.json'),failure)
  raise
