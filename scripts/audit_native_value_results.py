"""NumPy-only reanalysis of the restricted public T19 numerical release.

Recomputes published loss/KL/agreement/norm statistics, finite differences and
paired passage bootstraps. Verifies exported-byte commitments and available
suffix/current/anchor metrics. Does not reconstruct omitted prediction IDs,
attention states, vocabulary rows, checkpoints or private journal contents.
Statistical comparison helpers are retained from the independent primary audit
SHA735459860b4bbc79ef8d0ccc650ef536690fd3613f5402a39866cc8ba7179bd0.
No model, torch, network, calibration fitting or mutation of input artifacts.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import argparse,collections,hashlib,itertools,json,math,resource,time
from pathlib import Path
import numpy as np
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
BITS=tuple(format(i,'03b') for i in range(8))
MODES=tuple(b+'__'+op for b in BITS for op in ('baseline','current','remove','orth'))+('111__remove_scale','111__orth_scale')
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
NORM_VIEWS={'full':(0,510),'early':(0,126),'later':(126,510)}
FIELDS=('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
NORM_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
METRIC_KEYS=('loss','kl_c','kl_boundary','agreement_c','agreement_boundary')
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
def arrays_to_rows(directory,index,cmp):
 rows={};anchors={};normrows={};suffix_count=0;array_count=0
 expected_roster={(c,w,i) for c in CORPORA for w in range(4) for i in ('natural','bos')}
 require(len(index)==32 and {(x['metadata']['corpus'],x['metadata']['window'],x['metadata']['identity']) for x in index}==expected_roster,'exact32input roster')
 for entry in index:
  base=entry['metadata'];c,w,i=base['corpus'],base['window'],base['identity'];file=entry['file']
  require(file==f'{c}__{w}__{i}.npz' and base['length']==512,'canonical input identity')
  p=directory/'inputs'/file;require(sha(p)==entry['export_sha256'],'index binds exported bytes')
  require(read(p.with_suffix('.json'))['metadata']==base,'public sidecar identity')
  modes=MODES+(('native_repeat','native_identity') if i=='natural' and w==0 else ())
  with np.load(p,allow_pickle=False) as z:
   require(len(z.files)==len(set(z.files)) and set(z.files)==set(entry['arrays']),'exact per-file array roster')
   array_count+=len(z.files)
   for name in z.files:
    a=z[name];require(a.dtype in (np.dtype('float32'),np.dtype('float64'),np.dtype('bool')) and np.isfinite(a).all(),'only finite float/bool arrays')
   def metric(mode,key):
    a=z[f'metrics__{mode}__{key}'];want=np.bool_ if key.startswith('agreement_') else np.float32
    require(a.shape==(511,) and a.dtype==want,'metric shape/dtype');return a
   for mode in modes:
    boundary=mode.split('__')[0] if mode in MODES else '000';ref=boundary+'__baseline'
    a={k:metric(mode,k) for k in METRIC_KEYS};b={k:metric(ref,k) for k in METRIC_KEYS};control={k:metric('000__baseline',k) for k in METRIC_KEYS}
    if mode.endswith('__baseline'):require(np.count_nonzero(a['kl_boundary'])==0,'baselineKLzero')
    if mode.endswith('__current') or mode not in MODES:require(all(np.array_equal(a[k],b[k]) for k in METRIC_KEYS),'current/anchor exportedmetrics exact')
    if mode in MODES:
     for k in METRIC_KEYS:
      n=127 if k=='loss' else 128;ss=z[f'suffix__{mode}__{k}']
      require(ss.shape==(n,) and ss.dtype==a[k].dtype and np.array_equal(ss,a[k][:n]),'suffix exportedmetrics exact')
     suffix_count+=1
    for view,(lo,hi) in VIEWS.items():
     target=rows if mode in MODES else anchors
     target[c,i,w,view,mode]={'nll':float(a['loss'][lo:hi].astype(np.float64).mean()),'delta_nll_c':float((a['loss'][lo:hi].astype(np.float64)-control['loss'][lo:hi].astype(np.float64)).mean()),'delta_nll_boundary':float((a['loss'][lo:hi].astype(np.float64)-b['loss'][lo:hi].astype(np.float64)).mean()),**{k:float(a[k][lo:hi].astype(np.float64).mean()) for k in METRIC_KEYS if k!='loss'}}
    if mode=='native_repeat':continue
    vals={k:z[f'norms__{mode}__10__{k}'] for k in NORM_KEYS}
    for k,a in vals.items():
     require(a.shape==((12,) if k=='source_v_norm' else (511,)) and a.dtype==(np.float32 if k=='s' else np.float64),'norm shape/dtype')
     if k!='s':require((a>=0).all(),'norm nonnegative')
     if mode in MODES:
      ss=z[f'suffix_norms__{mode}__10__{k}'];require(np.array_equal(ss,a[:127]),'suffix exportednorm exact')
    for view,(lo,hi) in NORM_VIEWS.items():
     normrows[c,i,w,view,mode,10]={}
     for k,a in vals.items():
      x=(a if k=='source_v_norm' else a[lo:hi]).astype(np.float64)
      normrows[c,i,w,view,mode,10][k]={'scope':'source_heads_constant_across_query_views' if k=='source_v_norm' else 'scored_query_positions','mean':float(x.mean()),'rms':float(np.sqrt(np.mean(x*x))),'max_abs':float(np.abs(x).max())}
 require(suffix_count==1088 and array_count==41632,'complete suffix/array counts')
 return rows,anchors,normrows,{'numeric_arrays':array_count,'suffix_records':suffix_count}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 base=Path(__file__).resolve().parents[1]
 ap.add_argument('--results',type=Path,default=base/'results/gate_native_value')
 ap.add_argument('--inventory',type=Path,default=base/'manifests/native_value_numeric_inventory.json')
 ap.add_argument('--out',type=Path)
 a=ap.parse_args();start=time.monotonic();directory=a.results
 require(not directory.is_symlink(),'input root must not be symlink')
 inventory=read(a.inventory);files=inventory['files']
 actual=set()
 for p in directory.rglob('*'):
  require(not p.is_symlink(),'no symlinks in numerical release')
  if p.is_file():actual.add(p.relative_to(directory).as_posix())
 require(actual==set(files) and len(files)==83,'exact83file numericalrelease')
 for name,record in files.items():
  p=directory/name;require(p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],'release inventory '+name)
 export=read(directory/'export.json');m=read(directory/'manifest.json');analysis=read(directory/'native_value_analysis.json');index=read(directory/'index.json')
 require(m['status']=='completed' and m['execution_type']=='verified_composite_after_interruption','composite provenance explicit')
 require(m['forward_passes']==2184 and m['actual_cumulative_model_returns']==2272,'retained/actual forwards distinct')
 require(m['original_final_guards_available'] is False and m['parameter_versions_unchanged'] is None,'missing original endguards remain explicit')
 require(analysis['scope']['source_hashes']=={x['file']:x['raw_sha256'] for x in index},'analysis bound all32rawshards')
 require(all(export['raw_inventory']['inputs/'+x['file']]['sha256']==x['raw_sha256'] for x in index),'raw inventory commitment')
 cmp=Comparisons({'statistic_absolute_tolerance':1e-12,'norm_absolute_tolerance':1e-10,'norm_relative_tolerance':1e-12})
 rows,anchors,normrows,counts=arrays_to_rows(directory,index,cmp)
 ids=np.random.default_rng(20261021).integers(0,4,(2000,4))
 report={'status':'passed','scope':'Restricted public numerical reanalysis only; omitted prediction IDs, attention states, vocabulary rows and private journals are not revalidated.', 'numeric_export_sha256':sha(directory/'export.json'),'inventory_sha256':sha(a.inventory),'audit_source_sha256':sha(__file__),'statistics':statistics(analysis,rows,anchors,normrows,cmp,ids),'available_arrays':counts,'comparisons':cmp.report(),'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_forwards':0,'model_loaded':False}
 if a.out:put(a.out,report)
 print(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
