"""Prospective T22 NumPy-only saved-array analyzer. SOURCE ONLY pending review.

No model, fitting, scorer replacement, outcome selection or unsaved reconstruction.
Three canonical FP32 metric families remain signed, including tiny/negative KL.
The independent N audit is separately authored; P/R do not establish causality.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, itertools, json, math, time, traceback
from pathlib import Path
import numpy as np
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
BOUNDARIES=('000',)
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched_early','remove_l11_kv',
      'orth_write_matched_early','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
DONORS=('remove_write_matched_early','orth_write_matched_early')
RECIPIENTS=('remove_l11_kv','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
MODES=tuple('000__'+a for a in ARMS)
VIEWS={'full':(1,511),'early':(1,127),'later':(127,511)}
GATE_VIEWS={'full':(0,510),'early':(0,126),'later':(126,510)}
FIELDS=('nll','kl_baseline','agreement_baseline')
METRIC_FIELDS=('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')
GATE_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
WRITE_COMMON_KEYS={'actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'}
WRITE_MATCH_KEYS={'target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap'}
WRITE_KEYS={a:WRITE_COMMON_KEYS|(WRITE_MATCH_KEYS if a in DONORS else set()) for a in ARMS}
L11_KEYS={'q_norm','k_norm','v_norm','q_delta_baseline_norm','k_delta_baseline_norm','v_delta_baseline_norm','q_delta_original_norm','k_delta_original_norm','v_delta_original_norm'}
SEED=20261021
REPETITIONS=2000
BOOTSTRAP_SHA='9e2a313244e72f2bfb4300e2497a4b8590cc189411863bdabed15b4922e2d4bc'


def require(ok, message):
    if not ok:
        raise AssertionError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def exact(a,b):
    """Byte identity, including signed zero, dtype, shape, and no object arrays."""
    return (isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and
            not a.dtype.hasobject and a.dtype==b.dtype and a.shape==b.shape and
            a.tobytes(order='C')==b.tobytes(order='C'))

def load_npz(path):
    """Independent lossless tensor-container decoder; never uses pickle/torch."""
    with np.load(path,allow_pickle=False) as z:
        header=json.loads(z['metadata'].tobytes())
        require(header['schema']=='wiki_gate_lossless_npz_v1','lossless container schema')
        seen=set()
        def decode(node):
            kind=node['type']
            if kind=='scalar':return node['value']
            if kind in ('tuple','list'):
                value=[decode(x) for x in node['items']]
                return tuple(value) if kind=='tuple' else value
            if kind=='dict':
                pairs=[(decode(k),decode(v)) for k,v in node['items']]
                value=dict(pairs);require(len(value)==len(pairs),'no duplicate logical keys')
                return value
            require(kind=='tensor' and node['array'] not in seen,'unique tensor member')
            seen.add(node['array']);a=z[node['array']]
            require(not a.dtype.hasobject,'no object array')
            require(node['shape']==list(a.shape) and node['dtype']=='torch.'+a.dtype.name and
                    node['sha256']==hashlib.sha256(a.tobytes(order='C')).hexdigest(),'array identity')
            return a
        result=decode(header['object'])
        require(set(z.files)==seen|{'metadata'} and len(z.files)==len(set(z.files)),'exact NPZ member universe')
        return result

def paired(values,indices):
    x=np.asarray(values,dtype=np.float64)
    require(x.shape==(4,) and np.isfinite(x).all(),'four finite passage clusters')
    return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[indices].mean(axis=1),[.025,.975]).tolist(),'per_window':x.tolist()}

def aggregate(values,domain):
    x=np.asarray(values,dtype=np.float64)
    require(x.ndim==1 and x.size and np.isfinite(x).all(),'finite nonempty aggregate domain')
    return {'scope':domain,'mean':float(x.mean()),'rms':float(np.sqrt(np.mean(x*x))),'max_abs':float(np.abs(x).max())}

def contrast_specs():
    specs=[]
    def add(name,terms,primary=False):
        specs.append({'name':'000__'+name,'kind':'within_native_boundary','primary_view':'later' if primary else None,
                      'terms':{'000__'+arm:k for arm,k in terms.items()}})
    # The shared baseline is neither; each direction supplies donor/K/V/KV.
    for label,direction in (('PC','remove'),('orth','orth')):
        donor=direction+'_write_matched_early';k=direction+'_l11_k';v=direction+'_l11_v';kv=direction+'_l11_kv'
        add(label+'_donor_effect',{donor:1,'baseline':-1})
        add(label+'_K_effect',{k:1,'baseline':-1},True)
        add(label+'_V_effect',{v:1,'baseline':-1},True)
        add(label+'_KV_effect',{kv:1,'baseline':-1})
        add(label+'_K_minus_V',{k:1,v:-1},True)
        add(label+'_interaction',{kv:1,k:-1,v:-1,'baseline':1},True)
        add(label+'_KV_minus_donor',{kv:1,donor:-1})
    for path in ('K','V','KV'):
        add(path+'_PC_minus_orth',{'remove_l11_'+path.lower():1,'orth_l11_'+path.lower():-1})
    add('donor_PC_minus_orth',{'remove_write_matched_early':1,'orth_write_matched_early':-1})
    add('interaction_PC_minus_orth',{'remove_l11_kv':1,'remove_l11_k':-1,'remove_l11_v':-1,
                                   'orth_l11_kv':-1,'orth_l11_k':1,'orth_l11_v':1})
    require(len(specs)==19 and len({x['name'] for x in specs})==19,'19 unique explicit definitions')
    require(sum(s['primary_view']=='later' for s in specs)==8 and all(sum(s['terms'].values())==0 for s in specs),'eight later primary zero-sum definitions')
    return specs


def make_rows(records):
    expected={(c,w,i,'000') for c in CORPORA for w in range(4) for i in ('natural','bos')}
    require(set(records)==expected,'exact32 native input records')
    rows=[];anchors=[];gate_rows=[];write_rows=[];l11_rows=[]
    for c,w,i,b in sorted(records):
        r=records[c,w,i,b];raw=r['metrics'];suffix=r['suffix'];base=raw['baseline']
        expected_arms=set(ARMS)|({'native_repeat','native_identity'} if w==0 and i=='natural' else set())
        require(set(raw)==expected_arms and set(suffix)==set(ARMS),'exact scientific/anchor roster')
        for field in ('gate_norms','write_norms','l11_norms'):
            require(set(r[field])==expected_arms and set(r['suffix_'+field])==set(ARMS),'exact norm roster '+field)
        for arm,z in raw.items():
            require(set(z)==set(METRIC_FIELDS),'exact six canonical metric fields')
            for key,a in z.items():
                dtype=np.int64 if key=='pred' else np.bool_ if key.startswith('agreement') else np.float32
                require(a.shape==(511,) and a.dtype==dtype and np.isfinite(a).all(),'metric shape/dtype/finite')
            require(((z['pred']>=0)&(z['pred']<50257)).all(),'prediction range')
            require(exact(z['agreement_c'],z['pred']==base['pred']) and exact(z['agreement_boundary'],z['pred']==base['pred']),'baseline agreement identity')
            require(exact(z['kl_c'],z['kl_boundary']),'native000 KL co-family identity')
            if arm=='baseline':require(np.count_nonzero(z['kl_boundary'])==0,'baseline KL zero')
            if arm=='current' or arm not in ARMS:
                require(all(exact(z[k],base[k]) for k in METRIC_FIELDS),'current/anchor metric bytes')
            if arm in ARMS:
                ss=suffix[arm]
                require(set(ss)==set(METRIC_FIELDS)|{'target_indices','logit_query_indices'},'exact suffix metric fields')
                require(exact(ss['target_indices'],np.arange(1,128,dtype=np.int64)) and exact(ss['logit_query_indices'],np.arange(128,dtype=np.int64)),'suffix indices')
                for key in METRIC_FIELDS:require(exact(ss[key],z[key][:(127 if key=='loss' else 128)]),'suffix causal metric prefix bytes')
            for view,(lo,hi) in VIEWS.items():
                row={'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,
                     'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),
                     'kl_baseline':float(z['kl_boundary'][lo:hi].astype(np.float64).mean()),
                     'agreement_baseline':float(z['agreement_boundary'][lo:hi].astype(np.float64).mean())}
                (rows if arm in ARMS else anchors).append(row)
            if arm not in ARMS:continue
            g=r['gate_norms'][arm];sg=r['suffix_gate_norms'][arm]
            require(set(g)==set(sg)=={10} and set(g[10])==set(sg[10])==GATE_KEYS,'gate norm keysets')
            for name,a in g[10].items():
                require(a.shape==((12,) if name=='source_v_norm' else (511,)) and a.dtype==(np.float32 if name=='s' else np.float64) and np.isfinite(a).all(),'gate dtype/domain')
                if name!='s':require((a>=0).all(),'nonnegative gate norm')
                require(exact(sg[10][name],a[:127]),'suffix gate norm bytes')
            for view,(lo,hi) in GATE_VIEWS.items():
                gate_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{
                    name:aggregate(a if name=='source_v_norm' else a[lo:hi],'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in g[10].items()}})
            vals=r['write_norms'][arm];svals=r['suffix_write_norms'][arm]
            require(set(vals)==set(svals)==WRITE_KEYS[arm],'write-vector keysets')
            for name,a in vals.items():
                sa=svals[name]
                require(a.shape==sa.shape==(512,) and a.dtype==sa.dtype==np.float64 and np.isfinite(a).all() and np.isfinite(sa).all(),'write-vector full domain')
                if name!='signed_target_residual':require((a>=0).all() and (sa>=0).all(),'nonnegative write norm/amplitude')
                if name=='lambda':require((a<=1).all() and (sa<=1).all(),'unit-interval lambda')
                require(exact(sa[:128],a[:128]),'suffix write prefix bytes')
            if arm in ('baseline','current')+RECIPIENTS:
                require(all(np.count_nonzero(vals[n])==np.count_nonzero(svals[n])==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm')),'recipient/baseline L10 is actual baseline, no hypothetical edit')
            require(all(vals[n][0]==0 for n in ('actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm')),'source q0 norm invariance')
            for view,(lo,hi) in VIEWS.items():
                write_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,
                                   'values':{n:aggregate(a[lo:hi],'scored_query_positions') for n,a in vals.items()}})
            ln=r['l11_norms'][arm];sln=r['suffix_l11_norms'][arm]
            require(set(ln)==set(sln)==L11_KEYS,'nine L11 vector keys')
            for name,a in ln.items():
                sa=sln[name]
                require(a.shape==sa.shape==(512,) and a.dtype==sa.dtype==np.float64 and np.isfinite(a).all() and np.isfinite(sa).all() and (a>=0).all() and (sa>=0).all(),'finite nonnegative L11 full domain')
                require(exact(sa[:128],a[:128]),'suffix L11 prefix bytes')
            for view,(lo,hi) in VIEWS.items():
                l11_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':11,
                    'values':{n:aggregate(a[lo:hi],'scored_query_positions') for n,a in ln.items()},
                    'active_early_values':{n:aggregate(a[1:127],'active_E_queries1..126') for n,a in ln.items()}})
    require((len(rows),len(anchors),len(gate_rows),len(write_rows),len(l11_rows))==(1152,24,1152,1152,1152),'fixed wide metric/anchor/norm counts')
    return rows,anchors,gate_rows,write_rows,l11_rows


def summarize_rows(rows):
    lookup={tuple(x[k] for k in ('corpus','identity','window','view','mode')):x for x in rows}
    expected={(c,i,w,v,m) for c in CORPORA for i in ('natural','bos') for w in range(4) for v in VIEWS for m in MODES}
    require(set(lookup)==expected and len(rows)==len(lookup),'complete unique wide metric rows')
    indices=np.random.Generator(np.random.PCG64(SEED)).integers(0,4,(REPETITIONS,4),dtype=np.int64)
    require(hashlib.sha256(indices.tobytes()).hexdigest()==BOOTSTRAP_SHA,'fixed shared bootstrap matrix')
    summaries=[];contrasts=[];starts=[];specs=contrast_specs()
    for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
        for mode in MODES:
            metrics={key:paired([lookup[c,i,w,v,mode][key] for w in range(4)],indices) for key in FIELDS}
            delta=paired([lookup[c,i,w,v,mode]['nll']-lookup[c,i,w,v,'000__baseline']['nll'] for w in range(4)],indices)
            transforms={'perplexity':{'value':math.exp(metrics['nll']['mean']),'ci95':[math.exp(q) for q in metrics['nll']['ci95']]},
                        'ppl_ratio_baseline':{'value':math.exp(delta['mean']),'ci95':[math.exp(q) for q in delta['ci95']], 'delta_nll':delta}}
            summaries.append({'corpus':c,'identity':i,'view':v,'mode':mode,'metrics':metrics,'transforms':transforms})
        for metric,spec in itertools.product(FIELDS,specs):
            xs=[math.fsum(k*lookup[c,i,w,v,mode][metric] for mode,k in spec['terms'].items()) for w in range(4)]
            primary=spec['primary_view']==v and metric in ('nll','kl_baseline')
            contrasts.append({'corpus':c,'identity':i,'view':v,'metric':metric,**spec,'primary':primary,**paired(xs,indices)})
    for c,v,metric,spec in itertools.product(CORPORA,VIEWS,FIELDS,specs):
        xs=[math.fsum(k*(lookup[c,'bos',w,v,mode][metric]-lookup[c,'natural',w,v,mode][metric]) for mode,k in spec['terms'].items()) for w in range(4)]
        starts.append({'corpus':c,'view':v,'metric':metric,'name':'bos_minus_natural__'+spec['name'],'terms':spec['terms'],
                       'scope':'paired_start_interaction_secondary','primary':False,**paired(xs,indices)})
    require((len(summaries),len(contrasts),len(starts))==(288,1368,684) and sum(x['primary'] for x in contrasts)==128,'fixed expanded metric/primary counts')
    return {'summaries':summaries,'contrasts':contrasts,'paired_start_interactions':starts,'statistical_scope':{
        'bootstrap_seed':SEED,'bootstrap_repetitions':REPETITIONS,'shared_indices_sha256':BOOTSTRAP_SHA,
        'cluster':'four passages per corpus; starts/arms/views/metrics/contrasts share indices',
        'row_counts':{'window_wide':1152,'window_scalar_metric_entries':3456,'anchor_wide':24,'arm_summary_wide':288,'arm_summary_metric_records':864,'contrast_long':1368,'paired_start_long':684,'primary':128},
        'primary':'Eight predeclared K/V estimands x later NLL/baseline KL x eight source/start strata; all128 reported, no any-one-CI success rule.',
        'multiplicity':'Nominal conditional95% intervals, unadjusted and correlated; no fit/population uncertainty or independent-replication interpretation.',
        'numerical_scope':'Signed canonical FP32 KL retained. Selected R saved-logit FP64 diagnostics are not a whole-panel precision/sensitivity bound.',
        'limits':'Finite donor transplants, unequal natural K/V doses; no mediation fraction, intrinsic path importance, unique circuit, equivalence or method-priority claim.'}}


def matching_rows(records):
    reports=[];active=np.zeros(512,np.bool_);active[1:127]=True
    for (c,w,i,b),r in sorted(records.items()):
        for variant,field,opfield in (('main','write_norms','operations'),('suffix','suffix_write_norms','suffix_operations')):
            norms=r[field];left=norms[DONORS[0]];right=norms[DONORS[1]]
            target_full=np.minimum(norms['remove_full']['actual_write_delta_norm'],norms['orth_full']['actual_write_delta_norm'])
            target=np.where(active,target_full,0.);gap=np.abs(left['actual_write_delta_norm']-right['actual_write_delta_norm'])
            for direction,z,other in (('remove',left,'orth'),('orth',right,'remove')):
                own=norms[direction+'_full'];otherfull=norms[other+'_full']
                require(exact(z['own_full_norm'],own['actual_write_delta_norm']) and exact(z['other_full_norm'],otherfull['actual_write_delta_norm']),'full proposal norm references')
                require(exact(z['proposed_full_write_delta_norm'],own['actual_write_delta_norm']) and exact(z['proposed_pre_context_delta_norm'],own['proposed_pre_context_delta_norm']),'hypothetical full proposal diagnostics')
                scale=np.zeros(512,np.float64);positive=z['own_full_norm']>0
                scale[positive]=target_full[positive]/z['own_full_norm'][positive];scale=np.where(active,scale,0.)
                require(exact(z['target_norm'],target) and exact(z['lambda'],scale),'unchanged masked min-full recipe')
                require(exact(z['actual_write_norm'][~active],norms['baseline']['actual_write_norm'][~active]),'inactive donor write baseline norm')
                require(np.count_nonzero(z['rounding_budget'][~active])==np.count_nonzero(z['other_rounding_budget'][~active])==0,'inactive budgets masked')
                require(exact(z['signed_target_residual'],z['actual_write_delta_norm']-target) and exact(z['absolute_pair_gap'],gap),'actual residual and pair gap')
                require(np.all(np.abs(z['signed_target_residual'])<=z['rounding_budget']),'frozen target budget')
                require(np.count_nonzero(z['actual_write_delta_norm'][scale==0])==0 and exact(z['actual_write_delta_norm'][scale==1],z['own_full_norm'][scale==1]),'zero/unit endpoint norms')
                wc=r[opfield][direction+'_write_matched_early']['write_checks']
                require(wc['support']=='early' and wc['active_write_bytes_exact'] is True and wc['inactive_write_bytes_exact'] is True,'recorded donor support byte guards')
            require(exact(left['rounding_budget'],right['other_rounding_budget']) and exact(right['rounding_budget'],left['other_rounding_budget']),'mirrored budgets')
            require(np.all(gap<=left['rounding_budget']+right['rounding_budget']),'actual paired norm bound')
            domains={}
            for name,domain in (('all_nonsource',np.arange(1,512)),('primary_scored',np.arange(1,511))):
                den=float(np.sqrt(np.mean(target[domain]**2)));num=float(np.sqrt(np.mean(gap[domain]**2)))
                if den==0:require(np.count_nonzero(left['actual_write_delta_norm'][domain])==np.count_nonzero(right['actual_write_delta_norm'][domain])==0,'zero RMS target branch')
                else:require(num<=1e-5*den,'frozen relative RMS guard')
                selected=domain[active[domain]]
                domains[name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None,'active_count':int(len(selected)),
                               'active_only_target':aggregate(target[selected],'active_E'),'active_only_gap':aggregate(gap[selected],'active_E')}
            reports.append({'corpus':c,'window':w,'identity':i,'boundary':b,'variant':variant,'support':'early','active_queries':126,'active_scored_queries':126,
                            'domains':domains,'maximum_absolute_pair_gap':float(gap.max()),
                            'scope':'Actual rounded donor norms after both donors; full proposals hypothetical outside E. L11 K/V doses not matched.'})
    require(len(reports)==64,'32inputs times2variants actual donor pairs')
    return reports


def canonical_metrics(raw,operations,*,suffix=False):
    """Validate producer-only hidden metadata, retaining canonical numeric arrays.

    This boundary changes no metric, reduction or statistical helper. Full raw
    fingerprints remain in original shards/operations for the independent N audit.
    """
    require(isinstance(raw,dict) and 'baseline' in raw,'raw metric mapping with baseline')
    extra={'target_indices','logit_query_indices'} if suffix else set()
    expected=set(METRIC_FIELDS)|extra|{'final_hidden_sha256'}
    canonical={}
    def digest(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for arm,z in raw.items():
        require(isinstance(z,dict) and set(z)==expected,'exact producer metric arrays and hidden hash')
        h=z['final_hidden_sha256'];require(digest(h),'canonical64hex hidden hash')
        # Unhooked anchors have no owned hook observation; bind to baseline.
        if arm=='native_repeat':
            require(h==raw['baseline']['final_hidden_sha256'],'unhooked anchor hidden hash equals baseline')
        fields=operations.get(arm,{}).get('l11_records',{}).get('fields',{})
        if 'final_hidden' in fields:
            observed=fields['final_hidden'];require(isinstance(observed,dict) and digest(observed.get('sha256')) and observed['sha256']==h,'owned final-hidden observation matches metric hash')
        canonical[arm]={k:v for k,v in z.items() if k!='final_hidden_sha256'}
    return canonical


def analyze(out):
    manifest=json.loads((out/'manifest.json').read_text());inventory=json.loads((out/'numeric_inventory.json').read_text())
    require(manifest['status']=='completed' and manifest['execution_type']=='planned_sequential_segments','complete T22 planned segmented study')
    for marker in ('merge_failure.json','checkpoint_failure.json','failure_state.npz'):
        require(not (out/marker).exists(),'no composite failure marker '+marker)
    require(not list((out/'inputs').glob('*__partial.npz')),'no partial shards')
    for key,want in (('forward_passes',776),('forward_attempts',776),('fully_processed_forwards',776),('evaluation_forwards',384),('suffix_forwards',384),('anchor_forwards',8),('calibration_forwards',0)):
        require(manifest[key]==want,'literal T22 count '+key)
    require(manifest['boundaries']==['000'] and manifest['arms']==list(ARMS),'frozen native twelve-arm design')
    require(manifest['source_hashes'][Path(__file__).name]==sha(__file__),'frozen analyzer source identity')
    require(set(manifest['original_segments'])=={'A','B'},'two planned segment bindings')
    expected={f'inputs/{c}__{w}__{i}__000.npz' for c in CORPORA for w in range(4) for i in ('natural','bos')}
    require(set(inventory)==expected and {p.relative_to(out).as_posix() for p in (out/'inputs').glob('*.npz')}==expected,'exact32 raw shard paths')
    records={}
    for name in sorted(expected):
        path=out/name;entry=inventory[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'raw shard byte identity')
        shard=load_npz(path);meta=shard['metadata']
        require(shard['schema']=='l11_key_value_donor_explicit_v1' and meta['length']==512 and meta['boundary']=='000','T22 literal schema')
        key=tuple(meta[k] for k in ('corpus','window','identity','boundary'))
        require(name==f'inputs/{key[0]}__{key[1]}__{key[2]}__000.npz' and key not in records,'unique canonical raw input identity')
        records[key]={k:shard[k] for k in ('metrics','suffix','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms','l11_norms','suffix_l11_norms','operations','suffix_operations')}
        records[key]['metrics']=canonical_metrics(shard['metrics'],shard['operations'])
        records[key]['suffix']=canonical_metrics(shard['suffix'],shard['suffix_operations'],suffix=True)
        del shard
    rows,anchors,gates,writes,l11=make_rows(records)
    result={'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gates,'write_norm_window_summaries':writes,
            'l11_norm_window_summaries':l11,'matching_quality':matching_rows(records),**summarize_rows(rows),
            'scope':{'analyzer_sha256':sha(__file__),'raw_manifest_sha256':sha(out/'manifest.json'),'raw_inventory_sha256':sha(out/'numeric_inventory.json'),
                     'source_hashes':{Path(n).name:inventory[n]['sha256'] for n in sorted(expected)},
                     'norm_semantics':'L11 actual Q/K/V and deltas from baseline versus original natural projection are distinct. Recipient L10 is baseline, never a hypothetical full edit.',
                     'evidence':'Runtime protected-region/closure flags and fingerprints are not independent raw reconstruction. Selected5 L10/9 L11/12 endstate panels require separate N/P/R audits.',
                     'small_KL':'Signed FP32 metrics unchanged; selected saved-logit FP64 R diagnostics cannot bound unsaved rows or establish sensitivity at1e-6 from a1e-4 consistency PASS.',
                     'primary':'All128 finite co-primary comparisons reported; no any-one-CI study-success rule.'}}
    with (out/'l11_key_value_donor_analysis.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('T22_ANALYSIS_PASSED',len(rows),len(result['summaries']),len(result['contrasts']),len(result['matching_quality']))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:analyze(args.out)
    except BaseException as error:
        if args.out.is_dir():
            with (args.out/('l11_key_value_donor_analysis_failure_'+str(time.time_ns())+'.json')).open('x') as f:
                json.dump({'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
                           'analyzer_sha256':sha(__file__),'model_forwards':0,'gpu_used':False,'scope':'Analysis failure; original raw untouched.'},f,indent=2,allow_nan=False);f.write('\n')
        raise
if __name__=='__main__':main()
