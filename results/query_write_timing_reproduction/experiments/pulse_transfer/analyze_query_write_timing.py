"""Prospective T21 native000 query-write timing NumPy statistics; no model, fitting, or outcome-based choice.

Decoder/CLI uses the declared 32-shard writer contract; independent review and
source/input launch freeze remain required before model execution.
The public statistical backend consumes saved per-target arrays, not vocabulary
readouts, and distinguishes scored views from execution/matching domains.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, collections, hashlib, itertools, json, math, time, traceback
from pathlib import Path
import numpy as np

CORPORA = ('wiki_validation','utilitarianism','ethics','optimizers_new')
BOUNDARIES = ('000',)
ARMS = ('baseline','current','remove_full','orth_full',
        'remove_write_matched_early','remove_write_matched_later','remove_write_matched_both',
        'orth_write_matched_early','orth_write_matched_later','orth_write_matched_both')
SUPPORTS=('early','later','both')
MODES = tuple(b+'__'+a for b in BOUNDARIES for a in ARMS)
VIEWS = {'full':(1,511),'early':(1,127),'later':(127,511)}
GATE_VIEWS = {'full':(0,510),'early':(0,126),'later':(126,510)}
FIELDS = ('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
METRIC_FIELDS = ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary')
GATE_KEYS = {'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm','mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm','actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}
SEED = 20261021
REPETITIONS = 2000
# Draft hook API; independent source review must bind these before launch.
WRITE_COMMON_KEYS = {'actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm'}
WRITE_MATCH_KEYS = {'target_norm','lambda','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','signed_target_residual','absolute_pair_gap'}
WRITE_KEYS = {arm:WRITE_COMMON_KEYS | (WRITE_MATCH_KEYS if '_write_matched_' in arm else set()) for arm in ARMS}


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


def contrast_specs():
    # Explicit prospective17contrast universe. Baseline is shared neither.
    specs=[]
    def add(name,terms,primary=False):
        specs.append({'name':'000__'+name,'kind':'within_native_boundary',
            'primary_view':'later' if primary else None,
            'terms':{'000__'+arm:coef for arm,coef in terms.items()}})
    for direction in ('remove','orth'):
        for support in SUPPORTS:
            add(direction+'_'+support+'_effect',{direction+'_write_matched_'+support:1,'baseline':-1},support=='early')
    for direction in ('remove','orth'):
        add(direction+'_full_effect',{direction+'_full':1,'baseline':-1})
    for support in SUPPORTS:
        add(support+'_remove_minus_orth',{'remove_write_matched_'+support:1,'orth_write_matched_'+support:-1},support=='early')
    for direction in ('remove','orth'):
        add(direction+'_support_interaction',{direction+'_write_matched_both':1,
            direction+'_write_matched_early':-1,direction+'_write_matched_later':-1,'baseline':1})
    add('support_interaction_remove_minus_orth',{
        'remove_write_matched_both':1,'remove_write_matched_early':-1,'remove_write_matched_later':-1,
        'orth_write_matched_both':-1,'orth_write_matched_early':1,'orth_write_matched_later':1})
    for direction in ('remove','orth'):
        add(direction+'_both_minus_full',{direction+'_write_matched_both':1,direction+'_full':-1})
    add('both_direction_difference_minus_full',{'remove_write_matched_both':1,'orth_write_matched_both':-1,
        'remove_full':-1,'orth_full':1})
    require(len(specs)==17 and len({x['name'] for x in specs})==17,'17 unique definitions')
    require(sum(s['primary_view']=='later' for s in specs)==3 and all(sum(s['terms'].values())==0 for s in specs),'3 later-view primary zero-sum definitions')
    return specs


def paired(values,indices):
    x=np.asarray(values,dtype=np.float64)
    require(x.shape==(4,) and np.isfinite(x).all(),'four finite passage clusters')
    return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[indices].mean(axis=1),[.025,.975]).tolist(),'per_window':x.tolist()}


def aggregate(values,domain):
    x=np.asarray(values,dtype=np.float64)
    require(x.ndim==1 and x.size and np.isfinite(x).all(),'finite nonempty aggregate domain')
    return {'scope':domain,'mean':float(x.mean()),'rms':float(np.sqrt(np.mean(x*x))),'max_abs':float(np.abs(x).max())}


def make_rows(records):
    expected={(c,w,i,b) for c in CORPORA for w in range(4) for i in ('natural','bos') for b in BOUNDARIES}
    require(set(records)==expected,'exact32 native boundary/input records')
    rows=[];anchors=[];gate_rows=[];write_rows=[]
    for c,w,i,b in sorted(records):
        record=records[c,w,i,b];raw=record['metrics'];suffix=record['suffix']
        require(set(record['gate_norms'])>=set(ARMS) and set(record['write_norms'])>=set(ARMS) and set(record['suffix_gate_norms'])==set(ARMS) and set(record['suffix_write_norms'])==set(ARMS),'complete norm arm roster')
        expected_arms=set(ARMS)|({'native_repeat','native_identity'} if b=='000' and w==0 and i=='natural' else set())
        require(set(raw)==expected_arms and set(suffix)==set(ARMS),'exact arm and suffix roster')
        baseline=raw['baseline'];native=records[c,w,i,'000']['metrics']['baseline']
        for arm,z in raw.items():
            for key in METRIC_FIELDS:
                a=z[key];dtype=np.int64 if key=='pred' else np.bool_ if key.startswith('agreement') else np.float32
                require(a.shape==(511,) and a.dtype==dtype and np.isfinite(a).all(),'metric shape/dtype/finite')
            require(((z['pred']>=0)&(z['pred']<50257)).all(),'prediction vocabulary range')
            require(exact(z['agreement_c'],z['pred']==native['pred']) and exact(z['agreement_boundary'],z['pred']==baseline['pred']),'saved agreement identity')
            if arm=='baseline':require(np.count_nonzero(z['kl_boundary'])==0,'baseline KL exact zero')
            if arm=='current' or arm not in ARMS:
                require(all(exact(z[k],baseline[k]) for k in METRIC_FIELDS),'current/anchor metrics exact')
            if arm in ARMS:
                require(exact(suffix[arm]['target_indices'],np.arange(1,128,dtype=np.int64)) and exact(suffix[arm]['logit_query_indices'],np.arange(128,dtype=np.int64)),'suffix target/query indexing')
                for key in METRIC_FIELDS:
                    n=127 if key=='loss' else 128
                    require(exact(suffix[arm][key],z[key][:n]),'causal suffix metric bytes')
            for view,(lo,hi) in VIEWS.items():
                row={'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,
                    'nll':float(z['loss'][lo:hi].astype(np.float64).mean()),
                    'delta_nll_c':float((z['loss'][lo:hi].astype(np.float64)-native['loss'][lo:hi].astype(np.float64)).mean()),
                    'delta_nll_boundary':float((z['loss'][lo:hi].astype(np.float64)-baseline['loss'][lo:hi].astype(np.float64)).mean()),
                    **{key:float(z[key][lo:hi].astype(np.float64).mean()) for key in ('kl_c','kl_boundary','agreement_c','agreement_boundary')}}
                (rows if arm in ARMS else anchors).append(row)
            if arm not in ARMS:continue  # Anchor diagnostics remain raw, outside efficacy norm summaries.
            g=record['gate_norms'][arm];require(set(g)=={10} and set(g[10])==GATE_KEYS,'frozen gate norm keyset')
            for name,a in g[10].items():
                require(a.shape==((12,) if name=='source_v_norm' else (511,)) and a.dtype==(np.float32 if name=='s' else np.float64) and np.isfinite(a).all(),'gate norm dtype/domain')
                if name!='s':require((a>=0).all(),'nonnegative gate norm')
                require(exact(record['suffix_gate_norms'][arm][10][name],a[:127]),'causal suffix gate norm bytes')
            for view,(lo,hi) in GATE_VIEWS.items():
                gate_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,'values':{
                    name:aggregate(a if name=='source_v_norm' else a[lo:hi],
                        'source_heads_constant_across_query_views' if name=='source_v_norm' else 'scored_query_positions') for name,a in g[10].items()}})
            vals=record['write_norms'][arm];require(set(vals)==WRITE_KEYS[arm],'frozen write-vector keyset')
            for name,a in vals.items():
                require(a.shape==(512,) and a.dtype==np.float64 and np.isfinite(a).all(),'write vector dtype/domain')
                if name!='signed_target_residual':require((a>=0).all(),'nonnegative write norm or amplitude/budget')
                if name=='lambda':require((a<=1).all(),'matching scale unit interval')
                ss=record['suffix_write_norms'][arm][name]
                require(ss.shape==(512,) and ss.dtype==np.float64 and np.isfinite(ss).all(),'complete suffix write vector domain')
                require(exact(ss[:128],a[:128]),'causal suffix write vector prefix bytes')
            require(vals['actual_write_delta_norm'][0]==vals['proposed_full_write_delta_norm'][0]==vals['proposed_pre_context_delta_norm'][0]==0,'source query actual/proposed norm invariance')
            if arm in ('baseline','current'):
                require(np.count_nonzero(vals['actual_write_delta_norm'])==np.count_nonzero(vals['proposed_full_write_delta_norm'])==np.count_nonzero(vals['proposed_pre_context_delta_norm'])==0,'baseline/current zero edit norms')
            for view,(lo,hi) in VIEWS.items():
                write_rows.append({'corpus':c,'window':w,'identity':i,'view':view,'mode':b+'__'+arm,'layer':10,
                    'values':{name:aggregate(a[lo:hi],'scored_query_positions') for name,a in vals.items()}})
    require((len(rows),len(anchors),len(gate_rows),len(write_rows))==(960,24,960,960),'complete primary/anchor/norm row counts')
    return rows,anchors,gate_rows,write_rows


def summarize_rows(rows):
    lookup={tuple(x[k] for k in ('corpus','identity','window','view','mode')):x for x in rows}
    expected={(c,i,w,v,m) for c in CORPORA for i in ('natural','bos') for w in range(4) for v in VIEWS for m in MODES}
    require(set(lookup)==expected and len(rows)==len(lookup),'complete unique statistical rows')
    indices=np.random.Generator(np.random.PCG64(SEED)).integers(0,4,(REPETITIONS,4),dtype=np.int64)
    require(hashlib.sha256(indices.tobytes()).hexdigest()=='9e2a313244e72f2bfb4300e2497a4b8590cc189411863bdabed15b4922e2d4bc','fixed shared PCG64 bootstrap indices')
    summaries=[];contrasts=[];paired_starts=[];specs=contrast_specs()
    for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
        for mode in MODES:
            metrics={key:paired([lookup[c,i,w,v,mode][key] for w in range(4)],indices) for key in FIELDS}
            for key,label in (('nll','perplexity'),('delta_nll_c','ppl_ratio_c'),('delta_nll_boundary','ppl_ratio_boundary')):
                metrics[label]={'value':math.exp(metrics[key]['mean']),'ci95':[math.exp(q) for q in metrics[key]['ci95']]}
            summaries.append({'corpus':c,'identity':i,'view':v,'mode':mode,'metrics':metrics})
        for spec in specs:
            xs=[math.fsum(k*lookup[c,i,w,v,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
            contrasts.append({'corpus':c,'identity':i,'view':v,**spec,'primary':spec['primary_view']==v,**paired(xs,indices)})
    for c,v in itertools.product(CORPORA,VIEWS):
        for spec in specs:
            xs=[math.fsum(k*(lookup[c,'bos',w,v,mode]['nll']-lookup[c,'natural',w,v,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
            paired_starts.append({'corpus':c,'view':v,'name':'bos_minus_natural__'+spec['name'],'terms':spec['terms'],
                'scope':'paired_start_interaction_secondary',**paired(xs,indices)})
    require((len(summaries),len(contrasts),len(paired_starts))==(240,408,204),'fixed summary/contrast counts')
    return {'summaries':summaries,'contrasts':contrasts,'paired_start_interactions':paired_starts,
        'statistical_scope':{'bootstrap_seed':SEED,'bootstrap_repetitions':REPETITIONS,'shared_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
            'cluster':'four passages per corpus, paired starts/arms/boundaries share indices',
            'calibration':'one fixed F, no fit/population calibration uncertainty',
            'multiplicity':'unadjusted correlated contrasts, not independent replications',
            'primary':'later-view early-only PC effect, orth effect, paired difference; every corpus/start',
            'effects':'Support contrasts have unequal write counts/dose; no timing-versus-dose isolation, mediation, circuit uniqueness or equivalence claim.'}}


def active_support(support):
    require(support in SUPPORTS,'known prospective support')
    mask=np.zeros(512,dtype=np.bool_)
    if support in ('early','both'):mask[1:127]=True
    if support in ('later','both'):mask[127:512]=True
    return mask


def matching_rows(records):
    reports=[]
    for identity,record in sorted(records.items()):
        c,w,i,b=identity
        require(b=='000','native000 only')
        for variant,field,opfield in (('main','write_norms','operations'),('suffix','suffix_write_norms','suffix_operations')):
            norms=record[field];ops=record[opfield]
            baseline=norms['baseline'];full_target=np.minimum(norms['remove_full']['actual_write_delta_norm'],norms['orth_full']['actual_write_delta_norm'])
            for support in SUPPORTS:
                active=active_support(support)
                left=norms['remove_write_matched_'+support];right=norms['orth_write_matched_'+support]
                target=np.where(active,full_target,0.)
                a=left['actual_write_delta_norm'];bb=right['actual_write_delta_norm'];gap=np.abs(a-bb)
                require(exact(target,left['target_norm']) and exact(target,right['target_norm']),'masked min-full target exact bytes')
                require(exact(left['rounding_budget'],right['other_rounding_budget']) and exact(right['rounding_budget'],left['other_rounding_budget']),'mirrored support budgets')
                for direction,z,other_direction in (('remove',left,'orth'),('orth',right,'remove')):
                    arm=direction+'_write_matched_'+support;own=norms[direction+'_full'];other=norms[other_direction+'_full'];both=norms[direction+'_write_matched_both']
                    require(exact(z['own_full_norm'],own['actual_write_delta_norm']) and exact(z['other_full_norm'],other['actual_write_delta_norm']),'unmasked full candidate references')
                    require(exact(z['proposed_full_write_delta_norm'],own['actual_write_delta_norm']) and exact(z['proposed_pre_context_delta_norm'],own['proposed_pre_context_delta_norm']),'full hypothetical proposal diagnostic domain')
                    scale=np.zeros(512,np.float64);positive=z['own_full_norm']>0
                    scale[positive]=full_target[positive]/z['own_full_norm'][positive];scale=np.where(active,scale,0.)
                    require(exact(z['lambda'],scale) and ((scale>=0)&(scale<=1)).all(),'fixed full-recipe scale masked without renormalization')
                    for name in ('target_norm','lambda','rounding_budget','other_rounding_budget','actual_write_delta_norm'):
                        require(exact(z[name],np.where(active,both[name],0.)),'fixed both-recipe supported subset '+name)
                    require(exact(z['actual_write_norm'][~active],baseline['actual_write_norm'][~active]),'inactive actual write norm equals baseline')
                    require(exact(z['signed_target_residual'],z['actual_write_delta_norm']-target) and exact(z['absolute_pair_gap'],gap),'actual delivered residual/pair diagnostics')
                    require(np.all(np.abs(z['signed_target_residual'])<=z['rounding_budget']),'frozen per-query target budget')
                    require(np.count_nonzero(z['actual_write_delta_norm'][scale==0])==0,'zero endpoint delivered norm')
                    require(exact(z['actual_write_delta_norm'][scale==1],z['own_full_norm'][scale==1]),'unit endpoint delivered norm')
                    require(z['actual_write_delta_norm'][0]==target[0]==scale[0]==0,'source q0 baseline norm')
                    wc=ops[arm]['write_checks']
                    require(wc['support']==support and wc['target_scope']=='same_frozen_full_recipe_target_on_active_queries_zero_elsewhere','support metadata identity')
                    require(wc['paired_actual_delivery_complete'] is (direction=='orth'),'PC arrives first; orth completes actual pair')
                    require(wc['counterpart_norm_scope']==('actual_delivered_same_support' if direction=='orth' else 'expected_CPU_rounded_not_yet_delivered_same_support'),'honest actual versus expected counterpart metadata')
                    require(wc['active_write_bytes_exact'] is True and wc['inactive_write_bytes_exact'] is True,'recorded support byte-copy guards')
                require(np.all(gap<=left['rounding_budget']+right['rounding_budget']),'frozen per-query paired norm bound')
                domains={}
                for name,domain in (('all_nonsource',np.arange(1,512)),('primary_scored',np.arange(1,511))):
                    den=float(np.sqrt(np.mean(target[domain]**2)));num=float(np.sqrt(np.mean(gap[domain]**2)))
                    if den==0:require(np.count_nonzero(a[domain])==np.count_nonzero(bb[domain])==0,'zero RMS target exact branch')
                    else:require(num<=1e-5*den,'frozen support RMS quality')
                    selected=domain[active[domain]]
                    domains[name]={'RMS_target':den,'RMS_gap':num,'relative_RMS_gap':num/den if den else None,
                        'active_count':int(len(selected)),
                        'active_only_target':aggregate(target[selected],'active_supported_queries'),
                        'active_only_gap':aggregate(gap[selected],'active_supported_queries')}
                positive=target[1:]>0;relative=gap[1:][positive]/target[1:][positive]
                reports.append({'corpus':c,'window':w,'identity':i,'boundary':b,'variant':variant,'support':support,
                    'active_queries':int(active.sum()),'active_scored_queries':int(active[1:511].sum()),'domains':domains,
                    'maximum_absolute_pair_gap':float(gap.max()),'zero_target_queries':int(np.count_nonzero(target==0)),
                    'active_zero_target_queries':int(np.count_nonzero(active&(target==0))),
                    'positive_target_relative_gap_quantiles':np.quantile(relative,[0,.25,.5,.75,.95,.99,1]).tolist() if relative.size else None,
                    'relative_quantile_probabilities':[0,.25,.5,.75,.95,.99,1],
                    'quantized_to_zero_positive_targets':[int(np.count_nonzero((x==0)&(target>0))) for x in (a,bb)],
                    'proposal_scope':'Own/other full write and pre-context proposals remain hypothetical outside active support; actual delta/target/lambda/budgets are support-masked.',
                    'matching_scope':'Actual PC/orth delivered norms checked after all arms; support sets differ in count and dose.'})
    require(len(reports)==192,'32inputs times2variants times3supports')
    return reports
def analyze(out):
    manifest=json.loads((out/'manifest.json').read_text());inventory=json.loads((out/'numeric_inventory.json').read_text())
    require(manifest['status']=='completed' and manifest['execution_type']=='planned_sequential_segments','complete T21 planned segmented study required')
    for key,want in (('forward_passes',648),('forward_attempts',648),('fully_processed_forwards',648),('evaluation_forwards',320),('suffix_forwards',320),('anchor_forwards',8),('calibration_forwards',0)):
        require(manifest[key]==want,'literal T21 complete count '+key)
    require(manifest['boundaries']==['000'] and manifest['arms']==list(ARMS),'frozen native ten-arm design')
    require(manifest['source_hashes'][Path(__file__).name]==sha(__file__),'frozen T21 analyzer source identity')
    require(set(manifest['original_segments'])=={'A','B'},'two completed planned segment bindings')
    expected={f'inputs/{c}__{w}__{i}__000.npz' for c in CORPORA for w in range(4) for i in ('natural','bos')}
    require(set(inventory)==expected and {p.relative_to(out).as_posix() for p in (out/'inputs').glob('*.npz')}==expected,'exact32 native input shards')
    records={}
    for name in sorted(expected):
        path=out/name;entry=inventory[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'T21 raw shard inventory identity')
        shard=load_npz(path);meta=shard['metadata']
        require(shard['schema']=='query_write_timing_explicit_v1' and meta['length']==512 and meta['boundary']=='000','literal native timing schema')
        key=tuple(meta[k] for k in ('corpus','window','identity','boundary'))
        require(name==f'inputs/{key[0]}__{key[1]}__{key[2]}__000.npz' and key not in records,'canonical unique native raw identity')
        # Timing fingerprints/sampled raw guards and full snapshots/readouts have
        # their own independent numerical audit. This backend uses saved metrics,
        # norm vectors and actual/expected delivery metadata, not unsaved states.
        records[key]={k:shard[k] for k in ('metrics','suffix','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms','operations','suffix_operations')}
        del shard
    rows,anchors,gate_rows,write_rows=make_rows(records)
    result={'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gate_rows,
        'write_norm_window_summaries':write_rows,'matching_quality':matching_rows(records),**summarize_rows(rows),
        'scope':{'analyzer_sha256':sha(__file__),'raw_manifest_sha256':sha(out/'manifest.json'),'raw_inventory_sha256':sha(out/'numeric_inventory.json'),
            'source_hashes':{Path(n).name:inventory[n]['sha256'] for n in sorted(expected)},
            'primary':'Later-view early-only PC effect, orth effect and paired difference in every source/start stratum.',
            'primary_targets':'2..511; queries1..510. Primary mechanistic contrast view later targets128..511.',
            'secondary_views':'Early targets2..127 and full targets2..511 share fixed512 execution.',
            'matching_acceptance_domains':'All nonsourceq1..511 and scoredq1..510, main and full rotated suffix, three supports.',
            'support_counts':{'early':{'all_nonsource':126,'primary_scored':126},'later':{'all_nonsource':385,'primary_scored':384},'both':{'all_nonsource':511,'primary_scored':510}},
            'norm_semantics':'Targets/lambda/budgets and actual change norms are support-restricted; proposed full-write/pre-context norms remain hypothetical outside support.',
            'timing_evidence':'All-query timing controls are recorded runtime fingerprints/flags; bounded sampled timing states and10 full snapshot states require separate independent audit. No unsaved whole-state reconstruction claimed.',
            'limits':'Support sets differ in query count/natural dose. No timing-versus-dose isolation, mediation fraction, unique circuit, equivalence/absence, fidelity or generation claim.'}}
    with (out/'query_write_timing_analysis.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('T21_ANALYSIS_PASSED',len(rows),len(result['summaries']),len(result['contrasts']),len(result['matching_quality']))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='Completed T21 planned-segment composite directory')
    args=parser.parse_args()
    try:analyze(args.out)
    except BaseException as error:
        if args.out.is_dir():
            failure=args.out/('query_write_timing_analysis_failure_'+str(time.time_ns())+'.json')
            with failure.open('x') as f:
                json.dump({'status':'failed','error_type':type(error).__name__,'error':str(error),
                    'traceback':traceback.format_exc(),'analyzer_sha256':sha(__file__),
                    'scope':'Analysis failure; original raw shards untouched.','model_forwards':0,'gpu_used':False},f,indent=2,allow_nan=False)
                f.write('\n')
        raise

if __name__=='__main__':main()
