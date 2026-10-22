"""Prospective T20 NumPy statistics; no model, fitting, or outcome-based choice.

Decoder/CLI uses the declared 64-shard writer contract; independent review and
source/input launch freeze remain required before model execution.
The public statistical backend consumes saved per-target arrays, not vocabulary
readouts, and distinguishes scored views from execution/matching domains.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse, collections, hashlib, itertools, json, math
from pathlib import Path
import numpy as np

CORPORA = ('wiki_validation','utilitarianism','ethics','optimizers_new')
BOUNDARIES = ('000','111')
ARMS = ('baseline','current','remove_full','orth_full','remove_write_matched','orth_write_matched')
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
WRITE_KEYS = {arm:WRITE_COMMON_KEYS | (WRITE_MATCH_KEYS if arm.endswith('_write_matched') else set()) for arm in ARMS}


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
    # Each template is explicit; endpoint contrasts subtract the 000 operation
    # effect from the corresponding 111 effect, rather than subtract baselines.
    templates=(
        ('full_remove_effect',{'remove_full':1,'baseline':-1}),
        ('full_orth_effect',{'orth_full':1,'baseline':-1}),
        ('full_remove_minus_orth',{'remove_full':1,'orth_full':-1}),
        ('matched_remove_effect',{'remove_write_matched':1,'baseline':-1}),
        ('matched_orth_effect',{'orth_write_matched':1,'baseline':-1}),
        ('matched_remove_minus_orth',{'remove_write_matched':1,'orth_write_matched':-1}),
        ('matched_remove_minus_full',{'remove_write_matched':1,'remove_full':-1}),
        ('matched_orth_minus_full',{'orth_write_matched':1,'orth_full':-1}),
        ('matching_difference_of_differences',{'remove_write_matched':1,'orth_write_matched':-1,'remove_full':-1,'orth_full':1}))
    specs=[]
    for boundary in BOUNDARIES:
        for name,terms in templates:
            specs.append({'name':boundary+'__'+name,'kind':'within_boundary',
                'primary':boundary=='000' and name=='matched_remove_minus_orth',
                'terms':{boundary+'__'+arm:k for arm,k in terms.items()}})
    for name,terms in templates:
        specs.append({'name':'endpoint_111_minus_000__'+name,'kind':'endpoint_effect_difference','primary':False,
            'terms':{boundary+'__'+arm:sign*k for boundary,sign in (('111',1),('000',-1)) for arm,k in terms.items()}})
    require(len(specs)==27 and len({x['name'] for x in specs})==27,'27 unique contrasts')
    require(sum(x['primary'] for x in specs)==1 and all(sum(x['terms'].values())==0 and set(x['terms'])<=set(MODES) for x in specs),'primary and zero-sum definitions')
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
    require(set(records)==expected,'exact64 boundary/input records')
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
    require((len(rows),len(anchors),len(gate_rows),len(write_rows))==(1152,24,1152,1152),'complete primary/anchor/norm row counts')
    return rows,anchors,gate_rows,write_rows


def summarize_rows(rows):
    lookup={tuple(x[k] for k in ('corpus','identity','window','view','mode')):x for x in rows}
    expected={(c,i,w,v,m) for c in CORPORA for i in ('natural','bos') for w in range(4) for v in VIEWS for m in MODES}
    require(set(lookup)==expected and len(rows)==len(lookup),'complete unique statistical rows')
    indices=np.random.default_rng(SEED).integers(0,4,(REPETITIONS,4))
    summaries=[];contrasts=[];paired_starts=[];specs=contrast_specs()
    for c,i,v in itertools.product(CORPORA,('natural','bos'),VIEWS):
        for mode in MODES:
            metrics={key:paired([lookup[c,i,w,v,mode][key] for w in range(4)],indices) for key in FIELDS}
            for key,label in (('nll','perplexity'),('delta_nll_c','ppl_ratio_c'),('delta_nll_boundary','ppl_ratio_boundary')):
                metrics[label]={'value':math.exp(metrics[key]['mean']),'ci95':[math.exp(q) for q in metrics[key]['ci95']]}
            summaries.append({'corpus':c,'identity':i,'view':v,'mode':mode,'metrics':metrics})
        for spec in specs:
            xs=[math.fsum(k*lookup[c,i,w,v,mode]['nll'] for mode,k in spec['terms'].items()) for w in range(4)]
            contrasts.append({'corpus':c,'identity':i,'view':v,**spec,**paired(xs,indices)})
    for c,v in itertools.product(CORPORA,VIEWS):
        for spec in specs:
            xs=[math.fsum(k*(lookup[c,'bos',w,v,mode]['nll']-lookup[c,'natural',w,v,mode]['nll']) for mode,k in spec['terms'].items()) for w in range(4)]
            paired_starts.append({'corpus':c,'view':v,'name':'bos_minus_natural__'+spec['name'],'terms':spec['terms'],
                'scope':'paired_start_interaction_secondary',**paired(xs,indices)})
    require((len(summaries),len(contrasts),len(paired_starts))==(288,648,324),'fixed summary/contrast counts')
    return {'summaries':summaries,'contrasts':contrasts,'paired_start_interactions':paired_starts,
        'statistical_scope':{'bootstrap_seed':SEED,'bootstrap_repetitions':REPETITIONS,'shared_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
            'cluster':'four passages per corpus, paired starts/arms/boundaries share indices',
            'calibration':'one fixed F, no fit/population calibration uncertainty',
            'multiplicity':'unadjusted correlated contrasts, not independent replications',
            'primary':'native000 matched_remove_minus_orth by corpus/start fullview',
            'effects':'finite controlled operation contrasts; no mediation fraction/circuit uniqueness'}}


def matching_rows(records):
    reports=[]
    for identity,record in sorted(records.items()):
        c,w,i,b=identity
        for variant,field in (('main','write_norms'),('suffix','suffix_write_norms')):
            norms=record[field]
            left=norms['remove_write_matched'];right=norms['orth_write_matched']
            target=left['target_norm'];a=left['actual_write_delta_norm'];bb=right['actual_write_delta_norm']
            gap=np.abs(a-bb)
            require(exact(target,right['target_norm']),'same-state common norm target')
            require(exact(left['rounding_budget'],right['other_rounding_budget']) and exact(right['rounding_budget'],left['other_rounding_budget']),'mirrored per-arm rounding budgets')
            for match,own,other in ((left,norms['remove_full'],norms['orth_full']),(right,norms['orth_full'],norms['remove_full'])):
                for name in ('actual_write_norm','actual_write_delta_norm','proposed_full_write_delta_norm','proposed_pre_context_delta_norm','target_norm','own_full_norm','other_full_norm','rounding_budget','other_rounding_budget','absolute_pair_gap'):
                    require((match[name]>=0).all(),'nonnegative matching norm/budget')
                require(exact(match['own_full_norm'],own['actual_write_delta_norm']) and exact(match['other_full_norm'],other['actual_write_delta_norm']),'full-arm actual norm references')
                require(exact(match['proposed_full_write_delta_norm'],own['actual_write_delta_norm']),'replayed full proposal norm reference')
                require(exact(match['proposed_pre_context_delta_norm'],own['proposed_pre_context_delta_norm']),'replayed full proposed pre-context norm reference')
                require(exact(target,np.minimum(match['own_full_norm'],match['other_full_norm'])),'target min of actual candidate norms')
                scale=np.zeros(512,dtype=np.float64);positive=match['own_full_norm']>0
                scale[positive]=target[positive]/match['own_full_norm'][positive]
                require(exact(scale,match['lambda']) and ((scale>=0)&(scale<=1)).all(),'explicit zero/range/exact amplitude recipe')
                require(exact(match['signed_target_residual'],match['actual_write_delta_norm']-target),'actual signed target residual')
                require(exact(match['absolute_pair_gap'],gap),'actual two-delivered-arm pair gap')
                require(np.all(np.abs(match['signed_target_residual'])<=match['rounding_budget']),'per-query actual target error within frozen budget')
                require(np.count_nonzero(match['actual_write_delta_norm'][scale==0])==0,'zero endpoint actual delta norm')
                require(exact(match['actual_write_delta_norm'][scale==1],match['own_full_norm'][scale==1]),'one endpoint actual delta norm')
                require(match['actual_write_delta_norm'][0]==0 and target[0]==0 and scale[0]==0,'source query norm identity')
            require(np.all(gap<=left['rounding_budget']+right['rounding_budget']),'per-query actual paired norm budget')
            domains={}
            for name,sl in (('all_nonsource',slice(1,512)),('primary_scored',slice(1,511))):
                denominator=float(np.sqrt(np.mean(target[sl]*target[sl])))
                numerator=float(np.sqrt(np.mean(gap[sl]*gap[sl])))
                if denominator==0:
                    require(np.count_nonzero(a[sl])==np.count_nonzero(bb[sl])==0,'zero RMS target exact branch')
                else:
                    require(numerator<=1e-5*denominator,'frozen actual delivered RMS match-quality criterion')
                domains[name]={'RMS_target':denominator,'RMS_gap':numerator,'relative_RMS_gap':numerator/denominator if denominator else None}
            positive=target[1:]>0;relative=gap[1:][positive]/target[1:][positive]
            reports.append({'corpus':c,'window':w,'identity':i,'boundary':b,'variant':variant,'domains':domains,
                'maximum_absolute_pair_gap':float(gap.max()),'zero_target_queries':int(np.count_nonzero(target==0)),
                'positive_target_relative_gap_quantiles':np.quantile(relative,[0,.25,.5,.75,.95,.99,1]).tolist() if relative.size else None,
                'relative_quantile_probabilities':[0,.25,.5,.75,.95,.99,1],
                'quantized_to_zero_positive_targets':[int(np.count_nonzero((x==0)&(target>0))) for x in (a,bb)],
                'scope':'Actual saved norms of both delivered arms; norm equality is approximate in FP32, not identical downstream sensitivity.'})
    require(len(reports)==128,'all64 boundary/input times2 actual matching variants')
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True,help='Completed planned-segment composite directory')
    args=parser.parse_args();out=args.out
    manifest=json.loads((out/'manifest.json').read_text());inventory=json.loads((out/'numeric_inventory.json').read_text())
    require(manifest['status']=='completed' and manifest['execution_type']=='planned_sequential_segments','complete planned segmented study required')
    for key,want in (('forward_passes',776),('forward_attempts',776),('evaluation_forwards',384),('suffix_forwards',384),('anchor_forwards',8),('calibration_forwards',0)):
        require(manifest[key]==want,'literal complete forward count '+key)
    require(manifest['source_hashes'][Path(__file__).name]==sha(__file__),'frozen analyzer source identity')
    require(set(manifest['original_segments'])=={'A','B'},'two original completed segment bindings required')
    expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c in CORPORA for w in range(4) for i in ('natural','bos') for b in BOUNDARIES}
    require(set(inventory)==expected and {p.relative_to(out).as_posix() for p in (out/'inputs').glob('*.npz')}==expected,'exact64raw shard file universe')
    records={}
    for name in sorted(expected):
        path=out/name;entry=inventory[name]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],'raw shard inventory identity')
        shard=load_npz(path);meta=shard['metadata']
        require(shard['schema']=='post_write_norm_explicit_v1' and meta['length']==512,'literal boundary schema/shape')
        key=tuple(meta[k] for k in ('corpus','window','identity','boundary'))
        require(name==f'inputs/{key[0]}__{key[1]}__{key[2]}__{key[3]}.npz' and key not in records,'canonical unique raw identity')
        # Snapshot/projection identities and journal reconstruction have their
        # separate full numerical audit. Retain only analysis-relevant arrays.
        records[key]={k:shard[k] for k in ('metrics','suffix','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms')}
        del shard
    rows,anchors,gate_rows,write_rows=make_rows(records)
    result={'window_metrics':rows,'anchors':anchors,'gate_norm_window_summaries':gate_rows,
        'write_norm_window_summaries':write_rows,'matching_quality':matching_rows(records),**summarize_rows(rows),
        'scope':{'analyzer_sha256':sha(__file__),'raw_manifest_sha256':sha(out/'manifest.json'),'raw_inventory_sha256':sha(out/'numeric_inventory.json'),
            'source_hashes':{Path(n).name:inventory[n]['sha256'] for n in sorted(expected)},
            'primary_targets':'2..511; corresponding queries1..510','secondary_views':'early2..127 and later128..511 share512execution',
            'matching_acceptance_domains':'all nonsourceq1..511 and primaryscoredq1..510, both main and full rotatedsuffix',
            'limits':'Actual-write norm matching is approximate FP32 and does not match direction/downstream Jacobian; no mediation/fidelity/generation claim.'}}
    with (out/'post_write_norm_analysis.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('T20_ANALYSIS_PASSED',len(rows),len(result['summaries']),len(result['contrasts']),len(result['matching_quality']))

if __name__=='__main__':main()
