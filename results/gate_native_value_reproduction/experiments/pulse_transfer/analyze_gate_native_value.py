"""Prospective T19 passage-paired statistics; no model or calibration fitting.

Full-view loss effects are primary. Early/later views share execution. All
intervals condition on the frozen T18 F fit; no multiplicity adjustment.
"""
import argparse, hashlib, itertools, json, math
from pathlib import Path
import numpy as np
import torch
from gate_wiki_storage import read_npz

CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
BOUNDARIES=tuple(''.join(x) for x in itertools.product('01',repeat=3))
OPERATIONS=('baseline','current','remove','orth')
MODES=tuple(b+'__'+op for b in BOUNDARIES for op in OPERATIONS)+('111__remove_scale','111__orth_scale')
VIEWS={'full':slice(1,511),'early':slice(1,127),'later':slice(127,511)}
NORM_VIEWS={'full':slice(0,510),'early':slice(0,126),'later':slice(126,510)}
SEED=20261021
REPETITIONS=2000
FIELDS=('nll','delta_nll_c','delta_nll_boundary','kl_c','kl_boundary','agreement_c','agreement_boundary')
NORM_KEYS={'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm','logit_delta_norm',
    'mass_delta_norm','context_delta_norm','b_norm','target_b_norm','mass_norm','target_mass_norm',
    'actual_context_norm','delivered_context_norm','stable_mixture_norm','source_v_norm'}

def dump(path,value):
    with Path(path).open('x') as f: f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def paired(values):
    x=np.asarray(values,dtype=np.float64)
    assert x.shape==(4,) and np.isfinite(x).all()
    ids=np.random.default_rng(SEED).integers(0,4,(REPETITIONS,4))
    return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[ids].mean(1),[.025,.975]).tolist(),
            'per_window':x.tolist()}

def combine(terms):
    out={}
    for mode,coefficient in terms:
        out[mode]=out.get(mode,0)+coefficient
    return {m:c for m,c in out.items() if c}

def delta_terms(boundary,operation,sign=1):
    return [(boundary+'__'+operation,sign),(boundary+'__baseline',-sign)]

def contrast_specs():
    specs=[]
    def add(name,terms,kind,operation=None,factor=None):
        specs.append({'name':name,'kind':kind,'operation':operation,'factor':factor,'terms':combine(terms)})
    for boundary in BOUNDARIES:
        for operation in ('remove','orth'):
            add('effect_'+operation+'_'+boundary,delta_terms(boundary,operation),'within_boundary_effect',operation)
        add('matched_remove_minus_orth_'+boundary,[(boundary+'__remove',1),(boundary+'__orth',-1)],'matched_control')
    for operation in ('remove','orth'):
        add('endpoint_111_minus_000_'+operation,delta_terms('111',operation)+delta_terms('000',operation,-1),
            'endpoint_effect_difference',operation)
        for j,factor in enumerate('ULD'):
            others=[k for k in range(3) if k!=j]
            for fixed in itertools.product('01',repeat=2):
                b=['0']*3
                for k,v in zip(others,fixed): b[k]=v
                zero=''.join(b); b[j]='1'; one=''.join(b)
                add('factor_'+factor+'_'+one+'_minus_'+zero+'_'+operation,
                    delta_terms(one,operation)+delta_terms(zero,operation,-1),'controlled_factor_interaction',operation,factor)
        for js in itertools.combinations(range(3),2):
            k=next(k for k in range(3) if k not in js)
            for fixed in '01':
                terms=[]
                for values in itertools.product('01',repeat=2):
                    b=['0']*3; b[k]=fixed
                    for j,v in zip(js,values): b[j]=v
                    sign=1 if values[0]==values[1] else -1
                    terms+=delta_terms(''.join(b),operation,sign)
                factor=''.join('ULD'[j] for j in js)
                add('two_factor_'+factor+'_fixed_'+str(k)+'_'+fixed+'_'+operation,terms,
                    'two_factor_interaction',operation,factor)
        terms=[]
        for boundary in BOUNDARIES:
            terms+=delta_terms(boundary,operation,(-1)**(3-boundary.count('1')))
        add('three_factor_ULD_'+operation,terms,'three_factor_interaction',operation,'ULD')
    assert len(specs)==64 and len({s['name'] for s in specs})==64
    assert all(set(s['terms'])<=set(MODES) for s in specs)
    return specs

def summarize(shards):
    groups={}; rows=[]; anchors=[]; normrows=[]; seen=set(); suffix_records=0
    for shard in shards:
        base=shard['metadata']; ident=(base['corpus'],base['identity'],base['window'])
        assert ident not in seen; seen.add(ident)
        assert base['corpus'] in CORPORA and base['identity'] in ('natural','bos') and base['window'] in range(4)
        assert base['length']==512 and len(base['token_sha256'])==64
        raw=shard['metrics']; expected=set(MODES)
        if base['identity']=='natural' and base['window']==0: expected|={'native_repeat','native_identity'}
        assert set(raw)==expected and set(shard['suffix'])==set(MODES)
        assert set(shard['norms'])==expected
        c=raw['000__baseline']
        for mode,z in raw.items():
            boundary=mode.split('__')[0] if mode in MODES else '000'
            ref=raw[boundary+'__baseline']
            for key,dtype in [('loss',torch.float32),('kl_c',torch.float32),('kl_boundary',torch.float32),
                              ('pred',torch.int64),('agreement_c',torch.bool),('agreement_boundary',torch.bool)]:
                assert z[key].shape==(511,) and z[key].dtype==dtype,(mode,key)
            assert ((z['pred']>=0)&(z['pred']<50257)).all()
            assert all(torch.isfinite(z[k]).all() for k in ('loss','kl_c','kl_boundary'))
            assert torch.equal(z['agreement_c'],z['pred'].eq(c['pred']))
            assert torch.equal(z['agreement_boundary'],z['pred'].eq(ref['pred']))
            assert len(z['final_hidden_sha256'])==64
            if mode.endswith('__baseline'): assert torch.count_nonzero(z['kl_boundary'])==0
            if mode.endswith('__current') or mode not in MODES:
                assert all(torch.equal(z[k],ref[k]) for k in ('loss','kl_c','kl_boundary','pred','agreement_c','agreement_boundary'))
                assert z['final_hidden_sha256']==ref['final_hidden_sha256']
            for view,sl in VIEWS.items():
                row={'corpus':base['corpus'],'identity':base['identity'],'window':base['window'],
                    'view':view,'mode':mode,'nll':float(z['loss'][sl].double().mean()),
                    'delta_nll_c':float((z['loss'][sl].double()-c['loss'][sl].double()).mean()),
                    'delta_nll_boundary':float((z['loss'][sl].double()-ref['loss'][sl].double()).mean()),
                    **{k:float(z[k][sl].double().mean()) for k in ('kl_c','kl_boundary','agreement_c','agreement_boundary')}}
                if mode in MODES:
                    rows.append(row);groups.setdefault((base['corpus'],base['identity'],view),{}).setdefault(base['window'],{})[mode]=row
                else: anchors.append(row)
            if mode in MODES:
                suffix=shard['suffix'][mode]; suffix_records+=1
                for k in ('loss','kl_c','kl_boundary','pred','agreement_c','agreement_boundary'):
                    n=127 if k=='loss' else 128
                    assert suffix[k].shape==(n,) and suffix[k].dtype==z[k].dtype
                    assert torch.equal(suffix[k],z[k][:n]),('suffix mismatch',ident,mode,k)
                assert torch.equal(suffix['target_indices'],torch.arange(1,128))
                assert torch.equal(suffix['logit_query_indices'],torch.arange(128))
        for mode,states in shard['norms'].items():
            assert mode in expected
            if mode=='native_repeat':
                assert states=={}, 'Unhooked native anchor has no intervention norm captures'
                continue
            assert set(states)=={10}
            vals=states[10]
            assert set(vals)==NORM_KEYS
            for view,sl in NORM_VIEWS.items():
                row={'corpus':base['corpus'],'identity':base['identity'],'window':base['window'],'view':view,'mode':mode,'layer':10,'values':{}}
                for key,array in vals.items():
                    shape=(12,) if key=='source_v_norm' else (511,)
                    dtype=torch.float32 if key=='s' else torch.float64
                    assert array.shape==shape and array.dtype==dtype and torch.isfinite(array).all(),(mode,key,array.shape)
                    if key!='s': assert (array>=0).all(),(mode,key)
                    selected=array[sl] if array.shape==(511,) else array
                    row['values'][key]={'scope':'scored_query_positions' if array.shape==(511,) else 'source_heads_constant_across_query_views',
                        'mean':float(selected.double().mean()),'rms':float(selected.double().square().mean().sqrt()),'max_abs':float(selected.abs().max())}
                normrows.append(row)
    assert seen=={(c,i,w) for c in CORPORA for i in ('natural','bos') for w in range(4)}
    assert len(rows)==3264 and len(anchors)==24 and suffix_records==1088
    summaries=[];contrasts=[]
    for (corpus,identity,view),windows in groups.items():
        assert set(windows)==set(range(4)) and all(set(d)==set(MODES) for d in windows.values())
        for mode in MODES:
            metrics={f:paired([windows[w][mode][f] for w in range(4)]) for f in FIELDS}
            for f,label in [('nll','perplexity'),('delta_nll_c','ppl_ratio_c'),('delta_nll_boundary','ppl_ratio_boundary')]:
                metrics[label]={'value':math.exp(metrics[f]['mean']),'ci95':[math.exp(x) for x in metrics[f]['ci95']]}
            summaries.append({'corpus':corpus,'identity':identity,'view':view,'mode':mode,'metrics':metrics})
        for spec in contrast_specs():
            values=[sum(coef*windows[w][mode]['nll'] for mode,coef in spec['terms'].items()) for w in range(4)]
            contrasts.append({'corpus':corpus,'identity':identity,'view':view,**spec,**paired(values)})
    paired_starts=[]
    for corpus in CORPORA:
        for view in VIEWS:
            natural=groups[corpus,'natural',view];bos=groups[corpus,'bos',view]
            for spec in contrast_specs():
                values=[sum(coef*(bos[w][mode]['nll']-natural[w][mode]['nll']) for mode,coef in spec['terms'].items()) for w in range(4)]
                paired_starts.append({'corpus':corpus,'view':view,'name':'bos_minus_natural_'+spec['name'],
                    'scope':'paired_start_interaction_secondary','terms':spec['terms'],**paired(values)})
    assert len(summaries)==816 and len(contrasts)==1536 and len(paired_starts)==768
    return {'window_metrics':rows,'anchors':anchors,'summaries':summaries,'contrasts':contrasts,
            'paired_start_interactions':paired_starts,'norm_window_summaries':normrows}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run=json.loads((a.out/'manifest.json').read_text())
    assert run['status']=='completed' and run['forward_passes']==2184
    assert run['source_hashes'][Path(__file__).name]==sha(__file__), 'Analyzer differs from frozen study source'
    paths=sorted((a.out/'inputs').glob('*.npz')); assert len(paths)==32
    inventory=json.loads((a.out/'numeric_inventory.json').read_text())
    assert {p.relative_to(a.out).as_posix() for p in paths}==set(inventory)
    for p in paths:
        entry=inventory[p.relative_to(a.out).as_posix()]
        assert entry['sha256']==sha(p) and entry['bytes']==p.stat().st_size
    result=summarize(read_npz(p) for p in paths)
    result['scope']={'seed':SEED,'bootstrap_repetitions':REPETITIONS,'cluster':'four passages per corpus; paired starts share passage indices',
        'calibration':'fixed T18 F; no population calibration uncertainty','multiplicity':'no adjustment; correlated contrasts are not independent replications',
        'views':{'full':'targets2..511 primary','early':'targets2..127 secondary','later':'targets128..511 secondary'},
        'causal_contrasts':'finite controlled operation effects and interactions; no mediation fractions or circuit uniqueness',
        'source_hashes':{p.name:sha(p) for p in paths},'analyzer_sha256':sha(__file__)}
    dump(a.out/'native_value_analysis.json',result)
    print('NATIVE_VALUE_ANALYSIS_PASSED',len(result['window_metrics']),len(result['summaries']),len(result['contrasts']),flush=True)

if __name__=='__main__': main()
