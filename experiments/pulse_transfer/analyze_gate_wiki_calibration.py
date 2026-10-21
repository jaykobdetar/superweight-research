"""Prespecified paired Wiki4 center/direction factorial statistics, no model load.

Evaluation intervals condition on two fixed calibrations; n=4 passages, not
2040 independent observations. Early/later views share a512-position execution.
"""
import json,math,argparse
from pathlib import Path
import numpy as np
import torch
from gate_wiki_calibration import CONDITIONS,FAMILIES,CELLS,SETS
from gate_wiki_storage import read_npz
from run_transfer import dump,sha
VIEWS={'full':slice(1,511),'early':slice(1,127),'later':slice(127,511)}
SEED=20261021
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')

def paired(values):
    x=np.asarray(values,dtype=np.float64);assert x.shape==(4,)and np.isfinite(x).all()
    ids=np.random.default_rng(SEED).integers(0,4,(2000,4))
    return {'n':4,'mean':float(x.mean()),'ci95':np.quantile(x[ids].mean(1),[.025,.975]).tolist(),'per_window':x.tolist()}

def contrast_specs():
    out=[]
    for layer_set in SETS:
        for family in FAMILIES:
            def key(cell):return family+'_'+cell+'_'+layer_set
            for name,terms in (
                ('center_at_O',{'FO':1,'OO':-1}),('center_at_F',{'FF':1,'OF':-1}),
                ('direction_at_O',{'OF':1,'OO':-1}),('direction_at_F',{'FF':1,'FO':-1}),
                ('interaction',{'FF':1,'FO':-1,'OF':-1,'OO':1}),
                ('M8_minus_OO',{'M8':1,'OO':-1}),('M8_minus_FF',{'M8':1,'FF':-1})):
                out.append({'name':name,'family':family,'layer_set':layer_set,
                    'scope':'secondary_mixture_and_sample_count_confound'if name.startswith('M8')else'primary_within_Wiki4',
                    'terms':{key(k):v for k,v in terms.items()}})
        for cell in (*CELLS,'M8'):
            out.append({'name':'remove_minus_orthogonal_'+cell,'family':'matched_control','layer_set':layer_set,
                'scope':'secondary_M8'if cell=='M8'else'primary_within_Wiki4',
                'terms':{'remove_'+cell+'_'+layer_set:1,'orthogonal_'+cell+'_'+layer_set:-1}})
    return out

def summarize(shards):
    rows=[];normrows=[];groups={};seen=set()
    for shard in shards:
        base=shard['metadata'];base=base.get('base',base)
        ident=(base['corpus'],base['identity'],base['window']);assert ident not in seen;seen.add(ident)
        assert base['corpus']in CORPORA and base['identity']in ('natural','bos')and base['window']in range(4)
        prefix=f"gate__{base['corpus']}__{base['window']}__512__{base['identity']}__"
        assert all(k.startswith(prefix)and '__'not in k[len(prefix):]for k in shard['per_token'])
        raw={k.rsplit('__',1)[-1]:v for k,v in shard['per_token'].items()}
        assert set(raw)==set(CONDITIONS)|({'repeat','identity'}if base['identity']=='natural'and base['window']==0 else set())
        assert len(raw)==len(shard['per_token'])
        c,p=raw['C'],raw['P']
        assert torch.count_nonzero(c['kl'])==0 and torch.count_nonzero(p['kl_p'])==0
        for trial,reference in [('current','P'),('fullrank','P')]+([('repeat','C'),('identity','C')]if 'repeat'in raw else []):
            assert all(torch.equal(raw[trial][k],raw[reference][k])for k in ('loss','kl','kl_p','pred','agreement_p'))
            assert raw[trial]['final_hidden_sha256']==raw[reference]['final_hidden_sha256']
        for mode,z in raw.items():
            assert z['score_index']==1
            for k,dtype in [('loss',torch.float32),('kl',torch.float32),('kl_p',torch.float32),('pred',torch.int64),('agreement_p',torch.bool)]:
                assert z[k].shape==(511,)and z[k].dtype==dtype,(mode,k)
            assert ((z['pred']>=0)&(z['pred']<50257)).all()
            assert len(z['final_hidden_sha256'])==64 and z['token_sha256']==p['token_sha256']
            assert all(torch.isfinite(z[k]).all()for k in ('loss','kl','kl_p'))
            assert torch.equal(z['agreement_p'],z['pred'].eq(p['pred']))
            for view,sl in VIEWS.items():
                row={'corpus':base['corpus'],'identity':base['identity'],'window':base['window'],'view':view,'mode':mode,
                    'nll':float(z['loss'][sl].double().mean()),'delta_nll_p':float((z['loss'][sl].double()-p['loss'][sl].double()).mean()),
                    'delta_nll_c':float((z['loss'][sl].double()-c['loss'][sl].double()).mean()),
                    'kl_c':float(z['kl'][sl].double().mean()),'kl_p':float(z['kl_p'][sl].double().mean()),
                    'agreement_c':float(z['pred'][sl].eq(c['pred'][sl]).double().mean()),'agreement_p':float(z['agreement_p'][sl].double().mean())}
                rows.append(row);groups.setdefault((base['corpus'],base['identity'],view),{}).setdefault(base['window'],{})[mode]=row
        for mode,states in shard['norms'].items():
            for layer,vals in states.items():
                for view,sl in {'full':slice(0,510),'early':slice(0,126),'later':slice(126,510)}.items():
                    normrows.append({'corpus':base['corpus'],'identity':base['identity'],'window':base['window'],'mode':mode,'layer':layer,'view':view,
                        **{key:{'mean':float(v[sl].double().mean()),'rms':float(v[sl].double().square().mean().sqrt()),'max_abs':float(v[sl].abs().max())}for key,v in vals.items()}})
    assert seen=={(c,i,w)for c in CORPORA for i in ('natural','bos')for w in range(4)}
    summaries=[];contrasts=[]
    for (corpus,identity,view),windows in groups.items():
        assert set(windows)=={0,1,2,3}
        for mode in CONDITIONS:
            values=[windows[w][mode]for w in range(4)]
            d={k:paired([r[k]for r in values])for k in ('nll','delta_nll_p','delta_nll_c','kl_c','kl_p','agreement_c','agreement_p')}
            d['ppl_ratio_p']={'point':math.exp(d['delta_nll_p']['mean']),'ci95':[math.exp(x)for x in d['delta_nll_p']['ci95']]}
            summaries.append({'corpus':corpus,'identity':identity,'view':view,'mode':mode,**d})
        for spec in contrast_specs():
            contrasts.append({'corpus':corpus,'identity':identity,'view':view,**spec,
                'delta_nll':paired([sum(weight*windows[w][mode]['nll']for mode,weight in spec['terms'].items())for w in range(4)])})
    return {'window_metrics':rows,'summary':summaries,'contrasts':contrasts,'norm_summary':normrows,
        'statistical_scope':{'evaluation_passages_per_group':4,'bootstrap_repetitions':2000,'bootstrap_seed':SEED,'common_indices':True,'views_share_execution':True,
            'calibrations_fixed':True,'calibration_population_uncertainty_estimated':False,'multiple_comparison_adjustment':False,
            'estimand':'Finite center/direction interventions conditional on the P source-value-zero boundary; M8 comparisons confound sample count and mixture.'}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();torch.set_num_threads(2)
    manifest=json.loads((a.run/'manifest.json').read_text());assert manifest['status']=='completed'
    assert manifest['forward_passes']==manifest['forward_attempts']==1548 and manifest['records']==776
    assert manifest['calibration_forwards']==4 and manifest['evaluation_forwards']==776 and manifest['suffix_forwards']==768
    inventory=json.loads((a.run/'numeric_inventory.json').read_text())
    for name,record in inventory.items():assert sha(a.run/name)==record['sha256']and (a.run/name).stat().st_size==record['bytes']
    files=sorted((a.run/'inputs').glob('*.npz'));assert len(files)==32
    assert set('inputs/'+p.name for p in files)=={n for n in inventory if n.startswith('inputs/')}
    for name,h in manifest['source_hashes'].items():assert sha(a.run/'frozen_sources'/name)==h
    for name,h in manifest['input_dependencies'].items():assert sha(a.run/name)==h
    result=summarize(read_npz(p)for p in files)
    result['input_shards']={p.name:sha(p)for p in files};result['source_sha256']=sha(__file__)
    target=a.run/'wiki_factorial_analysis.json';assert not target.exists();dump(target,result)
    print('WIKI_FACTORIAL_ANALYZED',len(result['summary']),len(result['contrasts']))
if __name__=='__main__':main()
