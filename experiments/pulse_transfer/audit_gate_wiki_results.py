"""Saved-state T18 audit. No pretrained forward; runtime-only closures are labeled."""
import argparse,json,math,resource,time
from pathlib import Path
import torch
from gate_wiki_storage import read_npz,restore_snapshot,tensor_identity
from gate_wiki_calibration import CONDITIONS,SNAPSHOT_MODES,READOUT_ROWS,mode_spec,wiki_map,fit_geometry
from gpt2_gate_subspace import LAYERS,maps_digest
from gpt2_gate_subspace_v2 import fingerprint
from fit_gate_calibration import fit_maps_n
from run_transfer import sha,dump,th

def check_equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor)and a.dtype==b.dtype and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:check_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b)and len(a)==len(b)
        for x,y in zip(a,b):check_equal(x,y)
    else:assert a==b

def audit(run):
    started=time.time();torch.set_num_threads(4);manifest=json.loads((run/'manifest.json').read_text());assert manifest['status']=='completed'
    inventory=json.loads((run/'numeric_inventory.json').read_text())
    for name,record in inventory.items():assert sha(run/name)==record['sha256']and (run/name).stat().st_size==record['bytes']
    maps=read_npz(run/'calibration_maps.npz');ds=read_npz(run/'calibration_directions.npz');spectra=read_npz(run/'calibration_spectra.npz')
    assert {k:maps_digest(v)for k,v in maps.items()}==manifest['calibration_map_digests']
    cal={};calops=[]
    for path in sorted((run/'calibration').glob('*.npz')):
        d=read_npz(path);cal[path.stem]=d['raw'];calops.append(d['operation'])
    assert set(cal)=={'wiki_test__'+str(i)for i in range(4)}
    fresh,reports=fit_maps_n(cal,4);check_equal(fresh,maps['F']);check_equal(reports,spectra['F'])
    dd,geometry=fit_geometry(maps,spectra);check_equal(dd,ds);check_equal(geometry,read_npz(run/'calibration_geometry.npz'))
    tokens=torch.load(run/'tokens.pt',weights_only=True,map_location='cpu');lookup={(d['corpus'],d['window']):d['tokens']for d in tokens}
    dm=json.loads((run/'data_manifest.json').read_text());bos=dm['literals']['start_special']['id']
    metric_fields={'loss','pred','kl','kl_p','agreement_p'};totals={'inputs':0,'records':0,'suffix_records':0,'checks':0,'gate_checks':0,'snapshot_states':0,'norm_layer_states':0}
    extrema={};snap_errors={};allops=list(calops);seen=set()
    corpora=('wiki_validation','utilitarianism','ethics','optimizers_new')
    for path in sorted((run/'inputs').glob('*.npz')):
        d=read_npz(path);base=d['metadata'];identity=(base['corpus'],base['window'],base['identity'])
        assert identity not in seen;seen.add(identity)
        assert path.name==f"gate__{base['corpus']}__{base['window']}__512__{base['identity']}.npz"
        x=lookup[base['corpus'],base['window']].clone()
        if base['identity']=='bos':x[0]=bos
        y=x.clone();y[128:]=torch.roll(y[128:],1);raw={k.rsplit('__',1)[-1]:v for k,v in d['per_token'].items()}
        totals['inputs']+=1;totals['records']+=len(raw);totals['checks']+=len(d['checks'])
        assert len(d['operations'])==len(raw) and len(d['suffix_checks'])==len(d['suffix_operations'])==24
        for z in raw.values():assert z['token_sha256']==th(x)
        op={v['trial']:v for v in d['operations']};sop={v['trial']:v for v in d['suffix_operations']}
        assert set(op)==set(raw)and set(sop)==set(CONDITIONS)
        allops.extend(d['operations']);allops.extend(d['suffix_operations'])
        expected_checks={m+'_source_identity'for m in CONDITIONS if m!='C'}|{m+'_singleton'for m in CONDITIONS if len(mode_spec(m)[3])==1}|{'current_closure','fullrank_closure'}
        if base['identity']=='natural'and base['window']==0:expected_checks|={'repeat','identity'}
        assert len(d['checks'])==len(expected_checks)and {ch['trial']for ch in d['checks']}==expected_checks
        for check in d['checks']:
            if check['trial'].endswith('_source_identity'):required={'source_exact'}
            elif check['trial'].endswith('_singleton'):required={'b_exact','mass_exact','context_exact','coefficient_exact'}
            else:required={k+'_exact'for k in ('loss','kl','pred','kl_p','agreement_p','hidden','logits')}
            assert required<=set(check)
            assert all(check[k]==base[k]for k in ('corpus','window','identity','length'))
            assert all(value is True for k,value in check.items()if k.endswith('_exact')or k.endswith('_passed'))
            if 'source_exact'in check:assert check['reference_fingerprints']==check['trial_fingerprints']
            if 'reference_context'in check:assert check['reference_context']==check['trial_context']
            if 'logits_reference'in check:assert check['logits_reference']==check['logits_control']
        for mode in CONDITIONS:
            assert op[mode]['final_hidden_sha256']==raw[mode]['final_hidden_sha256']
            if mode!='C':assert op[mode]['source_fingerprints']==op['P']['source_fingerprints']
            affected=mode_spec(mode)[3]
            if len(affected)==1:
                for k in ('b','mass'):assert op[mode]['gate_fingerprints'][10][k]==op['P']['gate_fingerprints'][10][k]
                assert op[mode]['current_context_hashes'][10]==op['P']['current_context_hashes'][10]
        assert set(d['suffix_metrics'])==set(CONDITIONS)
        assert len({ch['trial']for ch in d['suffix_checks']})==24 and {ch['trial']for ch in d['suffix_checks']}==set(CONDITIONS)
        for check in d['suffix_checks']:
            mode=check['trial'];totals['suffix_records']+=1
            assert {k+'_exact'for k in ('logits','hidden','gates','source','mask','contexts','loss','pred','kl','kl_p','agreement_p')}<=set(check)
            assert check['original_token_sha256']==th(x)and check['control_token_sha256']==th(y)
            assert all(value is True for k,value in check.items()if k.endswith('_exact'))
            assert check['logits_reference']==check['logits_control']
            assert check['reference_context_hashes']==op[mode]['prefix_context_hashes']==check['control_context_hashes']==sop[mode]['prefix_context_hashes']
            assert op[mode]['mask_records']==sop[mode]['mask_records']
            sm=d['suffix_metrics'][mode];assert sm['token_sha256']==th(y)and sm['score_index']==1 and sm['scored_tokens']==126
            values=sm.get('raw',sm)
            for k in metric_fields:
                n=127 if k=='loss'else 128
                assert values[k].shape==(n,)and values[k].dtype==raw[mode][k].dtype and torch.equal(values[k],raw[mode][k][:n])
        assert set(d['norms'])==set(CONDITIONS)
        for mode,states in d['norms'].items():
            affected=mode_spec(mode)[3];assert set(states)==(set(LAYERS)if mode=='P'else set(affected))
            for layer,vals in states.items():
                totals['norm_layer_states']+=1
                expected={'logit_delta_norm','mass_delta_norm','context_delta_norm'}|({'s','hypothetical_pc1_norm','hypothetical_orthogonal_norm'}if mode!='P'else set())
                assert set(vals)==expected
                for k,v in vals.items():
                    assert v.shape==(511,)and torch.isfinite(v).all();assert fingerprint(v)==op[mode]['norm_fingerprints'][layer][k]
                    if k!='s':assert (v>=0).all()
                if mode=='P':assert all(torch.count_nonzero(v)==0 for v in vals.values())
        snapshot_input=(base['corpus'],base['window'],base['identity'])==('wiki_validation',0,'natural')
        if not snapshot_input:assert d['snapshots']is None and not d['sample_readouts'];continue
        assert set(d['snapshots']['states'])==set(d['sample_readouts'])==set(SNAPSHOT_MODES)
        for mode in SNAPSHOT_MODES:
            totals['snapshot_states']+=1;s=restore_snapshot(d['snapshots'],mode)
            q,k,v,mask=[s[n]for n in ('q','k','v','mask')]
            assert q.shape==k.shape==v.shape==(1,12,512,64)and q.dtype==k.dtype==v.dtype==torch.float32
            assert torch.count_nonzero(v[:,:,0])==0
            logits=q@k.transpose(-1,-2)/8
            logits=logits.masked_fill(torch.ones(512,512,dtype=torch.bool).triu(1),float('-inf'))
            logits=logits.masked_fill(~mask,float('-inf'))if mask.dtype==torch.bool else logits+mask
            a=logits.softmax(-1)[0,:,1:,1:].sum(-1);b=(logits[:,:,1:,1:].logsumexp(-1)-logits[:,:,1:,0])[0]
            u=(logits[:,:,1:,1:].softmax(-1)@v[:,:,1:])[0]
            native=(logits.softmax(-1)@v).transpose(1,2).reshape(1,512,768)
            localmaps={label:{key:t.float()for key,t in m[10].items()}for label,m in maps.items()}
            localdirs={label:{key:z[10][key].float()for key in ('u','v')}for label,z in ds.items()}
            sb,sa=s['gate_state']['b'],s['gate_state']['mass']
            target_b,target,_,_,nd,ne=wiki_map(mode,10,sb,sa,localmaps,localdirs)
            actual=s['actual_context'];out=actual.clone()
            if mode!='P':out[:,1:]=(actual[:,1:].reshape(511,12,64).transpose(0,1)*(target/sa)[...,None]).transpose(0,1).reshape(1,511,768)
            pairs={'b':(b,sb),'mass':(a,sa),'stable_mixture':(u,s['stable_mixture']),'native_context':(native,actual),
                'target_b':(target_b,s['gate_state']['target_b']),'target_mass':(target,s['gate_state']['target_mass']),'delivered_context':(out,s['delivered_context'])}
            errors={name:float((left-right).abs().max())for name,(left,right)in pairs.items()};assert max(errors.values())<=1e-4,(mode,errors)
            assert ne<=1e-6 and torch.equal(out[:,0],actual[:,0]);snap_errors[mode]=errors
            ro=d['sample_readouts'][mode];assert ro['hidden'].shape==(1,512,768)and ro['logits'].shape==(4,50257)
            assert fingerprint(ro['hidden'])==ro['hidden_fingerprint']and ro['hidden_fingerprint']['sha256']==raw[mode]['final_hidden_sha256']
            assert ro['query_rows'].tolist()==list(READOUT_ROWS)and torch.equal(ro['targets'],x[[q+1 for q in READOUT_ROWS]])
            assert ro['token_sha256']==th(x)
    assert seen=={(c,w,i)for c in corpora for w in range(4)for i in ('natural','bos')}
    assert len(allops)==1548
    for op in allops:
        expected_layers=set()if op['trial']in ('C','repeat','identity')else set(LAYERS)
        assert set(op['gate_checks'])==expected_layers
        assert op['projection_checks']==op['query_checks']==op['context_checks']==8 and op['writer_locality_checks']==30
        assert math.isfinite(op['max_context_error'])and op['max_context_error']<=1e-4
        for ch in op['gate_checks'].values():
            totals['gate_checks']+=1
            assert {k+'_error'for k in ('conditional','native_reconstruction','mixed_reconstruction','sigmoid_context','fullrank_logit','fullrank_mass','fullrank_context','formula')}<=set(ch)
            assert ch['finite']and ch['source_context_exact']and ch.get('unaffected_context_exact',True)and ch['min_mass']>0
            for key,value in ch.items():
                if key.endswith('_error'):
                    assert math.isfinite(value)and value<=(1e-6 if key=='norm_match_error'else 1e-4)
                    extrema[key]=max(extrema.get(key,0.),value)
            for value in ch.get('center_shift_errors',{}).values():assert value<=1e-4
    assert totals['inputs']==32 and totals['records']==776 and totals['suffix_records']==768 and totals['checks']==1128 and totals['snapshot_states']==11 and totals['gate_checks']==11808 and totals['norm_layer_states']==3136
    return {'status':'passed','totals':totals,'fresh_fit_and_geometry_exact':True,'maxima':extrema,'snapshot_cpu_errors':snap_errors,
        'scope':'Raw sampled attention and all per-target main/suffix metrics; other context/logit closures rely on saved runtime checks and hashes. Readout projection is a separate audit. No pretrained replay.',
        'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_sha256':sha(__file__)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args()
    result=audit(a.run);target=a.run/'saved_state_audit.json';assert not target.exists();dump(target,result);print(result['status'],result['totals'])
