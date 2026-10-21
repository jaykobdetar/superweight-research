"""Fabricated saved-state fixtures, not experimental data or model inference.

Exercise every aggregation/count/file path and reject scalar/closure corruption.
The separate independent review tests attention hooks and scientific arithmetic.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import hashlib,json,resource,subprocess,sys,tempfile,shutil
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));torch.set_num_threads(1)
from gpt2_gate_subspace import LAYERS,maps_digest,tensor_hash
from gpt2_gate_localization import CONDITIONS,SNAPSHOT_MODES,CORPUS_COUNTS,INPUT_HASHES,localization_map,mode_spec
from analyze_gate_localization import scalars,paired,nonadditivity
from gpt2_gate_subspace_v2 import relative_prefix,fingerprint
from run_transfer import dump,sha,th

with tempfile.TemporaryDirectory(prefix='gate-analyzer-fixture-') as temp:
    p=Path(temp);(p/'gates').mkdir();(p/'prefix').mkdir();(p/'suffix').mkdir();(p/'local_norms').mkdir()
    reference=root/'gpt2_gate_fixed512_run';ip=root/'localization_inputs_v2'
    for n in ('calibration_raw.pt','calibration_maps.pt','calibration_spectra.json','calibration_metrics.pt','calibration_inputs.json'):
        shutil.copyfile(reference/n,p/n)
    for n in INPUT_HASHES:shutil.copyfile(ip/n,p/n)
    maps=torch.load(p/'calibration_maps.pt',weights_only=True,map_location='cpu')
    cal=torch.load(p/'calibration_raw.pt',weights_only=True,map_location='cpu');cal0=next(iter(cal.values()))
    directions=torch.load(p/'directions.pt',weights_only=True,map_location='cpu')
    dump(p/'conditions.json',list(CONDITIONS))
    tokens=torch.load(p/'tokens.pt',weights_only=True,map_location='cpu');tokenmap={(d['corpus'],d['window']):d['tokens'] for d in tokens}
    bos=json.loads((p/'data_manifest.json').read_text())['literals']['start_special']['id']
    hs={n:torch.ones(1,n,768)for n in (128,512)};hhash={n:tensor_hash(v)for n,v in hs.items()}
    common={n:{}for n in(128,512)};ops_common={n:{}for n in(128,512)};normcommon={}
    for n in (512,):
        for trial in CONDITIONS:
            common[n][trial]={};ops_common[n][trial]={};normcommon[trial]={}
            if trial=='C':continue
            for l in LAYERS:
                b=cal0[l]['b'][:,:n-1].clone();a=b.sigmoid()
                tb,t,fe,fm,nd,ne=localization_map(trial,l,b,a,{k:v.float()for k,v in maps[l].items()}, {k:directions[l][k].float()for k in ('u','v')})
                nd.update(logit_delta_norm=(tb-b).T.double().norm(dim=-1),mass_delta_norm=(t-a).T.double().norm(dim=-1),context_delta_norm=(t-a).T.double().norm(dim=-1))
                normcommon[trial][l]=nd
                common[n][trial][l]={'b':b,'mass':a,'target_b':tb,'target_mass':t}
                ops_common[n][trial][l]={'finite':True,'source_context_exact':True,'unaffected_context_exact':True,'norm_match_error':ne,'active':l in mode_spec(trial)[1],'min_mass':float(a.min()),
                    'min_target_mass':float(t.min()),'max_target_mass':float(t.max()),'target_zeros':int((t==0).sum()),
                    'target_ones':int((t==1).sum()),'max_gain':float((t/a).max()),
                    **{k:0. for k in('conditional_error','native_reconstruction_error','mixed_reconstruction_error',
                     'sigmoid_context_error','fullrank_logit_error','fullrank_mass_error','fullrank_context_error','formula_error')}}
    rows=[];raw={};traces={};checks=[];ops=[];sources={};snaps={};gh={};ph={};pc=[];suffixchecks=[];suffixops=[];suffixfiles={};normfiles={}
    source={'q':{l:torch.zeros(1,12,1,64)for l in LAYERS},
            'kv':{l:{'k':torch.zeros(1,12,1,64),'v':torch.zeros(1,12,1,64)}for l in LAYERS}}
    for c,nw in CORPUS_COUNTS.items():
        for w in range(nw):
            for identity in ('natural','bos'):
                x=tokenmap[c,w].clone()
                if identity=='bos':x[0]=bos
                y=x.clone();y[128:]=torch.roll(y[128:],1)
                trials=list(CONDITIONS)+(['repeat','identity']if identity=='natural'and w==0 else [])
                for n in (512,):
                    stem=f'gate__{c}__{w}__{n}__{identity}';zs={}
                    for trial in trials:
                        canonical='P'if trial in ('P_native','current','fullrank')else'C'if trial in('repeat','identity')else trial
                        shift=CONDITIONS.index(canonical)*.001
                        z={'loss':torch.arange(n-1).float()*.0001+2+shift,'pred':torch.zeros(n-1,dtype=torch.long),
                           'kl':torch.full((n-1,),0. if canonical=='C'else.02),
                           'kl_p':torch.full((n-1,),0. if canonical=='P'else.03),'agreement_p':torch.ones(n-1,dtype=torch.bool),
                           'score_index':1,'token_sha256':th(x[:n]),'final_hidden_sha256':hhash[n]}
                        zs[trial]=z;raw[stem+'__'+trial]=z;traces[stem+'__'+trial]={}
                    gs={}
                    for trial in trials:
                        z=zs[trial];base={'panel':'gate','corpus':c,'identity':identity,'length':n,'window':w,'trial':trial,'exposed_passage':False}
                        rows.append({**base,'key':stem+'__'+trial,'input_tokens':n,'target_start':2,'score_index':1,
                            'scored_tokens':n-2,'token_sha256':th(x[:n]),'q_contribution':None if n==128 else 0.,
                            **scalars(z,zs['C'],zs['P'],slice(1,None))})
                        gates=common[n].get(trial,{});gs[trial]=gates;sources[stem+'__'+trial]=source
                        op={'projection_checks':8,'query_checks':8,'context_checks':8,'writer_locality_checks':30,
                            'gate_checks':ops_common[n].get(trial,{}),'max_context_error':0.,
                            'prefix_context_hashes':{l:{k:fingerprint(torch.ones(1,128,768))for k in('current','delivered')}for l in LAYERS},
                            'mask_records':{l:{'kind':'implicit_causal'}for l in LAYERS}}
                        ops.append({**base,**op})
                        if gates:checks.append({**base,'trial':trial+'_source_identity','source_kv_exact':True,'source_q_exact':True})
                        if trial in('current','fullrank','repeat','identity'):checks.append({**base,'final_hidden_exact':True,'loss_exact':True})
                        if c=='wiki_validation'and w==0 and identity=='natural'and trial in SNAPSHOT_MODES:
                            snaps[stem+'__'+trial]={4:{},11:{}}
                    for trial in CONDITIONS:
                        if len(mode_spec(trial)[1])==1:checks.append({'corpus':c,'identity':identity,'length':n,'window':w,'trial':trial+'_singleton','b_exact':True,'mass_exact':True,'context_exact':True,'reference_context':{'fake':'same'},'trial_context':{'fake':'same'}})
                    file=p/'local_norms'/(stem+'.pt');torch.save({t:normcommon.get(t,{})for t in trials},file);normfiles[file.name]=sha(file)
                    file=p/'gates'/(stem+'.pt');torch.save(gs,file);gh[file.name]=sha(file)
                pairs={trial:{'hidden':hs[128]}for trial in trials}
                file=p/'prefix'/f'gate__{c}__{w}__{identity}.pt';torch.save(pairs,file);ph[file.name]=sha(file)
                suffix={};stem=f'gate__{c}__{w}__512__{identity}'
                for trial in CONDITIONS:
                    z=raw[stem+'__'+trial];originalop=next(o for o in reversed(ops)if o['corpus']==c and o['window']==w and o['identity']==identity and o['length']==512 and o['trial']==trial)
                    suffixops.append(originalop);lf={'sha256':'f'*64,'shape':[128,50257],'dtype':'torch.float32'}
                    suffixchecks.append({'corpus':c,'window':w,'identity':identity,'length':512,'trial':trial,
                        'original_token_sha256':th(x),'control_token_sha256':th(y),'logits_reference':lf,'logits_control':lf,
                        'reference_context_hashes':originalop['prefix_context_hashes'],'control_context_hashes':originalop['prefix_context_hashes'],
                        **{k:True for k in('logits_exact','hidden_exact','gates_exact','source_q_exact','source_kv_exact','mask_exact','contexts_exact',
                                          'loss_exact','pred_exact','kl_exact','kl_p_exact','agreement_p_exact')}})
                    suffix[trial]={'hidden':hs[128],'gates':common[512][trial],'source':source,'token_sha256':th(y),
                        'score_index':1,'scored_tokens':126,'raw':{k:z[k][:(127 if k=='loss'else 128)].clone()for k in('loss','pred','kl','kl_p','agreement_p')}}
                    if c=='wiki_validation'and w==0 and identity=='natural'and trial in SNAPSHOT_MODES:
                        snaps[stem+'__suffix__'+trial]={4:{},11:{}}
                file=p/'suffix'/f'gate__{c}__{w}__{identity}.pt';torch.save(suffix,file);suffixfiles[file.name]=sha(file)

    torch.save(raw,p/'per_token.pt');torch.save(traces,p/'traces.pt');torch.save(sources,p/'source_states.pt');torch.save(snaps,p/'attention_snapshots.pt')
    (p/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows))
    dump(p/'suffix_checks.json',suffixchecks);dump(p/'suffix_operations.json',suffixops);dump(p/'suffix_files.json',suffixfiles)
    dump(p/'norm_files.json',normfiles);dump(p/'checks.json',checks);dump(p/'operation_checks.json',ops);dump(p/'gate_files.json',gh);dump(p/'prefix_files.json',ph);dump(p/'prefix_checks.json',pc)
    dump(p/'calibration_operations.json',[{'trial':'calibration','projection_checks':8,'query_checks':8,'context_checks':8,
        'writer_locality_checks':30,'gate_checks':ops_common[512]['P'],'max_context_error':0.}for _ in range(8)])
    m={'records':938,'evaluation_forwards':938,'forward_passes':1868,'forward_attempts':1868,'calibration_forwards':0,'inherited_calibration_forwards':8,'suffix_forwards':930,'phase':'completed','status':'completed',
       'completed_utc':'fixture2','calibration_frozen_utc':'fixture1','model_hash_before':'fixture','model_hash_after':'fixture',
       'parameter_versions_unchanged':True,'calibration_maps_unchanged':True,'context_tolerance':1e-4,'study':'gate_localization','physical_execution_lengths':[512],'cross_length_comparison':False,'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},
       'passages_per_corpus':CORPUS_COUNTS,'direction_sha256':INPUT_HASHES['directions.pt'],'input_dependencies':INPUT_HASHES,'calibration_maps_tensor_digest':maps_digest(maps),'source_hashes':{n:sha(root/n)for n in('gpt2_gate_localization.py','analyze_gate_localization.py','GPT2_GATE_LOCALIZATION.md','gpt2_gate_subspace_v2.py','analyze_gate_subspace_v2.py','GPT2_GATE_SUBSPACE_V2.md','gpt2_gate_subspace.py','analyze_gate_subspace.py','analyze_transfer.py')}}
    for name,field in(('conditions.json','conditions_sha256'),('calibration_inputs.json','calibration_inputs_sha256'),
                     ('data_manifest.json','evaluation_inputs_sha256'),('calibration_maps.pt','calibration_maps_sha256'),('calibration_raw.pt','calibration_raw_sha256')):m[field]=sha(p/name)
    dump(p/'calibration_dependency.json',{'inherited_calibration_forwards':8,'new_calibration_forwards':0,'map_tensor_digest':maps_digest(maps),
        'copied_files':{n:sha(p/n)for n in('calibration_raw.pt','calibration_maps.pt','calibration_inputs.json','data_manifest.json','tokens.pt')}})
    m['calibration_dependency_sha256']=sha(p/'calibration_dependency.json');dump(p/'manifest.json',m)
    def analyze():return subprocess.run([sys.executable,str(root/'analyze_gate_localization.py'),str(p)],capture_output=True,text=True)
    good=analyze();assert good.returncode==0,good.stderr
    audit=json.loads((p/'audit.json').read_text());assert audit['status']=='passed'
    ps=json.loads((p/'paired_summary.json').read_text());assert len(ps)==216 and all(r['n']==CORPUS_COUNTS[r['corpus']]for r in ps)
    assert audit['summaries']==248 and audit['segment_summaries']==496 and audit['contrasts']==810
    assert audit['nonadditivity_records']==90 and audit['nonadditivity_summaries']==24
    assert audit['norm_audit']['score_aligned_rows']==5040
    ni={('c','natural',512,w):{'P':{'nll':2.},**{f+'_L'+str(l):{'nll':2.+(l-3)*.01}for f in ('remove_pc1','orthogonal_control','position')for l in range(4,12)},**{f+'_all':{'nll':2.5}for f in ('remove_pc1','orthogonal_control','position')}}for w in range(3)}
    nr,ns=nonadditivity(ni);assert len(nr)==9 and len(ns)==3 and all(abs(r['joint_minus_sum_delta_nll']-.14)<1e-12 for r in nr)
    import numpy as np
    xx=np.array([-.01,.02,.13]);rr=np.random.default_rng(20261021);bb=xx[rr.integers(0,3,(2000,3))].mean(1)
    assert paired(xx)['ci95_delta_nll']==np.quantile(bb,[.025,.975]).tolist()
    original=(p/'windows.jsonl').read_text();rows[0]['nll']+=.1
    (p/'windows.jsonl').write_text(''.join(json.dumps(r)+'\n'for r in rows));bad_scalar=analyze();assert bad_scalar.returncode!=0
    assert json.loads((p/'audit.json').read_text())['status']=='checking'
    (p/'windows.jsonl').write_text(original)
    key=next(k for k in raw if k.endswith('__current'));raw[key]['final_hidden_sha256']='a'*64
    torch.save(raw,p/'per_token.pt');bad_closure=analyze();assert bad_closure.returncode!=0
    result={'status':'passed','fabricated_data_only':True,'pretrained_model_loaded':False,'gpu_used':False,
        'complete_fixture_audit':audit,'scalar_corruption_rejected':True,'closure_corruption_rejected':True,'actual_n3_bootstrap_exact':True,'all930_suffix_controls_audited':True,'actual_n3_code_n4_prose':True,'all_three_family_nonadditivity':True,
        'stale_pass_cleared':True,'source_hashes':m['source_hashes'],
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_child_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    assert max(result['peak_rss_kib'],result['peak_child_rss_kib'])<3*1024**2
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
