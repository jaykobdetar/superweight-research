"""Fixed-512 gate study; no cross-execution-length comparison or tolerance."""
import argparse
import contextlib
import json
import resource
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from gpt2_gate_subspace import (CONDITIONS, ERROR_TOLERANCE, LAYERS, RANKS,
    SNAPSHOT_MODES, GateRunner, add_p_metrics, maps_digest, tensor_hash)
from gpt2_layer_writes import LayerWrites
from native_prefix_diagnostic import precision_settings
from run_transfer import ROOT, dump, modelhash, sha, th

from gpt2_gate_subspace_v2 import (GateV2, evaluate_control, fingerprint,
    REFERENCE_RAW, REFERENCE_MAPS, REFERENCE_DIGEST)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--calibration-reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;reference=args.calibration_reference
    assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    for name in('gates','prefix','suffix'):(out/name).mkdir()
    refmeta=json.loads((reference/'manifest.json').read_text())
    assert sha(reference/'calibration_raw.pt')==REFERENCE_RAW and sha(reference/'calibration_maps.pt')==REFERENCE_MAPS
    maps=torch.load(reference/'calibration_maps.pt',weights_only=True,map_location='cpu')
    assert maps_digest(maps)==REFERENCE_DIGEST==refmeta['calibration_maps_tensor_digest']
    dependency_names=('calibration_tokens.pt','calibration_inputs.json','calibration_raw.pt','calibration_maps.pt',
        'calibration_spectra.json','calibration_metrics.pt','calibration_operations.json','tokens.pt','data_manifest.json','start_identity_summary.json')
    dependencies={n:sha(reference/n)for n in dependency_names}
    for name in dependency_names:shutil.copyfile(reference/name,out/name)
    assert all(sha(out/n)==h for n,h in dependencies.items())
    assert dependencies['calibration_inputs.json']==refmeta['calibration_inputs_sha256']
    assert dependencies['data_manifest.json']==refmeta['evaluation_inputs_sha256']
    caltokens=torch.load(out/'calibration_tokens.pt',weights_only=True,map_location='cpu')
    calinputs=json.loads((out/'calibration_inputs.json').read_text())
    assert len(caltokens)==8 and all(th(i['tokens'])==next(p['window_token_sha256']for p in calinputs['passages']
        if p['corpus']==i['corpus']and p['window']==i['window'])for i in caltokens)
    items=torch.load(out/'tokens.pt',weights_only=True,map_location='cpu');assert len(items)==16
    dm=json.loads((out/'data_manifest.json').read_text());bos=dm['literals']['start_special']['id']
    assert all(len(i['tokens'])==512 and th(i['tokens'])==next(p['window_token_sha256']for p in dm['passages']
        if p['corpus']==i['corpus']and p['window']==i['window'])for i in items)
    assert all(len(v)==32 for v in dm['old_used_spans'].values())
    for name,h in refmeta['source_hashes'].items():assert sha(ROOT/name)==h,name
    dump(out/'calibration_dependency.json',{'source_manifest_sha256':sha(reference/'manifest.json'),
        'inherited_calibration_forwards':8,'new_calibration_forwards':0,'copied_files':dependencies,
        'map_tensor_digest':REFERENCE_DIGEST,'reference_source_hashes':refmeta['source_hashes'],
        'note':'Exact frozen maps; no new fit or calibration model forward in the fixed-512 study.'})
    parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'selection.json')==old['selection_sha256']
    selected=json.loads((parent/'selection.json').read_text())['selected']
    assert selected['layer']==2 and selected['row']==447
    assert cfg['models']['gpt2']['revision']==old['revision']==refmeta['revision']
    own=('gpt2_gate_fixed512.py','analyze_gate_fixed512.py','GPT2_GATE_FIXED512.md',
         'gpt2_gate_subspace_v2.py','analyze_gate_subspace_v2.py','GPT2_GATE_SUBSPACE_V2.md',
         'native_prefix_diagnostic.py','GPT2_PREFIX_DIAGNOSTIC.md','analyze_gate_subspace.py','analyze_transfer.py',
         'gpt2_gate_subspace.py','gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py',
         'gpt2_writes.py','llama_source_kv.py','run_transfer.py')
    names=tuple(dict.fromkeys(own+tuple(refmeta['source_hashes'])))
    started=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
    dump(out/'conditions.json',list(CONDITIONS))
    settings=precision_settings();assert settings['float32_matmul_precision']=='highest'and not settings['cuda_matmul_allow_tf32']
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'phase':'evaluation','status':'running','protocol_version':1,'study':'gate_fixed512','physical_execution_lengths':[512],
        'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa','seed':20261021,
        'qr_seed':20261025,'layers':list(LAYERS),'ranks':list(RANKS),'source_hashes':{n:sha(ROOT/n)for n in names},
        'parent_manifest_sha256':sha(parent/'manifest.json'),'calibration_dependency_sha256':sha(out/'calibration_dependency.json'),
        'conditions_sha256':sha(out/'conditions.json'),'calibration_inputs_sha256':dependencies['calibration_inputs.json'],
        'evaluation_inputs_sha256':dependencies['data_manifest.json'],'calibration_raw_sha256':REFERENCE_RAW,
        'calibration_maps_sha256':REFERENCE_MAPS,'calibration_maps_tensor_digest':REFERENCE_DIGEST,
        'calibration_frozen_utc':refmeta['calibration_frozen_utc'],'inherited_calibration_forwards':8,'calibration_forwards':0,
        'context_tolerance':ERROR_TOLERANCE,'cross_length_comparison':False,
        'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},
        'snapshot_modes':list(SNAPSHOT_MODES),'precision_settings_before':settings,
        'runtime_versions':refmeta['runtime_versions'],'exposed_passages':[{'corpus':'wiki_validation','window':0}],
        'scope':'New fixed-512 study after two failed cross-length studies; neither failure is a pass.',
        'suffix_rule':'Rotate positions128..511 by one; every512 input and all21 conditions; no changed-target scoring.'}
    import platform
    from importlib.metadata import version
    meta['runtime_versions']={'python':platform.python_version(),'torch':str(torch.__version__),
        'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(cfg['models']['gpt2']['path'],dtype=torch.float32,
        device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.requires_grad_(False)
    before=modelhash(model);assert before==old['model_hash_before'];versions={n:p._version for n,p in model.named_parameters()}
    probe=LayerWrites(model,selected,[447,138]);gate=GateV2(model);gate.install_maps(maps,REFERENCE_MAPS);run=GateRunner(model,probe,out)
    hidden={};ops=[];snapshots={};source_states={};gate_files={};prefix_files={};suffix_files={}
    suffix_checks=[];suffix_ops=[];active_input={};last_forward={};record_target={};check_target={};record_bundle=None;input_gates={};prefix_references={};input_suffix={}
    suffix_failure=None;pair_reference=None;pair_original_logits=None;profile_sampled=False;control_computation_failure={};forward_attempts=0
    def capture_hidden(mod,a,o):hidden['tensor']=o.detach().cpu().clone()
    gate.handles.append(model.transformer.ln_f.register_forward_hook(capture_hidden))
    def ev(x,mode='P',zero=True,mask='causal',bl=None,pl=None,special=None,snapshot=False,suffix=False):
        nonlocal profile_sampled,forward_attempts,last_forward,check_target
        assert len(x)==512
        last_forward=dict(active_input);check_target={**last_forward,"stage":"forward"}
        forward_attempts+=1
        probe.reset_layers(E=False,blocked=[],special=special);gate.reset_gate(mode,zero,mask,snapshot);hidden.clear()
        profiling=not profile_sampled
        if suffix:control_computation_failure.clear()
        ctx=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])if profiling else contextlib.nullcontext()
        with ctx as profiler:
            z,ll,cap,norms=evaluate_control(run,x,bl,pl,control_computation_failure)if suffix else run.ev(x,baseline=bl,keep=True)
        if profiling:
            events=[{'operator':e.key,'count':e.count}for e in profiler.key_averages()if 'scaled_dot_product' in e.key]
            meta['sdpa_profiler_events']=events;meta['sdpa_backend_observed']=any('_scaled_dot_product'in e['operator']for e in events)
            profile_sampled=True
        cap['selected']['writers']=probe.vectors;h=hidden['tensor'];z['final_hidden_sha256']=tensor_hash(h)
        if pl is not None and not suffix:add_p_metrics(z,ll,pl)
        assert probe.locality==30 and probe.writer_zero_checks==0
        assert gate.query_checks==gate.context_checks==gate.projection_checks==8
        assert len(gate.gate_checks)==(8 if zero else 0)
        contexts=gate.prefix_contexts;hashes={l:{k:fingerprint(v)for k,v in d.items()}for l,d in contexts.items()}
        bundle={'z':z,'cap':cap,'norms':norms,'hidden':h,'gates':gate.gate_cache,'snapshots':gate.snapshots,
            'prefix_contexts':contexts,'masks':gate.mask_records,
            'source':{'kv':gate.cache,'q':{l:d['source_q']for l,d in gate.diag.items()}},
            'operation':{'projection_checks':gate.projection_checks,'query_checks':gate.query_checks,'context_checks':gate.context_checks,
                'writer_locality_checks':probe.locality,'gate_checks':gate.gate_checks,
                'max_context_error':max(d['context_max_abs_error']for d in gate.diag.values()),
                'prefix_context_hashes':hashes,'mask_records':gate.mask_records}}
        return bundle,ll
    def equality(base,name,z,ref):
        nonlocal check_target
        check_target={**base,"trial":name,"stage":"identity_check"}
        check={**base,'trial':name,**{k+'_exact':torch.equal(z[k][1:],ref[k][1:])for k in('loss','kl','pred','kl_p','agreement_p')},
               'final_hidden_exact':z['final_hidden_sha256']==ref['final_hidden_sha256']}
        run.checks.append(check);assert all(v for k,v in check.items()if k.endswith('_exact')),check
    def checkpoint():
        run.checkpoint();torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt')
        dump(out/'operation_checks.json',ops);dump(out/'gate_files.json',gate_files);dump(out/'prefix_files.json',prefix_files)
        dump(out/'suffix_files.json',suffix_files)
        dump(out/'suffix_checks.json',suffix_checks);dump(out/'suffix_operations.json',suffix_ops)
    try:
        for item in items:
            for identity in('natural','bos'):
                whole=item['tokens'].clone()
                if identity=='bos':whole[0]=bos
                rotated=whole.clone();rotated[128:]=torch.roll(rotated[128:],1)
                assert torch.equal(rotated[:128],whole[:128])and not torch.equal(rotated[128:],whole[128:])
                prefix_references={};pairkey=f"gate__{item['corpus']}__{item['window']}__{identity}"
                for length in(512,):
                    x=whole[:length].clone();base={'panel':'gate','corpus':item['corpus'],'window':item['window'],'length':length,
                        'identity':identity,'exposed_passage':item['corpus']=='wiki_validation'and item['window']==0}
                    key=f"gate__{item['corpus']}__{item['window']}__{length}__{identity}"
                    input_gates={};input_suffix={};refs={};active_input={**base,'phase':'evaluation','trial':'C'}
                    cb,cl=ev(x,mode='C',zero=False,mask='native');active_input={**base,'phase':'evaluation','trial':'P'}
                    snap=item['corpus']=='wiki_validation'and item['window']==0 and identity=='natural'
                    pb,pl=ev(x,bl=cl,snapshot=snap);add_p_metrics(cb['z'],cl,pl);add_p_metrics(pb['z'],pl,pl);src=pb['source']
                    def record(trial,bundle):
                        nonlocal record_target,check_target,record_bundle
                        record_target={**base,"trial":trial,"stage":"record"};check_target=dict(record_target);record_bundle=bundle
                        z=bundle['z'];full=key+'__'+trial
                        extra={'p_baseline_nll':float(pb['z']['loss'][1:].double().mean()),
                               'kl_p':float(z['kl_p'][1:].double().mean()),'agreement_p':float(z['agreement_p'][1:].double().mean())}
                        run.rec(full,{**base,'trial':trial,**extra},x,z,cb['z'],bundle['cap'],bundle['norms']);refs[trial]=z
                        ops.append({**base,'trial':trial,'final_hidden_sha256':z['final_hidden_sha256'],**bundle['operation']})
                        source_states[full]=bundle['source'];input_gates[trial]=bundle['gates']
                        if bundle['snapshots']:snapshots[full]=bundle['snapshots']
                        if bundle['gates']:
                            st=bundle['source'];check={**base,'trial':trial+'_source_identity',
                                'source_kv_exact':all(torch.equal(v,src['kv'][l][k])for l,d in st['kv'].items()for k,v in d.items()),
                                'source_q_exact':all(torch.equal(v,src['q'][l])for l,v in st['q'].items())}
                            run.checks.append(check);assert check['source_kv_exact']and check['source_q_exact']
                        prefix_references[trial]={'hidden':bundle['hidden'][:,:128].clone()}
                        record_bundle=None
                    def control(trial,original,original_logits):
                        nonlocal active_input,pair_reference,pair_original_logits,suffix_failure,check_target
                        active_input={**base,'phase':'suffix_control','trial':trial};pair_reference=original;pair_original_logits=original_logits[:128]
                        altered,ll=ev(rotated,mode=trial,zero=trial!='C',mask='native'if trial in('C','P_native')else'causal',
                            bl=cl,pl=pl,snapshot=snap and trial in SNAPSHOT_MODES,suffix=True)
                        check_target={**base,'trial':trial,'stage':'suffix_comparison'}
                        a,b=original['z'],altered['z'];st,tt=original['source'],altered['source']
                        gates_exact=all((v is None and altered['gates'][l][k]is None)or(v is not None and
                            torch.equal(v[:,:127],altered['gates'][l][k][:,:127]))for l,d in original['gates'].items()for k,v in d.items())
                        check={**base,'trial':trial,'original_token_sha256':th(x),'control_token_sha256':th(rotated),
                            'logits_reference':fingerprint(original_logits[:128]),'logits_control':fingerprint(ll),
                            'logits_exact':torch.equal(original_logits[:128],ll),'hidden_exact':torch.equal(original['hidden'][:,:128],altered['hidden'][:,:128]),
                            'gates_exact':gates_exact,'source_q_exact':all(torch.equal(v,tt['q'][l])for l,v in st['q'].items()),
                            'source_kv_exact':all(torch.equal(v,tt['kv'][l][k])for l,d in st['kv'].items()for k,v in d.items()),
                            'mask_exact':original['masks']==altered['masks'],
                            'contexts_exact':all(torch.equal(v,altered['prefix_contexts'][l][k])for l,d in original['prefix_contexts'].items()for k,v in d.items()),
                            'reference_context_hashes':original['operation']['prefix_context_hashes'],
                            'control_context_hashes':altered['operation']['prefix_context_hashes'],
                            **{k+'_exact':torch.equal(a[k][1:127],b[k][1:127])for k in('loss','pred','kl','kl_p','agreement_p')}}
                        suffix_checks.append(check);suffix_ops.append({**base,'trial':trial,**altered['operation']})
                        input_suffix[trial]={'hidden':altered['hidden'][:,:128].clone(),'gates':altered['gates'],'source':altered['source'],
                            'token_sha256':th(rotated),'score_index':1,'scored_tokens':126,
                            'raw':{k:v for k,v in b.items()if k in('loss','pred','kl','kl_p','agreement_p')}}
                        if altered['snapshots']:snapshots[key+'__suffix__'+trial]=altered['snapshots']
                        if not all(v for k,v in check.items()if k.endswith('_exact')):
                            suffix_failure={'check':check,'reference':original,'control':altered,
                                'reference_logits':original_logits[:128].detach().cpu().clone(),'control_logits':ll.detach().cpu().clone(),
                                'original_tokens':x,'control_tokens':rotated}
                            raise AssertionError(('fixed-shape suffix control failed',check))
                        pair_reference=None;pair_original_logits=None;del ll,altered
                    record('C',cb);control('C',cb,cl);record('P',pb);control('P',pb,pl)
                    for trial in CONDITIONS:
                        if trial in('C','P'):continue
                        active_input={**base,'phase':'evaluation','trial':trial}
                        bundle,ll=ev(x,mode=trial,mask='native'if trial=='P_native'else'causal',bl=cl,pl=pl,
                            snapshot=snap and trial in SNAPSHOT_MODES)
                        record(trial,bundle);control(trial,bundle,ll)
                        if trial in('current','fullrank'):equality(base,trial+'_closure',bundle['z'],pb['z'])
                        del ll,bundle
                    check_target={**base,'trial':'mask_reference','stage':'mask_check'}
                    run.checks.append({**base,'trial':'mask_reference','max_loss_drift':float((refs['P']['loss'][1:]-refs['P_native']['loss'][1:]).abs().max()),
                        'argmax_agreement':float(refs['P']['pred'][1:].eq(refs['P_native']['pred'][1:]).double().mean())})
                    if identity=='natural'and item['window']==0:
                        for trial,special in(('repeat',None),('identity','identity')):
                            active_input={**base,'phase':'evaluation','trial':trial}
                            bundle,ll=ev(x,mode='C',zero=False,mask='native',bl=cl,pl=pl,special=special)
                            record(trial,bundle);equality(base,trial,bundle['z'],cb['z']);del ll,bundle
                    gp=out/'gates'/(key+'.pt');torch.save(input_gates,gp);gate_files[gp.name]=sha(gp)
                    sp=out/'suffix'/(pairkey+'.pt');torch.save(input_suffix,sp);suffix_files[sp.name]=sha(sp)
                    del cl,pl,cb,pb;input_gates={};input_suffix={}
                pp=out/'prefix'/(pairkey+'.pt');torch.save(prefix_references,pp);prefix_files[pp.name]=sha(pp)
                prefix_references={};checkpoint()
            print('GATE_FIXED512_DONE',item['corpus'],item['window'],run.count,flush=True)
        assert maps_digest(maps)==REFERENCE_DIGEST and all(sha(out/n)==h for n,h in dependencies.items())
        assert meta['source_hashes']=={n:sha(ROOT/n)for n in names}and precision_settings()==settings
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        assert forward_attempts==run.count==1352 and len(run.raw)==680 and len(suffix_checks)==672 and len(suffix_ops)==672
        assert len(run.checks)==744 and len(gate_files)==32 and len(prefix_files)==len(suffix_files)==32
        meta.update(status='completed',phase='completed',completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,forward_attempts=forward_attempts,
            records=len(run.raw),evaluation_forwards=680,suffix_forwards=672,calibration_forwards=0,
            model_hash_before=before,model_hash_after=after,parameter_versions_unchanged=True,calibration_maps_unchanged=True,
            precision_settings_after=precision_settings(),wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(out/'manifest.json',meta);print('COMPLETE_GATE_FIXED512',run.count,flush=True)
    except BaseException as error:
        checkpoint()
        torch.save({'last_forward_identity':last_forward,'active_record_target':record_target,'active_check_target':check_target,
            'pending_record_bundle':record_bundle,'current_tensor_identity':last_forward,'calibration_sha256':REFERENCE_MAPS,'gate':gate.failure_gate,
            'reconstruction':gate.failure_reconstruction,'current_gates':gate.gate_cache,'current_contexts':gate.prefix_contexts,
            'current_snapshots':gate.snapshots,'current_source':gate.cache,'current_input_gates':input_gates,
            'current_input_suffix':input_suffix,'prefix_references':prefix_references,'current_hidden':hidden.get('tensor'),
            'suffix_failure':suffix_failure,'pending_reference':pair_reference,
            'control_computation_failure':control_computation_failure,
            'pending_reference_logits':None if pair_original_logits is None else pair_original_logits.detach().cpu().clone()},out/'failure_state.pt')
        meta.pop('completed_utc',None);meta.update(status='failed',phase='failed',active_failure_phase=check_target.get('stage'),
            last_forward_identity=last_forward,active_record_target=record_target,active_check_target=check_target,
            failure_type=type(error).__name__,failure_message=str(error),wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,forward_attempts=forward_attempts,records=len(run.raw),suffix_records=len(suffix_checks))
        dump(out/'manifest.json',meta);raise
    finally:gate.close();probe.close()


if __name__=='__main__':main()
