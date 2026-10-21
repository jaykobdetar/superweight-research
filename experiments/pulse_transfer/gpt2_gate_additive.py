"""Draft additive-logit study. Execution requires frozen inputs and independent readiness."""
import argparse
import contextlib
import json
import math
import resource
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from gpt2_gate_subspace import (ERROR_TOLERANCE, LAYERS, GateRunner,
    add_p_metrics, maps_digest, tensor_hash, mapped_gate)
from gpt2_query_source import QuerySource
from gpt2_layer_writes import LayerWrites
from native_prefix_diagnostic import precision_settings
from run_transfer import ROOT, dump, modelhash, sha, th
from gpt2_gate_subspace_v2 import (GateV2, evaluate_control, fingerprint,
    REFERENCE_RAW, REFERENCE_MAPS, REFERENCE_DIGEST)

from gpt2_gate_localization import GateLocalization, mode_spec, FAMILIES, SETS, CONDITIONS, NORM_TOLERANCE
from gate_additive_readout import AdditiveReadout, loss64
from gate_additive_failure import save_capsule,accumulator_state,singleton_state,suffix_state,logit_pair,CAPSULE_LIMIT

SNAPSHOT_MODES = ('P','remove_pc1_all')
CORPUS_COUNTS = {'wiki_validation':4,'mill':4,'wollstonecraft':4,'optimizers_new':4}
INPUT_HASHES = {'preflight.json': 'd0fa737595f63ff7c48507925a4ddb500013f5301a245878f85ee72beabee66b', 'tokens.pt': 'e009ddfbcee00ec08e013ef2cddc9c745cc2f8f90ca042a2e83d5e2c89120510', 'data_manifest.json': '931f8c698aefa46f2b595121ef9de27764b8e9afdf34a81ff55043279ee16ebf', 'directions.pt': '4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3', 'directions.json': '0a27dcc857b221a9fdbd1265be678aa341d59f914eb3030cff5eb44690ff852e', 'exclusion_inventory.json': '5722677bc774efda1b0b3e7bc8a8d943d3c34a032d33a991854e33bfc3432686', 'historical_manifest_scan.json': '41d8a119d94e875113bc5280fa20703f028488b34799ed6519c2c31cd10d616c', 'start_identity_summary.json': '15c7b0994209c68189c63184fb1049d0990d64c720dcf5662d02cfcfb28c49b4'}
READOUT_ROWS = (0,1,126,127,254,510)
ESTIMATED_OUTPUT_BYTES = 2 * 1024**3
DISK_RESERVE_BYTES = 2 * 1024**3
ANALYSIS_ALLOWANCE_BYTES = 32 * 1024**2
NORMAL_OUTPUT_LIMIT = ESTIMATED_OUTPUT_BYTES-CAPSULE_LIMIT-ANALYSIS_ALLOWANCE_BYTES

def next_group_allowance(corpus,window,identity):
    if window==0 and identity=='natural':return 146_000_000 if corpus=='wiki_validation' else 104_000_000
    return 58_000_000
LOSS64_FP32_SANITY_TOLERANCE = 1e-4

def compact_gates(trial,states,prefix=False):
    affected=mode_spec(trial)[1] if trial in CONDITIONS else ()
    keep=LAYERS if trial=='P' else affected
    return {l:{k:(None if v is None else v[:,:127].clone() if prefix else v.clone())
        for k,v in states[l].items()} for l in keep}

class EvidenceRunner(GateRunner):
    def ev(self,x,baseline=None,keep=False):
        self.p.cache={};self.p.norms={};device=next(self.m.parameters()).device
        with torch.inference_mode():
            result=self.m(x[None].to(device),use_cache=False);self.count+=1
            logits=result.logits[0,:-1].clone();del result
            losses=[];kls=[]
            for a in range(0,len(logits),32):
                b=min(a+32,len(logits));lp=logits[a:b].float().log_softmax(-1)
                losses.append(-lp.gather(-1,x[a+1:b+1,None].to(device)).squeeze(-1).cpu())
                if baseline is None:kls.append(torch.zeros(b-a))
                else:
                    lb=baseline[a:b].float().log_softmax(-1);kls.append((lb.exp()*(lb-lp)).sum(-1).cpu())
            z={'loss':torch.cat(losses),'kl':torch.cat(kls),'pred':logits.argmax(-1).cpu()}
        assert all(torch.isfinite(z[k]).all()for k in ('loss','kl'))
        return z,logits if keep else None,dict(self.p.cache),dict(self.p.norms)
    def rec(self,*args,**kwargs):
        super().rec(*args,**kwargs);self.traces.clear()
    def checkpoint(self):
        torch.save(self.raw,self.out/'per_token.pt');dump(self.out/'checks.json',self.checks)
        dump(self.out/'progress.json',{'forward_passes':self.count,'records':len(self.raw),'utc':time.strftime('%FT%TZ',time.gmtime())})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--calibration-reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    preflight_started=time.time()
    args=ap.parse_args();out=args.out;reference=args.calibration_reference
    assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    for name in('gates','prefix','suffix','hidden','readout_rows','additive'):(out/name).mkdir()
    free=shutil.disk_usage(out).free
    disk={'free_before_bytes':free,'estimated_output_bytes':ESTIMATED_OUTPUT_BYTES,'reserve_bytes':DISK_RESERVE_BYTES,
        'required_bytes':ESTIMATED_OUTPUT_BYTES+DISK_RESERVE_BYTES,'passed':free>=ESTIMATED_OUTPUT_BYTES+DISK_RESERVE_BYTES}
    dump(out/'disk_preflight.json',disk)
    if not disk['passed']:
        dump(out/'manifest.json',{'status':'failed','phase':'preflight','active_failure_phase':'disk_preflight',
            'wall_seconds':time.time()-preflight_started,'failure_type':'InsufficientDisk','failure_message':'Need2GiB estimated output plus2GiB reserve before loading model',
            'disk_preflight':disk,'forward_attempts':0,'forward_passes':0,'records':0,'model_loaded':False,
            'peak_gpu_allocated_bytes':0,'peak_gpu_reserved_bytes':0,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'failed_utc':time.strftime('%FT%TZ',time.gmtime())})
        raise AssertionError(('Insufficient disk before model load',disk))
    model_loaded=False
    try:
        cfg=json.loads(args.config.read_text())
        assert len(INPUT_HASHES)>=5,'Inputs not frozen; execution prohibited'
        assert all(sha(args.inputs/n)==h for n,h in INPUT_HASHES.items())
        preflight=json.loads((args.inputs/'preflight.json').read_text())
        assert preflight['status']=='passed' and preflight['model_forwards']==0 and preflight['passages_per_corpus']==CORPUS_COUNTS
        input_dependencies={n:sha(args.inputs/n) for n in (*INPUT_HASHES,'directions.json','start_identity_summary.json')}
        for n in input_dependencies:shutil.copyfile(args.inputs/n,out/n)
        directions=torch.load(out/'directions.pt',weights_only=True,map_location='cpu')
        refmeta=json.loads((reference/'manifest.json').read_text())
        assert sha(reference/'calibration_raw.pt')==REFERENCE_RAW and sha(reference/'calibration_maps.pt')==REFERENCE_MAPS
        maps=torch.load(reference/'calibration_maps.pt',weights_only=True,map_location='cpu')
        assert maps_digest(maps)==REFERENCE_DIGEST==refmeta['calibration_maps_tensor_digest']
        dependency_names=('calibration_tokens.pt','calibration_inputs.json','calibration_raw.pt','calibration_maps.pt',
            'calibration_spectra.json','calibration_metrics.pt','calibration_operations.json')
        dependencies={n:sha(reference/n)for n in dependency_names}
        for name in dependency_names:shutil.copyfile(reference/name,out/name)
        assert all(sha(out/n)==h for n,h in dependencies.items())
        assert dependencies['calibration_inputs.json']==refmeta['calibration_inputs_sha256']
        caltokens=torch.load(out/'calibration_tokens.pt',weights_only=True,map_location='cpu')
        calinputs=json.loads((out/'calibration_inputs.json').read_text())
        assert len(caltokens)==8 and all(th(i['tokens'])==next(p['window_token_sha256']for p in calinputs['passages']
            if p['corpus']==i['corpus']and p['window']==i['window'])for i in caltokens)
        items=torch.load(out/'tokens.pt',weights_only=True,map_location='cpu');assert len(items)==16
        dm=json.loads((out/'data_manifest.json').read_text());bos=dm['literals']['start_special']['id']
        assert all(len(i['tokens'])==512 and th(i['tokens'])==next(p['window_token_sha256']for p in dm['passages']
            if p['corpus']==i['corpus']and p['window']==i['window'])for i in items)
        assert {c:sum(i['corpus']==c for i in items) for c in CORPUS_COUNTS}==CORPUS_COUNTS
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
        own=('gpt2_gate_additive.py','gate_additive_readout.py','gate_additive_failure.py','analyze_gate_additive.py','GPT2_GATE_ADDITIVE_PROPOSAL.md','GPT2_GATE_ADDITIVE_STORAGE_ADDENDUM.md','gate_additive_inputs.py','gpt2_gate_localization.py','analyze_gate_localization.py','GPT2_GATE_LOCALIZATION.md',
             'gate_localization_inputs.py','GPT2_GATE_LOCALIZATION_PROPOSAL.md','gpt2_gate_fixed512.py','analyze_gate_fixed512.py','GPT2_GATE_FIXED512.md',
             'gpt2_gate_subspace_v2.py','analyze_gate_subspace_v2.py','GPT2_GATE_SUBSPACE_V2.md',
             'native_prefix_diagnostic.py','GPT2_PREFIX_DIAGNOSTIC.md','analyze_gate_subspace.py','analyze_transfer.py',
             'gpt2_gate_subspace.py','gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py',
             'gpt2_writes.py','llama_source_kv.py','run_transfer.py')
        names=tuple(dict.fromkeys(own+tuple(refmeta['source_hashes'])))
        started=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
        dump(out/'conditions.json',list(CONDITIONS))
        settings=precision_settings();assert settings['float32_matmul_precision']=='highest'and not settings['cuda_matmul_allow_tf32']
        meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'phase':'evaluation','status':'running','protocol_version':1,'study':'gate_additive','physical_execution_lengths':[512],
            'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa','seed':20261021,
            'qr_seed':20261025,'layers':list(LAYERS),'families':list(FAMILIES),'layer_sets':list(SETS),'norm_tolerance':NORM_TOLERANCE,'source_hashes':{n:sha(ROOT/n)for n in names},
            'parent_manifest_sha256':sha(parent/'manifest.json'),'calibration_dependency_sha256':sha(out/'calibration_dependency.json'),
            'conditions_sha256':sha(out/'conditions.json'),'calibration_inputs_sha256':dependencies['calibration_inputs.json'],
            'evaluation_inputs_sha256':input_dependencies['data_manifest.json'],'input_dependencies':input_dependencies,'direction_sha256':INPUT_HASHES['directions.pt'],'passages_per_corpus':CORPUS_COUNTS,'calibration_raw_sha256':REFERENCE_RAW,
            'calibration_maps_sha256':REFERENCE_MAPS,'calibration_maps_tensor_digest':REFERENCE_DIGEST,
            'calibration_frozen_utc':refmeta['calibration_frozen_utc'],'inherited_calibration_forwards':8,'calibration_forwards':0,
            'context_tolerance':ERROR_TOLERANCE,'cross_length_comparison':False,
            'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},
            'snapshot_modes':list(SNAPSHOT_MODES),'precision_settings_before':settings,
            'runtime_versions':refmeta['runtime_versions'],'exposed_passages':[],
            'scope':'Fresh verified passages; singleton and joint frozen-direction screen conditional on zero-source-V boundary and fixed calibration.',
            'disk_preflight':disk,'snapshot_states':4,'norm_score_views':{'full':[1,510],'early':[1,126],'later':[127,510]},
            'suffix_rule':'Rotate positions128..511 by one; every512 input and all31 conditions; no changed-target scoring.'}
        import platform
        from importlib.metadata import version
        meta['runtime_versions']={'python':platform.python_version(),'torch':str(torch.__version__),
            'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda}
        frozen=out/'frozen_sources';frozen.mkdir()
        for name,h in meta['source_hashes'].items():
            shutil.copyfile(ROOT/name,frozen/name);assert sha(frozen/name)==h
        dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(cfg['models']['gpt2']['path'],dtype=torch.float32,
            device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.requires_grad_(False)
        model_loaded=True
        before=modelhash(model);assert before==old['model_hash_before'];versions={n:p._version for n,p in model.named_parameters()}
        probe=LayerWrites(model,selected,[447,138]);gate=GateLocalization(model);gate.install_maps(maps,REFERENCE_MAPS);gate.install_directions(directions,INPUT_HASHES['directions.pt']);gate.reset_gate();run=EvidenceRunner(model,probe,out)
        hidden={};ops=[];snapshots={};source_states={};gate_files={};prefix_files={};suffix_files={};norm_files={}
        hidden_files={};readout_files={};additive_files={};input_hidden={};input_readout={};input_additive={};accumulator=None;readout_check={}
        disk_checks=[];suffix_checks=[];suffix_ops=[];active_input={};last_forward={};record_target={};check_target={};record_bundle=None;input_gates={};prefix_references={};input_suffix={};input_norms={};singleton_failure=None
        suffix_failure=None;identity_failure=None;pair_reference=None;pair_original_logits=None;profile_sampled=False;control_computation_failure={};forward_attempts=0
        def capture_hidden(mod,a):hidden['tensor']=a[0].detach().cpu().clone()
        gate.handles.append(model.lm_head.register_forward_pre_hook(capture_hidden))
        meta['readout']={'capture':'Exact tensor entering lm_head after final LayerNorm','weight':fingerprint(model.lm_head.weight),'bias':None if model.lm_head.bias is None else fingerprint(model.lm_head.bias),'query_rows':list(READOUT_ROWS),'hidden_distinct_conditions':29,'output_cap_bytes':ESTIMATED_OUTPUT_BYTES,'normal_output_limit_bytes':NORMAL_OUTPUT_LIMIT,'analysis_allowance_bytes':ANALYSIS_ALLOWANCE_BYTES,'failure_allowance_bytes':CAPSULE_LIMIT}
        meta['readout'].update(weight_tensor_name='lm_head.weight',tied_embedding_name='transformer.wte.weight',
            tied_embedding_storage=model.lm_head.weight.data_ptr()==model.transformer.wte.weight.data_ptr(),
            weight_parameter_version_before=model.lm_head.weight._version)
        assert model.lm_head.bias is None
        dump(out/'manifest.json',meta)
    except BaseException as error:
        setup_meta=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else {}
        setup_meta.update(status='failed',phase='preflight',active_failure_phase='setup',
            failure_type=type(error).__name__,failure_message=str(error),model_loaded=model_loaded,
            forward_attempts=0,forward_passes=0,records=0,wall_seconds=time.time()-preflight_started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            failed_utc=time.strftime('%FT%TZ',time.gmtime()),
            failure_arrays='No model forward attempted; no forward activations exist.')
        dump(out/'manifest.json',setup_meta)
        raise
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
            'prefix_contexts':contexts,'current_contexts':gate.current_contexts,'localization_norms':gate.localization_norms,'masks':gate.mask_records,
            'source':{'kv':gate.cache,'q':{l:d['source_q']for l,d in gate.diag.items()}},
            'operation':{'projection_checks':gate.projection_checks,'query_checks':gate.query_checks,'context_checks':gate.context_checks,
                'writer_locality_checks':probe.locality,'gate_checks':gate.gate_checks,
                'source_fingerprints':{'kv':{l:{k:fingerprint(v)for k,v in d.items()}for l,d in gate.cache.items()},'q':{l:fingerprint(d['source_q'])for l,d in gate.diag.items()}},
                'max_context_error':max(d['context_max_abs_error']for d in gate.diag.values()),
                'prefix_context_hashes':hashes,'mask_records':gate.mask_records,
                'current_context_hashes':{l:fingerprint(v) for l,v in gate.current_contexts.items()},
                'gate_fingerprints':{l:{k:None if v is None else fingerprint(v)for k,v in d.items()}for l,d in gate.gate_cache.items()},
                'context_delta_norm_summary':{l:{view:{'mean':float(d['context_delta_norm'][sl].mean()),'rms':float(d['context_delta_norm'][sl].square().mean().sqrt())}for view,sl in [('full',slice(0,510)),('early',slice(0,126)),('later',slice(126,510))]}for l,d in gate.localization_norms.items()}}}
        return bundle,ll
    def equality(base,name,z,ref,ll,ref_logits):
        nonlocal check_target,identity_failure
        check_target={**base,"trial":name,"stage":"identity_check"}
        check={**base,'trial':name,**{k+'_exact':torch.equal(z[k][1:],ref[k][1:])for k in('loss','kl','pred','kl_p','agreement_p')},
               'final_hidden_exact':z['final_hidden_sha256']==ref['final_hidden_sha256'],
               'logits_exact':torch.equal(ll,ref_logits),'logits_reference':fingerprint(ref_logits),'logits_control':fingerprint(ll)}
        run.checks.append(check)
        if not all(v for k,v in check.items()if k.endswith('_exact')):
            identity_failure={'check':check,'logits':logit_pair(ll,ref_logits)}
            raise AssertionError(('identity closure failed',check))
    def checkpoint():
        run.checkpoint();torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt')
        dump(out/'operation_checks.json',ops);dump(out/'gate_files.json',gate_files);dump(out/'prefix_files.json',prefix_files)
        dump(out/'suffix_files.json',suffix_files)
        dump(out/'hidden_files.json',hidden_files);dump(out/'readout_files.json',readout_files);dump(out/'additive_files.json',additive_files)
        dump(out/'disk_checks.json',disk_checks);dump(out/'suffix_checks.json',suffix_checks);dump(out/'suffix_operations.json',suffix_ops)
    try:
        for item in items:
            for identity in('natural','bos'):
                check_target={'corpus':item['corpus'],'window':item['window'],'identity':identity,'stage':'disk_guard'}
                disk_now={'corpus':item['corpus'],'window':item['window'],'identity':identity,
                    'free_bytes':shutil.disk_usage(out).free,'reserve_bytes':DISK_RESERVE_BYTES,'forward_passes':run.count}
                disk_now['output_bytes']=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
                disk_now['next_group_allowance_bytes']=next_group_allowance(item['corpus'],item['window'],identity)
                disk_now['normal_output_limit_bytes']=NORMAL_OUTPUT_LIMIT
                disk_checks.append(disk_now)
                assert disk_now['output_bytes']+disk_now['next_group_allowance_bytes']<=NORMAL_OUTPUT_LIMIT,('Projected normal-output allowance exceeded',disk_now)
                assert disk_now['free_bytes']-disk_now['next_group_allowance_bytes']>=DISK_RESERVE_BYTES,('Projected disk reserve exhausted before next input',disk_now)
                whole=item['tokens'].clone()
                if identity=='bos':whole[0]=bos
                rotated=whole.clone();rotated[128:]=torch.roll(rotated[128:],1)
                assert torch.equal(rotated[:128],whole[:128])and not torch.equal(rotated[128:],whole[128:])
                prefix_references={};pairkey=f"gate__{item['corpus']}__{item['window']}__{identity}"
                for length in(512,):
                    x=whole[:length].clone();base={'panel':'gate','corpus':item['corpus'],'window':item['window'],'length':length,
                        'identity':identity,'exposed_passage':False}
                    key=f"gate__{item['corpus']}__{item['window']}__{length}__{identity}"
                    input_gates={};input_suffix={};input_norms={};input_hidden={};input_readout={};input_additive={};accumulator=None;refs={};active_input={**base,'phase':'evaluation','trial':'C'}
                    snap=item['corpus']=='wiki_validation'and item['window']==0 and identity=='natural'
                    cb,cl=ev(x,mode='C',zero=False,mask='native',snapshot=False);active_input={**base,'phase':'evaluation','trial':'P'}
                    pb,pl=ev(x,bl=cl,snapshot=snap);add_p_metrics(cb['z'],cl,pl);add_p_metrics(pb['z'],pl,pl);src=pb['source']
                    def record(trial,bundle):
                        nonlocal record_target,check_target,record_bundle,singleton_failure
                        record_target={**base,"trial":trial,"stage":"record"};check_target=dict(record_target);record_bundle=bundle
                        z=bundle['z'];full=key+'__'+trial
                        extra={'p_baseline_nll':float(pb['z']['loss'][1:].double().mean()),
                               'kl_p':float(z['kl_p'][1:].double().mean()),'agreement_p':float(z['agreement_p'][1:].double().mean())}
                        run.rec(full,{**base,'trial':trial,**extra},x,z,cb['z'],bundle['cap'],bundle['norms']);refs[trial]=z
                        ops.append({**base,'trial':trial,'final_hidden_sha256':z['final_hidden_sha256'],**bundle['operation']})
                        if trial=='P':source_states[full]=bundle['source']
                        input_gates[trial]=compact_gates(trial,bundle['gates'])
                        if trial not in('current','fullrank','repeat','identity'):input_hidden[trial]=bundle['hidden'].clone()
                        else:assert torch.equal(bundle['hidden'],pb['hidden']if trial in('current','fullrank')else cb['hidden'])
                        if bundle['snapshots']:snapshots[full]=bundle['snapshots']
                        if bundle['gates']:
                            st=bundle['source'];check={**base,'trial':trial+'_source_identity',
                                'source_kv_exact':all(torch.equal(v,src['kv'][l][k])for l,d in st['kv'].items()for k,v in d.items()),
                                'source_q_exact':all(torch.equal(v,src['q'][l])for l,v in st['q'].items())}
                            run.checks.append(check);assert check['source_kv_exact']and check['source_q_exact']
                        family,affected=mode_spec(trial) if trial in CONDITIONS else (None,())
                        if len(affected)==1:
                            layer=affected[0];state=bundle['gates'][layer];ref=pb['gates'][layer]
                            sc={**base,'trial':trial+'_singleton', 'layer':layer,
                                'b_exact':torch.equal(state['b'],ref['b']),'mass_exact':torch.equal(state['mass'],ref['mass']),
                                'context_exact':torch.equal(bundle['current_contexts'][layer],pb['current_contexts'][layer]),
                                'reference_context':fingerprint(pb['current_contexts'][layer]),
                                'trial_context':fingerprint(bundle['current_contexts'][layer])}
                            if family in ('remove_pc1','orthogonal_control'):
                                sc['s_exact']=torch.equal(bundle['localization_norms'][layer]['s'],pb['localization_norms'][layer]['s'])
                            run.checks.append(sc)
                            if not all(v for k,v in sc.items() if k.endswith('_exact')):
                                singleton_failure={'check':sc,'reference':pb,'trial':bundle}
                                raise AssertionError(('singleton incoming-state mismatch',sc))
                        prefix_references[trial]={'hidden':fingerprint(bundle['hidden'][:,:128])}
                        record_bundle=None
                    def control(trial,original,original_logits):
                        nonlocal active_input,pair_reference,pair_original_logits,suffix_failure,check_target
                        active_input={**base,'phase':'suffix_control','trial':trial};pair_reference=original;pair_original_logits=original_logits[:128]
                        altered,ll=ev(rotated,mode=trial,zero=trial!='C',mask='native'if trial=='C'else'causal',
                            bl=cl,pl=pl,snapshot=False,suffix=True)
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
                        input_suffix[trial]={'hidden':fingerprint(altered['hidden'][:,:128]),'gates':compact_gates(trial,altered['gates'],prefix=True),'source_fingerprints':{'kv':{l:{k:fingerprint(v)for k,v in d.items()}for l,d in altered['source']['kv'].items()},'q':{l:fingerprint(v)for l,v in altered['source']['q'].items()}},
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
                    save_rows=item['window']==0 and identity=='natural'
                    if save_rows:
                        input_readout.update(query_rows=torch.tensor(READOUT_ROWS),targets=x[torch.tensor(READOUT_ROWS)+1].clone(),C=cl[list(READOUT_ROWS)].detach().cpu().clone(),P=pl[list(READOUT_ROWS)].detach().cpu().clone())
                    for trial in CONDITIONS:
                        if trial in('C','P'):continue
                        active_input={**base,'phase':'evaluation','trial':trial}
                        bundle,ll=ev(x,mode=trial,mask='causal',bl=cl,pl=pl,
                            snapshot=snap and trial in SNAPSHOT_MODES)
                        record(trial,bundle);control(trial,bundle,ll)
                        if save_rows:input_readout[trial]=ll[list(READOUT_ROWS)].detach().cpu().clone()
                        family,affected=mode_spec(trial)
                        if family is not None:
                            if len(affected)==1:
                                if affected[0]==4:assert accumulator is None;accumulator=AdditiveReadout(pl,x[1:])
                                accumulator.add_singleton(affected[0],ll)
                            else:
                                readout_check={'corpus':item['corpus'],'window':item['window'],'identity':identity,'family':family}
                                assert accumulator is not None
                                input_additive[family]=accumulator.finish(ll,cl)
                                ad=input_additive[family]
                                pairs=[('P',ad['loss_p']), (trial,ad['loss_joint'])]+[(family+'_L'+str(l),ad['singleton_losses'][i])for i,l in enumerate(LAYERS)]
                                diffs={name:float((value-refs[name]['loss'].double()).abs().max())for name,value in pairs}
                                ad['fp64_vs_fp32_loss_sanity']={'max_absolute_errors':diffs,'threshold':LOSS64_FP32_SANITY_TOLERANCE,'interpretation':'Distinct numerical readouts; conservative sanity guard, not exact equality.'}
                                assert max(diffs.values())<=LOSS64_FP32_SANITY_TOLERANCE,('FP64/FP32 loss sanity failure',readout_check,diffs)
                                input_additive[family]['baseline_logit_fingerprint']=fingerprint(pl)
                                input_additive[family]['joint_logit_fingerprint']=fingerprint(ll)
                                input_additive[family]['additive_logit_fingerprint']=fingerprint(accumulator.additive)
                                if save_rows:input_readout[family+'_additive']=accumulator.additive[list(READOUT_ROWS)].clone()
                                accumulator=None
                        if trial in('current','fullrank'):equality(base,trial+'_closure',bundle['z'],pb['z'],ll,pl)
                        del ll,bundle
                    if identity=='natural'and item['window']==0:
                        for trial,special in(('repeat',None),('identity','identity')):
                            active_input={**base,'phase':'evaluation','trial':trial}
                            bundle,ll=ev(x,mode='C',zero=False,mask='native',bl=cl,pl=pl,special=special)
                            record(trial,bundle);equality(base,trial,bundle['z'],cb['z'],ll,cl);del ll,bundle
                    gp=out/'gates'/(key+'.pt');torch.save(input_gates,gp);gate_files[gp.name]=sha(gp)
                    hp=out/'hidden'/(key+'.pt');torch.save(input_hidden,hp);hidden_files[hp.name]=sha(hp)
                    ap=out/'additive'/(key+'.pt');torch.save(input_additive,ap);additive_files[ap.name]=sha(ap)
                    assert len(input_hidden)==29 and set(input_additive)==set(FAMILIES)
                    if save_rows:
                        rp=out/'readout_rows'/(key+'.pt');torch.save(input_readout,rp);readout_files[rp.name]=sha(rp)
                    sp=out/'suffix'/(pairkey+'.pt');torch.save(input_suffix,sp);suffix_files[sp.name]=sha(sp)
                    del cl,pl,cb,pb;input_gates={};input_suffix={};input_norms={}
                pp=out/'prefix'/(pairkey+'.json');dump(pp,prefix_references);prefix_files[pp.name]=sha(pp)
                prefix_references={};input_hidden={};input_readout={};input_additive={};checkpoint()
                bytes_now=sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
                assert bytes_now<=NORMAL_OUTPUT_LIMIT,('Normal-output allowance exceeded',bytes_now)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=4*1024**2,'Production RSS ceiling exceeded'
                assert shutil.disk_usage(out).free>=DISK_RESERVE_BYTES,'Disk reserve exhausted after checkpoint'
            print('GATE_ADDITIVE_DONE',item['corpus'],item['window'],run.count,flush=True)
        assert all(sha(out/n)==h for n,h in input_dependencies.items())
        assert maps_digest(maps)==REFERENCE_DIGEST and all(sha(out/n)==h for n,h in dependencies.items())
        assert meta['source_hashes']=={n:sha(ROOT/n)for n in names}and precision_settings()==settings
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        assert fingerprint(model.lm_head.weight)==meta['readout']['weight']
        meta['readout']['weight_parameter_version_after']=model.lm_head.weight._version
        assert forward_attempts==run.count==1992 and len(run.raw)==1000 and len(suffix_checks)==992 and len(suffix_ops)==992
        assert len(run.checks)==1800 and len(gate_files)==32 and len(prefix_files)==len(suffix_files)==len(hidden_files)==len(additive_files)==32 and len(readout_files)==4
        meta.update(status='completed',phase='completed',completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,forward_attempts=forward_attempts,
            records=len(run.raw),evaluation_forwards=1000,suffix_forwards=992,calibration_forwards=0,
            model_hash_before=before,model_hash_after=after,parameter_versions_unchanged=True,calibration_maps_unchanged=True,
            precision_settings_after=precision_settings(),wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(out/'manifest.json',meta);print('COMPLETE_GATE_ADDITIVE',run.count,flush=True)
    except BaseException as error:
        checkpoint()
        # Current input shards are part of the preallocated normal evidence budget.
        # Preserve partial completed work once, without duplicating it in the capsule.
        for folder,data,index in (('hidden',input_hidden,hidden_files),('gates',input_gates,gate_files),
                                  ('suffix',input_suffix,suffix_files),('readout_rows',input_readout,readout_files),('additive',input_additive,additive_files)):
            if data:
                name=(pairkey if folder=='suffix'else key)+'.pt';fp=out/folder/name
                torch.save(data,fp);index[name]=sha(fp)
        checkpoint()
        payload={'last_forward_identity':last_forward,'active_record_target':record_target,'active_check_target':check_target,
            'current_tensor_identity':last_forward,'calibration_sha256':REFERENCE_MAPS,'gate':gate.failure_gate,
            'reconstruction':gate.failure_reconstruction,'current_gates':gate.gate_cache,'current_contexts':gate.prefix_contexts,
            'current_source':gate.cache,'current_hidden':hidden.get('tensor'),'current_localization_norms':gate.localization_norms,
            'direction_sha256':INPUT_HASHES['directions.pt'],'readout_check':readout_check,'accumulator':accumulator_state(accumulator),
            'singleton_failure':singleton_state(singleton_failure),'suffix_failure':suffix_state(suffix_failure),'identity_failure':identity_failure,
            'control_computation_metrics':control_computation_failure.get('metrics'),
            'control_computation_logits':logit_pair(control_computation_failure.get('prefix_logits')),
            'pending_reference_logits':logit_pair(pair_original_logits),
            'pending_record_metrics':None if record_bundle is None else record_bundle['z']}
        try:meta['failure_capsule_bytes']=save_capsule(out/'failure_state.pt',payload)
        except BaseException as capture_error:
            dump(out/'failure_capture_error.json',{'type':type(capture_error).__name__,'message':str(capture_error),'original_error':str(error),'partial_shards_preserved':True})
            meta['failure_capture_error']=str(capture_error)
        meta.pop('completed_utc',None);meta.update(status='failed',phase='failed',active_failure_phase=check_target.get('stage'),
            last_forward_identity=last_forward,active_record_target=record_target,active_check_target=check_target,
            failure_type=type(error).__name__,failure_message=str(error),wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,forward_attempts=forward_attempts,records=len(run.raw),suffix_records=len(suffix_checks))
        dump(out/'manifest.json',meta);raise
    finally:gate.close();probe.close()


if __name__=='__main__':main()
