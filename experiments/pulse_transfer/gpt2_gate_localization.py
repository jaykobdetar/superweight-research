"""Prospectively frozen singleton/joint gate localization. See protocol before launch."""
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

FAMILIES = ('remove_pc1', 'orthogonal_control', 'position')
SETS = tuple('L'+str(l) for l in LAYERS) + ('all',)
CONDITIONS = ('C', 'P', 'current', 'fullrank') + tuple(f+'_'+s for f in FAMILIES for s in SETS)
SNAPSHOT_MODES = ('C', 'P', 'current', 'fullrank') + tuple(f+'_'+s for f in FAMILIES for s in ('all','L4','L11'))
CORPUS_COUNTS = {'wiki_validation':4,'dubois':4,'hobbes':4,'optimizers':3}
NORM_TOLERANCE = 1e-6
INPUT_HASHES = {
    'preflight.json':'d1500843cd0c12c37e2c1210a0ce047fd526d320668950bef422cd2862f829d5',
    'tokens.pt':'cac494daff1f70b1787430e80a32a35af6e6ac8939d94f8acafe5d423e306a84',
    'data_manifest.json':'387d214aa87c8533f59171ab0e8179ffb4b3adf3ab2494dcacec39a732671fc7',
    'directions.pt':'4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3',
    'exclusion_inventory.json':'cbc58d05d3eb00903140e788be80d500773693e2d3bae82bb8fe6c76c35f509d'}
ESTIMATED_OUTPUT_BYTES = 4 * 1024**3
DISK_RESERVE_BYTES = 3 * 1024**3


def mode_spec(mode):
    if mode in ('C','P','current','fullrank'): return None, ()
    family, suffix = mode.rsplit('_',1)
    assert family in FAMILIES and suffix in SETS, mode
    return family, LAYERS if suffix == 'all' else (int(suffix[1:]),)


def localization_map(mode, layer, b, mass, mapping, direction):
    """Current-state mapping; no pooling across query positions or suffix data."""
    family, affected = mode_spec(mode)
    logq = torch.arange(1,b.shape[1]+1,device=b.device,dtype=b.dtype).log()[:,None]
    centered = b.T-logq-mapping['alpha']
    s = centered @ direction['u']
    du = s[:,None]*direction['u']; dv = s[:,None]*direction['v']
    # Guard real FP32 hypothetical perturbations, diagnose norms in FP64.
    nu, nv = du.double().norm(dim=-1), dv.double().norm(dim=-1)
    norm_error = float(((nu-nv).abs()/torch.maximum(torch.ones_like(nu),torch.maximum(nu,nv))).max())
    if mode == 'fullrank':
        target_b, target, logerr, masserr = mapped_gate(mode,b,mass,mapping)
    elif layer not in affected:
        target_b,target,logerr,masserr = b,mass,0.,0.
    else:
        proposed = b.T-du if family == 'remove_pc1' else b.T-dv if family == 'orthogonal_control' else mapping['alpha']+logq
        target_b,target,logerr,masserr = proposed.T,proposed.sigmoid().T,0.,0.
    return target_b,target,logerr,masserr,{'s':s,'hypothetical_pc1_norm':nu,
        'hypothetical_orthogonal_norm':nv},norm_error


class GateLocalization(GateV2):
    def install_directions(self,directions,digest):
        self.directions=directions;self.direction_sha256=digest;self.device_directions={}
        assert set(directions)==set(LAYERS)
        for d in directions.values():
            u,v=d['u'],d['v']
            assert u.dtype==v.dtype==torch.float64 and u.shape==v.shape==(12,)
            assert torch.isfinite(u).all() and torch.isfinite(v).all()
            assert abs(float(u.norm())-1)<=1e-12 and abs(float(v.norm())-1)<=1e-12 and abs(float(u@v))<=1e-12
            uf,vf=u.float(),v.float()
            assert abs(float(uf.double().norm()-vf.double().norm()))<=NORM_TOLERANCE
            assert abs(float(uf.double()@vf.double()))<=NORM_TOLERANCE

    def reset_gate(self,mode='P',zero=True,mask='causal',snapshot=False):
        assert mode in CONDITIONS
        super().reset_gate('C' if mode=='C' else 'P',zero,mask,snapshot)
        self.mode=mode;self.current_contexts={};self.localization_norms={}

    def context(self, layer, actual):
        raw_inputs = self.reconstruction_inputs[layer]
        QuerySource.context(self, layer, actual)
        self.current_contexts[layer]=actual.detach().cpu().clone()
        self.prefix_contexts[layer]={"current":actual[:,:128].detach().cpu().clone(),"delivered":actual[:,:128].detach().cpu().clone()}
        if not self.zero_values:
            return None
        b, a, u = self.gate_pending.pop(layer)
        n = actual.shape[1]
        c = actual.reshape(1,n,12,64)[0].transpose(0,1)[:,1:]
        mode=self.mode;family,affected=mode_spec(mode);active=layer in affected
        assert self.maps is not None and self.calibration_sha256 and self.direction_sha256
        if layer not in self.device_maps:
            self.device_maps[layer]={k:v.to(device=b.device,dtype=b.dtype) for k,v in self.maps[layer].items()}
            self.device_directions[layer]={k:self.directions[layer][k].to(device=b.device,dtype=b.dtype) for k in ('u','v')}
        proposed_b,target,full_error,full_mass_error,nd,norm_error=localization_map(
            mode,layer,b,a,self.device_maps[layer],self.device_directions[layer])
        good = bool(torch.isfinite(b).all() and torch.isfinite(a).all() and (a>0).all()
                    and torch.isfinite(c).all() and torch.isfinite(u).all() and torch.isfinite(target).all())
        if proposed_b is not None: good = good and bool(torch.isfinite(proposed_b).all())
        if active: good = good and bool((target>=0).all() and (target<=1).all())
        errors = {'conditional_error': float('inf'), 'native_reconstruction_error': float('inf'),
                  'mixed_reconstruction_error': float('inf'), 'sigmoid_context_error': float('inf'),
                  'fullrank_logit_error': full_error, 'fullrank_mass_error': full_mass_error,
                  'fullrank_context_error': 0., 'formula_error': 0.}
        result = actual; gain = float('inf')
        if good:
            ratio = target/a
            gain = float(ratio.max())
            mixed = c*ratio[...,None] if active else c
            errors.update(conditional_error=float((c/a[...,None]-u).abs().max()),
                          native_reconstruction_error=float((c-a[...,None]*u).abs().max()),
                          mixed_reconstruction_error=float((mixed-target[...,None]*u).abs().max()),
                          sigmoid_context_error=float((c-b.sigmoid()[...,None]*u).abs().max()))
            if mode == 'fullrank':
                alternate = c*(proposed_b.sigmoid()/a)[...,None]
                errors['fullrank_context_error'] = float((alternate-c).abs().max())
            if active or mode in ('current','fullrank'):
                result = actual.clone()
                result[:,1:] = mixed.transpose(0,1).reshape(1,n-1,768)
                errors['formula_error'] = float((result[:,1:].reshape(n-1,12,64).transpose(0,1)-mixed).abs().max())
            good = bool(torch.isfinite(mixed).all() and torch.isfinite(ratio).all())
        good = good and math.isfinite(norm_error) and math.isfinite(gain) and all(math.isfinite(v) for v in errors.values())
        check = {'min_mass':float(a.min()), 'max_mass':float(a.max()),
                 'min_b':float(b.min()), 'max_b':float(b.max()),
                 'min_target_mass':float(target.min()), 'max_target_mass':float(target.max()),
                 'min_target_b':None if proposed_b is None else float(proposed_b.min()),
                 'max_target_b':None if proposed_b is None else float(proposed_b.max()),
                 'max_gain':gain, 'target_zeros':int((target==0).sum()),
                 'target_ones':int((target==1).sum()), 'finite':good,
                 'source_context_exact':torch.equal(result[:,0],actual[:,0]),
                 'unaffected_context_exact':active or torch.equal(result,actual),
                 'norm_match_error':norm_error,'active':active, **errors}
        delivered=result.reshape(1,n,12,64)[0].transpose(0,1)[:,1:]
        nd.update(logit_delta_norm=(proposed_b-b).T.double().norm(dim=-1),
                  mass_delta_norm=(target-a).T.double().norm(dim=-1),
                  context_delta_norm=(delivered-c).transpose(0,1).reshape(n-1,768).double().norm(dim=-1))
        self.localization_norms[layer]={k:v.detach().cpu().clone() for k,v in nd.items()}
        self.prefix_contexts[layer]['delivered']=result[:,:128].detach().cpu().clone()
        state = {'b':b.detach().cpu().clone(), 'mass':a.detach().cpu().clone(),
                 'target_b':None if proposed_b is None else proposed_b.detach().cpu().clone(),
                 'target_mass':target.detach().cpu().clone()}
        self.gate_cache[layer] = state; self.gate_checks[layer] = check
        if layer in self.snapshots:
            self.snapshots[layer].update(gate_state=state, stable_mixture=u.detach().cpu().clone(),
                                         delivered_context=result.detach().cpu().clone())
        if not good or max(errors.values())>ERROR_TOLERANCE or norm_error>NORM_TOLERANCE or not check['source_context_exact'] or not check['unaffected_context_exact']:
            self.failure_gate = {'layer':layer,'mode':self.mode,'calibration_sha256':self.calibration_sha256,
                'state':state,'check':check,'norms':self.localization_norms[layer],'direction_sha256':self.direction_sha256,'raw_inputs':{k:None if v is None else v.detach().cpu().clone()
                for k,v in zip(('q','k','v','mask'),raw_inputs)},'stable_mixture':u.detach().cpu().clone(),
                'actual_context':actual.detach().cpu().clone(),'delivered_context':result.detach().cpu().clone()}
            raise AssertionError(('gate diagnostic failed',layer,self.mode,check))
        return (result,) if active or mode in ('current','fullrank') else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--calibration-reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    preflight_started=time.time()
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;reference=args.calibration_reference
    assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True)
    for name in('gates','prefix','suffix','local_norms'):(out/name).mkdir()
    free=shutil.disk_usage(out).free
    disk={'free_before_bytes':free,'estimated_output_bytes':ESTIMATED_OUTPUT_BYTES,'reserve_bytes':DISK_RESERVE_BYTES,
        'required_bytes':ESTIMATED_OUTPUT_BYTES+DISK_RESERVE_BYTES,'passed':free>=ESTIMATED_OUTPUT_BYTES+DISK_RESERVE_BYTES}
    dump(out/'disk_preflight.json',disk)
    if not disk['passed']:
        dump(out/'manifest.json',{'status':'failed','phase':'preflight','active_failure_phase':'disk_preflight',
            'wall_seconds':time.time()-preflight_started,'failure_type':'InsufficientDisk','failure_message':'Need4GiB estimated output plus3GiB reserve before loading model',
            'disk_preflight':disk,'forward_attempts':0,'forward_passes':0,'records':0,'model_loaded':False,
            'peak_gpu_allocated_bytes':0,'peak_gpu_reserved_bytes':0,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'failed_utc':time.strftime('%FT%TZ',time.gmtime())})
        raise AssertionError(('Insufficient disk before model load',disk))
    model_loaded=False
    try:
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
        items=torch.load(out/'tokens.pt',weights_only=True,map_location='cpu');assert len(items)==15
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
        own=('gpt2_gate_localization.py','analyze_gate_localization.py','GPT2_GATE_LOCALIZATION.md',
             'gate_localization_inputs.py','GPT2_GATE_LOCALIZATION_PROPOSAL.md','gpt2_gate_fixed512.py','analyze_gate_fixed512.py','GPT2_GATE_FIXED512.md',
             'gpt2_gate_subspace_v2.py','analyze_gate_subspace_v2.py','GPT2_GATE_SUBSPACE_V2.md',
             'native_prefix_diagnostic.py','GPT2_PREFIX_DIAGNOSTIC.md','analyze_gate_subspace.py','analyze_transfer.py',
             'gpt2_gate_subspace.py','gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py',
             'gpt2_writes.py','llama_source_kv.py','run_transfer.py')
        names=tuple(dict.fromkeys(own+tuple(refmeta['source_hashes'])))
        started=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
        dump(out/'conditions.json',list(CONDITIONS))
        settings=precision_settings();assert settings['float32_matmul_precision']=='highest'and not settings['cuda_matmul_allow_tf32']
        meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'phase':'evaluation','status':'running','protocol_version':1,'study':'gate_localization','physical_execution_lengths':[512],
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
            'disk_preflight':disk,'snapshot_states':52,'norm_score_views':{'full':[1,510],'early':[1,126],'later':[127,510]},
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
        probe=LayerWrites(model,selected,[447,138]);gate=GateLocalization(model);gate.install_maps(maps,REFERENCE_MAPS);gate.install_directions(directions,INPUT_HASHES['directions.pt']);gate.reset_gate();run=GateRunner(model,probe,out)
        hidden={};ops=[];snapshots={};source_states={};gate_files={};prefix_files={};suffix_files={};norm_files={}
        disk_checks=[];suffix_checks=[];suffix_ops=[];active_input={};last_forward={};record_target={};check_target={};record_bundle=None;input_gates={};prefix_references={};input_suffix={};input_norms={};singleton_failure=None
        suffix_failure=None;pair_reference=None;pair_original_logits=None;profile_sampled=False;control_computation_failure={};forward_attempts=0
        def capture_hidden(mod,a,o):hidden['tensor']=o.detach().cpu().clone()
        gate.handles.append(model.transformer.ln_f.register_forward_hook(capture_hidden))
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
                'max_context_error':max(d['context_max_abs_error']for d in gate.diag.values()),
                'prefix_context_hashes':hashes,'mask_records':gate.mask_records,
                'current_context_hashes':{l:fingerprint(v) for l,v in gate.current_contexts.items()}}}
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
        dump(out/'suffix_files.json',suffix_files);dump(out/'norm_files.json',norm_files)
        dump(out/'disk_checks.json',disk_checks);dump(out/'suffix_checks.json',suffix_checks);dump(out/'suffix_operations.json',suffix_ops)
    try:
        for item in items:
            for identity in('natural','bos'):
                check_target={'corpus':item['corpus'],'window':item['window'],'identity':identity,'stage':'disk_guard'}
                disk_now={'corpus':item['corpus'],'window':item['window'],'identity':identity,
                    'free_bytes':shutil.disk_usage(out).free,'reserve_bytes':DISK_RESERVE_BYTES,'forward_passes':run.count}
                disk_checks.append(disk_now)
                assert disk_now['free_bytes']>=DISK_RESERVE_BYTES,('Disk reserve exhausted before next input',disk_now)
                whole=item['tokens'].clone()
                if identity=='bos':whole[0]=bos
                rotated=whole.clone();rotated[128:]=torch.roll(rotated[128:],1)
                assert torch.equal(rotated[:128],whole[:128])and not torch.equal(rotated[128:],whole[128:])
                prefix_references={};pairkey=f"gate__{item['corpus']}__{item['window']}__{identity}"
                for length in(512,):
                    x=whole[:length].clone();base={'panel':'gate','corpus':item['corpus'],'window':item['window'],'length':length,
                        'identity':identity,'exposed_passage':False}
                    key=f"gate__{item['corpus']}__{item['window']}__{length}__{identity}"
                    input_gates={};input_suffix={};input_norms={};refs={};active_input={**base,'phase':'evaluation','trial':'C'}
                    snap=item['corpus']=='wiki_validation'and item['window']==0 and identity=='natural'
                    cb,cl=ev(x,mode='C',zero=False,mask='native',snapshot=snap);active_input={**base,'phase':'evaluation','trial':'P'}
                    pb,pl=ev(x,bl=cl,snapshot=snap);add_p_metrics(cb['z'],cl,pl);add_p_metrics(pb['z'],pl,pl);src=pb['source']
                    def record(trial,bundle):
                        nonlocal record_target,check_target,record_bundle,singleton_failure
                        record_target={**base,"trial":trial,"stage":"record"};check_target=dict(record_target);record_bundle=bundle
                        z=bundle['z'];full=key+'__'+trial
                        extra={'p_baseline_nll':float(pb['z']['loss'][1:].double().mean()),
                               'kl_p':float(z['kl_p'][1:].double().mean()),'agreement_p':float(z['agreement_p'][1:].double().mean())}
                        run.rec(full,{**base,'trial':trial,**extra},x,z,cb['z'],bundle['cap'],bundle['norms']);refs[trial]=z
                        ops.append({**base,'trial':trial,'final_hidden_sha256':z['final_hidden_sha256'],**bundle['operation']})
                        source_states[full]=bundle['source'];input_gates[trial]=bundle['gates'];input_norms[trial]=bundle['localization_norms']
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
                        prefix_references[trial]={'hidden':bundle['hidden'][:,:128].clone()}
                        record_bundle=None
                    def control(trial,original,original_logits):
                        nonlocal active_input,pair_reference,pair_original_logits,suffix_failure,check_target
                        active_input={**base,'phase':'suffix_control','trial':trial};pair_reference=original;pair_original_logits=original_logits[:128]
                        altered,ll=ev(rotated,mode=trial,zero=trial!='C',mask='native'if trial=='C'else'causal',
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
                        bundle,ll=ev(x,mode=trial,mask='causal',bl=cl,pl=pl,
                            snapshot=snap and trial in SNAPSHOT_MODES)
                        record(trial,bundle);control(trial,bundle,ll)
                        if trial in('current','fullrank'):equality(base,trial+'_closure',bundle['z'],pb['z'])
                        del ll,bundle
                    if identity=='natural'and item['window']==0:
                        for trial,special in(('repeat',None),('identity','identity')):
                            active_input={**base,'phase':'evaluation','trial':trial}
                            bundle,ll=ev(x,mode='C',zero=False,mask='native',bl=cl,pl=pl,special=special)
                            record(trial,bundle);equality(base,trial,bundle['z'],cb['z']);del ll,bundle
                    gp=out/'gates'/(key+'.pt');torch.save(input_gates,gp);gate_files[gp.name]=sha(gp)
                    np=out/'local_norms'/(key+'.pt');torch.save(input_norms,np);norm_files[np.name]=sha(np)
                    sp=out/'suffix'/(pairkey+'.pt');torch.save(input_suffix,sp);suffix_files[sp.name]=sha(sp)
                    del cl,pl,cb,pb;input_gates={};input_suffix={};input_norms={}
                pp=out/'prefix'/(pairkey+'.pt');torch.save(prefix_references,pp);prefix_files[pp.name]=sha(pp)
                prefix_references={};checkpoint()
            print('GATE_LOCALIZATION_DONE',item['corpus'],item['window'],run.count,flush=True)
        assert all(sha(out/n)==h for n,h in input_dependencies.items())
        assert maps_digest(maps)==REFERENCE_DIGEST and all(sha(out/n)==h for n,h in dependencies.items())
        assert meta['source_hashes']=={n:sha(ROOT/n)for n in names}and precision_settings()==settings
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        assert forward_attempts==run.count==1868 and len(run.raw)==938 and len(suffix_checks)==930 and len(suffix_ops)==930
        assert len(run.checks)==1688 and len(gate_files)==30 and len(prefix_files)==len(suffix_files)==len(norm_files)==30
        meta.update(status='completed',phase='completed',completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,forward_attempts=forward_attempts,
            records=len(run.raw),evaluation_forwards=938,suffix_forwards=930,calibration_forwards=0,
            model_hash_before=before,model_hash_after=after,parameter_versions_unchanged=True,calibration_maps_unchanged=True,
            precision_settings_after=precision_settings(),wall_seconds=time.time()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dump(out/'manifest.json',meta);print('COMPLETE_GATE_LOCALIZATION',run.count,flush=True)
    except BaseException as error:
        checkpoint()
        torch.save({'last_forward_identity':last_forward,'active_record_target':record_target,'active_check_target':check_target,
            'pending_record_bundle':record_bundle,'current_tensor_identity':last_forward,'calibration_sha256':REFERENCE_MAPS,'gate':gate.failure_gate,
            'reconstruction':gate.failure_reconstruction,'current_gates':gate.gate_cache,'current_contexts':gate.prefix_contexts,
            'current_snapshots':gate.snapshots,'current_source':gate.cache,'current_input_gates':input_gates,
            'current_input_suffix':input_suffix,'prefix_references':prefix_references,'current_hidden':hidden.get('tensor'),
            'current_localization_norms':gate.localization_norms,'current_input_norms':input_norms,'direction_sha256':INPUT_HASHES['directions.pt'],
            'singleton_failure':singleton_failure,'suffix_failure':suffix_failure,'pending_reference':pair_reference,
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
