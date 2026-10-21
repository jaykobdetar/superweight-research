"""Draft T18 Wiki4 factorial runner. Execution is blocked until input/storage freeze.

Fresh calibration is four P forwards. Original W_old4/M8 are derived/reused
from preserved raw data, never rerun. All evaluation uses physical length512.
"""
import argparse
import contextlib
import io
import json
import math
import platform
import resource
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from run_transfer import ROOT, dump, modelhash, sha, th
from gpt2_gate_subspace import GateRunner, LAYERS, ERROR_TOLERANCE, add_p_metrics, maps_digest, tensor_hash
from gpt2_gate_subspace_v2 import GateV2, evaluate_control, fingerprint, REFERENCE_RAW, REFERENCE_MAPS, REFERENCE_DIGEST
from gpt2_layer_writes import LayerWrites
from native_prefix_diagnostic import precision_settings
from fit_gate_calibration import fit_maps_n
from gate_wiki_calibration import GateWiki, CONDITIONS, SNAPSHOT_MODES, READOUT_ROWS, mode_spec, fit_geometry, wiki_map
from gate_wiki_storage import write_npz, SnapshotStore
from gate_additive_failure import logit_pair, singleton_state, suffix_state

INPUT_HASHES = {'calibration_inputs.json': 'b58e0dfc03cc6125cf618c9d66bd38b375db102e418971bf0a48b4fe2a58df04', 'calibration_tokens.pt': 'dcdc07d717bac319889491f637e0c98fd8f4c05dc663fb58aef994a2ca7faae5', 'data_manifest.json': '9dc7298938b81c4f434f95efbb631a81d2854292d07c557dcff531e0ecdcd38f', 'exclusion_inventory.json': '4ac2a6107ea6c7f75b453e79954877a88ccbf9355c01651bf507bf020c49a35e', 'historical_manifest_scan.json': '6b55bdab8a64dd6d0d5789dd2cc76b420ebd394a76b3e914103a0f95f913af0e', 'mixture_directions.json': '0a27dcc857b221a9fdbd1265be678aa341d59f914eb3030cff5eb44690ff852e', 'mixture_directions.pt': '4bf8fc396fd57e2146933131b928ace0a49498b3d295d749b91626a1b70260b3', 'preflight.json': 'e579f67d97e8f8adcbb250857657e956c15861bdd71558df3ff8dbf5e5532c46', 'prior_exclusion_inventory.json': 'ce87628cbc3507e51d785b0eb39ae496886f1166683c283e7f3a92aa5f22f41c', 'prior_historical_manifest_scan.json': 'cf7ec1ca442d3f0aaad4dd89f82da6661fca9b044252099e9bfc1fa76618d7e7', 'prior_olmo_source_mapping.json': '7b5f39d5be210bf4997c684ba01249b711561878385997642f7db341fa787515', 'prior_preflight.json': '0106ad5125408c21dc6d37941afd77f3de80c74ebc13013a18b6f98e1738201a', 'start_identity_summary.json': '9b124da72c1ddf98ebfca33503c388c1ecee55c0f589bc5932ce7a6d2a0e14a2', 'tokens.pt': '78b5b9a3594bd04a0abfe902fadb5c62dae6ef3f760aa67d28c9ebe912a6111b'}
STORAGE_READY = True  # Complete writer and failure measurement passed; independent final approval still required.
NORMAL_LIMIT = 176*1024**2  # Measured normal layout includes16MiB metadata/analysis allowance.
FAILURE_LIMIT = 48*1024**2
DISK_RESERVE = 2*1024**3
GROUP_ALLOWANCES = {'calibration':1*1024**2,'snapshot':58*1024**2,'ordinary':4*1024**2}
CORPUS_COUNTS = {'wiki_validation':4,'utilitarianism':4,'ethics':4,'optimizers_new':4}
OLD_WIKI_DIGEST = 'e59fab6ed96e92d34a3286680b989f8ca12608edee7923baaf088f5d1b40573d'
SNAPSHOT_INPUT = ('wiki_validation',0,'natural')
SOURCE_NAMES = (
 'gpt2_gate_wiki_calibration.py','gate_wiki_calibration.py','gate_wiki_storage.py',
 'fit_gate_calibration.py','gate_calibration_factorial_maps.py','gate_factorial_algebra.py','gate_direction_geometry.py','gate_wiki_calibration_inputs.py',
 'GPT2_GATE_WIKI_CALIBRATION_REVISION.md','gate_calibration_factorial_inputs.py',
 'GPT2_GATE_CALIBRATION_FACTORIAL_PROPOSAL.md','gate_calibration_factorial_history_sources.json',
 'gate_localization_inputs.py','gpt2_gate_localization.py','gpt2_gate_subspace.py',
 'gpt2_gate_subspace_v2.py','gpt2_query_source.py','gpt2_regrowth_kv.py',
 'gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py',
 'native_prefix_diagnostic.py','gate_additive_failure.py','analyze_transfer.py',
 'analyze_gate_wiki_calibration.py','audit_gate_wiki_results.py','gate_inputs.py','fresh_inputs.py','fresh_inputs_v2.py')

def fp_nested(value):
    if isinstance(value,torch.Tensor):return fingerprint(value)
    if isinstance(value,dict):return {k:fp_nested(v)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [fp_nested(v)for v in value]
    return value

def prefix_gates_equal(a,b):
    return set(a)==set(b)and all(set(d)==set(b[l])and all(
        (v is None and b[l][k]is None)or(v is not None and b[l][k]is not None and torch.equal(v[:,:127],b[l][k][:,:127]))
        for k,v in d.items())for l,d in a.items())

def source_equal(a,b):
    return (set(a['q'])==set(b['q'])and set(a['kv'])==set(b['kv'])
        and all(torch.equal(v,b['q'][l])for l,v in a['q'].items())
        and all(set(d)==set(b['kv'][l])and all(torch.equal(v,b['kv'][l][k])for k,v in d.items())for l,d in a['kv'].items()))

def selected_norms(trial,norms):
    affected=mode_spec(trial)[3]if trial in CONDITIONS else ()
    keep=LAYERS if trial=='P'else affected
    return {l:norms[l]for l in keep}

def pack_input_payload(base,key,raw,checks,operations,suffix_checks,suffix_operations,norms,source,snapshots,readouts,suffix_metrics):
    """Single immutable input shard; shared by runner and complete-size fixture."""
    return {'metadata':base,'per_token':{k:v for k,v in raw.items()if k.startswith(key+'__')},
        'checks':checks,'operations':operations,'suffix_checks':suffix_checks,'suffix_operations':suffix_operations,
        'norms':{k.removeprefix(key+'__'):v for k,v in norms.items()if k.startswith(key+'__')},
        'source_reference':source,'snapshots':snapshots,'sample_readouts':readouts,
        'suffix_metrics':{k.removeprefix(key+'__'):v for k,v in suffix_metrics.items()if k.startswith(key+'__')}}

class NumericRunner(GateRunner):
    """Count completed forwards before scoring; retain no unrelated probe traces."""
    def ev(self,x,baseline=None,keep=False):
        self.p.cache={};self.p.norms={};device=next(self.m.parameters()).device
        with torch.inference_mode():
            result=self.m(x[None].to(device),use_cache=False);self.count+=1
            logits=result.logits[0,:-1].clone();self.latest_logits=logits;del result
            losses=[];kls=[]
            for start in range(0,len(logits),32):
                end=min(start+32,len(logits));lp=logits[start:end].float().log_softmax(-1)
                losses.append(-lp.gather(-1,x[start+1:end+1,None].to(device)).squeeze(-1).cpu())
                if baseline is None:kls.append(torch.zeros(end-start))
                else:
                    lb=baseline[start:end].float().log_softmax(-1);kls.append((lb.exp()*(lb-lp)).sum(-1).cpu())
            z={'loss':torch.cat(losses),'kl':torch.cat(kls),'pred':logits.argmax(-1).cpu()}
        assert z['loss'].shape==(511,)and all(torch.isfinite(z[k]).all()for k in('loss','kl'))
        return z,logits if keep else None,dict(self.p.cache),dict(self.p.norms)
    def rec(self,*args,**kwargs):
        super().rec(*args,**kwargs);self.traces.clear()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--calibration-reference',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out
    assert not out.exists()or not any(out.iterdir()),'Preserve prior outputs; use a new directory'
    out.mkdir(parents=True,exist_ok=True);started=time.time()
    (out/'inputs').mkdir();(out/'calibration').mkdir()
    meta={'status':'preparing','phase':'setup','study':'gate_wiki_calibration','model_loaded':False,
        'forward_attempts':0,'forward_passes':0,'records':0,'calibration_forwards':0,'evaluation_forwards':0,'suffix_forwards':0,
        'started_utc':time.strftime('%FT%TZ',time.gmtime()),'normal_limit_bytes':NORMAL_LIMIT,
        'failure_limit_bytes':FAILURE_LIMIT,'disk_reserve_bytes':DISK_RESERVE}
    run=None;probe=None;gate=None;model=None;hidden={};attempts=0;last_forward={};last_completed={};current_attempt_tokens=None;record_target={};check_target={}
    calraw={};calmetrics={};calops=[];ops=[];suffix_ops=[];suffix_checks=[];norms_raw={};sample_readouts={}
    source_references={};suffix_numeric={};disk_checks=[];snapshot_store=SnapshotStore();profiled=False
    bundle=None;cb=None;pb=None;pair_reference=None;pair_logits=None;control_failure={}
    singleton_failure=None;suffix_failure=None;identity_failure=None;partial_record=None;pending_logits=None
    maps=None;dirs=None;spectra=None;frozen_maps=None;active_input={};model_loaded=False
    source_hashes={};input_hashes={};output_inventory={};gate_failure=None
    group_marks=None;group_key=None;group_base=None;group_committed=True;group_snapshot=False
    dump(out/'manifest.json',meta)

    def bytes_now():return sum(p.stat().st_size for p in out.rglob('*')if p.is_file())
    def resources():
        return {'wall_seconds':time.time()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()if torch.cuda.is_initialized()else 0,
            'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved()if torch.cuda.is_initialized()else 0}
    def disk_guard(stage,projected):
        nonlocal check_target
        check_target={**active_input,'stage':stage}
        check={'stage':stage,**active_input,'free_bytes':shutil.disk_usage(out).free,'output_bytes':bytes_now(),
            'projected_growth_bytes':projected,'normal_limit_bytes':NORMAL_LIMIT,'failure_allowance_bytes':FAILURE_LIMIT,'reserve_bytes':DISK_RESERVE,
            'forward_attempts':attempts,'forward_passes':0 if run is None else run.count}
        disk_checks.append(check)
        assert check['output_bytes']+projected<=NORMAL_LIMIT,('Normal output allowance',check)
        assert check['free_bytes']-projected>=DISK_RESERVE+FAILURE_LIMIT,('Preserve reserve and failure allowance',check)
    def save(name,obj):
        remaining=min(NORMAL_LIMIT-bytes_now(),shutil.disk_usage(out).free-DISK_RESERVE-FAILURE_LIMIT)
        assert remaining>0,('No normal storage headroom',name,remaining)
        write_npz(out/name,obj,limit_bytes=remaining)
        output_inventory[name]={'sha256':sha(out/name),'bytes':(out/name).stat().st_size}
    def checkpoint(preserving_failure=False):
        nonlocal check_target
        if not preserving_failure:check_target={**active_input,'stage':'checkpoint','checkpoint_phase':meta['phase']}
        dump(out/'disk_checks.json',disk_checks);dump(out/'numeric_inventory.json',output_inventory)
        dump(out/'progress.json',{'forward_attempts':attempts,'forward_passes':0 if run is None else run.count,
            'records':0 if run is None else len(run.raw),'suffix_records':len(suffix_checks),'phase':meta['phase'],'utc':time.strftime('%FT%TZ',time.gmtime())})
        assert bytes_now()<=NORMAL_LIMIT and shutil.disk_usage(out).free>=DISK_RESERVE+FAILURE_LIMIT
    def group_payload():
        assert group_marks is not None and run is not None
        return pack_input_payload(group_base,group_key,run.raw,run.checks[group_marks['checks']:],
            ops[group_marks['operations']:],suffix_checks[group_marks['suffix_checks']:],
            suffix_ops[group_marks['suffix_operations']:],norms_raw,source_references.get(group_key),
            snapshot_store.payload if group_snapshot else None,sample_readouts if group_snapshot else {},suffix_numeric)
    def capture_hidden(module,inputs):hidden['tensor']=inputs[0].detach().cpu().clone()
    def install_hidden_hook():gate.handles.append(model.lm_head.register_forward_pre_hook(capture_hidden))
    def ev(x,mode='P',zero=True,mask='causal',bl=None,pl=None,special=None,snapshot=False,suffix=False):
        nonlocal attempts,last_forward,last_completed,current_attempt_tokens,check_target,profiled,pending_logits
        assert len(x)==512
        last_forward=dict(active_input);current_attempt_tokens=x.detach().cpu().clone()
        check_target={**last_forward,'stage':'forward'};attempts+=1
        probe.reset_layers(E=False,blocked=[],special=special);gate.reset_gate(mode,zero,mask,snapshot);hidden.clear();pending_logits=None
        if suffix:control_failure.clear()
        context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])if not profiled else contextlib.nullcontext()
        count_before=run.count;run.latest_logits=None
        try:
            with context as profiler:
                z,ll,cap,pnorm=evaluate_control(run,x,bl,pl,control_failure)if suffix else run.ev(x,baseline=bl,keep=True)
        finally:
            counter='suffix_forwards'if suffix else'calibration_forwards'if active_input['phase']=='calibration'else'evaluation_forwards'
            completed=run.count-count_before;assert completed in(0,1)
            meta[counter]+=completed
            if completed:last_completed=dict(last_forward)
        pending_logits=ll
        if not profiled:
            events=[{'operator':e.key,'count':e.count}for e in profiler.key_averages()if 'scaled_dot_product'in e.key]
            meta['sdpa_profiler_events']=events;meta['sdpa_backend_observed']=any('_scaled_dot_product'in e['operator']for e in events);profiled=True
        h=hidden['tensor'];z['final_hidden_sha256']=tensor_hash(h)
        if pl is not None and not suffix:add_p_metrics(z,ll,pl)
        assert probe.locality==30 and probe.writer_zero_checks==0
        assert gate.query_checks==gate.projection_checks==gate.context_checks==8 and len(gate.gate_checks)==(8 if zero else 0)
        cap['selected']['writers']=probe.vectors
        state={'kv':gate.cache,'q':{l:d['source_q']for l,d in gate.diag.items()}}
        contexts=gate.prefix_contexts;current=getattr(gate,'current_contexts',{})
        norm=getattr(gate,'localization_norms',{})
        operation={'projection_checks':gate.projection_checks,'query_checks':gate.query_checks,'context_checks':gate.context_checks,
            'writer_locality_checks':probe.locality,'gate_checks':gate.gate_checks,'mask_records':gate.mask_records,
            'max_context_error':max(d['context_max_abs_error']for d in gate.diag.values()),
            'source_fingerprints':fp_nested(state),'prefix_context_hashes':fp_nested(contexts),
            'current_context_hashes':fp_nested(current),'gate_fingerprints':fp_nested(gate.gate_cache),
            'norm_fingerprints':fp_nested(norm)}
        result={'z':z,'cap':cap,'norms':pnorm,'hidden':h,'gates':gate.gate_cache,'snapshots':gate.snapshots,
            'prefix_contexts':contexts,'current_contexts':current,'localization_norms':norm,'masks':gate.mask_records,'source':state,'operation':operation}
        return result,ll
    def equality(base,name,left,right,ll,rr):
        nonlocal check_target,identity_failure
        check_target={**base,'trial':name,'stage':'identity_check'}
        check={**base,'trial':name,**{k+'_exact':torch.equal(left['z'][k][1:],right['z'][k][1:])for k in('loss','kl','pred','kl_p','agreement_p')},
            'hidden_exact':torch.equal(left['hidden'],right['hidden']),'logits_exact':torch.equal(ll,rr),
            'logits_reference':fingerprint(rr),'logits_control':fingerprint(ll)}
        run.checks.append(check)
        if not all(v for k,v in check.items()if k.endswith('_exact')):
            identity_failure={'check':check,'left_hidden':left['hidden'],'right_hidden':right['hidden'],'logits':logit_pair(ll,rr)}
            raise AssertionError(('identity closure failed',check))
    try:
        assert INPUT_HASHES and STORAGE_READY and GROUP_ALLOWANCES,'Execution prohibited until reviewed inputs and measured storage are frozen'
        cfg=json.loads(args.config.read_text());torch.set_num_threads(4);torch.manual_seed(20261021)
        assert len(CONDITIONS)==24 and len(SNAPSHOT_MODES)==11
        disk_guard('pre_model_complete_projection',max(0,NORMAL_LIMIT-bytes_now()))
        assert all(sha(args.inputs/n)==h for n,h in INPUT_HASHES.items())
        pre=json.loads((args.inputs/'preflight.json').read_text())
        assert pre['status']=='passed'and pre['model_forwards']==0 and pre['calibration_passages']==4 and pre['evaluation_passages']==16
        assert pre['evaluation_passages_per_corpus']==CORPUS_COUNTS
        input_hashes={n:sha(args.inputs/n)for n in INPUT_HASHES}
        for name in input_hashes:shutil.copyfile(args.inputs/name,out/name)
        inputs=torch.load(args.inputs/'tokens.pt',weights_only=True,map_location='cpu')
        calibration=torch.load(args.inputs/'calibration_tokens.pt',weights_only=True,map_location='cpu')
        dm=json.loads((args.inputs/'data_manifest.json').read_text());cm=json.loads((args.inputs/'calibration_inputs.json').read_text())
        assert len(inputs)==16 and len(calibration)==4
        for xs,records in((inputs,dm['passages']),(calibration,cm['passages'])):
            assert len(records)==len(xs)
            for item in xs:
                rec=next(p for p in records if p['corpus']==item['corpus']and p['window']==item['window'])
                assert item['tokens'].shape==(512,)and item['tokens'].dtype==torch.long and th(item['tokens'])==rec['window_token_sha256']
        assert {c:sum(i['corpus']==c for i in inputs)for c in CORPUS_COUNTS}==CORPUS_COUNTS
        assert [(i['corpus'],i['window'])for i in calibration]==[('wiki_test',i)for i in range(4)]
        bos=dm['literals']['start_special']['id'];ref=args.calibration_reference
        assert sha(ref/'calibration_raw.pt')==REFERENCE_RAW and sha(ref/'calibration_maps.pt')==REFERENCE_MAPS
        raw_old=torch.load(ref/'calibration_raw.pt',weights_only=True,map_location='cpu')
        mixture=torch.load(ref/'calibration_maps.pt',weights_only=True,map_location='cpu');assert maps_digest(mixture)==REFERENCE_DIGEST
        assert sha(ref/'calibration_spectra.json')=='2910c3de477e1d137fb35dbb7f7a55fec583e795cb6c243332372b0c69bba138'
        mixture_spectra=json.loads((ref/'calibration_spectra.json').read_text());mixture_spectra={int(k):v for k,v in mixture_spectra.items()}
        old_subset={f'wiki_test__{i}':raw_old[f'wiki_test__{i}']for i in range(4)}
        check_target={'stage':'old_wiki_fit','phase':'setup','model_forwards':0}
        old_wiki,old_spectra=fit_maps_n(old_subset,4);assert maps_digest(old_wiki)==OLD_WIKI_DIGEST
        del raw_old,old_subset
        parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256']
        selected=json.loads((parent/'selection.json').read_text())['selected'];assert selected['layer']==2 and selected['row']==447
        assert cfg['models']['gpt2']['revision']==old['revision']
        refmeta=json.loads((ref/'manifest.json').read_text());assert refmeta['revision']==old['revision']
        source_hashes={n:sha(ROOT/n)for n in SOURCE_NAMES}
        assert sha(ROOT/'fit_gate_calibration.py')=='9cc6ea695f6c3753f8878a5d5146fa9a371be264685c2e26d8f831cd2b147c7e'
        frozen=out/'frozen_sources';frozen.mkdir()
        for name,h in source_hashes.items():shutil.copyfile(ROOT/name,frozen/name);assert sha(frozen/name)==h
        torch.set_num_threads(4);torch.manual_seed(20261021);settings=precision_settings()
        assert settings['float32_matmul_precision']=='highest'and not settings['cuda_matmul_allow_tf32']
        meta.update(model_id=old['model_id'],revision=old['revision'],dtype='float32',attention='sdpa',seed=20261021,qr_seed=20261025,
            source_hashes=source_hashes,input_dependencies=input_hashes,parent_manifest_sha256=sha(parent/'manifest.json'),
            reference_manifest_sha256=sha(ref/'manifest.json'),reference_raw_sha256=REFERENCE_RAW,reference_maps_sha256=REFERENCE_MAPS,
            reference_map_digest=REFERENCE_DIGEST,old_wiki_map_digest=OLD_WIKI_DIGEST,
            reference_spectra_sha256=sha(ref/'calibration_spectra.json'),inherited_calibration_forwards=8,
            physical_execution_lengths=[512],passages_per_corpus=CORPUS_COUNTS,conditions=list(CONDITIONS),
            snapshot_input=list(SNAPSHOT_INPUT),snapshot_modes=list(SNAPSHOT_MODES),snapshot_layer=10,readout_rows=list(READOUT_ROWS),
            precision_settings_before=settings,context_tolerance=ERROR_TOLERANCE,
            calibration_scope={'O':'original Wiki4 from preserved raw','F':'fresh Wiki4 exact retained passages','M':'frozen original Wiki/Hume8'},
            calibration_metrics_reference='self/no-native-C',score_views={'full':[2,511],'early':[2,127],'later':[128,511]},
            runtime_versions={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda})
        dump(out/'conditions.json',list(CONDITIONS));dump(out/'manifest.json',meta)
        torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(cfg['models']['gpt2']['path'],dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
        model.requires_grad_(False);model_loaded=True;meta['model_loaded']=True
        before=modelhash(model);assert before==old['model_hash_before'];versions={n:p._version for n,p in model.named_parameters()}
        meta['model_hash_before']=before;meta['model_hash_before_verified']=True
        dump(out/'manifest.json',meta)
        meta['readout']={'capture':'exact lm_head input after final LayerNorm','weight':fingerprint(model.lm_head.weight),
            'bias':None if model.lm_head.bias is None else fingerprint(model.lm_head.bias),'weight_tensor_name':'lm_head.weight',
            'tied_embedding_storage':model.lm_head.weight.data_ptr()==model.transformer.wte.weight.data_ptr()}
        probe=LayerWrites(model,selected,[447,138]);run=NumericRunner(model,probe,out)
        gate=GateV2(model);install_hidden_hook();meta['phase']='calibration';meta['status']='running';dump(out/'manifest.json',meta)
        for item in calibration:
            active_input={'phase':'calibration','corpus':item['corpus'],'window':item['window'],'identity':'natural','trial':'calibration'}
            disk_guard('next_calibration',GROUP_ALLOWANCES['calibration'])
            bundle,ll=ev(item['tokens'],mode='calibration');key=f"{item['corpus']}__{item['window']}"
            record_target={**active_input,'stage':'calibration_record'};check_target=dict(record_target)
            calraw[key]={l:{k:d[k]for k in('b','mass')}for l,d in bundle['gates'].items()}
            calmetrics[key]={'token_sha256':th(item['tokens']),'kl_reference':'self/no-native-C',**bundle['z']}
            calops.append({**active_input,**bundle['operation']})
            check_target={**active_input,'stage':'calibration_save'}
            save('calibration/'+key+'.npz',{'raw':calraw[key],'metrics':calmetrics[key],'operation':calops[-1]})
            del ll;bundle=None;pending_logits=None;checkpoint()
        gate.close();gate=None;hidden.clear();meta['phase']='fit_before_evaluation'
        active_input={'phase':'fit_before_evaluation','trial':'W_old4_W_fresh4_M8'}
        record_target={};check_target={**active_input,'stage':'fresh_wiki_fit'}
        fresh,fresh_spectra=fit_maps_n(calraw,4);maps={'O':old_wiki,'F':fresh,'M':mixture};spectra={'O':old_spectra,'F':fresh_spectra,'M':mixture_spectra}
        check_target={**active_input,'stage':'calibration_geometry'}
        dirs,geometry=fit_geometry(maps,spectra)
        check_target={**active_input,'stage':'calibration_map_save'}
        frozen_maps={k:maps_digest(v)for k,v in maps.items()};save('calibration_maps.npz',maps);save('calibration_directions.npz',dirs)
        save('calibration_spectra.npz',spectra);save('calibration_geometry.npz',geometry)
        set_hash=sha(out/'calibration_maps.npz');meta.update(calibration_frozen_utc=time.strftime('%FT%TZ',time.gmtime()),
            calibration_map_digests=frozen_maps,calibration_maps_sha256=set_hash,calibration_directions_sha256=sha(out/'calibration_directions.npz'))
        meta['phase']='evaluation';dump(out/'manifest.json',meta)
        gate=GateWiki(model);gate.install_sets(maps,dirs,set_hash);install_hidden_hook()
        for item in inputs:
            for identity in('natural','bos'):
                active_input={'corpus':item['corpus'],'window':item['window'],'identity':identity,'phase':'evaluation'}
                is_snapshot=(item['corpus'],item['window'],identity)==SNAPSHOT_INPUT
                disk_guard('next_input',GROUP_ALLOWANCES['snapshot'if is_snapshot else'ordinary'])
                x=item['tokens'].clone()
                if identity=='bos':x[0]=bos
                rotated=x.clone();rotated[128:]=torch.roll(rotated[128:],1)
                assert torch.equal(x[:128],rotated[:128])and not torch.equal(x[128:],rotated[128:])
                base={'panel':'gate','corpus':item['corpus'],'window':item['window'],'identity':identity,'length':512,'exposed_passage':False}
                key=f"gate__{item['corpus']}__{item['window']}__512__{identity}";snap=(item['corpus'],item['window'],identity)==SNAPSHOT_INPUT
                group_key=key;group_base=base;group_snapshot=snap;group_committed=False
                group_marks={'checks':len(run.checks),'operations':len(ops),'suffix_checks':len(suffix_checks),'suffix_operations':len(suffix_ops)}
                active_input={**base,'phase':'evaluation','trial':'C'};cb,cl=ev(x,mode='C',zero=False,mask='native')
                active_input={**base,'phase':'evaluation','trial':'P'};pb,pl=ev(x,bl=cl,snapshot=snap)
                add_p_metrics(cb['z'],cl,pl);add_p_metrics(pb['z'],pl,pl);source_references[key]=pb['source']
                def record(trial,data,ll):
                    nonlocal record_target,check_target,partial_record,singleton_failure
                    record_target={**base,'trial':trial,'stage':'record'};check_target=dict(record_target);partial_record=data
                    z=data['z'];full=key+'__'+trial
                    run.rec(full,{**base,'trial':trial,'p_baseline_nll':float(pb['z']['loss'][1:].double().mean()),
                        'kl_p':float(z['kl_p'][1:].double().mean()),'agreement_p':float(z['agreement_p'][1:].double().mean())},x,z,cb['z'],data['cap'],data['norms'])
                    ops.append({**base,'trial':trial,'final_hidden_sha256':z['final_hidden_sha256'],**data['operation']})
                    if data['gates']:
                        check={**base,'trial':trial+'_source_identity','source_exact':source_equal(data['source'],pb['source']),
                            'reference_fingerprints':fp_nested(pb['source']),'trial_fingerprints':fp_nested(data['source'])}
                        run.checks.append(check);assert check['source_exact'],check
                    affected=mode_spec(trial)[3]if trial in CONDITIONS else ()
                    if len(affected)==1:
                        layer=affected[0];state=data['gates'][layer];ref=pb['gates'][layer]
                        # Same incoming state implies same coefficient under this
                        # arm's own center/basis, not the P mode's empty coefficient.
                        device=next(model.parameters()).device
                        expected=wiki_map(trial,layer,ref['b'].to(device),ref['mass'].to(device),
                            gate.map_device[layer],gate.direction_device[layer])[4]['s'].detach().cpu()
                        actual=data['localization_norms'][layer]['s']
                        check={**base,'trial':trial+'_singleton','layer':layer,
                            'b_exact':torch.equal(state['b'],ref['b']),'mass_exact':torch.equal(state['mass'],ref['mass']),
                            'context_exact':torch.equal(data['current_contexts'][layer],pb['current_contexts'][layer]),
                            'coefficient_exact':torch.equal(expected,actual),
                            'coefficient_max_abs_error':float((expected-actual).abs().max()),
                            'reference_context':fingerprint(pb['current_contexts'][layer]),'trial_context':fingerprint(data['current_contexts'][layer])}
                        run.checks.append(check)
                        if not all(v for k,v in check.items()if k.endswith('_exact')or k.endswith('_passed')):
                            singleton_failure={'check':check,'reference':pb,'trial':data};raise AssertionError(('singleton check failed',check))
                    if trial in CONDITIONS:norms_raw[full]=selected_norms(trial,data['localization_norms'])
                    if snap and trial in SNAPSHOT_MODES:
                        assert set(data['snapshots'])=={10};snapshot_store.add(trial,data['snapshots'][10])
                        sample_readouts[trial]={'hidden':data['hidden'],'query_rows':torch.tensor(READOUT_ROWS),
                            'logits':ll[list(READOUT_ROWS)].detach().cpu().clone(),'targets':x[[q+1 for q in READOUT_ROWS]],
                            'token_sha256':th(x),'hidden_fingerprint':fingerprint(data['hidden'])}
                    else:assert not data['snapshots']
                    partial_record=None
                def control(trial,original,original_logits):
                    nonlocal active_input,pair_reference,pair_logits,suffix_failure,check_target,bundle
                    active_input={**base,'phase':'suffix_control','trial':trial};pair_reference=original;pair_logits=original_logits[:128]
                    altered,ll=ev(rotated,mode=trial,zero=trial!='C',mask='native'if trial=='C'else'causal',bl=cl,pl=pl,suffix=True)
                    bundle=altered;check_target={**base,'trial':trial,'stage':'suffix_comparison'}
                    check={**base,'trial':trial,'original_token_sha256':th(x),'control_token_sha256':th(rotated),
                        'logits_exact':torch.equal(original_logits[:128],ll),'logits_reference':fingerprint(original_logits[:128]),'logits_control':fingerprint(ll),
                        'hidden_exact':torch.equal(original['hidden'][:,:128],altered['hidden'][:,:128]),
                        'gates_exact':prefix_gates_equal(original['gates'],altered['gates']),
                        'source_exact':source_equal(original['source'],altered['source']),'mask_exact':original['masks']==altered['masks'],
                        'contexts_exact':all(torch.equal(v,altered['prefix_contexts'][l][k])for l,d in original['prefix_contexts'].items()for k,v in d.items()),
                        'reference_context_hashes':original['operation']['prefix_context_hashes'],'control_context_hashes':altered['operation']['prefix_context_hashes'],
                        **{k+'_exact':torch.equal(original['z'][k][:127 if k=='loss' else 128],altered['z'][k])for k in('loss','pred','kl','kl_p','agreement_p')}}
                    suffix_checks.append(check);suffix_ops.append({**base,'trial':trial,**altered['operation']})
                    suffix_numeric[key+'__'+trial]={'token_sha256':th(rotated),'original_token_sha256':th(x),
                        'score_index':1,'scored_tokens':126,'hidden_prefix_fingerprint':fingerprint(altered['hidden'][:,:128]),
                        **{k:altered['z'][k]for k in('loss','pred','kl','kl_p','agreement_p')}}
                    if not all(v for k,v in check.items()if k.endswith('_exact')):
                        suffix_failure={'check':check,'reference':original,'control':altered,'reference_logits':original_logits[:128],
                            'control_logits':ll,'original_tokens':x,'control_tokens':rotated};raise AssertionError(('suffix closure failed',check))
                    assert not altered['snapshots'];pair_reference=None;pair_logits=None;bundle=None
                record('C',cb,cl);control('C',cb,cl);record('P',pb,pl);control('P',pb,pl)
                for trial in CONDITIONS:
                    if trial in('C','P'):continue
                    active_input={**base,'phase':'evaluation','trial':trial}
                    current,ll=ev(x,mode=trial,bl=cl,pl=pl,snapshot=snap and trial in SNAPSHOT_MODES)
                    bundle=current;record(trial,current,ll);control(trial,current,ll)
                    if trial in('current','fullrank'):equality(base,trial+'_closure',current,pb,ll,pl)
                    del current,ll;bundle=None
                if identity=='natural'and item['window']==0:
                    for trial,special in(('repeat',None),('identity','identity')):
                        active_input={**base,'phase':'evaluation','trial':trial}
                        current,ll=ev(x,mode='C',zero=False,mask='native',bl=cl,pl=pl,special=special)
                        bundle=current;record(trial,current,ll);equality(base,trial,current,cb,ll,cl);del current,ll;bundle=None
                save('inputs/'+key+'.npz',group_payload());group_committed=True
                cb=None;pb=None;pending_logits=None;del cl,pl;checkpoint()
            print('WIKI_CALIBRATION_DONE',item['corpus'],item['window'],run.count,flush=True)
        check_target={'stage':'final_invariants'}
        assert attempts==run.count==1548 and len(run.raw)==776 and len(suffix_checks)==768 and len(suffix_ops)==768
        assert len(calraw)==4 and len(calops)==4 and len(run.checks)==1128 and len(ops)==776
        assert len(norms_raw)==768 and len(sample_readouts)==11 and len(source_references)==32
        assert len(suffix_numeric)==768
        assert {k:maps_digest(v)for k,v in maps.items()}==frozen_maps and maps_digest(maps['M'])==REFERENCE_DIGEST
        assert all(sha(args.inputs/n)==h for n,h in input_hashes.items())and all(sha(ROOT/n)==h for n,h in source_hashes.items())
        assert precision_settings()==settings
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        checkpoint();meta.update(status='completed',phase='completed',completed_utc=time.strftime('%FT%TZ',time.gmtime()),
            forward_attempts=attempts,forward_passes=run.count,records=len(run.raw),model_hash_before=before,model_hash_after=after,
            parameter_versions_unchanged=True,calibration_maps_unchanged=True,precision_settings_after=precision_settings(),**resources())
        assert meta['calibration_forwards']==4 and meta['evaluation_forwards']==776 and meta['suffix_forwards']==768
        dump(out/'manifest.json',meta);print('COMPLETE_WIKI_CALIBRATION',run.count,flush=True)
    except BaseException as error:
        # Pending numerical rows/norms are already in normal objects. Save them
        # first, then a separately measured capsule; never dump full logits.
        preservation={}
        try:
            if group_marks is not None and not group_committed:
                save('inputs/'+group_key+'__partial.npz',group_payload());preservation['partial_input']='saved'
            checkpoint(preserving_failure=True);preservation['partial_checkpoint']='saved'
        except BaseException as save_error:preservation['partial_checkpoint_error']=repr(save_error)
        payload={'last_forward_identity':last_forward,'last_attempted_forward_identity':last_forward,
            'last_completed_forward_identity':last_completed,'current_attempt_input':current_attempt_tokens,
            'active_record_target':record_target,'active_check_target':check_target,
            'current_tensor_identity':last_forward,'gate_failure':None if gate is None else gate.failure_gate,
            'reconstruction_failure':None if gate is None else gate.failure_reconstruction,
            'current_hidden':hidden.get('tensor'),'current_gate_state':None if gate is None else gate.gate_cache,
            'current_source':None if gate is None else {'kv':gate.cache,'q':{l:d['source_q']for l,d in gate.diag.items()}},
            'singleton_failure':singleton_state(singleton_failure),
            'suffix_failure':suffix_state(suffix_failure),'identity_failure':identity_failure,
            'control_computation_failure':{k:v for k,v in control_failure.items()if k!='prefix_logits'},
            'control_logits':logit_pair(control_failure.get('prefix_logits')),
            'pending_logits':logit_pair(pending_logits if pending_logits is not None else None if run is None else getattr(run,'latest_logits',None)),
            'partial_calibration_raw':calraw if meta['phase']in('calibration','fit_before_evaluation')else{},
            'partial_calibration_metrics':calmetrics if meta['phase']in('calibration','fit_before_evaluation')else{},
            'partial_calibration_operations':calops if meta['phase']in('calibration','fit_before_evaluation')else[],
            'map_digests':frozen_maps,
            'snapshot_dedup_failure':getattr(error,'payload',None)}
        try:
            serialized=io.BytesIO();torch.save(payload,serialized);n=serialized.tell()
            assert n<=FAILURE_LIMIT,('Capsule exceeds reviewed bound',n,FAILURE_LIMIT)
            assert shutil.disk_usage(out).free-n>=DISK_RESERVE,('Capsule would violate disk reserve',n)
            (out/'failure_state.pt').write_bytes(serialized.getbuffer());preservation.update(failure_capsule_bytes=n,failure_capsule_sha256=sha(out/'failure_state.pt'))
        except BaseException as save_error:preservation['failure_capsule_error']=repr(save_error)
        meta.pop('completed_utc',None);meta.update(status='failed',phase='failed',active_failure_phase=check_target.get('stage',meta['phase']),
            failure_type=type(error).__name__,failure_message=str(error),last_forward_identity=last_forward,
            last_attempted_forward_identity=last_forward,last_completed_forward_identity=last_completed,
            active_record_target=record_target,active_check_target=check_target,model_loaded=model_loaded,
            forward_attempts=attempts,forward_passes=0 if run is None else run.count,records=0 if run is None else len(run.raw),
            failed_utc=time.strftime('%FT%TZ',time.gmtime()),preservation=preservation,**resources())
        dump(out/'manifest.json',meta);raise
    finally:
        if gate is not None:gate.close()
        if probe is not None:probe.close()

if __name__=='__main__':main()
