"""T19 frozen-F native-value 2x2x2 runner. No calibration or axis selection.

Requires literal input hashes, measured storage and independent freeze before
launch. 34 main and 34 fixed512 suffix trials per32 inputs, plus8 anchors=2184.
Native-repeat anchors remove all intervention hooks; output_hidden_states only
returns already computed states and use_cache stays False on every forward.
"""
import argparse
import copy
import contextlib
import json
import math
import platform
import resource
import shutil
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
import torch
from gate_native_value import GateNativeValue, BOUNDARIES, fingerprint, error_metrics
from gate_native_value_storage import (NORMAL_LIMIT,FAILURE_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,READOUT_ROWS,
    EvidenceWriter,empty_payload,exact,nested_fp,summarize_diag,trim_gate,trim_norm,
    failure_bundle,logit_rows,source_closure,sha,json_write,read_npz)

ROOT=Path(__file__).resolve().parent
CORPUS_COUNTS={'wiki_validation':4,'utilitarianism':4,'ethics':4,'optimizers_new':4}
BOUNDARY_ORDER=('000','111','001','010','011','100','101','110')
CONDITIONS=tuple(b+'__'+op for b in BOUNDARY_ORDER for op in (('baseline','remove','remove_scale','orth','orth_scale','current') if b=='111' else ('baseline','current','remove','orth')))
SNAPSHOT_MODES=tuple(b+'__'+op for b in ('000','111') for op in (('baseline','remove','orth','remove_scale','orth_scale') if b=='111' else ('baseline','remove','orth')))
SNAPSHOT_INPUT=('wiki_validation',0,'natural')
METADATA_PINS={'gpt2_run/manifest.json':'60909cdc2af2240ba99bf103c3cbf62146bddffd43c4f8f6810d092632fb5515','gpt2_run/selection.json':'56bbb1076fc342ae1f504bf8bca0d72b521ff1b69269e51575375a144eba50ae','../superweight-causal-circuits/manifests/transfer_models.json':'4cb4e6934a62ceccf9c062b8d85b171f2704d0891c40d8d2cd4f51ab1f75e90c'}
F_DIGEST='8b0ca7a45d01258d92bfd20069e4cd915437d8ce8a5175dda9ed4c17098fe2c8'
# Populated only after the independent tokenizer/input audit. Empty means blocked.
INPUT_HASHES={'data_manifest.json': 'b72641752c80b3408873791c03767477c68831c6e2e7fe69d5892e96b4df9f3b', 'exclusion_inventory.json': '277595ce6209b6e027954e9e236c8ea0c37540d726a58706c1176ebce0c16557', 'historical_manifest_scan.json': '351dbebe91387f1c228a2a35938cc621ec88eb56c49912e15683cc030897d20b', 'preflight.json': 'e1e677cffe33987196c1be314407551506d322ad727ab13d2d8df9476d387306', 'start_identity_summary.json': '7b07f9d72ff912488184ff94238a7c81845be38924e757c5f1d65a20f6b592b7', 'tokens.pt': '962c2ad61d248370731a5a80d776f456aae57018dd65111c56cbe21b2bdd615b'}
STORAGE_READY=True
GROUP_ALLOWANCES={'snapshot': 107379754, 'ordinary': 11137197}
BRIDGE_POLICY={'context_max_abs':1e-4,'target_loss_max_abs':1e-3,'passage_mean_abs':1e-5}

def pair_grid():
    return [(c,w,i) for c,n in CORPUS_COUNTS.items() for w in range(n) for i in ('natural','bos')]

def metric_rows(logits,x,c_logits=None,b_logits=None,*,suffix=False):
    """Suffix keeps logits0..127, losses for unchanged targets1..127 only."""
    n=128 if suffix else 511;logits=logits[:n];pred=logits.argmax(-1).cpu()
    result={k:[] for k in ('loss','kl_c','kl_boundary')}
    for start in range(0,n,32):
        end=min(start+32,n);lp=logits[start:end].float().log_softmax(-1)
        target_end=min(end,127) if suffix else end
        if target_end>start:result['loss'].append(-lp[:target_end-start].gather(-1,x[start+1:target_end+1,None].to(logits.device)).squeeze(-1).cpu())
        for name,reference in (('kl_c',c_logits),('kl_boundary',b_logits)):
            if reference is None:result[name].append(torch.zeros(end-start))
            else:
                lb=reference[start:end].float().log_softmax(-1)
                result[name].append((lb.exp()*(lb-lp)).sum(-1).cpu())
    result={k:torch.cat(v) for k,v in result.items()};result['pred']=pred
    for name,reference in (('agreement_c',c_logits),('agreement_boundary',b_logits)):
        result[name]=torch.ones(n,dtype=torch.bool) if reference is None else pred.eq(reference[:n].argmax(-1).cpu())
    assert result['loss'].shape==((127,) if suffix else (511,))
    assert all(torch.isfinite(result[k]).all() for k in ('loss','kl_c','kl_boundary'))
    if suffix:result.update(target_indices=torch.arange(1,128),logit_query_indices=torch.arange(128))
    return result

def model_forward(model,x,on_return,on_invoke=None):
    """Invoke one model; completion callback precedes all output/scoring access."""
    with torch.inference_mode():
        prepared=x[None].to(next(model.parameters()).device)
        if on_invoke is not None:on_invoke()
        output=model(prepared,use_cache=False,output_hidden_states=True)
    on_return()
    return output


def capture_gate(gate,hidden,metrics):
    gate.finish()
    return {'metrics':metrics,'hidden':hidden.detach().cpu().clone(),'source':copy.deepcopy(gate.source_states),
        'gates':copy.deepcopy(gate.gate_cache),'norms':copy.deepcopy(gate.localization_norms),
        'snapshots':copy.deepcopy(gate.snapshots),'prefix_contexts':copy.deepcopy(gate.prefix_contexts),
        'operations':{'diag':summarize_diag(gate.diag),'gate_checks':copy.deepcopy(gate.gate_checks),
            'projection_records':copy.deepcopy(gate.projection_records),'context_records':copy.deepcopy(gate.context_records),
            'mask_records':copy.deepcopy(gate.mask_records),'source_fingerprints':nested_fp(gate.source_states),
            'gate_fingerprints':nested_fp(gate.gate_cache),'prefix_context_fingerprints':nested_fp(gate.prefix_contexts),
            'projection_checks':gate.projection_checks,'context_checks':gate.context_checks,'query_checks':gate.query_checks},
        'diag_l10_vectors':{k:v.clone() for k,v in gate.diag[10].items() if isinstance(v,torch.Tensor) and k!='source_q'}}

def pack_record(payload,mode,bundle,logits,x,*,suffix=False):
    metrics=dict(bundle['metrics']);metrics['final_hidden_sha256']=fingerprint(bundle['hidden'])['sha256']
    if suffix:
        payload['suffix'][mode]=metrics;payload['suffix_gates'][mode]=trim_gate(bundle['gates'])
        payload['suffix_norms'][mode]=trim_norm(bundle['norms']);payload['suffix_operations'][mode]=bundle['operations']
    else:
        payload['metrics'][mode]=metrics;payload['gates'][mode]=bundle['gates'];payload['norms'][mode]=bundle['norms']
        payload['operations'][mode]=bundle['operations']
        if tuple(payload['metadata'][k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT and mode in SNAPSHOT_MODES:
            assert set(bundle['snapshots'])=={10}
            payload['snapshots'][mode]=bundle['snapshots'][10]
            payload['readouts'][mode]={'hidden':bundle['hidden'],'query_rows':torch.tensor(READOUT_ROWS),
                'logits':logits[list(READOUT_ROWS)].detach().cpu().clone(),'targets':x[[q+1 for q in READOUT_ROWS]],
                'token_sha256':fingerprint(x)['sha256'],'hidden_fingerprint':fingerprint(bundle['hidden'])}

def identity_check(left,right,ll,rr,label):
    result={'kind':label,'logits_exact':torch.equal(ll,rr),'hidden_exact':exact(left['hidden'],right['hidden'])}
    for k in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
        result[k+'_exact']=exact(left['metrics'][k],right['metrics'][k])
    return result

def suffix_check(left,right,ll,rr):
    out={'kind':'suffix','logits_exact':torch.equal(ll[:128],rr[:128]),
        'hidden_exact':exact(left['hidden'][:,:128],right['hidden'][:,:128]),
        'source_exact':exact(left['source'],right['source']),
        'contexts_exact':exact(left['prefix_contexts'],right['prefix_contexts']),
        'gates_exact':exact(trim_gate(left['gates']),trim_gate(right['gates'])),
        'mask_exact':exact(left['operations']['mask_records'],right['operations']['mask_records'])}
    for k in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
        n=127 if k=='loss' else 128
        out[k+'_exact']=exact(left['metrics'][k][:n],right['metrics'][k])
    return out

def check_passed(check):
    return all(v for k,v in check.items() if k.endswith('_exact') or k.endswith('_passed'))

def bridge_check(delta,scale,left,right):
    ctx=error_metrics(delta['snapshots'][10]['delivered_context'],scale['snapshots'][10]['delivered_context'])
    loss=scale['metrics']['loss'][1:].double()-delta['metrics']['loss'][1:].double()
    kls=[[],[]];mismatch=[]
    for start in range(0,len(left),32):
        a=left[start:start+32].detach().cpu().double();b=right[start:start+32].detach().cpu().double()
        la,lb=a.log_softmax(-1),b.log_softmax(-1)
        kls[0].append((la.exp()*(la-lb)).sum(-1));kls[1].append((lb.exp()*(lb-la)).sum(-1))
        pa,pb=a.argmax(-1),b.argmax(-1);inds=(pa!=pb).nonzero().flatten()
        for q in inds.tolist():
            ar,br=a[q].topk(2).values,b[q].topk(2).values;drift=float((a[q]-b[q]).abs().max())
            mismatch.append({'query':start+q,'delta_id':int(pa[q]),'scale_id':int(pb[q]),
                'delta_margin':float(ar[0]-ar[1]),'scale_margin':float(br[0]-br[1]),
                'row_max_abs_logit_drift':drift,'compatible_with_drift':float(ar[0]-ar[1])<=2*drift})
    return {'kind':'arithmetic_bridge','delivered_context':ctx,'hidden':error_metrics(delta['hidden'],scale['hidden']),
        'logits':error_metrics(left.detach().cpu(),right.detach().cpu()),'loss_difference':loss,
        'kl_delta_to_scale':torch.cat(kls[0]),'kl_scale_to_delta':torch.cat(kls[1]),'argmax_mismatches':mismatch,
        'context_passed':ctx['max_abs']<=BRIDGE_POLICY['context_max_abs'],
        'target_loss_passed':float(loss.abs().max())<=BRIDGE_POLICY['target_loss_max_abs'],
        'passage_mean_passed':abs(float(loss.mean()))<=BRIDGE_POLICY['passage_mean_abs']}

class OrderMismatch(AssertionError):
    def __init__(self,check,reference,trial):
        self.check=check;self.reference=reference;self.trial=trial
        super().__init__(check)

def order_checks(refs,boundary,op,bundle):
    """Exact common-state checks; preserve actual compared sides on failure."""
    u,l,d=boundary;mode=boundary+'__'+op;result=[];gate=bundle['gates'][10]
    raw=failure_bundle(bundle)
    def checked(check,reference):
        if not check_passed(check):raise OrderMismatch(check,reference,raw)
        result.append(check)
    incoming={k:bundle['operations']['projection_records'][10]['before'][k] for k in ('q','k','nonsource_v','source_v')}
    incoming.update({k:fingerprint(gate[k]) for k in ('b','mass','s','target_b','target_mass')})
    incoming_key=(u,op)
    if incoming_key in refs['incoming']:
        old=refs['incoming'][incoming_key]
        checked({'kind':'L10_incoming_causal_order','mode':mode,'reference_mode':old['mode'],
                 'reference_fingerprints':old['summary'],'trial_fingerprints':incoming,
                 'incoming_exact':exact(incoming,old['summary'])},old['raw'])
    else:refs['incoming'][incoming_key]={'mode':mode,'summary':incoming,'raw':raw}
    delivered_key=(u,l,op);delivered=bundle['operations']['context_records'][10]
    if delivered_key in refs['delivered']:
        old=refs['delivered'][delivered_key]
        checked({'kind':'D_causal_order','mode':mode,'reference_mode':old['mode'],
                 'reference_fingerprints':old['summary'],'trial_fingerprints':delivered,
                 'delivered_exact':exact(delivered,old['summary'])},old['raw'])
    else:refs['delivered'][delivered_key]={'mode':mode,'summary':delivered,'raw':raw}
    if not op.endswith('_scale'):
        snap=bundle['snapshots'][10];delta=snap['delivered_context'].double()-snap['actual_context'].double()
        state={'mode':mode,'delta':delta,'v0':snap['v'][0,:,:1].double(),
               'gain_delta':gate['target_mass'].double()-gate['mass'].double(),
               'local_reconstruction_residuals':{k:bundle['operations']['gate_checks'][10][k]
                    for k in ('native_reconstruction_error','explicit_source_reconstruction_error')}}
        raw['direct_state']=state
        pair_key=(u,d,op);other=refs['direct'].get(pair_key)
        if other is not None and other[0]!=l:
            native,zero=(state,other[1]) if l=='0' else (other[1],state)
            heads,_,head_dim=snap['v'][0].shape
            observed=(native['delta']-zero['delta'])[:,1:].reshape(1,511,heads,head_dim)[0].transpose(0,1)
            target=-native['gain_delta'][...,None]*native['v0'];err=error_metrics(observed,target)
            checked({'kind':'direct_local_delta_identity','mode':mode,'reference_mode':other[1]['mode'],
                     'fp64':err,'reference_reconstruction_residuals':other[1]['local_reconstruction_residuals'],
                     'trial_reconstruction_residuals':state['local_reconstruction_residuals'],
                     'identity_passed':err['max_abs']<=1e-4},other[2])
        else:refs['direct'][pair_key]=(l,state,raw)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--calibration-reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    writer=EvidenceWriter(a.out);started=time.time();model=gate=probe=None;attempts=completed=0;main_count=suffix_count=anchor_count=0
    active={};last_attempt={};last_completed={};current_tokens=None;latest_logits=None;latest_hidden=None;pair_left=pair_right=None;pair_logits=None;payload=None;group_name=None;group_committed=True
    meta={'study':'gate_native_value','status':'preparing','phase':'setup','model_loaded':False,'calibration_forwards':0,
          'normal_limit_bytes':NORMAL_LIMIT,'failure_limit_bytes':FAILURE_LIMIT,'disk_reserve_bytes':DISK_RESERVE,
          'forward_attempts':0,'forward_passes':0,'started_utc':time.strftime('%FT%TZ',time.gmtime()),'bridge_policy':BRIDGE_POLICY,
          'score_views':{'full':[2,511],'early':[2,127],'later':[128,511]},'conditions':list(CONDITIONS),'snapshot_modes':list(SNAPSHOT_MODES),
          'anchors':{'native_repeat':'all gate/writer hooks removed; native forward returns final hidden',
                     'native_identity':'transient LayerWrites identity plus000baseline observer; no parameter edits'}}
    def progress():return {'phase':meta['phase'],'forward_attempts':attempts,'forward_passes':completed,'evaluation_forwards':main_count,'suffix_forwards':suffix_count,'anchor_forwards':anchor_count,'active':active,'last_attempted':last_attempt,'last_completed':last_completed}
    def resources():return {'wall_seconds':time.time()-started,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
        'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0}
    def checkpoint(*,final=False):
        meta.update(progress(),**resources());writer.checkpoint(progress(),manifest=meta,final=final)
    def ev(x,boundary='000',op='baseline',c_logits=None,b_logits=None,*,suffix=False,unhooked=False,anchor=False):
        nonlocal attempts,completed,last_attempt,last_completed,current_tokens,latest_logits,latest_hidden,main_count,suffix_count,anchor_count
        assert x.dtype==torch.long and x.shape==(512,)
        current_tokens=x.clone();latest_logits=latest_hidden=None
        writer.journal({'event':'forward_intent','next_model_attempt':attempts+1,'identity':active,'token_fingerprint':fingerprint(x)})
        if not unhooked:gate.reset(boundary,op,snapshot=True)
        def invoked():
            nonlocal attempts,last_attempt
            writer.journal({'event':'forward_attempt','attempt':attempts+1,'identity':active,'token_fingerprint':fingerprint(x)})
            attempts+=1;last_attempt=dict(active)
        def returned():
            nonlocal completed,last_completed,anchor_count,suffix_count,main_count
            completed+=1;last_completed=dict(active)
            if anchor:anchor_count+=1
            elif suffix:suffix_count+=1
            else:main_count+=1
            writer.journal({'event':'model_returned','attempt':attempts,'completed':completed,'identity':active})
        profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if completed==0 else contextlib.nullcontext()
        with torch.inference_mode():
            with profile as prof:output=model_forward(model,x,returned,invoked)
            if completed==1:
                events=[{'operator':e.key,'count':e.count} for e in prof.key_averages() if 'scaled_dot_product' in e.key]
                meta['sdpa_profiler_events']=events
                assert any('_scaled_dot_product' in e['operator'] for e in events),'Expected observed SDPA backend'
            # Count the successful model return before any cloning/scoring can fail.
            latest_logits=output.logits[0,:511].detach().clone();hidden=output.hidden_states[-1].detach().cpu().clone();latest_hidden=hidden;del output
            metrics=metric_rows(latest_logits,x,c_logits,b_logits,suffix=suffix)
        bundle={'metrics':metrics,'hidden':hidden,'source':{},'gates':{},'norms':{},'snapshots':{},'prefix_contexts':{},'operations':{},'diag_l10_vectors':{}} if unhooked else capture_gate(gate,hidden,metrics)
        writer.journal({'event':'forward_completed','attempt':attempts,'completed':completed,'identity':active,'hidden_fingerprint':fingerprint(hidden),'logits_fingerprint':fingerprint(latest_logits)})
        return bundle,latest_logits
    def assert_check(check,left=None,right=None,ll=None,rr=None):
        nonlocal pair_left,pair_right,pair_logits
        payload['checks'].append(check)
        if not check_passed(check):
            pair_left=left;pair_right=right;pair_logits=logit_rows(ll,rr) if ll is not None else None
            raise AssertionError(check)
    writer.checkpoint(progress(),manifest=meta)
    try:
        assert INPUT_HASHES and STORAGE_READY and GROUP_ALLOWANCES,'Launch blocked: independent input/storage/source freeze not complete'
        assert len(CONDITIONS)==34 and len(SNAPSHOT_MODES)==8
        writer.guard('pre_model_complete_projection',NORMAL_LIMIT-writer.used())
        assert all(sha(a.inputs/n)==h for n,h in INPUT_HASHES.items())
        pre=json.loads((a.inputs/'preflight.json').read_text());assert pre['status']=='passed' and pre['model_forwards']==0 and pre['evaluation_passages']==16
        dm=json.loads((a.inputs/'data_manifest.json').read_text());inputs=torch.load(a.inputs/'tokens.pt',map_location='cpu',weights_only=True)
        assert len(inputs)==16 and {c:sum(i['corpus']==c for i in inputs) for c in CORPUS_COUNTS}==CORPUS_COUNTS
        assert {(i['corpus'],i['window']) for i in inputs}=={(c,w) for c,n in CORPUS_COUNTS.items() for w in range(n)}
        for item in inputs:
            rec=next(r for r in dm['passages'] if (r['corpus'],r['window'])==(item['corpus'],item['window']))
            assert item['tokens'].dtype==torch.long and item['tokens'].shape==(512,) and fingerprint(item['tokens'])['sha256']==rec['window_token_sha256']
        for name in INPUT_HASHES:shutil.copyfile(a.inputs/name,a.out/name)
        ref=a.calibration_reference;assert sha(ref/'manifest.json')==dm['calibration_reference_manifest_sha256']
        refmeta=json.loads((ref/'manifest.json').read_text())
        assert refmeta['status']=='completed' and refmeta['calibration_map_digests']['F']==F_DIGEST
        for name in ('calibration_maps.npz','calibration_directions.npz'):
            assert sha(ref/name)==dm['calibration_reference_files'][name]
            shutil.copyfile(ref/name,a.out/name)
        maps=read_npz(ref/'calibration_maps.npz');directions=read_npz(ref/'calibration_directions.npz')
        from gpt2_gate_subspace import maps_digest
        from run_transfer import modelhash
        from native_prefix_diagnostic import precision_settings
        assert maps_digest(maps['F'])==F_DIGEST
        alpha=maps['F'][10]['alpha'];u=directions['F'][10]['u'];v=directions['F'][10]['v']
        closure=source_closure(ROOT,['gpt2_gate_native_value.py','gate_native_value.py','gate_native_value_storage.py','analyze_gate_native_value.py','gate_native_value_inputs.py','GPT2_NATIVE_VALUE_BOUNDARY_PROPOSAL.md','GPT2_NATIVE_VALUE_IMPLEMENTATION_ADDENDUM.md','gpt2_layer_writes.py'])
        for name,h in closure.items():shutil.copyfile(ROOT/name,a.out/'frozen_sources'/name);assert sha(a.out/'frozen_sources'/name)==h
        cfg=json.loads(a.config.read_text())
        for name,h in METADATA_PINS.items():
            assert sha(ROOT/name)==h,('Changed historical metadata',name)
            destination=a.out/'metadata_dependencies'/name.removeprefix('../').replace('/','__')
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
            assert sha(destination)==h
        old=json.loads((ROOT/'gpt2_run/manifest.json').read_text())
        assert cfg['models']['gpt2']['revision']==refmeta['revision']==old['revision']
        checkpoint_spec=json.loads((ROOT.parent/'superweight-causal-circuits/manifests/transfer_models.json').read_text())['gpt2']
        assert cfg['models']['gpt2']['model_id']==checkpoint_spec['model_id']==refmeta['model_id']==old['model_id']
        assert cfg['models']['gpt2']['revision']==checkpoint_spec['revision']==refmeta['revision']==old['revision']
        modelpath=Path(cfg['models']['gpt2']['path'])
        for name,spec in checkpoint_spec['checkpoint_files'].items():assert sha(modelpath/name)==spec['sha256']
        # No GPU computation or weights before all prospective freezes/resources pass.
        torch.set_num_threads(4);torch.manual_seed(20261021);settings=precision_settings()
        assert settings['float32_matmul_precision']=='highest' and not settings['cuda_matmul_allow_tf32']
        assert settings==refmeta['precision_settings_before'],'Precision settings must match the frozen T18 reference'
        meta.update(input_dependencies=INPUT_HASHES,source_hashes=closure,metadata_dependencies=METADATA_PINS,calibration_map_digest=F_DIGEST,calibration_fit=False,
            calibration_reference_sha256=sha(ref/'manifest.json'),physical_execution_lengths=[512],passages_per_corpus=CORPUS_COUNTS,
            precision_settings_before=settings,revision=old['revision'],model_id=old['model_id'],dtype='float32',attention='sdpa',
            runtime_versions={'python':platform.python_version(),'torch':str(torch.__version__),'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda})
        checkpoint()
        from transformers import AutoModelForCausalLM
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
        assert not gpu_processes,('GPU compute worker already present',gpu_processes)
        meta['gpu_compute_processes_before_load']=gpu_processes
        torch.cuda.reset_peak_memory_stats()
        model=AutoModelForCausalLM.from_pretrained(modelpath,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
        meta['model_loaded']=True;before=modelhash(model);assert before==old['model_hash_before'];versions={n:p._version for n,p in model.named_parameters()}
        meta['model_hash_before']=before
        gate=GateNativeValue(model,alpha,u,v,F_DIGEST)
        for item in inputs:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'corpus':item['corpus'],'window':item['window'],'identity':identity,'length':512,'token_sha256':fingerprint(x)['sha256']}
                group_name=f"{item['corpus']}__{item['window']}__{identity}";snap=tuple(base[k] for k in ('corpus','window','identity'))==SNAPSHOT_INPUT
                writer.guard('input_group:'+group_name,GROUP_ALLOWANCES['snapshot' if snap else 'ordinary']+METADATA_ALLOWANCE)
                payload=empty_payload(base);group_committed=False;meta['phase']='evaluation';refs={'incoming':{},'delivered':{},'direct':{}}
                rotated=x.clone();rotated[128:]=x[128:].roll(1);c_bundle=c_logits=None
                def anchors():
                    nonlocal gate,probe,active
                    original_active=dict(active)
                    for anchor in ('native_repeat','native_identity'):
                        active={**base,'mode':anchor,'phase':'anchor'}
                        if anchor=='native_repeat':
                            gate.close()
                            try:trial,ll=ev(x,c_logits=c_logits,b_logits=c_logits,unhooked=True,anchor=True)
                            finally:gate=GateNativeValue(model,alpha,u,v,F_DIGEST)
                        else:
                            from gpt2_layer_writes import LayerWrites
                            sel=json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected']
                            probe=LayerWrites(model,sel,[447,138]);probe.reset_layers(E=False,blocked=[],special='identity')
                            try:trial,ll=ev(x,c_logits=c_logits,b_logits=c_logits,anchor=True)
                            finally:probe.close();probe=None
                        pack_record(payload,anchor,trial,ll,x)
                        assert_check(identity_check(c_bundle,trial,c_logits,ll,anchor),c_bundle,trial,c_logits,ll)
                        del trial,ll
                    active=original_active
                for boundary in BOUNDARY_ORDER:
                    own_base=own_logits=None;pending_bridge=None;pending_bridge_logits=None
                    for mode in (m for m in CONDITIONS if m.startswith(boundary+'__')):
                        op=mode.split('__')[1];active={**base,'mode':mode,'phase':'evaluation'}
                        trial,ll=ev(x,boundary,op,c_logits,own_logits)
                        if boundary=='000' and op=='baseline':c_bundle,c_logits=trial,ll
                        if op=='baseline':own_base,own_logits=trial,ll;payload['source_reference'][boundary]=trial['source']
                        # Backfill self-references only through the same scoring path.
                        trial['metrics']=metric_rows(ll,x,c_logits,own_logits)
                        pack_record(payload,mode,trial,ll,x)
                        if mode=='000__baseline' and identity=='natural' and item['window']==0:anchors()
                        if op!='baseline':
                            assert_check({'kind':'source_q0','mode':mode,'source_exact':exact(trial['source'],own_base['source'])},own_base,trial,own_logits,ll)
                        if op=='current':assert_check({'mode':mode,**identity_check(own_base,trial,own_logits,ll,'current')},own_base,trial,own_logits,ll)
                        for check in order_checks(refs,boundary,op,trial):assert_check(check,own_base,trial,own_logits,ll)
                        if op.endswith('_scale'):
                            check={'mode':mode,**bridge_check(pending_bridge,trial,pending_bridge_logits,ll)}
                            assert_check(check,pending_bridge,trial,pending_bridge_logits,ll);pending_bridge=pending_bridge_logits=None
                        # Every main condition has its own exactly fixed-shape suffix run.
                        active={**base,'mode':mode,'phase':'suffix_control'}
                        altered,al=ev(rotated,boundary,op,c_logits,own_logits,suffix=True)
                        pack_record(payload,mode,altered,al,rotated,suffix=True)
                        assert_check({'mode':mode,**suffix_check(trial,altered,ll,al)},trial,altered,ll[:128],al[:128])
                        if boundary=='111' and op in ('remove','orth'):pending_bridge,pending_bridge_logits=trial,ll
                        del altered,al
                        if op!='baseline' and not (boundary=='111' and op in ('remove','orth')):del trial,ll
                    del own_base,own_logits
                assert set(payload['metrics'])==set(CONDITIONS)|({'native_repeat','native_identity'} if identity=='natural' and item['window']==0 else set())
                assert set(payload['suffix'])==set(CONDITIONS)
                writer.save('inputs/'+group_name+'.npz',payload);group_committed=True
                payload=None;c_bundle=c_logits=latest_logits=None;refs={};checkpoint()
                print('NATIVE_VALUE_DONE',group_name,completed,flush=True)
        assert attempts==completed==2184 and main_count==suffix_count==1088 and anchor_count==8
        assert all(sha(a.inputs/n)==h for n,h in INPUT_HASHES.items()) and all(sha(ROOT/n)==h for n,h in closure.items())
        assert all(sha(ROOT/n)==h for n,h in METADATA_PINS.items())
        assert all(sha(ref/n)==dm['calibration_reference_files'][n] for n in ('calibration_maps.npz','calibration_directions.npz'))
        assert maps_digest(maps['F'])==F_DIGEST and precision_settings()==settings
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        meta.update(status='completed',phase='completed',model_hash_after=after,parameter_versions_unchanged=True,precision_settings_after=precision_settings(),completed_utc=time.strftime('%FT%TZ',time.gmtime()))
        checkpoint(final=True);print('COMPLETE_NATIVE_VALUE',completed,flush=True)
    except BaseException as error:
        if isinstance(error,OrderMismatch):
            pair_left,pair_right=error.reference,error.trial
            if payload is not None:payload['checks'].append(error.check)
        preservation={}
        try:
            if payload is not None and not group_committed:
                writer.save('inputs/'+group_name+'__partial.npz',payload);preservation['partial_input']='saved'
        except BaseException as save_error:preservation['partial_input_error']=repr(save_error)
        capsule={'error_type':type(error).__name__,'error':str(error),'progress':progress(),'current_attempt_tokens':current_tokens,'current_input_identity':active,
            'gate_failure':None if gate is None else gate.failure,'gate_state':None if gate is None else gate.gate_cache,
            'source_states':None if gate is None else gate.source_states,'left':failure_bundle(pair_left),'right':failure_bundle(pair_right),
            'comparison_logits':pair_logits,'latest_logits':logit_rows(latest_logits),'current_hidden':latest_hidden,
            'uncommitted_input':payload if 'partial_input_error' in preservation else None}
        try:preservation['failure_capsule']=writer.save('failure_state.npz',capsule,failure=True)
        except BaseException as save_error:preservation['failure_capsule_error']=repr(save_error)
        meta.update(progress(),**resources(),status='failed',phase='failed',failure_type=type(error).__name__,failure_message=str(error),preservation=preservation)
        try:writer.checkpoint(progress(),manifest=meta,failure=True,final=True)
        except BaseException as save_error:json_write(a.out/'checkpoint_failure.json',{'error':repr(save_error)})
        raise
    finally:
        if gate is not None:gate.close()
        if probe is not None:probe.close()

if __name__=='__main__':main()
