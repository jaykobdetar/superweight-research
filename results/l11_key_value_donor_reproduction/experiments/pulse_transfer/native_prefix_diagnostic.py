"""Native-only shape/backend diagnostic; frozen gate study remains untouched."""
import argparse
import contextlib
import json
import math
import platform
import resource
import time
from pathlib import Path

import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM

from run_transfer import ROOT, dump, modelhash, sha, th

VARIANTS = ('short128','short128_repeat','full512','full512_repeat',
            'suffix_rotated512','suffix_constant512','padded512')
CASES = [('default','hookfree_short128','none'),('default','hookfree_full512','none')]
CASES += [(backend,variant,'observer') for backend in ('default','math') for variant in VARIANTS]
CASES += [('default','study_short128','study'),('default','study_full512','study')]


def tensor(x):
    return x.detach().cpu().clone()


def first(x):
    return x[0] if isinstance(x,(tuple,list)) else x


class PrefixProbe:
    """Observer hooks return None: they never replace an input or output."""
    def __init__(self,model):
        self.handles=[];self.states={};self.masks={}
        def post(name):
            def hook(module,args,output): self.states[name]=tensor(first(output)[:,:128])
            return hook
        def pre(name):
            def hook(module,args): self.states[name]=tensor(args[0][:,:128])
            return hook
        for l,block in enumerate(model.transformer.h):
            self.handles.append(block.register_forward_pre_hook(pre(f'L{l}.input')))
            self.handles.append(block.register_forward_hook(post(f'L{l}.output')))
            self.handles.append(block.ln_1.register_forward_hook(post(f'L{l}.ln1')))
            self.handles.append(block.attn.c_attn.register_forward_hook(post(f'L{l}.qkv')))
            self.handles.append(block.attn.c_proj.register_forward_pre_hook(pre(f'L{l}.context')))
            self.handles.append(block.attn.register_forward_hook(post(f'L{l}.attention_output')))
            self.handles.append(block.mlp.register_forward_hook(post(f'L{l}.mlp_output')))
            def mask_hook(module,args,kwargs,l=l):
                mask=kwargs.get('attention_mask');n=args[0].shape[1] if args else kwargs['hidden_states'].shape[1]
                info={'kind':'implicit_causal' if mask is None else 'explicit','input_length':n,'prefix':None}
                if mask is not None:
                    assert mask.ndim==4
                    prefixmask=mask[...,:128,:].expand(*mask.shape[:-2],128,mask.shape[-1])
                    info.update(shape=list(mask.shape),dtype=str(mask.dtype),prefix=tensor(prefixmask[...,:128]))
                    future=torch.arange(mask.shape[-1],device=mask.device)[None,:]>torch.arange(128,device=mask.device)[:,None]
                    visible=prefixmask if prefixmask.dtype==torch.bool else prefixmask>-1e30
                    info['future_allowed_entries']=int((visible&future).sum())
                    assert not (visible&future).any(), 'Captured explicit mask exposes future keys to prefix queries'
                self.masks[l]=info
            self.handles.append(block.attn.register_forward_pre_hook(mask_hook,with_kwargs=True))
        self.handles.append(model.transformer.ln_f.register_forward_pre_hook(pre('final_ln_input')))
        self.handles.append(model.transformer.ln_f.register_forward_hook(post('final_ln_output')))

    def reset(self): self.states={};self.masks={}

    def close(self):
        for h in self.handles:h.remove()


def evaluate_native(model,x,attention_mask=None,probe=None,profile=True):
    device=next(model.parameters()).device
    if probe is not None:probe.reset()
    ctx=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) if profile else contextlib.nullcontext()
    with ctx as prof,torch.inference_mode():
        out=model(x[None].to(device),attention_mask=None if attention_mask is None else attention_mask[None].to(device),
                  use_cache=False,output_hidden_states=True)
        logits=out.logits[0,:128];hidden=out.hidden_states[-1][:,:128]
        # Same FP32 chunk32 loss arithmetic as the failed runner; shared target IDs.
        loss=[]
        for start in range(0,127,32):
            end=min(start+32,127);lp=logits[start:end].float().log_softmax(-1)
            loss.append(-lp.gather(-1,x[start+1:end+1,None].to(device)).squeeze(-1).cpu())
        result={'logits':tensor(logits),'hidden':tensor(hidden),'loss':torch.cat(loss),
                'pred':tensor(logits.argmax(-1)),'token_sha256':th(x),'input_tokens':len(x),
                'states':{} if probe is None else probe.states,'masks':{} if probe is None else probe.masks}
    result['sdpa_profiler_events']=[] if not profile else [
        {'operator':event.key,'count':event.count}for event in prof.key_averages()
        if 'scaled_dot_product' in event.key or 'flash_attention' in event.key or 'efficient_attention' in event.key]
    result['sdpa_backend_events']=[event for event in result['sdpa_profiler_events']
        if '_scaled_dot_product' in event['operator'] or 'flash_attention' in event['operator'] or 'efficient_attention' in event['operator']]
    result['profiler_backend_observed']=bool(result['sdpa_backend_events'])
    assert result['logits'].shape[0]==128 and result['hidden'].shape[1]==128
    assert all(torch.isfinite(result[k]).all()for k in('logits','hidden','loss'))
    if probe is not None:assert torch.equal(probe.states['final_ln_output'],result['hidden'])
    return result


def difference(reference,trial):
    """FP64 descriptive error; norm-relative error avoids near-zero coordinates."""
    assert reference.shape==trial.shape
    assert torch.isfinite(reference).all()and torch.isfinite(trial).all()
    a,b=reference.double(),trial.double();d=b-a;flat=d.abs().reshape(-1);index=int(flat.argmax())
    ar=float(a.square().mean().sqrt());dr=float(d.square().mean().sqrt())
    av=float(a.reshape(-1)[index]);bv=float(b.reshape(-1)[index])
    coords=list(torch.unravel_index(torch.tensor(index),a.shape));coords=[int(z)for z in coords]
    return {'exact':torch.equal(reference,trial),'max_abs':float(flat[index]),'rms_error':dr,
            'reference_rms':ar,'relative_l2':None if ar==0 else dr/ar,'reference_max_abs':float(a.abs().max()),
            'mean_signed_error':float(d.mean()),'count_abs_gt_1e_minus_3':int((flat>1e-3).sum()),
            'elements':d.numel(),'maximum_index':coords,'reference_at_maximum':av,'trial_at_maximum':bv,
            'relative_at_maximum':None if av==0 else abs(bv-av)/abs(av)}


def compare(reference,trial):
    sl=slice(1,127);kl=[]
    for start in range(1,127,8):
        end=min(start+8,127);a=reference['logits'][start:end].double().log_softmax(-1)
        b=trial['logits'][start:end].double().log_softmax(-1);kl.append((a.exp()*(a-b)).sum(-1))
    kl=torch.cat(kl)
    margins={}
    for name,state in (('reference',reference),('trial',trial)):
        top=state['logits'][sl].double().topk(2,dim=-1).values;gap=top[:,0]-top[:,1]
        margins[name]={'min_top_two_logit_gap':float(gap.min()),'mean_top_two_logit_gap':float(gap.mean())}
    result={'hidden':difference(reference['hidden'],trial['hidden']),
            'source_q0_hidden':difference(reference['hidden'][:,0],trial['hidden'][:,0]),
            'query127_hidden':difference(reference['hidden'][:,127],trial['hidden'][:,127]),
            'source_q0_logits':difference(reference['logits'][0],trial['logits'][0]),
            'query127_logits':difference(reference['logits'][127],trial['logits'][127]),
            'logits_all_128_queries':difference(reference['logits'],trial['logits']),
            'logits_scored_queries':difference(reference['logits'][sl],trial['logits'][sl]),
            'loss':difference(reference['loss'][1:],trial['loss'][1:]),
            'mean_delta_nll':float((trial['loss'][1:].double()-reference['loss'][1:].double()).mean()),
            'mean_kl_reference_to_trial_fp64':float(kl.mean()),'max_kl_reference_to_trial_fp64':float(kl.max()),
            'argmax_agreement':float(reference['pred'][sl].eq(trial['pred'][sl]).double().mean()),
            'top_two_margins':margins,
            'argmax_mismatches':[{'query':int(q),'reference_id':int(reference['pred'][q]),'trial_id':int(trial['pred'][q])}
                for q in torch.arange(1,127)[reference['pred'][sl]!=trial['pred'][sl]]],
            'scored_queries':[1,126],'scored_targets':[2,127],'scored_tokens':126,
            'layer_states':{k:difference(a,trial['states'][k])for k,a in reference['states'].items() if k in trial['states']}}
    return result


def score_diagnostic(state,layer,heads):
    qkv=state['states'][f'L{layer}.qkv'].double();width=qkv.shape[-1]//3;dim=width//heads
    q,k,v=[x.reshape(1,128,heads,dim).transpose(1,2)for x in qkv.split(width,-1)]
    scores=q@k.transpose(-1,-2)/math.sqrt(dim)
    legal=torch.ones(128,128,dtype=torch.bool).tril()[None,None]
    mask=state['masks'][layer]['prefix']
    if mask is not None:
        if mask.dtype==torch.bool:legal=legal&mask
        else:
            legal=legal&(mask>-1e30);scores=scores+mask.double()
    scores=scores.masked_fill(~legal,float('-inf'));prob=scores.softmax(-1)
    manual=(prob@v).transpose(1,2).reshape(1,128,width)
    visible=legal.expand_as(scores)
    assert torch.isfinite(scores[visible]).all()and torch.isfinite(prob).all()
    return scores[visible],prob,manual


def precision_settings():
    return {'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),'memory_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp_enabled':torch.backends.cuda.math_sdp_enabled(),'cudnn_sdp_enabled':torch.backends.cuda.cudnn_sdp_enabled()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--failed-run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out;failed=args.failed_run
    assert not out.exists()or not any(out.iterdir());out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir()
    cfg=json.loads(args.config.read_text());fm=json.loads((failed/'manifest.json').read_text())
    prefix=json.loads((failed/'prefix_checks.json').read_text());assert fm['status']=='failed'and len(prefix)==1
    assert sha(failed/'data_manifest.json')==fm['evaluation_inputs_sha256']
    event=prefix[0];assert event['trial']=='C'and event['identity']=='natural'and event['length']==512
    assert event['max_hidden_drift']>event['threshold']and event['threshold']==1e-3
    for name,h in fm['source_hashes'].items():assert sha(ROOT/name)==h,name
    items=torch.load(failed/'tokens.pt',map_location='cpu',weights_only=True)
    item=next(x for x in items if x['corpus']==event['corpus']and x['window']==event['window'])
    original=item['tokens'];assert original.shape==(512,)
    prior_raw=torch.load(failed/'per_token.pt',map_location='cpu',weights_only=True)
    for n in(128,512):
        prior_key=f"gate__{event['corpus']}__{event['window']}__{n}__natural__C"
        assert prior_raw[prior_key]['token_sha256']==th(original[:n])
    del prior_raw
    dm=json.loads((failed/'data_manifest.json').read_text());bos=dm['literals']['start_special']['id']
    passage=next(z for z in dm['passages']if z['corpus']==event['corpus']and z['window']==event['window'])
    assert passage['window_token_sha256']==th(original)
    rotated=original.clone();rotated[128:]=torch.roll(rotated[128:],1)
    constant=original.clone();constant[128:]=bos
    assert not torch.equal(rotated[128:],original[128:])and not torch.equal(constant[128:],original[128:])
    xs={'short128':original[:128].clone(),'full512':original.clone(),
        'suffix_rotated512':rotated,'suffix_constant512':constant,'padded512':constant.clone()}
    mask=torch.cat((torch.ones(128,dtype=torch.long),torch.zeros(384,dtype=torch.long)))
    torch.save({'inputs':xs,'padded_attention_mask':mask},out/'inputs.pt')
    old=json.loads((ROOT/'gpt2_run/manifest.json').read_text());selected=json.loads((ROOT/'gpt2_run/selection.json').read_text())['selected']
    assert sha(ROOT/'gpt2_run/selection.json')==old['selection_sha256']
    assert fm['revision']==old['revision']==cfg['models']['gpt2']['revision']
    sources=('native_prefix_diagnostic.py','GPT2_PREFIX_DIAGNOSTIC.md','run_transfer.py')
    started=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'status':'running','model_id':old['model_id'],'revision':old['revision'],
        'dtype':'float32','attention':'sdpa','seed':20261021,'expected_forwards':18,'cases':CASES,
        'source_hashes':{n:sha(ROOT/n)for n in sources},'frozen_study_source_hashes':fm['source_hashes'],
        'failed_run_manifest_sha256':sha(failed/'manifest.json'),'failed_prefix_checks_sha256':sha(failed/'prefix_checks.json'),
        'failed_tokens_sha256':sha(failed/'tokens.pt'),'failed_failure_state_sha256':sha(failed/'failure_state.pt'),
        'failed_native_event':event,'input_sha256':sha(out/'inputs.pt'),'original_token_sha256':th(original),
        'precision_settings_before':precision_settings(),'runtime_versions':{'python':platform.python_version(),
        'torch':str(torch.__version__),'transformers':transformers.__version__,'cuda_runtime':torch.version.cuda}}
    inventory=ROOT/'review/gpt2_gate_failure_inventory.json'
    if inventory.exists():meta['failed_bundle_inventory_sha256']=sha(inventory)
    snapshot=failed/'frozen_sources'
    if snapshot.exists():
        assert all(sha(snapshot/name)==h for name,h in fm['source_hashes'].items())
        meta['failed_source_snapshot_verified']=True
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(cfg['models']['gpt2']['path'],dtype=torch.float32,
        device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.requires_grad_(False)
    assert model.config.n_layer==12 and model.config.n_head==12 and model.config.n_embd==768
    before=modelhash(model);assert before==old['model_hash_before'];versions={n:p._version for n,p in model.named_parameters()}
    meta.update(model_hash_before=before,gpu_name=torch.cuda.get_device_name(),gpu_capability=list(torch.cuda.get_device_capability()))
    files={};operations=[];active=None;observer=None;study=None;writes=None;completed=0
    try:
        for backend,name,kind in CASES:
            active={'backend':backend,'name':name,'capture':kind};base=name.removeprefix('hookfree_').removeprefix('study_').removesuffix('_repeat')
            x=xs[base];assert torch.equal(x[:128],original[:128])
            if kind!='none':observer=PrefixProbe(model)
            if kind=='study':
                from gpt2_gate_subspace import GateSubspace
                from gpt2_layer_writes import LayerWrites
                writes=LayerWrites(model,selected,[447,138]);study=GateSubspace(model)
                writes.reset_layers(E=False,blocked=[],special=None);study.reset_gate(mode='C',zero=False,mask='native')
            ctx=sdpa_kernel(SDPBackend.MATH) if backend=='math'else contextlib.nullcontext()
            start=time.time()
            with ctx:
                settings=precision_settings()
                result=evaluate_native(model,x,mask if base=='padded512'else None,observer,profile=True)
            completed+=1
            if study is not None:
                assert writes.locality==30 and study.query_checks==study.context_checks==study.projection_checks==8
                result['frozen_study_context_errors']={l:d['context_max_abs_error']for l,d in study.diag.items()}
                study.close();writes.close();study=None;writes=None
            if observer is not None:observer.close();observer=None
            assert all(not mod._forward_hooks and not mod._forward_pre_hooks for mod in model.modules())
            key=f'{backend}__{name}';file=out/'cases'/(key+'.pt');torch.save(result,file);files[key]={'filename':file.name,'sha256':sha(file)}
            operations.append({**active,'seconds_including_observers':time.time()-start,'precision_settings':settings,
                'sdpa_profiler_events':result['sdpa_profiler_events'],'profiler_backend_observed':result['profiler_backend_observed']})
            dump(out/'case_files.json',files);dump(out/'operations.json',operations)
            meta['forward_passes']=completed;dump(out/'manifest.json',meta);del result
            print('PREFIX_DIAGNOSTIC',key,completed,flush=True)
        def load(key):return torch.load(out/'cases'/files[key]['filename'],map_location='cpu',weights_only=True)
        pairs=[]
        for backend in ('default','math'):
            for left,right in (('short128','short128_repeat'),('full512','full512_repeat'),('short128','full512'),
                               ('full512','suffix_rotated512'),('full512','suffix_constant512'),('short128','padded512'),
                               ('suffix_constant512','padded512')):pairs.append((backend+'__'+left,backend+'__'+right))
        pairs += [('default__'+v,'math__'+v)for v in VARIANTS]
        pairs += [('default__hookfree_'+v,'default__'+v)for v in('short128','full512')]
        pairs += [('default__'+v,'default__study_'+v)for v in('short128','full512')]
        assert len(pairs)==25
        comparisons=[];scorecomparisons=[];reconstructions=[]
        for a,b in pairs:
            left,right=load(a),load(b);comparisons.append({'reference':a,'trial':b,**compare(left,right)})
            if left['states']and right['states']:
                for l in range(12):
                    ls,lp,_=score_diagnostic(left,l,12);rs,rp,_=score_diagnostic(right,l,12)
                    scorecomparisons.append({'reference':a,'trial':b,'layer':l,
                        'visible_causal_scores_fp64':difference(ls,rs),'attention_probabilities_fp64':difference(lp,rp)})
            del left,right
        for key in files:
            result=load(key)
            for l in result['masks']:
                _,_,manual=score_diagnostic(result,l,12)
                reconstructions.append({'case':key,'layer':l,'actual_context_vs_fp64_prefix_reference':
                    difference(manual,result['states'][f'L{l}.context'])})
            del result
        oldraw=torch.load(failed/'per_token.pt',map_location='cpu',weights_only=True)
        failure=torch.load(failed/'failure_state.pt',map_location='cpu',weights_only=True)
        saved=failure['prefix_pairs']['C'];alignment=[]
        for n,field in ((128,'short_hidden'),(512,'long_prefix_hidden')):
            key=f"gate__{event['corpus']}__{event['window']}__{n}__natural__C";z=oldraw[key]
            assert z['token_sha256']==th(original[:n])
            for new in ('default__hookfree_', 'default__', 'default__study_', 'math__'):
                name=new+('short128'if n==128 else'full512');result=load(name)
                alignment.append({'saved_input_length':n,'case':name,'hidden':difference(saved[field],result['hidden']),
                    'loss':difference(z['loss'][1:127],result['loss'][1:]),
                    'argmax_agreement':float(z['pred'][1:127].eq(result['pred'][1:127]).double().mean())})
        dump(out/'comparisons.json',comparisons);dump(out/'score_comparisons.json',scorecomparisons)
        dump(out/'context_reconstructions.json',reconstructions);dump(out/'failed_run_alignment.json',alignment)
        after=modelhash(model);assert before==after and versions=={n:p._version for n,p in model.named_parameters()}
        assert precision_settings()==meta['precision_settings_before']
        assert meta['source_hashes']=={n:sha(ROOT/n)for n in sources}
        assert meta['frozen_study_source_hashes']=={n:sha(ROOT/n)for n in fm['source_hashes']}
        assert completed==18 and len(comparisons)==25 and len(scorecomparisons)==276 and len(reconstructions)==192
        meta.update(status='completed',completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=completed,
            model_hash_after=after,parameter_versions_unchanged=True,precision_settings_after=precision_settings(),
            wall_seconds=time.time()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            pairwise_comparisons=25,layer_score_comparisons=276,context_reconstructions=192)
        dump(out/'manifest.json',meta);print('COMPLETE_NATIVE_DIAGNOSTIC',completed,flush=True)
    except BaseException as error:
        torch.save({'active_case':active,'partial_states':None if observer is None else observer.states,
            'partial_masks':None if observer is None else observer.masks,
            'study_failure':None if study is None else study.failure_reconstruction},out/'failure_state.pt')
        dump(out/'case_files.json',files);dump(out/'operations.json',operations)
        meta.pop('completed_utc',None);meta.update(status='failed',forward_passes=completed,failure_type=type(error).__name__,
            failure_message=str(error),failed_utc=time.strftime('%FT%TZ',time.gmtime()))
        dump(out/'manifest.json',meta);raise
    finally:
        for obj in(observer,study,writes):
            if obj is not None:obj.close()


if __name__=='__main__':main()
