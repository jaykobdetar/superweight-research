"""Frozen calibrated gate maps; see GPT2_GATE_SUBSPACE.md before execution."""
import argparse
import hashlib
import json
import math
import platform
import resource
import time
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gate_inputs import prepare_gate_inputs
from gpt2_layer_writes import LayerWrites
from gpt2_query_source import QuerySource
from run_transfer import ROOT, Runner, dump, modelhash, sha, th

LAYERS = tuple(range(4, 12))
RANKS = (1, 3, 6)
CONDITIONS = ('C', 'P_native', 'P', 'current', 'fullrank', 'meanlogit',
              'meanmass', 'position', 'global1') + tuple(
    f'{basis}_{action}{k}' for basis in ('pca', 'qr')
    for action in ('keep', 'remove') for k in RANKS)
SNAPSHOT_MODES = ('P', 'fullrank', 'meanlogit', 'meanmass', 'position',
                  'global1', 'pca_keep1', 'pca_remove1', 'qr_keep1', 'qr_remove1')
ERROR_TOLERANCE = 1e-4
PREFIX_TOLERANCE = 1e-3
QR_SEED = 20261025


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def maps_digest(maps):
    h = hashlib.sha256()
    for layer in sorted(maps):
        for name, x in sorted(maps[layer].items()):
            h.update(f'{layer}:{name}:{x.dtype}:{tuple(x.shape)}'.encode())
            h.update(x.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def spectrum(x):
    """Centered sample covariance, denominator N-1, and right singular vectors."""
    x = x.reshape(-1, 12).double()
    assert x.shape[0] > 12 and torch.isfinite(x).all()
    mean = x.mean(0)
    centered = x - mean
    cov = centered.T @ centered / (len(x) - 1)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh.T.contiguous()
    # Sign canonicalization does not change the subspace/projector.
    pivots = basis.abs().argmax(0)
    signs = torch.sign(basis[pivots, torch.arange(12)])
    signs[signs == 0] = 1
    basis = basis * signs
    eigenvalues = singular.square() / (len(x) - 1)
    direct = torch.linalg.eigvalsh(cov).flip(0)
    err = float((direct - eigenvalues).abs().max())
    ortho = float((basis.T @ basis - torch.eye(12, dtype=torch.float64)).abs().max())
    reconstruction = float((centered @ basis @ basis.T - centered).abs().max())
    assert err <= 1e-9 and ortho <= 1e-10 and reconstruction <= 1e-9
    total = float(eigenvalues.sum())
    explained = eigenvalues.cumsum(0) / total if total > 0 else torch.zeros_like(eigenvalues)
    report = {'vectors': len(x), 'covariance_denominator': len(x)-1,
              'mean': mean.tolist(), 'covariance': cov.tolist(),
              'singular_values': singular.tolist(), 'eigenvalues': eigenvalues.tolist(),
              'cumulative_explained_variance': explained.tolist(),
              'eigenvalue_svd_max_error': err, 'orthogonality_max_error': ortho,
              'full_reconstruction_max_error': reconstruction}
    return mean, basis, report


def fit_maps(calibration):
    """Eight equal-length passages; no evaluation data or outcome selection."""
    assert len(calibration) == 8
    generator = torch.Generator(device='cpu').manual_seed(QR_SEED)
    maps, reports = {}, {}
    logq = torch.arange(1, 511, dtype=torch.float64).log()[None, :, None]
    for layer in LAYERS:
        b = torch.stack([v[layer]['b'][:, :510].T for v in calibration.values()]).double()
        a = torch.stack([v[layer]['mass'][:, :510].T for v in calibration.values()]).double()
        assert b.shape == a.shape == (8, 510, 12)
        assert torch.isfinite(b).all() and torch.isfinite(a).all() and (a > 0).all()
        r = b - logq
        spaces = {'mass': a, 'b': b, 'r': r, 'position_centered_b': b-b.mean(0, keepdim=True)}
        specs = {name: spectrum(value) for name, value in spaces.items()}
        q, qr_r = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.float64))
        signs = torch.sign(qr_r.diag()); signs[signs == 0] = 1
        q = q * signs
        assert (q.T @ q-torch.eye(12, dtype=torch.float64)).abs().max() <= 1e-10
        maps[layer] = {'alpha': specs['r'][0], 'mean_b': specs['b'][0],
                       'mean_mass': specs['mass'][0], 'basis_r': specs['r'][1],
                       'basis_b': specs['b'][1], 'basis_qr': q}
        # Native summed mass may have tiny roundoff above1, but the fitted
        # constant gate must be in [0,1]. Never silently clip this mean.
        assert (maps[layer]['mean_mass'] > 0).all() and (maps[layer]['mean_mass'] <= 1).all()
        reports[layer] = {name: value[2] for name, value in specs.items()}
        reports[layer]['leave_one_passage_out'] = []
        for omit, name in enumerate(calibration):
            keep = [i for i in range(8) if i != omit]
            record = {'omitted_passage': name, 'spaces': {}}
            for space in ('mass', 'b', 'r', 'position_centered_b'):
                values = spaces[space][keep]
                if space == 'position_centered_b':
                    values = b[keep]-b[keep].mean(0, keepdim=True)
                _, ub, report = spectrum(values)
                report['principal_cosines_to_full'] = {
                    str(k): torch.linalg.svdvals(specs[space][1][:, :k].T @ ub[:, :k]).tolist()
                    for k in RANKS}
                record['spaces'][space] = report
            reports[layer]['leave_one_passage_out'].append(record)
    return maps, reports


def mapped_gate(mode, b, current_mass, mapping):
    """One row/query at a time across heads; never pool evaluation positions."""
    assert b.ndim == 2 and b.shape[0] == 12 and current_mass.shape == b.shape
    if mode in ('P', 'P_native', 'current'):
        return b, current_mass, 0., 0.
    logq = torch.arange(1, b.shape[1]+1, device=b.device, dtype=b.dtype).log()[:, None]
    x = b.T
    centered = x-logq-mapping['alpha']
    full_error = full_context_mass_error = 0.
    if mode == 'fullrank':
        u = mapping['basis_r']
        proposed = mapping['alpha']+logq+(centered @ u) @ u.T
        full_error = float((proposed-x).abs().max())
        full_context_mass_error = float((proposed.sigmoid().T-current_mass).abs().max())
        return proposed.T, current_mass, full_error, full_context_mass_error
    if mode == 'meanmass':
        return None, mapping['mean_mass'][:, None].expand_as(b), 0., 0.
    if mode == 'meanlogit':
        proposed = mapping['mean_b'][None, :].expand_as(x)
    elif mode == 'position':
        proposed = mapping['alpha']+logq
    elif mode == 'global1':
        u = mapping['basis_b'][:, :1]
        proposed = mapping['mean_b']+((x-mapping['mean_b']) @ u) @ u.T
    else:
        basis, operation = mode.split('_')
        assert basis in ('pca', 'qr')
        action = 'keep' if operation.startswith('keep') else 'remove'
        rank = int(operation[len(action):]); assert rank in RANKS
        u = mapping['basis_r' if basis == 'pca' else 'basis_qr'][:, :rank]
        part = (centered @ u) @ u.T
        proposed = mapping['alpha']+logq+(part if action == 'keep' else centered-part)
    return proposed.T, proposed.sigmoid().T, full_error, full_context_mass_error


class GateSubspace(QuerySource):
    def __init__(self, model):
        self.maps = None; self.device_maps = {}; self.calibration_sha256 = None
        self.gate_pending = {}; self.gate_cache = {}; self.gate_checks = {}
        self.mode = 'P'; self.failure_gate = None
        super().__init__(model)

    def install_maps(self, maps, digest):
        self.maps = maps; self.calibration_sha256 = digest; self.device_maps = {}

    def reset_gate(self, mode='P', zero=True, mask='causal', snapshot=False):
        self.reset_query(zero, 'present', mask, snapshot)
        self.mode = mode; self.gate_pending = {}; self.gate_cache = {}; self.gate_checks = {}
        self.failure_gate = None
        assert mode in CONDITIONS or mode == 'calibration'

    def combined(self, layer, output):
        output = super().combined(layer, output)
        if self.zero_values:
            q, k, v, mask = self.reconstruction_inputs[layer]
            n = q.shape[-2]
            logits = q @ k.transpose(-1, -2) / 8
            logits = logits.masked_fill(torch.ones(n,n,device=q.device,dtype=torch.bool).triu(1),float('-inf'))
            if mask is not None:
                logits = logits.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else logits+mask
            mass = logits.softmax(-1)[0, :, 1:, 1:].sum(-1)
            b = (logits[:, :, 1:, 1:].logsumexp(-1)-logits[:, :, 1:, 0])[0]
            stable = (logits[:, :, 1:, 1:].softmax(-1) @ v[:, :, 1:])[0]
            self.gate_pending[layer] = (b, mass, stable)
        return output

    def context(self, layer, actual):
        raw_inputs = self.reconstruction_inputs[layer]
        super().context(layer, actual)
        if not self.zero_values:
            return None
        b, a, u = self.gate_pending.pop(layer)
        n = actual.shape[1]
        c = actual.reshape(1,n,12,64)[0].transpose(0,1)[:,1:]
        mode = 'P' if self.mode == 'calibration' else self.mode
        active = mode not in ('P', 'P_native', 'current', 'fullrank')
        proposed_b, target, full_error, full_mass_error = b, a, 0., 0.
        if mode not in ('P', 'P_native', 'current'):
            assert self.maps is not None and self.calibration_sha256
            if layer not in self.device_maps:
                self.device_maps[layer] = {k: v.to(device=b.device,dtype=b.dtype) for k,v in self.maps[layer].items()}
            proposed_b, target, full_error, full_mass_error = mapped_gate(mode,b,a,self.device_maps[layer])
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
        good = good and math.isfinite(gain) and all(math.isfinite(v) for v in errors.values())
        check = {'min_mass':float(a.min()), 'max_mass':float(a.max()),
                 'min_b':float(b.min()), 'max_b':float(b.max()),
                 'min_target_mass':float(target.min()), 'max_target_mass':float(target.max()),
                 'min_target_b':None if proposed_b is None else float(proposed_b.min()),
                 'max_target_b':None if proposed_b is None else float(proposed_b.max()),
                 'max_gain':gain, 'target_zeros':int((target==0).sum()),
                 'target_ones':int((target==1).sum()), 'finite':good,
                 'source_context_exact':torch.equal(result[:,0],actual[:,0]), **errors}
        state = {'b':b.detach().cpu().clone(), 'mass':a.detach().cpu().clone(),
                 'target_b':None if proposed_b is None else proposed_b.detach().cpu().clone(),
                 'target_mass':target.detach().cpu().clone()}
        self.gate_cache[layer] = state; self.gate_checks[layer] = check
        if layer in self.snapshots:
            self.snapshots[layer].update(gate_state=state, stable_mixture=u.detach().cpu().clone(),
                                         delivered_context=result.detach().cpu().clone())
        if not good or max(errors.values())>ERROR_TOLERANCE or not check['source_context_exact']:
            self.failure_gate = {'layer':layer,'mode':self.mode,'calibration_sha256':self.calibration_sha256,
                'state':state,'check':check,'raw_inputs':{k:None if v is None else v.detach().cpu().clone()
                for k,v in zip(('q','k','v','mask'),raw_inputs)},'stable_mixture':u.detach().cpu().clone(),
                'actual_context':actual.detach().cpu().clone(),'delivered_context':result.detach().cpu().clone()}
            raise AssertionError(('gate diagnostic failed',layer,self.mode,check))
        return (result,) if active or mode in ('current','fullrank') else None


def add_p_metrics(z, logits, p_logits):
    """KL(P || trial), using the same chunking and direction as native-C KL."""
    vals=[]
    for start in range(0,len(logits),32):
        end=min(start+32,len(logits))
        lp=logits[start:end].float().log_softmax(-1)
        pp=p_logits[start:end].float().log_softmax(-1)
        vals.append((pp.exp()*(pp-lp)).sum(-1).cpu())
    z['kl_p']=torch.cat(vals);z['agreement_p']=z['pred'].eq(p_logits.argmax(-1).cpu())
    assert torch.isfinite(z['kl_p']).all()


class GateRunner(Runner):
    def rec(self,key,meta,x,z,b,cap,norms,score_index=1):
        """The inherited probe's query128 diagnostic is absent at length128."""
        assert key not in self.raw
        self.raw[key]={'token_sha256':th(x),'score_index':score_index,**z}
        self.traces[key]={'capture':cap.get('selected',{}),'norms':norms}
        zz={k:v[score_index:]for k,v in z.items()if k in('loss','pred','kl')}
        bb={k:v[score_index:]for k,v in b.items()if k in('loss','pred','kl')}
        d=cap.get('selected',{});weight=self.p.selected['weight'];aq=d.get('aq')
        row={**meta,'key':key,'token_sha256':th(x),'input_tokens':len(x),'score_index':score_index,
             'target_start':score_index+1,'scored_tokens':len(zz['loss']),
             'nll':float(zz['loss'].double().mean()),'baseline_nll':float(bb['loss'].double().mean()),
             'agreement':float(zz['pred'].eq(bb['pred']).double().mean()),'kl':float(zz['kl'].double().mean()),
             'source_contribution':d.get('a0',0)*weight,'q_contribution':None if aq is None else aq*weight}
        with(self.out/'windows.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out
    assert not out.exists() or not any(out.iterdir()), 'Output directory must be new or empty'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('gates','prefix'): (out/name).mkdir()
    parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'selection.json')==old['selection_sha256']
    selected=json.loads((parent/'selection.json').read_text())['selected'];assert selected['layer']==2 and selected['row']==447
    started=time.time();torch.set_num_threads(4);torch.manual_seed(20261021)
    path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    cal,items,dm=prepare_gate_inputs(cfg,tok,out);assert len(CONDITIONS)==21
    dump(out/'conditions.json',list(CONDITIONS))
    sources=('gpt2_gate_subspace.py','analyze_gate_subspace.py','analyze_transfer.py','gate_inputs.py','GPT2_GATE_SUBSPACE.md',
             'gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py','fresh_inputs_v2.py')
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],
          'dtype':'float32','attention':'sdpa','seed':20261021,'qr_seed':QR_SEED,'layers':list(LAYERS),
          'ranks':list(RANKS),'source_hashes':{n:sha(ROOT/n)for n in sources},
          'parent_manifest_sha256':sha(parent/'manifest.json'),'conditions_sha256':sha(out/'conditions.json'),
          'calibration_inputs_sha256':sha(out/'calibration_inputs.json'),'evaluation_inputs_sha256':sha(out/'data_manifest.json'),
          'context_tolerance':ERROR_TOLERANCE,'prefix_tolerance':PREFIX_TOLERANCE,'snapshot_modes':list(SNAPSHOT_MODES)}
    meta['runtime_versions']={'python':platform.python_version(),'torch':str(torch.__version__),
        'transformers':version('transformers'),'numpy':version('numpy'),'cuda_runtime':torch.version.cuda}
    meta['calibration_metrics_reference']='No external baseline; saved kl is Runner self-reference zero, not KL to native C.'
    dump(out/'manifest.json',meta)
    calstarts=[int(v['tokens'][0])for v in cal];evalstarts=[int(v['tokens'][0])for v in items];bos=dm['literals']['start_special']['id']
    dump(out/'start_identity_summary.json',{'calibration_first_token_ids':calstarts,'evaluation_natural_first_token_ids':evalstarts,
         'bos_id':bos,'evaluation_natural_matches_calibration':[x in set(calstarts)for x in evalstarts],
         'bos_seen_in_calibration':bos in set(calstarts),'note':'Natural/BOS are not assumed to be unseen token identities.'})
    torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    model.eval();model.requires_grad_(False);before=modelhash(model);assert before==old['model_hash_before']
    versions={n:p._version for n,p in model.named_parameters()}
    probe=LayerWrites(model,selected,[447,138]);gate=GateSubspace(model);run=GateRunner(model,probe,out)
    hidden={};ops=[];calops=[];calraw={};calmetrics={};snapshots={};source_states={}
    gate_files={};prefix_files={};prefix_checks=[];active_input={};input_gates={};prefix_pairs={};short={};maps=None;frozen_digest=None
    def capture_hidden(mod,a,o):hidden['tensor']=o.detach().cpu().clone()
    gate.handles.append(model.transformer.ln_f.register_forward_hook(capture_hidden))
    def ev(x,mode='P',zero=True,mask='causal',bl=None,pl=None,special=None,snapshot=False):
        probe.reset_layers(E=False,blocked=[],special=special);gate.reset_gate(mode,zero,mask,snapshot)
        hidden.clear();z,ll,cap,norms=run.ev(x,baseline=bl,keep=True);cap['selected']['writers']=probe.vectors
        h=hidden['tensor'];z['final_hidden_sha256']=tensor_hash(h)
        if pl is not None:add_p_metrics(z,ll,pl)
        assert probe.locality==30 and probe.writer_zero_checks==0
        assert gate.query_checks==gate.context_checks==gate.projection_checks==8
        assert len(gate.gate_checks)==(8 if zero else 0)
        bundle={'z':z,'cap':cap,'norms':norms,'hidden':h,'gates':gate.gate_cache,'snapshots':gate.snapshots,
                'source':{'kv':gate.cache,'q':{l:d['source_q']for l,d in gate.diag.items()}},
                'operation':{'projection_checks':gate.projection_checks,'query_checks':gate.query_checks,
                 'context_checks':gate.context_checks,'writer_locality_checks':probe.locality,
                 'gate_checks':gate.gate_checks,'max_context_error':max(d['context_max_abs_error']for d in gate.diag.values())}}
        return bundle,ll
    def equality(base,name,z,ref):
        check={**base,'trial':name,**{k+'_exact':torch.equal(z[k][1:],ref[k][1:])for k in('loss','kl','pred','kl_p','agreement_p')},
               'final_hidden_exact':z['final_hidden_sha256']==ref['final_hidden_sha256']}
        run.checks.append(check);assert all(v for k,v in check.items()if k.endswith('_exact')),check
    def checkpoint():
        run.checkpoint()
        if 'calibration_frozen_utc'not in meta:
            torch.save(calraw,out/'calibration_raw.pt');torch.save(calmetrics,out/'calibration_metrics.pt')
        dump(out/'calibration_operations.json',calops);torch.save(snapshots,out/'attention_snapshots.pt')
        torch.save(source_states,out/'source_states.pt');dump(out/'operation_checks.json',ops)
        dump(out/'gate_files.json',gate_files);dump(out/'prefix_files.json',prefix_files);dump(out/'prefix_checks.json',prefix_checks)
    try:
        stage=time.time();meta['phase']='calibration'
        for item in cal:
            active_input={'phase':'calibration','corpus':item['corpus'],'window':item['window'],'length':512,'identity':'natural','trial':'calibration'}
            key=f"{item['corpus']}__{item['window']}";bundle,ll=ev(item['tokens'],mode='calibration')
            calraw[key]={l:{k:d[k]for k in('b','mass')}for l,d in bundle['gates'].items()}
            calmetrics[key]={'token_sha256':th(item['tokens']),'kl_reference':'self/no external C reference',**bundle['z']};calops.append({**active_input,**bundle['operation']})
            del ll,bundle;checkpoint()
        meta['calibration_phase_seconds']=time.time()-stage
        stage=time.time();maps,spectra=fit_maps(calraw);torch.save(maps,out/'calibration_maps.pt');dump(out/'calibration_spectra.json',spectra)
        frozen_digest=maps_digest(maps);meta['calibration_maps_sha256']=sha(out/'calibration_maps.pt')
        meta['calibration_raw_sha256']=sha(out/'calibration_raw.pt');meta['calibration_maps_tensor_digest']=frozen_digest
        meta['calibration_fit_seconds']=time.time()-stage;meta['calibration_frozen_utc']=time.strftime('%FT%TZ',time.gmtime())
        meta['phase']='evaluation';dump(out/'manifest.json',meta);gate.install_maps(maps,meta['calibration_maps_sha256'])
        eval_started=time.time()
        for item in items:
            for identity in ('natural','bos'):
                whole=item['tokens'].clone()
                if identity=='bos':whole[0]=bos
                short={};prefix_pairs={}
                pairkey=f"gate__{item['corpus']}__{item['window']}__{identity}"
                for length in (128,512):
                    x=whole[:length].clone();base={'panel':'gate','corpus':item['corpus'],'window':item['window'],'length':length,'identity':identity}
                    key=f"gate__{item['corpus']}__{item['window']}__{length}__{identity}"
                    input_gates={};refs={};active_input={**base,'phase':'evaluation','trial':'C'}
                    cb,cl=ev(x,mode='C',zero=False,mask='native')
                    active_input={**base,'phase':'evaluation','trial':'P'}
                    snap=item['corpus']=='wiki_validation' and item['window']==0 and identity=='natural'
                    pb,pl=ev(x,bl=cl,snapshot=snap);add_p_metrics(cb['z'],cl,pl);add_p_metrics(pb['z'],pl,pl)
                    src=pb['source']
                    def record(trial,bundle):
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
                        if length==128:
                            short[trial]={'hidden':bundle['hidden'],'z':z,'gates':bundle['gates']}
                        else:
                            a=short[trial];h=bundle['hidden'][:,:128]
                            lossdrift=float((a['z']['loss'][1:]-z['loss'][1:127]).abs().max())
                            hiddendrift=float((a['hidden']-h).abs().max())
                            gate_b=max((float((a['gates'][l]['b']-d['b'][:,:127]).abs().max())for l,d in bundle['gates'].items()),default=0.)
                            gate_mass=max((float((a['gates'][l]['target_mass']-d['target_mass'][:,:127]).abs().max())for l,d in bundle['gates'].items()),default=0.)
                            check={**base,'trial':trial,'shared_scored_tokens':126,'max_loss_drift':lossdrift,'max_hidden_drift':hiddendrift,
                                   'max_gate_b_drift':gate_b,'max_target_mass_drift':gate_mass,
                                   'argmax_agreement':float(a['z']['pred'][1:].eq(z['pred'][1:127]).double().mean()),'threshold':PREFIX_TOLERANCE}
                            prefix_checks.append(check)
                            prefix_pairs[trial]={'short_hidden':a['hidden'],'long_prefix_hidden':h.clone()}
                            assert all(math.isfinite(v)for v in(lossdrift,hiddendrift,gate_b,gate_mass))
                            assert lossdrift<=PREFIX_TOLERANCE and hiddendrift<=PREFIX_TOLERANCE,check
                    record('C',cb);record('P',pb)
                    for trial in CONDITIONS:
                        if trial in ('C','P'):continue
                        active_input={**base,'phase':'evaluation','trial':trial}
                        bundle,ll=ev(x,mode=trial,mask='native'if trial=='P_native'else'causal',bl=cl,pl=pl,
                                     snapshot=snap and trial in SNAPSHOT_MODES)
                        record(trial,bundle)
                        if trial in ('current','fullrank'):equality(base,trial+'_closure',bundle['z'],pb['z'])
                        del ll,bundle
                    run.checks.append({**base,'trial':'mask_reference','max_loss_drift':float((refs['P']['loss'][1:]-refs['P_native']['loss'][1:]).abs().max()),
                                       'argmax_agreement':float(refs['P']['pred'][1:].eq(refs['P_native']['pred'][1:]).double().mean())})
                    if identity=='natural'and item['window']==0:
                        for trial,special in(('repeat',None),('identity','identity')):
                            active_input={**base,'phase':'evaluation','trial':trial}
                            bundle,ll=ev(x,mode='C',zero=False,mask='native',bl=cl,pl=pl,special=special)
                            record(trial,bundle);equality(base,trial,bundle['z'],cb['z']);del ll,bundle
                    gp=out/'gates'/(key+'.pt');torch.save(input_gates,gp);gate_files[gp.name]=sha(gp)
                    del cl,pl,cb,pb;input_gates={}
                pp=out/'prefix'/(pairkey+'.pt');torch.save(prefix_pairs,pp);prefix_files[pp.name]=sha(pp)
                short={};prefix_pairs={};checkpoint()
            print('GATE_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta['evaluation_phase_seconds']=time.time()-eval_started
        assert maps_digest(maps)==frozen_digest and sha(out/'calibration_maps.pt')==meta['calibration_maps_sha256']
        assert sha(out/'calibration_raw.pt')==meta['calibration_raw_sha256']
        assert meta['source_hashes']=={n:sha(ROOT/n)for n in sources}
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),phase='completed',forward_passes=run.count,
            calibration_forwards=8,evaluation_forwards=run.count-8,records=len(run.raw),wall_seconds=time.time()-started,
            model_hash_before=before,model_hash_after=modelhash(model),parameter_versions_unchanged=versions=={n:p._version for n,p in model.named_parameters()},
            calibration_maps_unchanged=maps_digest(maps)==frozen_digest,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==1368 and len(run.raw)==1360 and len(run.checks)==1488 and len(prefix_checks)==680
        assert len(gate_files)==64 and len(prefix_files)==32 and before==meta['model_hash_after']and meta['parameter_versions_unchanged']
        dump(out/'manifest.json',meta);print('COMPLETE',run.count,meta['wall_seconds'],flush=True)
    except BaseException as error:
        checkpoint()
        torch.save({'active_input':active_input,'calibration_sha256':gate.calibration_sha256,'gate':gate.failure_gate,
                    'reconstruction':gate.failure_reconstruction,'current_gates':gate.gate_cache,'current_snapshots':gate.snapshots,
                    'current_source':gate.cache,'current_input_gates':input_gates,'prefix_pairs':prefix_pairs,'short_prefix_states':short,
                    'current_hidden':hidden.get('tensor')},out/'failure_state.pt')
        meta.pop('completed_utc',None);meta.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),
            failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw))
        dump(out/'manifest.json',meta);raise
    finally:gate.close();probe.close()


if __name__=='__main__':main()
