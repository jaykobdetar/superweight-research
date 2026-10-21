"""Independent T19 saved-readout audit. CPU only; no model/analyzer imports.

Requires a COMPLETED run. Loads only wte.weight from safetensors,
then audits eight predetermined saved arms at queries1,126,127,510. Full-vocab
GPU readouts remain authoritative; CPU FP64 multiplication diagnoses rounding.
Every output is exclusively created; no cleanup, deletion or replacement.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import struct
import time
import traceback
# Standalone process: set numerical-library thread limits before importing NumPy.
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np

ROWS=(1,126,127,510)
MODES=('000__baseline','000__remove','000__orth','111__baseline','111__remove',
       '111__orth','111__remove_scale','111__orth_scale')
BRIDGE_POLICY={'context_max_abs':1e-4,'target_loss_max_abs':1e-3,'passage_mean_abs':1e-5}
ROUNDTRIP_METRIC_TOLERANCE=1e-10
# Frozen before reading any T19 numeric outcomes; offline saved-metric consistency only.
RAW_METRIC_MAX_ABS=1e-4
DTYPES={np.dtype('float32'):'torch.float32',np.dtype('float64'):'torch.float64',
        np.dtype('int64'):'torch.int64',np.dtype('bool'):'torch.bool',np.dtype('uint8'):'torch.uint8'}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def exclusive_json(path,value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def read_lossless_npz(path):
    """Independent schema decoder; verifies every tensor identity and membership."""
    with np.load(path,allow_pickle=False) as archive:
        header=json.loads(archive['metadata'].tobytes())
        assert header['schema']=='wiki_gate_lossless_npz_v1'
        seen=set()
        def decode(node):
            kind=node['type']
            if kind=='tensor':
                name=node['array'];assert name not in seen;seen.add(name)
                value=archive[name]
                assert not value.dtype.hasobject
                assert list(value.shape)==node['shape'] and DTYPES[value.dtype]==node['dtype']
                assert array_sha(value)==node['sha256']
                return value
            if kind=='dict':
                pairs=[(decode(k),decode(v)) for k,v in node['items']]
                value=dict(pairs);assert len(value)==len(pairs)
                return value
            if kind in ('list','tuple'):
                value=[decode(item) for item in node['items']]
                return tuple(value) if kind=='tuple' else value
            assert kind=='scalar'
            return node['value']
        result=decode(header['object'])
        assert set(archive.files)==seen|{'metadata'}
        return result


def drift(left,right):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d=a-b;den=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)))
    return {'max_abs':float(np.max(np.abs(d))),'rms':float(np.sqrt(np.mean(d*d))),
            'symmetric_relative_l2':float(np.linalg.norm(d))/den if den else 0.}


def log_prob(logits):
    x=np.asarray(logits,dtype=np.float64)
    assert x.ndim==2 and np.isfinite(x).all()
    centered=x-np.max(x,axis=1,keepdims=True)
    return centered-np.log(np.exp(centered).sum(axis=1,keepdims=True))


def row_stats(logits,targets,reference=None):
    lp=log_prob(logits);prob=np.exp(lp)
    assert np.max(np.abs(prob.sum(1)-1))<1e-12
    prediction=np.argmax(logits,axis=1)
    top=np.partition(np.asarray(logits,dtype=np.float64),-2,axis=1)[:,-2:]
    margin=top.max(1)-top.min(1)
    result={'nll':-lp[np.arange(len(targets)),targets], 'pred':prediction,'margin':margin}
    if reference is not None:
        lr=log_prob(reference);pr=np.exp(lr)
        result['kl_reference_to_trial']=np.sum(pr*(lr-lp),axis=1)
        result['kl_trial_to_reference']=np.sum(prob*(lp-lr),axis=1)
        result['agreement_reference']=prediction==np.argmax(reference,axis=1)
    return result


def json_value(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(v) for v in value]
    return value


def mismatch_rows(left,right):
    a=np.asarray(left,dtype=np.float64);b=np.asarray(right,dtype=np.float64)
    pa,pb=np.argmax(a,1),np.argmax(b,1);out=[]
    for local in np.flatnonzero(pa!=pb):
        aa=np.partition(a[local],-2)[-2:];bb=np.partition(b[local],-2)[-2:]
        bound=float(np.max(np.abs(a[local]-b[local])))
        ma=float(aa.max()-aa.min());mb=float(bb.max()-bb.min())
        out.append({'query':ROWS[local],'left_id':int(pa[local]),'right_id':int(pb[local]),
                    'left_margin':ma,'right_margin':mb,'row_max_abs_drift':bound,
                    'compatible_with_observed_drift':ma<=2*bound+1e-12 and mb<=2*bound+1e-12})
    return out


def embedding_readout(hidden,embedding):
    """FP64 dot products with one FP32 embedding tensor, bounded temporary chunks."""
    h=np.asarray(hidden,dtype=np.float64)
    output=np.empty((len(h),len(embedding)),dtype=np.float64)
    magnitude=np.empty_like(output)
    for start in range(0,len(embedding),4096):
        stop=min(start+4096,len(embedding))
        block=np.asarray(embedding[start:stop],dtype=np.float64)
        output[:,start:stop]=h@block.T
        magnitude[:,start:stop]=np.abs(h)@np.abs(block).T
    # Conservative separate-product/addition FP32 dot-product bound. No fitted
    # tolerance and no claim that downstream sensitivity follows this bound.
    unit_roundoff=np.finfo(np.float32).eps/2
    gamma=(2*h.shape[1]*unit_roundoff)/(1-2*h.shape[1]*unit_roundoff)
    return output,gamma*magnitude,gamma


def audit(args):
    started=time.time();root=args.run
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['forward_passes']==2184
    execution_type=manifest.get('execution_type','single_uninterrupted_run')
    assert execution_type in ('single_uninterrupted_run','verified_composite_after_interruption')
    composite=execution_type=='verified_composite_after_interruption'
    assert manifest['evaluation_forwards']==1088 and manifest['suffix_forwards']==1088 and manifest['anchor_forwards']==8
    inventory=json.loads((root/'numeric_inventory.json').read_text())
    key='inputs/wiki_validation__0__natural.npz';path=root/key
    assert inventory[key]['sha256']==file_sha(path) and inventory[key]['bytes']==path.stat().st_size
    spec=json.loads(args.checkpoint_manifest.read_text())['gpt2']
    checkpoint=args.checkpoint
    assert checkpoint.stat().st_size==spec['checkpoint_files']['model.safetensors']['bytes']
    assert file_sha(checkpoint)==spec['checkpoint_files']['model.safetensors']['sha256']
    assert manifest['revision']==spec['revision'] and manifest['model_id']==spec['model_id']
    # No torch/weight access before the completed-run and file identity gates.
    import torch
    torch.set_num_threads(1)
    # Read only one tensor payload using the public safetensors container format.
    # Avoids a model loader and any dependency on the safetensors Python package.
    with checkpoint.open('rb') as handle:
        header_length=struct.unpack('<Q',handle.read(8))[0]
        assert 0<header_length<16*1024**2
        tensor_header=json.loads(handle.read(header_length))['wte.weight']
        assert tensor_header['dtype']=='F32' and tensor_header['shape']==[50257,768]
        begin,end=tensor_header['data_offsets']
        assert end-begin==50257*768*4
        handle.seek(8+header_length+begin)
        embedding_values=np.fromfile(handle,dtype='<f4',count=50257*768).reshape(50257,768)
    embedding=torch.from_numpy(embedding_values)
    assert embedding.dtype==torch.float32 and embedding.shape==(50257,768) and embedding.device.type=='cpu'
    embedding_array=embedding.numpy()
    data=read_lossless_npz(path)
    assert data['schema']=='native_value_explicit_v1'
    assert tuple(data['metadata'][k] for k in ('corpus','window','identity'))==('wiki_validation',0,'natural')
    assert set(data['readouts'])==set(data['snapshots'])==set(MODES)
    tokens_file=root/'tokens.pt';assert file_sha(tokens_file)==manifest['input_dependencies']['tokens.pt']
    token_records=torch.load(tokens_file,map_location='cpu',weights_only=True)
    tokens=next(item['tokens'].numpy() for item in token_records if item['corpus']=='wiki_validation' and item['window']==0)
    assert tokens.dtype==np.int64 and tokens.shape==(512,) and array_sha(tokens)==data['metadata']['token_sha256']
    report={'schema':'T19_independent_saved_readout_audit_v2','status':'running','preselected_input':data['metadata'],
        'scope':'Eight saved arms and four full-vocabulary readout queries; full final hidden/context and stored bridge target losses. Full511-row GPU logits were not saved and are not independently reconstructed.',
        'execution_provenance':{'execution_type':execution_type,'retained_forward_passes':manifest['forward_passes'],
            'scope':('Verified composite retains committed original and resumed evidence. The interrupted original worker lacks final parameter/precision guards; this audit does not assert a global uninterrupted-run guard pass. Compatibility-gate provenance belongs to the composite audit.' if composite else 'Completed single execution; run guards are retained as recorded, not independently replayed by this readout audit.')},
        'source_sha256':file_sha(__file__),'run_manifest_sha256':file_sha(root/'manifest.json'),
        'inventory_sha256':file_sha(root/'numeric_inventory.json'),'snapshot_shard_sha256':file_sha(path),
        'checkpoint_sha256':spec['checkpoint_files']['model.safetensors']['sha256'],
        'weight_tensors_loaded':['wte.weight'],'embedding_sha256':array_sha(embedding_array),
        'model_forwards':0,'gpu_computation':False,'cpu_threads':1,'file_deletions':0,
        'query_rows':ROWS,'target_positions':[q+1 for q in ROWS],
        'counts':{'arms':8,'full_vocabulary_rows_per_arm':4,'total_full_vocabulary_rows':32,
                  'independently_checked_bridges':2,'full_saved_hidden_queries_per_arm':512},
        'prospective_offline_saved_metric_max_abs':RAW_METRIC_MAX_ABS,
        'arms':{},'bridges':{}}
    for mode in MODES:
        saved=data['readouts'][mode];raw=data['metrics'][mode]
        assert saved['hidden'].shape==(1,512,768) and saved['hidden'].dtype==np.float32
        assert saved['logits'].shape==(4,50257) and saved['logits'].dtype==np.float32
        assert saved['query_rows'].dtype==np.int64 and np.array_equal(saved['query_rows'],ROWS)
        assert saved['targets'].dtype==np.int64 and np.array_equal(saved['targets'],tokens[np.asarray(ROWS)+1])
        assert saved['token_sha256']==array_sha(tokens)
        assert saved['hidden_fingerprint']['sha256']==raw['final_hidden_sha256']==array_sha(saved['hidden'])
        hidden_rows=saved['hidden'][0,list(ROWS)]
        fp64,bound,gamma=embedding_readout(hidden_rows,embedding_array)
        fp32=(torch.from_numpy(hidden_rows)@embedding.T).numpy()
        readout_error=np.abs(saved['logits'].astype(np.float64)-fp64)
        assert np.all(readout_error<=bound+1e-12),('outside conservative FP32 dot-product bound',mode)
        reference_c=data['readouts']['000__baseline']['logits']
        own=data['readouts'][mode[:3]+'__baseline']
        reference_b=own['logits']
        actual=row_stats(saved['logits'],saved['targets'],reference_c)
        own_stats=row_stats(saved['logits'],saved['targets'],reference_b)
        idx=np.asarray(ROWS)
        assert np.array_equal(actual['pred'],raw['pred'][idx])
        assert np.array_equal(actual['agreement_reference'],raw['agreement_c'][idx])
        assert np.array_equal(own_stats['agreement_reference'],raw['agreement_boundary'][idx])
        assert np.array_equal(saved['hidden'][:,0],own['hidden'][:,0]),('own-boundary source hidden drift',mode)
        for independent,stored in ((actual['nll'],raw['loss'][idx]),
                                   (actual['kl_reference_to_trial'],raw['kl_c'][idx]),
                                   (own_stats['kl_reference_to_trial'],raw['kl_boundary'][idx])):
            assert np.max(np.abs(independent-stored))<=RAW_METRIC_MAX_ABS,('saved metric mismatch',mode)
        report['arms'][mode]={'nll':actual['nll'],'prediction':actual['pred'],'margin':actual['margin'],
            'kl_c_to_trial':actual['kl_reference_to_trial'],'kl_trial_to_c':actual['kl_trial_to_reference'],
            'kl_boundary_to_trial':own_stats['kl_reference_to_trial'],'kl_trial_to_boundary':own_stats['kl_trial_to_reference'],
            'saved_fp32_metric_error':{'nll':drift(actual['nll'],raw['loss'][idx]),
                'kl_c':drift(actual['kl_reference_to_trial'],raw['kl_c'][idx]),
                'kl_boundary':drift(own_stats['kl_reference_to_trial'],raw['kl_boundary'][idx])},
            'readout_gpu_vs_cpu_fp64':drift(saved['logits'],fp64),'readout_gpu_vs_cpu_fp32':drift(saved['logits'],fp32),
            'fp32_dot_product_bound':{'gamma_2n':gamma,'max_abs_bound':float(bound.max()),
                'max_error_to_bound_ratio':float((readout_error/np.maximum(bound,np.finfo(np.float64).tiny)).max())},
            'gpu_vs_cpu_fp64_argmax_mismatches':mismatch_rows(saved['logits'],fp64),
            'own_boundary_source_hidden_exact':True}
    checks={check['mode']:check for check in data['checks'] if check.get('kind')=='arithmetic_bridge'}
    assert set(checks)=={'111__remove_scale','111__orth_scale'}
    assert manifest['bridge_policy']==BRIDGE_POLICY
    for family in ('remove','orth'):
        left='111__'+family;right=left+'_scale';recorded=checks[right]
        aa,bb=data['readouts'][left],data['readouts'][right]
        stats=row_stats(aa['logits'],aa['targets'],bb['logits'])
        reverse=stats['kl_reference_to_trial'];forward=stats['kl_trial_to_reference']
        idx=np.asarray(ROWS)
        assert np.max(np.abs(forward-recorded['kl_delta_to_scale'][idx]))<=ROUNDTRIP_METRIC_TOLERANCE
        assert np.max(np.abs(reverse-recorded['kl_scale_to_delta'][idx]))<=ROUNDTRIP_METRIC_TOLERANCE
        losses=data['metrics'][right]['loss'][1:].astype(np.float64)-data['metrics'][left]['loss'][1:].astype(np.float64)
        assert np.array_equal(losses,recorded['loss_difference'])
        context=drift(data['snapshots'][left]['delivered_context'],data['snapshots'][right]['delivered_context'])
        hidden=drift(aa['hidden'],bb['hidden'])
        for label,computed in (('delivered_context',context),('hidden',hidden)):
            for field in computed:assert abs(computed[field]-recorded[label][field])<=ROUNDTRIP_METRIC_TOLERANCE
        passing={'context_passed':context['max_abs']<=BRIDGE_POLICY['context_max_abs'],
            'target_loss_passed':float(np.abs(losses).max())<=BRIDGE_POLICY['target_loss_max_abs'],
            'passage_mean_passed':abs(float(losses.mean()))<=BRIDGE_POLICY['passage_mean_abs']}
        assert all(passing.values()) and all(passing[k]==recorded[k] for k in passing)
        mismatch=mismatch_rows(aa['logits'],bb['logits'])
        saved_mismatch={row['query']:row for row in recorded['argmax_mismatches'] if row['query'] in ROWS}
        assert set(saved_mismatch)=={row['query'] for row in mismatch}
        for row in mismatch:
            old=saved_mismatch[row['query']]
            assert row['left_id']==old['delta_id'] and row['right_id']==old['scale_id']
            assert abs(row['left_margin']-old['delta_margin'])<=ROUNDTRIP_METRIC_TOLERANCE
            assert abs(row['right_margin']-old['scale_margin'])<=ROUNDTRIP_METRIC_TOLERANCE
            assert abs(row['row_max_abs_drift']-old['row_max_abs_logit_drift'])<=ROUNDTRIP_METRIC_TOLERANCE
        report['bridges'][family]={'full_saved_context_drift':context,'full_saved_final_hidden_drift':hidden,
            'four_row_saved_logit_drift':drift(aa['logits'],bb['logits']),
            'scored_target_loss_max_abs':float(np.abs(losses).max()),'passage_mean_loss_difference':float(losses.mean()),
            'four_row_kl_delta_to_scale':forward,'four_row_kl_scale_to_delta':reverse,
            'four_row_argmax_mismatches':mismatch,'recorded_all_query_argmax_mismatches':recorded['argmax_mismatches'],
            'all_query_mismatches_scope':'Runtime records outside the four saved full-vocabulary rows were retained but are not independently checked.',
            'prospective_bridge_criteria':passing}
    report['maxima']={
        'gpu_vs_cpu_fp64_readout_max_abs':max(value['readout_gpu_vs_cpu_fp64']['max_abs'] for value in report['arms'].values()),
        'gpu_vs_cpu_fp32_readout_max_abs':max(value['readout_gpu_vs_cpu_fp32']['max_abs'] for value in report['arms'].values()),
        'saved_nll_max_abs_error':max(value['saved_fp32_metric_error']['nll']['max_abs'] for value in report['arms'].values()),
        'saved_kl_max_abs_error':max(value['saved_fp32_metric_error'][key]['max_abs'] for value in report['arms'].values() for key in ('kl_c','kl_boundary')),
        'bridge_context_max_abs':max(value['full_saved_context_drift']['max_abs'] for value in report['bridges'].values()),
        'bridge_scored_target_loss_max_abs':max(value['scored_target_loss_max_abs'] for value in report['bridges'].values()),
        'bridge_absolute_passage_mean_loss_difference':max(abs(value['passage_mean_loss_difference']) for value in report['bridges'].values())}
    report.update(status='passed',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  elapsed_seconds=time.time()-started)
    report['counts']['bridge_recorded_all_query_argmax_mismatches']=sum(len(v['recorded_all_query_argmax_mismatches']) for v in report['bridges'].values())
    report['counts']['bridge_independently_checked_argmax_mismatches']=sum(len(v['four_row_argmax_mismatches']) for v in report['bridges'].values())
    report['resources']={'peak_rss_bytes':report['peak_rss_bytes'],'elapsed_seconds':report['elapsed_seconds'],
                         'cpu_threads':1,'model_forwards':0,'gpu_computation':False}
    assert report['peak_rss_bytes']<3*1024**3
    return json_value(report)



def markdown_report(report):
    maximum=report['maxima'];counts=report['counts']
    lines=['# T19 independent saved-readout audit', '', '**Status: PASS**', '',
        f"Audited {counts['arms']} saved arms and {counts['total_full_vocabulary_rows']} full-vocabulary rows at queries1,126,127,510. "
        'Only the pinned GPT2 token-embedding tensor was read; no model was loaded or run and no GPU computation occurred.', '',
        f"Maximum saved-GPU versus CPU-FP64 readout error: {maximum['gpu_vs_cpu_fp64_readout_max_abs']:.8g}. "
        f"Maximum independently recomputed NLL/KL errors: {maximum['saved_nll_max_abs_error']:.8g}/{maximum['saved_kl_max_abs_error']:.8g} nats. "
        'Readout errors satisfy the conservative FP32 dot-product rounding bound. Saved predictions, agreements, targets, hidden fingerprints and own-boundary source hidden states were checked.', '',
        '| Bridge | Full saved context max error | Scored-target loss max difference | Absolute passage-mean loss difference |',
        '|---|---:|---:|---:|']
    for name,value in report['bridges'].items():
        lines.append(f"| {name} | {value['full_saved_context_drift']['max_abs']:.8g} | {value['scored_target_loss_max_abs']:.8g} | {abs(value['passage_mean_loss_difference']):.8g} |")
    lines+=['', 'Both bridges meet the original context/target-loss/passage-mean thresholds. Full saved hidden/context drift and all stored scored-target loss differences were independently recomputed. Both KL directions and argmax margins were independently checked at the four saved readout queries.', '',
        f"The runtime recorded {counts['bridge_recorded_all_query_argmax_mismatches']} bridge argmax mismatches across all queries; "
        f"{counts['bridge_independently_checked_argmax_mismatches']} fall in the independently checked four-row subset. "
        'Mismatch margins and compatibility with observed row drift are retained in the JSON. Runtime records outside this subset were not independently verified against full-vocabulary rows.', '',
        f"Peak RSS: {report['resources']['peak_rss_bytes']} bytes; CPU threads:1. "
        'This single predetermined Wiki input does not establish corpus-level efficacy or general native-value harmlessness. Finite boundary interactions do not identify mediation fractions or a unique circuit.', '']
    lines+=['Execution provenance: '+report['execution_provenance']['scope'], '']
    return '\n'.join(lines)


def self_test():
    values=np.array([[10000.,9999.,-10000.],[-10000.,-10000.,-10001.]],dtype=np.float64)
    stats=row_stats(values,np.array([0,2]),values)
    assert np.isfinite(stats['nll']).all() and np.array_equal(stats['pred'],[0,0])
    assert np.max(np.abs(stats['kl_reference_to_trial']))==0
    assert np.allclose(np.exp(log_prob(values)).sum(1),1,rtol=0,atol=1e-15)
    assert drift(np.zeros(2),np.zeros(2))['symmetric_relative_l2']==0
    sample=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
    header={'schema':'wiki_gate_lossless_npz_v1','object':{'type':'dict','items':[
        [{'type':'scalar','value':'fixture'},{'type':'tensor','array':'a0','shape':[2,2],
         'dtype':'torch.float32','sha256':array_sha(sample)}]]}}
    container=io.BytesIO()
    np.savez_compressed(container,a0=sample,metadata=np.frombuffer(json.dumps(header).encode(),dtype=np.uint8))
    container.seek(0)
    assert np.array_equal(read_lossless_npz(container)['fixture'],sample)
    synthetic_h=np.array([[.1,.2,.3],[1.,-2.,3.]],dtype=np.float32)
    synthetic_e=np.array([[1.,2.,3.],[-1.,.5,2.],[.2,-.5,.1],[1.,1.,1.]],dtype=np.float32)
    precise,bound,gamma=embedding_readout(synthetic_h,synthetic_e)
    assert np.all(np.abs((synthetic_h@synthetic_e.T).astype(np.float64)-precise)<=bound+1e-12)
    assert gamma>0
    print('PASS independent stable-softmax/KL/argmax/zero-norm CPU self-test; no model/weights/output data')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--checkpoint-manifest',type=Path);ap.add_argument('--out',type=Path)
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    assert all((args.run,args.checkpoint,args.checkpoint_manifest,args.out))
    assert not args.out.exists() and not args.out.with_suffix('.md').exists()
    try:
        report=audit(args)
        exclusive_json(args.out,report)
        with args.out.with_suffix('.md').open('x') as f:f.write(markdown_report(report))
        print('READOUT_AUDIT_PASSED',len(report['arms']),len(report['bridges']),report['peak_rss_bytes'])
    except BaseException as error:
        failure={'status':'failed','error_type':type(error).__name__,'error':str(error),
                 'traceback':traceback.format_exc(),'auditor_sha256':file_sha(__file__),
                 'scope':'Audit failure; original run/checkpoint/evidence untouched.'}
        destination=args.out.with_name(args.out.stem+'_failure_'+str(time.time_ns())+'.json')
        exclusive_json(destination,failure)
        raise

if __name__=='__main__':main()
