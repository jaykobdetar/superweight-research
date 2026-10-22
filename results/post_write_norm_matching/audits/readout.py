"""Prospective T20 saved-readout auditor; CPU-only, no model/analyzer imports.

Actual checkpoint payload access requires completed sequential segments and a
root execution review binding this exact source. --self-test is header-free.
Only wte.weight may be decoded. All writes are exclusive and failures retained.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import struct
import time
import traceback
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
ROWS=(1,126,127,510)
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched','orth_write_matched')
BOUNDARIES=('000','111')
PROTOCOL_SHA='e562fd57097228fb3348d8a7d564010456651c9dd6bde4540496a11bae338aff'
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
        assert len(set(archive.files))==len(archive.files),'duplicate archive member names'
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


def byte_exact(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def validate_embedding_header(header, payload_bytes):
    entry=header['wte.weight']
    assert entry['dtype']=='F32' and entry['shape']==[50257,768]
    begin,end=entry['data_offsets']
    assert type(begin) is int and type(end) is int
    assert 0<=begin<end<=payload_bytes and end-begin==50257*768*4
    return begin,end


def read_embedding(checkpoint):
    # Public safetensors layout. No model loader, torch state dict or GPU.
    with checkpoint.open('rb') as handle:
        prefix=handle.read(8);assert len(prefix)==8
        length=struct.unpack('<Q',prefix)[0]
        assert 0<length<16*1024**2
        header_bytes=handle.read(length);assert len(header_bytes)==length
        begin,end=validate_embedding_header(json.loads(header_bytes),checkpoint.stat().st_size-8-length)
        handle.seek(8+length+begin)
        values=np.fromfile(handle,dtype='<f4',count=50257*768)
        assert values.size==50257*768 and np.isfinite(values).all()
        return values.reshape(50257,768)


def validate_readout(saved,raw,tokens):
    hidden=saved['hidden'];logits=saved['logits'];idx=np.asarray(ROWS)
    assert hidden.shape==(1,512,768) and hidden.dtype==np.float32 and np.isfinite(hidden).all()
    assert logits.shape==(4,50257) and logits.dtype==np.float32 and np.isfinite(logits).all()
    assert saved['query_rows'].dtype==np.int64 and np.array_equal(saved['query_rows'],ROWS)
    assert saved['targets'].dtype==np.int64 and np.array_equal(saved['targets'],tokens[idx+1])
    assert saved['token_sha256']==array_sha(tokens)
    assert saved['hidden_fingerprint']['sha256']==raw['final_hidden_sha256']==array_sha(hidden)
    assert saved['hidden_fingerprint']['shape']==list(hidden.shape)
    assert saved['hidden_fingerprint']['dtype']==DTYPES[hidden.dtype]
    for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
        assert raw[key].shape==(511,) and np.isfinite(raw[key]).all()


def audit_state(saved,raw,tokens,embedding,reference_c,own):
    validate_readout(saved,raw,tokens)
    hidden=saved['hidden'];idx=np.asarray(ROWS)
    fp64,bound,gamma=embedding_readout(hidden[0,list(ROWS)],embedding)
    error=np.abs(saved['logits'].astype(np.float64)-fp64)
    assert np.all(error<=bound+1e-12),'outside conservative FP32 dot-product bound'
    actual=row_stats(saved['logits'],saved['targets'],reference_c)
    own_stats=row_stats(saved['logits'],saved['targets'],own['logits'])
    for computed,key in ((actual['pred'],'pred'),(actual['agreement_reference'],'agreement_c'),
                         (own_stats['agreement_reference'],'agreement_boundary')):
        assert np.array_equal(computed,raw[key][idx]),key
    assert byte_exact(hidden[:,0],own['hidden'][:,0]),'source hidden bytes'
    metric_errors={}
    for computed,key in ((actual['nll'],'loss'),(actual['kl_reference_to_trial'],'kl_c'),
                         (own_stats['kl_reference_to_trial'],'kl_boundary')):
        metric_errors[key]=drift(computed,raw[key][idx])
        assert metric_errors[key]['max_abs']<=RAW_METRIC_MAX_ABS,('saved metric mismatch',key)
    return {'nll':actual['nll'],'prediction':actual['pred'],'margin':actual['margin'],
        'kl_c_to_trial':actual['kl_reference_to_trial'],
        'kl_boundary_to_trial':own_stats['kl_reference_to_trial'],
        'saved_metric_errors':metric_errors,'readout_vs_cpu_fp64':drift(saved['logits'],fp64),
        'dot_product_bound':{'gamma_2n':gamma,'max_abs_bound':float(bound.max()),
          'max_error_to_bound_ratio':float(np.divide(error,bound,out=np.zeros_like(error),where=bound!=0).max())},
        'argmax_mismatches':mismatch_rows(saved['logits'],fp64),'source_hidden_byte_exact':True}


def validate_shard_contract(shard,boundary):
    assert shard['schema']=='post_write_norm_explicit_v1'
    assert tuple(shard['metadata'][k] for k in ('corpus','window','identity','boundary'))==('wiki_validation',0,'natural',boundary)
    assert set(shard['readouts'])==set(shard['snapshots'])==set(ARMS)
    expected_metrics=set(ARMS)|({'native_repeat','native_identity'} if boundary=='000' else set())
    assert set(shard['metrics'])==expected_metrics


def audit(args):
    started=time.time();root=args.run
    # Reviewed exact source is a mandatory gate before any checkpoint bytes.
    review=json.loads(args.execution_review.read_text())
    assert review['approved_readout_execution'] is True
    assert review['auditor_sha256']==file_sha(__file__) and review['protocol_sha256']==PROTOCOL_SHA
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['execution_type']=='planned_sequential_segments'
    for key,count in (('forward_passes',776),('forward_attempts',776),('fully_processed_forwards',776),
                      ('evaluation_forwards',384),('suffix_forwards',384),('anchor_forwards',8),('calibration_forwards',0)):
        assert manifest[key]==count,(key,manifest.get(key))
    assert review['run_manifest_sha256']==file_sha(root/'manifest.json')
    inventory=json.loads((root/'numeric_inventory.json').read_text())
    expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c in ('wiki_validation','utilitarianism','ethics','optimizers_new')
              for w in range(4) for i in ('natural','bos') for b in BOUNDARIES}
    assert {k for k in inventory if k.startswith('inputs/')}==expected
    # Inventory-byte validation streams every shard without decoding non-selected data.
    for key in sorted(expected):
        path=root/key;assert path.stat().st_size==inventory[key]['bytes'] and file_sha(path)==inventory[key]['sha256']
    tokens_file=root/'tokens.pt'
    assert file_sha(tokens_file)==manifest['input_dependencies']['tokens.pt']
    spec=json.loads(args.checkpoint_manifest.read_text())['gpt2']
    assert spec['model_id']=='openai-community/gpt2' and spec['revision']=='75e09b43581151bd1d9ef6700faa605df408979f'
    assert manifest['model_id']==spec['model_id'] and manifest['revision']==spec['revision']
    pin=spec['checkpoint_files']['model.safetensors']
    assert pin['sha256']=='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
    assert args.checkpoint.stat().st_size==pin['bytes'] and file_sha(args.checkpoint)==pin['sha256']
    embedding=read_embedding(args.checkpoint)
    # Only private current-study tokens, after full identity/completion gates.
    import torch
    torch.set_num_threads(1)
    records=torch.load(tokens_file,map_location='cpu',weights_only=True)
    tokens=next(x['tokens'].numpy() for x in records if x['corpus']=='wiki_validation' and x['window']==0)
    assert tokens.dtype==np.int64 and tokens.shape==(512,)
    data={}
    for boundary in BOUNDARIES:
        shard=read_lossless_npz(root/f'inputs/wiki_validation__0__natural__{boundary}.npz')
        assert shard['schema']=='post_write_norm_explicit_v1'
        assert tuple(shard['metadata'][k] for k in ('corpus','window','identity','boundary'))==('wiki_validation',0,'natural',boundary)
        assert shard['metadata']['length']==512 and shard['metadata']['token_sha256']==array_sha(tokens)
        validate_shard_contract(shard,boundary)
        data[boundary]=shard
    report={'schema':'T20_saved_readout_audit_v1','status':'running','auditor_sha256':file_sha(__file__),
      'protocol_sha256':PROTOCOL_SHA,'run_manifest_sha256':file_sha(root/'manifest.json'),
      'execution_review_sha256':file_sha(args.execution_review),'inventory_sha256':file_sha(root/'numeric_inventory.json'),
      'checkpoint_sha256':pin['sha256'],'embedding_sha256':array_sha(embedding),'weight_tensors_loaded':['wte.weight'],
      'head':'bias-free tied embedding dot product','query_rows':ROWS,'target_positions':[q+1 for q in ROWS],
      'counts':{'states':12,'full_vocabulary_rows_per_state':4,'full_vocabulary_rows':48},
      'saved_metric_max_abs':RAW_METRIC_MAX_ABS,'states':{},
      'scope':'Four predetermined full-vocabulary rows and full final hidden fingerprints per state. No ordinary context, hidden, or unsaved vocabulary rows independently reconstructed; runtime guards are provenance, not replayed.',
      'model_forwards':0,'gpu_computation':False,'file_deletions':0}
    for boundary in BOUNDARIES:
        shard=data[boundary];own=shard['readouts']['baseline']
        current=shard['readouts']['current']
        for key in ('hidden','logits'):assert byte_exact(current[key],own[key]),('baseline/current',boundary,key)
        for key in ('loss','pred','kl_c','kl_boundary','agreement_c','agreement_boundary'):
            assert byte_exact(shard['metrics']['current'][key],shard['metrics']['baseline'][key]),('baseline/current metric',boundary,key)
        for arm in ARMS:
            report['states'][boundary+'__'+arm]=audit_state(shard['readouts'][arm],shard['metrics'][arm],tokens,
                embedding,data['000']['readouts']['baseline']['logits'],own)
    report.update(status='passed',peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  elapsed_seconds=time.time()-started)
    assert report['peak_rss_bytes']<3*1024**3
    return json_value(report)


def self_test():
    # Header-free and checkpoint-free: ordinary dictionaries and tiny arrays only.
    values=np.array([[10000.,9999.,-10000.],[-10000.,-10000.,-10001.]])
    result=row_stats(values,np.array([0,2]),values)
    assert np.array_equal(result['pred'],[0,0]) and np.max(np.abs(result['kl_reference_to_trial']))==0
    h=np.array([[.1,.2,.3],[1.,-2.,3.]],dtype=np.float32)
    e=np.array([[1.,2.,3.],[-1.,.5,2.],[.2,-.5,.1],[1.,1.,1.]],dtype=np.float32)
    precise,bound,_=embedding_readout(h,e)
    assert np.all(np.abs((h@e.T).astype(np.float64)-precise)<=bound+1e-12)
    assert not byte_exact(np.array([0.],np.float32),np.array([-0.],np.float32))
    entry={'wte.weight':{'dtype':'F32','shape':[50257,768],'data_offsets':[0,50257*768*4]}}
    validate_embedding_header(entry,50257*768*4)
    rejected=0
    for bad in ({'transformer.wte.weight':entry['wte.weight']},
                {'wte.weight':dict(entry['wte.weight'],dtype='F16')},
                {'wte.weight':dict(entry['wte.weight'],data_offsets=[-1,50257*768*4-1])}):
        try:validate_embedding_header(bad,50257*768*4)
        except (AssertionError,KeyError):rejected+=1
    assert rejected==3
    sample=np.array([0.,-0.,np.nextafter(np.float32(0),np.float32(1))],np.float32)
    header={'schema':'wiki_gate_lossless_npz_v1','object':{'type':'tensor','array':'a0','shape':[3],
      'dtype':'torch.float32','sha256':array_sha(sample)}}
    container=io.BytesIO();np.savez_compressed(container,a0=sample,metadata=np.frombuffer(json.dumps(header).encode(),dtype=np.uint8));container.seek(0)
    assert byte_exact(read_lossless_npz(container),sample)
    # Exercise exact production shapes and each saved metric with a deterministic,
    # bounded allocation synthetic embedding. This is not a pretrained tensor.
    tokens=np.arange(512,dtype=np.int64)
    hidden=np.zeros((1,512,768),dtype=np.float32);hidden[:,:,0]=1
    embedding=np.zeros((50257,768),dtype=np.float32)
    embedding[:,0]=np.linspace(-1,1,50257,dtype=np.float32)
    logits=np.tile(embedding[:,0],(4,1))
    saved={'hidden':hidden,'logits':logits,'query_rows':np.array(ROWS,np.int64),
      'targets':tokens[np.array(ROWS)+1],'token_sha256':array_sha(tokens),
      'hidden_fingerprint':{'sha256':array_sha(hidden),'shape':list(hidden.shape),'dtype':'torch.float32'}}
    stats=row_stats(logits,saved['targets'],logits)
    raw={'final_hidden_sha256':array_sha(hidden),'loss':np.zeros(511,np.float32),
      'kl_c':np.zeros(511,np.float32),'kl_boundary':np.zeros(511,np.float32),
      'pred':np.full(511,50256,np.int64),'agreement_c':np.ones(511,np.bool_),
      'agreement_boundary':np.ones(511,np.bool_)}
    raw['loss'][list(ROWS)]=stats['nll']
    for boundary in BOUNDARIES:
        for arm in ARMS:audit_state(saved,raw,tokens,embedding,logits,saved)
    # Full selected-shard ABI: six snapshots/readouts, eight versus six metrics.
    shard_contract_rejected=0
    for boundary in BOUNDARIES:
        metrics={arm:raw for arm in ARMS}
        if boundary=='000':metrics.update(native_repeat=raw,native_identity=raw)
        shard={'schema':'post_write_norm_explicit_v1',
          'metadata':{'corpus':'wiki_validation','window':0,'identity':'natural','boundary':boundary},
          'readouts':{arm:saved for arm in ARMS},'snapshots':{arm:{} for arm in ARMS},'metrics':metrics}
        validate_shard_contract(shard,boundary)
        bad=dict(shard,metrics={**metrics,'unexpected_anchor':raw})
        try:validate_shard_contract(bad,boundary)
        except AssertionError:shard_contract_rejected+=1
        if boundary=='000':
            bad=dict(shard,metrics={arm:raw for arm in ARMS})
        else:bad=dict(shard,metrics={**metrics,'native_repeat':raw,'native_identity':raw})
        try:validate_shard_contract(bad,boundary)
        except AssertionError:shard_contract_rejected+=1
    assert shard_contract_rejected==4
    duplicate=io.BytesIO()
    import zipfile,warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        with zipfile.ZipFile(duplicate,'w') as z:
            for _ in range(2):
                a=io.BytesIO();np.save(a,np.array([1]));z.writestr('a.npy',a.getvalue())
    duplicate.seek(0)
    try:read_lossless_npz(duplicate)
    except AssertionError as error:assert str(error)=='duplicate archive member names'
    else:raise AssertionError('duplicate archive accepted')
    hostile=0
    bad_saved=dict(saved,targets=saved['targets']+1)
    bad_raw=dict(raw,loss=raw['loss'].copy());bad_raw['loss'][ROWS[0]]+=1
    bad_logits=dict(saved,logits=logits.copy());bad_logits['logits'][0,0]+=1
    for state,metrics in ((bad_saved,raw),(saved,bad_raw),(bad_logits,raw)):
        try:audit_state(state,metrics,tokens,embedding,logits,saved)
        except AssertionError:hostile+=1
    assert hostile==3
    print(json.dumps({'status':'passed','fixture':'header_free_synthetic_v1',
      'shard_metric_counts':[8,6],'shard_contract_rejections':shard_contract_rejected,'duplicate_archive_rejected':True,
      'synthetic_states_checked':12,'synthetic_vocabulary_rows_checked':48,'hostile_readouts_rejected':hostile,'pretrained_payload_read':False,
      'model_forwards':0,'gpu_computation':False,'rejected_header_contracts':rejected,
      'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','checkpoint','checkpoint-manifest','execution-review','out'):ap.add_argument('--'+name,type=Path)
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:self_test();return
    assert all((args.run,args.checkpoint,args.checkpoint_manifest,args.execution_review,args.out))
    assert not args.out.exists()
    try:exclusive_json(args.out,audit(args))
    except BaseException as error:
        exclusive_json(args.out.with_name(args.out.stem+'_failure_'+str(time.time_ns())+'.json'),
          {'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
           'auditor_sha256':file_sha(__file__),'scope':'Original run/checkpoint untouched; no deletion or replacement.'})
        raise

if __name__=='__main__':main()
