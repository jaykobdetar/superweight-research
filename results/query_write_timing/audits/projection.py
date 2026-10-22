"""Prospective CPU T21 L10 Conv1D projection audit; no model forward.

Checkpoint access is unavailable without a later source/contract/decoder/run-
manifest bound execution review. Self-test uses synthetic arrays only.
"""
import argparse
import hashlib
import importlib.util
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

CONTRACT_SHA='1f5dbb80a5e082385ec4d8d9669de9f4068f720ad09fbcf6b069f05f736bf00c'
DECODER_SHA='f89320872515c133262d0bf1c4b55e152ef1a55e3d23265576040c6a88fd2b6b'
CHECKPOINT_SHA='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
CHECKPOINT_BYTES=548105171
HEADER_SHA='a41dc5e8c2f45cd5f533b4f081e7777de1f8faa3b2527134d4230ae531127b21'
PROTOCOL_SHA='a17e3e7e8150935950d58df1d2b266dac2bf91a6c65c4a8539d7e28fcf3e0819'
D=768
K=1537
U32=2.**-24
U64=2.**-53
N=2.**-126
K_E=4614


def up(value):return np.nextafter(np.asarray(value,dtype=np.float64),np.float64(np.inf))


def down(value):return np.nextafter(np.asarray(value,dtype=np.float64),np.float64(-np.inf))


def up_add(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.all(a>=0) and np.all(b>=0)
    return np.where((a==0)&(b==0),np.float64(0),up(a+b))


def up_multiply(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.all(a>=0) and np.all(b>=0)
    return np.where((a==0)|(b==0),np.float64(0),up(a*b))


def up_divide(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    assert np.all(a>=0) and np.all(b>0)
    return np.where(a==0,np.float64(0),up(a/b))


def down_positive(value):
    result=down(value);assert np.all(result>0)
    return result


GAMMA32=up_divide(K*U32,down_positive(1-K*U32))
GAMMA64=up_divide(K*U64,down_positive(1-K*U64))
GAMMA64_E=up_divide(K_E*U64,down_positive(1-K_E*U64))
ARMS=('baseline','current','remove_full','orth_full',
      'remove_write_matched_early','remove_write_matched_later','remove_write_matched_both',
      'orth_write_matched_early','orth_write_matched_later','orth_write_matched_both')
ENTRIES={
 'h.10.attn.c_proj.weight':{'dtype':'F32','shape':[768,768],'data_offsets':[178808832,181168128]},
 'h.10.attn.c_proj.bias':{'dtype':'F32','shape':[768],'data_offsets':[181168128,181171200]}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def exclusive_json(path,value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def load_decoder():
    path=Path(__file__).with_name('query_write_timing_readout_audit.py')
    assert sha(path)==DECODER_SHA,'reviewed decoder source changed'
    spec=importlib.util.spec_from_file_location('t20_reviewed_readout_decoder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def validate_header(header,payload_bytes):
    for key,expected in ENTRIES.items():
        assert header[key]==expected,('projection header mismatch',key)
        start,end=header[key]['data_offsets']
        assert 0<=start<end<=payload_bytes
        assert end-start==4*int(np.prod(header[key]['shape']))


def read_projection(checkpoint):
    # Invoked only after all source/review/run/file gates in audit().
    with checkpoint.open('rb') as handle:
        prefix=handle.read(8);assert len(prefix)==8
        length=struct.unpack('<Q',prefix)[0];assert 0<length<16*1024**2
        raw=handle.read(length);assert len(raw)==length
        assert hashlib.sha256(raw).hexdigest()==HEADER_SHA
        header=json.loads(raw);validate_header(header,checkpoint.stat().st_size-8-length)
        tensors={}
        for key,entry in ENTRIES.items():
            start,end=entry['data_offsets'];handle.seek(8+length+start)
            value=np.fromfile(handle,dtype='<f4',count=(end-start)//4)
            assert value.size==(end-start)//4 and np.isfinite(value).all()
            tensors[key]=value.reshape(entry['shape'])
    return tensors['h.10.attn.c_proj.weight'],tensors['h.10.attn.c_proj.bias']


def finite_f64(value,label):
    assert value.dtype==np.float64 and np.isfinite(value).all(),('nonfinite FP64 intermediate',label)
    return value


def validate_no_failure_paths(relative_paths):
    markers={'merge_failure.json','checkpoint_failure.json','failure_state.npz'}
    assert not any(Path(name).name in markers or Path(name).name.endswith('__partial.npz')
                   for name in relative_paths),'Preserved failure marker or partial shard'


def validate_precision_provenance(manifest):
    before=manifest['precision_settings_before']
    assert before['float32_matmul_precision']=='highest'
    assert before['cuda_matmul_allow_tf32'] is False
    assert manifest['precision_settings_after']==before
    assert manifest['all_final_guards_passed'] is True
    segments=manifest['original_segments'];assert set(segments)=={'A','B'}
    for label in ('A','B'):
        segment=segments[label]
        assert segment['all_final_guards_passed'] is True
        assert segment['precision_settings_before']==segment['precision_settings_after']==before
        for key in ('forward_attempts','forward_passes','fully_processed_forwards'):
            assert segment[key]==324
    return before


def projection_reference_bound(context,weight,bias):
    """Contract v2; directed positive FP64 enclosures, no arbitrary epsilon."""
    for value in (context,weight,bias):
        assert value.dtype==np.float32 and np.isfinite(value).all()
    assert context.ndim==2 and context.shape[1]==D
    assert weight.shape==(D,D) and bias.shape==(D,)
    x=context.astype(np.float64);w=weight.astype(np.float64);b=bias.astype(np.float64)
    ax=np.abs(x);aw=np.abs(w);ab=np.abs(b)
    reference=finite_f64(x@w,'reference dot')
    reference=finite_f64(reference+b,'reference bias add')
    magnitude=finite_f64(ax@aw,'magnitude dot')
    magnitude=finite_f64(magnitude+ab,'magnitude bias add')
    dx=np.where(ax<N,ax,0.)
    dw=np.where(aw<N,aw,0.)
    db=np.where(ab<N,ab,0.)
    flush=finite_f64(dx@aw,'flush dot1')
    flush=finite_f64(flush+finite_f64(ax@dw,'flush dot2'),'flush add1')
    flush=finite_f64(flush+finite_f64(dx@dw,'flush dot3'),'flush add2')
    flush=finite_f64(flush+db,'flush bias add')
    magnitude_upper=finite_f64(up_divide(magnitude,down_positive(1-GAMMA64)),'S upper')
    flush_upper=finite_f64(up_divide(flush,down_positive(1-GAMMA64_E)),'E upper')
    dot_budget=finite_f64(up_multiply(up_add(GAMMA32,GAMMA64),magnitude_upper),'dot budget')
    input_budget=finite_f64(up_multiply(up_add(1,GAMMA32),flush_upper),'input budget')
    additive_budget=finite_f64(up_divide(K*N,down_positive(1-K*U32)),'additive budget')
    bound=finite_f64(up_add(dot_budget,up_add(input_budget,additive_budget)),'coordinate bound')
    assert np.isfinite(reference).all() and np.isfinite(bound).all() and np.all(bound>=0)
    return reference,bound,flush_upper


def compare_projection(context,saved,weight,bias):
    assert saved.dtype==np.float32 and saved.shape==context.shape and np.isfinite(saved).all()
    reference,bound,flush=projection_reference_bound(context,weight,bias)
    difference=finite_f64(np.abs(saved.astype(np.float64)-reference),'coordinate difference')
    error=finite_f64(np.where(difference==0,np.float64(0),up(difference)),'coordinate error upper')
    failures=error>bound
    # Contract additive normal budget means bound is positive in practice;
    # retain explicit null policy for zero-bound coordinates.
    ratio=np.divide(error,bound,out=np.zeros_like(error),where=bound!=0)
    nonzero=bound!=0
    stats={'coordinates':int(error.size),'failed_coordinates':int(failures.sum()),
      'max_abs_error':float(error.max()),'rms_error':float(np.sqrt(np.mean(error*error))),
      'max_bound':float(bound.max()),'min_bound':float(bound.min()),
      'max_input_flush_budget_upper':float(flush.max()),
      'error_convention':'FP64 abs difference upward enclosed; exact zero retained',
      'max_error_to_bound_ratio':float(ratio[nonzero].max()) if nonzero.any() else None,
      'zero_bound_coordinates':int((~nonzero).sum()),
      'zero_bound_ratio':'null where bound zero','passed':not failures.any()}
    assert stats['passed'],json.dumps(stats)
    return stats


def validate_snapshot(snapshot):
    incoming=snapshot['incoming_reference']
    for value in (incoming['actual_context'],incoming['delivered_context'],
                  snapshot['baseline_write'],snapshot['proposed_full_write']):
        assert value.dtype==np.float32 and value.shape==(1,512,D) and np.isfinite(value).all()


def audit(args):
    started=time.time()
    assert sha(args.contract)==CONTRACT_SHA,'projection contract changed'
    contract=json.loads(args.contract.read_text())
    assert contract['protocol_sha256']==PROTOCOL_SHA and contract['weight_header']==ENTRIES
    review=json.loads(args.execution_review.read_text())
    assert review['approved_projection_execution'] is True
    assert review['auditor_sha256']==sha(__file__) and review['contract_sha256']==CONTRACT_SHA
    assert review['decoder_sha256']==DECODER_SHA and review['protocol_sha256']==PROTOCOL_SHA
    decoder=load_decoder()
    root=args.run;manifest=json.loads((root/'manifest.json').read_text())
    assert review['run_manifest_sha256']==sha(root/'manifest.json')
    assert manifest['status']=='completed' and manifest['execution_type']=='planned_sequential_segments'
    precision=validate_precision_provenance(manifest)
    validate_no_failure_paths(str(path.relative_to(root)) for path in root.rglob('*'))
    for key,count in (('forward_attempts',648),('forward_passes',648),('fully_processed_forwards',648),
      ('evaluation_forwards',320),('suffix_forwards',320),('anchor_forwards',8),('calibration_forwards',0)):
        assert manifest[key]==count,(key,manifest.get(key))
    inventory=json.loads((root/'numeric_inventory.json').read_text())
    expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c in ('wiki_validation','utilitarianism','ethics','optimizers_new')
      for w in range(4) for i in ('natural','bos') for b in ('000',)}
    assert {key for key in inventory if key.startswith('inputs/')}==expected
    for key in sorted(expected):
        path=root/key
        assert path.stat().st_size==inventory[key]['bytes'] and sha(path)==inventory[key]['sha256']
    spec=json.loads(args.checkpoint_manifest.read_text())['gpt2']
    assert manifest['model_id']==spec['model_id']=='openai-community/gpt2'
    assert manifest['revision']==spec['revision']=='75e09b43581151bd1d9ef6700faa605df408979f'
    checkpoint_pin=spec['checkpoint_files']['model.safetensors']
    assert checkpoint_pin['sha256']==CHECKPOINT_SHA and checkpoint_pin['bytes']==CHECKPOINT_BYTES
    assert args.checkpoint.stat().st_size==CHECKPOINT_BYTES and sha(args.checkpoint)==CHECKPOINT_SHA
    weight,bias=read_projection(args.checkpoint)
    report={'schema':'T21_offline_projection_audit_v1','status':'running',
      'auditor_sha256':sha(__file__),'contract_sha256':CONTRACT_SHA,'decoder_sha256':DECODER_SHA,
      'protocol_sha256':PROTOCOL_SHA,'execution_review_sha256':sha(args.execution_review),
      'run_manifest_sha256':sha(root/'manifest.json'),'inventory_sha256':sha(root/'numeric_inventory.json'),
      'checkpoint_sha256':CHECKPOINT_SHA,'header_sha256':HEADER_SHA,
      'weight_tensors_loaded':list(ENTRIES),'weight_sha256':array_sha(weight),'bias_sha256':array_sha(bias),
      'numeric_constants':{'D':D,'k':K,'u32':U32,'u64':U64,'N':N,'gamma32_upper':float(GAMMA32),'gamma64_upper':float(GAMMA64),'k_E':K_E,'gamma64_E_upper':float(GAMMA64_E)},
      'orientation':'context@weight+bias (Conv1D)',
      'precision_settings_before':precision,'segment_final_precision_guards_checked':['A','B'],
      'arithmetic_scope':{'assumptions':['Native GEMM uses full FP32 products/accumulation under recorded highest matmul precision and CUDA matmul TF32 disabled.',
        'Ordinary IEEE rounding/FTZ arithmetic with no nonfinite GPU intermediates; GPU intermediate finiteness is an assumption, not independently observed by this saved-array audit.',
        'CPU one-thread FP64 GEMM uses stated operation counts and separately grouped bias/E additions.'],
        'observed_checks':['Original X/W/b/saved output finite FP32.','All CPU FP64 reference/magnitude/flush/enclosure/bound/error intermediates finite.'],
        'cudnn_tf32_condition':'No condition; unrelated to Conv1D addmm.'},'model_forwards':0,'gpu_computation':False,
      'file_deletions':0,'states':{},'shards':{},
      'scope':'10 selected states, two pre-interpolation context projections each; repeated contexts are correlated. Actual matched returned-write interpolation belongs to the norm audit. No full-model/internal reconstruction.'}
    for boundary in ('000',):
        key=f'inputs/wiki_validation__0__natural__{boundary}.npz'
        shard=decoder.read_lossless_npz(root/key);decoder.validate_shard_contract(shard,boundary)
        report['shards'][boundary]=inventory[key]['sha256']
        for arm in ARMS:
            snapshot=shard['snapshots'][arm];validate_snapshot(snapshot)
            state={}
            for context_name,write_name in (('actual_context','baseline_write'),('delivered_context','proposed_full_write')):
                context=snapshot['incoming_reference'][context_name];saved=snapshot[write_name]
                state[context_name+'__'+write_name]=dict(compare_projection(context[0],saved[0],weight,bias),
                  context_sha256=array_sha(context),write_sha256=array_sha(saved))
            report['states'][boundary+'__'+arm]=state
    report.update(status='passed',states_checked=10,projection_comparisons=20,
      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,elapsed_seconds=time.time()-started)
    assert report['peak_rss_bytes']<3*1024**3
    return report


def self_test():
    tests=[];rejected=0
    from fractions import Fraction
    exact=lambda x:Fraction(float(x))
    for a,b in ((0.,0.),(1.,2.**-53),(2.**-126,2.**-149),(1e38,1e-38)):
        assert exact(up_add(a,b))>=exact(a)+exact(b)
        assert exact(up_multiply(a,b))>=exact(a)*exact(b)
        if b:assert exact(up_divide(a,b))>=exact(a)/exact(b)
    assert up_add(0,0)==0 and up_multiply(0,1)==0 and up_divide(0,1)==0
    assert exact(GAMMA32)>=Fraction(K)*exact(U32)/(1-Fraction(K)*exact(U32))
    assert exact(GAMMA64)>=Fraction(K)*exact(U64)/(1-Fraction(K)*exact(U64))
    assert exact(GAMMA64_E)>=Fraction(K_E)*exact(U64)/(1-Fraction(K_E)*exact(U64))
    tests.append('directed_scalar_enclosures_against_exact_fractions')
    settings={'float32_matmul_precision':'highest','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True}
    segment={'all_final_guards_passed':True,'precision_settings_before':settings,'precision_settings_after':settings,
      'forward_attempts':324,'forward_passes':324,'fully_processed_forwards':324}
    manifest={'precision_settings_before':settings,'precision_settings_after':settings,
      'all_final_guards_passed':True,'original_segments':{'A':segment,'B':segment}}
    validate_precision_provenance(manifest)
    import copy
    precision_rejected=0
    for modify in ('precision','tf32','segment_precision','segment_final','composite_final'):
        bad=copy.deepcopy(manifest)
        if modify=='precision':bad['precision_settings_before']=dict(settings,float32_matmul_precision='high')
        if modify=='tf32':bad['precision_settings_before']=dict(settings,cuda_matmul_allow_tf32=True)
        if modify=='segment_precision':bad['original_segments']['B']=dict(segment,precision_settings_after={})
        if modify=='segment_final':bad['original_segments']['B']=dict(segment,all_final_guards_passed=False)
        if modify=='composite_final':bad['all_final_guards_passed']=False
        try:validate_precision_provenance(bad)
        except AssertionError:precision_rejected+=1
    assert precision_rejected==5
    validate_no_failure_paths(['manifest.json','inputs/wiki_validation__0__natural__000.npz'])
    markers_rejected=0
    for name in ('merge_failure.json','checkpoint_failure.json','failure_state.npz','inputs/a__partial.npz'):
        try:validate_no_failure_paths(['manifest.json',name])
        except AssertionError:markers_rejected+=1
    assert markers_rejected==4
    for value in (np.array([np.nan],np.float64),np.array([np.inf],np.float64)):
        try:finite_f64(value,'hostile synthetic')
        except AssertionError:pass
        else:raise AssertionError('Nonfinite FP64 accepted')
    tests.append('precision_segment_final_and_failure_marker_hostile_guards')
    # Exact signed/asymmetric identity verifies orientation, bias and cancellation.
    x=np.zeros((3,D),np.float32);x[0,:3]=[1,-2,3];x[1,:3]=[-1,2,-3]
    w=np.eye(D,dtype=np.float32);w[0,1]=2;w[2,0]=-4
    b=np.zeros(D,np.float32);b[:3]=[.5,-.5,1]
    y=x@w+b
    compare_projection(x,y,w,b);tests.append('asymmetric_orientation_signed_bias')
    for wrong in (x@w.T+b,y+np.float32(1)):
        try:compare_projection(x,wrong,w,b)
        except AssertionError:rejected+=1
    # Full physical512*768 projection with exact coordinatewise identity.
    full=np.zeros((512,D),np.float32);full[:,0]=np.arange(512,dtype=np.float32)
    identity=np.eye(D,dtype=np.float32);zero=np.zeros(D,np.float32)
    compare_projection(full,full,identity,zero);tests.append('physical_512x768_projection')
    snapshot={'incoming_reference':{'actual_context':full[None],'delivered_context':full[None]},
      'baseline_write':full[None],'proposed_full_write':full[None]}
    for boundary in ('000',):
        for arm in ARMS:
            validate_snapshot(snapshot)
            for context_name,write_name in (('actual_context','baseline_write'),('delivered_context','proposed_full_write')):
                compare_projection(snapshot['incoming_reference'][context_name][0],snapshot[write_name][0],identity,zero)
    tests.append('10_snapshot_shape_contracts')
    # Explicit operand flush plus additive flush budget exercised independently.
    tiny=np.nextafter(np.float32(0),np.float32(1));sub=np.float32(2.**-127)
    sx=np.zeros((2,D),np.float32);sx[0,0]=sub;sx[1,0]=tiny
    sw=np.zeros((D,D),np.float32);sw[0,0]=1;sw[1,1]=sub
    sb=np.zeros(D,np.float32);sb[1]=sub
    ref,bound,flush=projection_reference_bound(sx,sw,sb)
    assert flush.max()>0 and np.all(bound>0)
    compare_projection(sx,np.zeros_like(sx),sw,sb);tests.append('subnormal_input_weight_bias_flush')
    compare_projection(np.zeros((2,D),np.float32),np.zeros((2,D),np.float32),np.zeros((D,D),np.float32),zero)
    tests.append('zero_reference_positive_additive_budget')
    rng=np.random.default_rng(20261021)
    rx=rng.normal(0,.05,(4,D)).astype(np.float32);rw=rng.normal(0,.05,(D,D)).astype(np.float32)
    rb=rng.normal(0,.01,D).astype(np.float32)
    compare_projection(rx,rx@rw+rb,rw,rb);tests.append('regular_fp32_matmul_rounding')
    # Exact rational selected-coordinate checks independently validate positive
    # magnitude and input-flush enclosure, including subnormal operands.
    for xx,ww,bb in ((rx,rw,rb),(sx,sw,sb)):
        ax=np.abs(xx.astype(np.float64));aw=np.abs(ww.astype(np.float64));ab=np.abs(bb.astype(np.float64))
        sf=ax@aw;sf=sf+ab
        su=up_divide(sf,down_positive(1-GAMMA64))
        dx=np.where(ax<N,ax,0.);dw=np.where(aw<N,aw,0.);db=np.where(ab<N,ab,0.)
        ef=dx@aw;ef=ef+ax@dw;ef=ef+dx@dw;ef=ef+db
        eu=up_divide(ef,down_positive(1-GAMMA64_E))
        for row,col in ((0,0),(0,1),(1,0),(1,1)):
            se=sum((exact(ax[row,j])*exact(aw[j,col]) for j in range(D)),exact(ab[col]))
            ee=sum((exact(dx[row,j])*exact(aw[j,col])+exact(ax[row,j])*exact(dw[j,col])+exact(dx[row,j])*exact(dw[j,col]) for j in range(D)),exact(db[col]))
            assert exact(su[row,col])>=se and exact(eu[row,col])>=ee
    tests.append('S_E_selected_coordinate_enclosures_against_exact_fractions')
    for args in ((x.astype(np.float64),y,w,b),(x,np.full_like(y,np.nan),w,b),
                 (x,y,np.full_like(w,np.inf),b)):
        try:compare_projection(*args)
        except AssertionError:rejected+=1
    for header in ({},dict(ENTRIES,**{'h.10.attn.c_proj.bias':dict(ENTRIES['h.10.attn.c_proj.bias'],dtype='F16')})):
        try:validate_header(header,CHECKPOINT_BYTES)
        except (AssertionError,KeyError):rejected+=1
    assert rejected==7
    return {'status':'passed','fixture':'T21_projection_header_free_v1_literal_layout','tests':tests,'hostile_cases_rejected':rejected,
      'precision_hostile_cases_rejected':precision_rejected,'failure_markers_rejected':markers_rejected,
      'states_shape_checked':10,'production_shape_projection_comparisons':20,'production_shape':[512,D],'synthetic_only':True,'checkpoint_headers_read':False,
      'pretrained_payload_read':False,'model_forwards':0,'gpu_computation':False,'file_deletions':0,
      'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','contract','execution-review','checkpoint','checkpoint-manifest','out'):
        ap.add_argument('--'+name,type=Path)
    ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    assert all((args.run,args.contract,args.execution_review,args.checkpoint,args.checkpoint_manifest,args.out))
    assert not args.out.exists()
    try:exclusive_json(args.out,audit(args))
    except BaseException as error:
        exclusive_json(args.out.with_name(args.out.stem+'_failure_'+str(time.time_ns())+'.json'),
          {'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
           'auditor_sha256':sha(__file__),'scope':'Original study/checkpoint untouched; failure retained.'})
        raise

if __name__=='__main__':main()
