"""Reconstruct every actual T22 candidate independently; no model/GPU."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import copy,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
import pyarrow as pa
import pyarrow.ipc as ipc
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'l11_key_value_donor_inputs_v1'
CORPORA=('wiki_validation','utilitarianism','ethics','optimizers_new')
PHYSICAL=('wiki_validation','utilitarianism','ethics','adadelta','adamax','asgd')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def th(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
def textsha(s):return hashlib.sha256(s.encode()).hexdigest()
def overlap(lo,hi,a,b):return max(lo,a)<min(hi,b)
def grid(target):
    yield target
    for distance in range(1,1001):
        if target-distance>=0:yield target-distance
        if target+distance<=1000:yield target+distance
def main():
    start=time.monotonic();torch.set_num_threads(1)
    pre=read(OUT/'preflight.json');dm=read(OUT/'data_manifest.json');inv=read(OUT/'exclusion_inventory.json')
    assert pre['status']=='passed'and pre['model_forwards']==0 and pre['evaluation_passages']==16 and pre['evaluation_inputs']==32
    assert pre['script_sha256']==sha(ROOT/'l11_key_value_donor_inputs.py')
    assert pre['history_hash_count']==777 and pre['pretrained_loaded'] is False and pre['gpu_used'] is False
    for name,h in pre['output_sha256'].items():assert sha(OUT/name)==h
    assert {p.name for p in OUT.iterdir()}==set(pre['output_sha256'])|{'preflight.json'}
    assert dm['exclusion_inventory_sha256']==sha(OUT/'exclusion_inventory.json')
    package=ROOT/'query_write_timing_portable_package_candidate_v1';previous=ROOT/'gpt2_query_write_timing_composite_run';base=read(previous/'exclusion_inventory.json');template=read(package/'metadata/data_manifest.template.json')
    chain=read(OUT/'known_history_chain.json');review=read(OUT/'known_history_review.json')
    assert sha(OUT/'known_history_chain.json')=='2ac54ea99d9e991434d364e19ceb483440e8bfeb85b7213bf19061df9a70edef'
    assert sha(OUT/'known_history_review.json')=='453d5caee74c3730c40f8d87616322e55b8bd658800f99fa7510cd3893acfaff'
    assert review['status']=='passed' and review['no_unaccounted_model_experiments'] is True
    assert review['known_history_chain_sha256']==sha(OUT/'known_history_chain.json')
    assert review['proposal_sha256']==chain['proposal_sha256']==pre['proposal_sha256']=='6238ea83f978705b0069e3b476f3e867ba95f08fbae2864d6e92827b88e2f159'
    assert sha(package/'identity.json')=='bc58c7b4ee0d77e4d96363ba67678d9b7f154d328006a238daa210343dff8cfd'
    assert sha(previous/'manifest.json')==chain['inherited_bindings']['T21_completed_manifest_sha256']=='c4de3d3473eb93556bcdeb1161d1fc7d88f8c4e20c9fb38ad10b7096058299a9'
    prior=read(previous/'manifest.json');identity=read(package/'identity.json')
    assert prior['status']=='completed' and prior['all_final_guards_passed'] is True
    for name in ('data_manifest.json','exclusion_inventory.json','known_history_chain.json'):
        assert sha(previous/name)==prior['input_dependencies'][name]==identity['canonical_private_input_sha256'][name]
    assert sha(previous/'exclusion_inventory.json')==chain['inherited_bindings']['T21_exclusion_inventory_sha256']
    assert base['exclusions']==read(package/'metadata/exclusion_inventory.json')['exclusions']
    assert sha(previous/'data_manifest.json')==chain['inherited_bindings']['T21_data_manifest_sha256']=='f0ff4671117fb5b1f540dded7329a21fc640e524d9f08648c721b37156011dd8'
    prior_dm=read(previous/'data_manifest.json')
    assert chain['inherited_known_history_chain']==read(previous/'known_history_chain.json')
    assert sha(previous/'known_history_chain.json')==chain['inherited_bindings']['T21_known_history_chain_sha256']=='67106b29dc1991dc86c5f7af4e03e4accc7d09c18fcb50f1415f0f9dd71d103e'
    assert sha(package/'metadata/history_attestation.json')==chain['inherited_bindings']['T21_safe_history_attestation_sha256']
    for row in chain['T21_used_spans']:
        r=next(r for r in prior_dm['passages'] if (r['corpus'],r['window'])==(row['corpus'],row['window']))
        assert (row['char_start'],row['char_end'],row['token_sha256'],row['BOS_token_sha256'])==(r['char_start'],r['used_char_end'],r['window_token_sha256'],r['bos_window_token_sha256'])
    assert review['coordinator_attestation_sha256']=='a43749272a73f137449f7c399fa4b3126c2e6b6830604a1a9706c72be42a2947'
    assert sha(ROOT/'review/t22_coordinator_known_history_attestation_v1.json')==review['coordinator_attestation_sha256']
    assert sha(ROOT/'review/t22_history_build_execution_bookkeeping.json')==review['build_bookkeeping_sha256']=='7066fd980f9b5de9f20348631ca0807639fd550cccb2214381ca516e91d097c5'
    # Activation changes exactly the three reviewed HOLD/hash literals.
    import ast
    def activated_ast(text):
        tree=ast.parse(text)
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('PREPARATION_APPROVED','APPROVED_HISTORY_CHAIN_SHA','APPROVED_HISTORY_REVIEW_SHA') for x in node.targets):node.value=ast.Constant('REVIEWED_ACTIVATION_ONLY')
        return ast.dump(tree,include_attributes=False)
    original=ROOT/'review/t22_input_wrapper_before_selection.py'
    assert sha(original)=='db2d459b9650acbc0bdace44d0e93232b5449a93453a7cb676ce0b75c33c5127'
    assert sha(ROOT/'l11_key_value_donor_inputs.py')=='26dad5cd0fde2c6ee4c2bfa5c984e825fb50e30a197da4ceb7098539954914f4'
    assert activated_ast(original.read_text())==activated_ast((ROOT/'l11_key_value_donor_inputs.py').read_text())
    assert pre['known_history_chain_sha256']==inv['known_history_chain_sha256']==sha(OUT/'known_history_chain.json')
    assert pre['known_history_review_sha256']==inv['known_history_review_sha256']==sha(OUT/'known_history_review.json')
    exclusions=copy.deepcopy(base['exclusions'])
    for row in chain['T21_used_spans']:exclusions[row['corpus']].append(row)
    assert inv['exclusions']==exclusions and inv['retained_calibration_buffers']==base['retained_calibration_buffers']==chain['retained_calibration_buffers']
    assert len(inv['retained_calibration_buffers'])==4 and len(chain['T21_used_spans'])==16
    for b in inv['retained_calibration_buffers']:
        assert b['corpus']=='wiki_test' and b['char_end']-b['char_start']==8192
        assert any(r['char_start']<=b['char_start'] and b['char_end']<=r['char_end'] for r in exclusions['wiki_test'])
    history=set(chain['history_token_hashes']);old_history=set(base['history_token_hashes'])
    attested={(r['corpus'],r['window'],r['identity']):r['token_sha256'] for r in chain['T21_identity_attestations']}
    assert len(attested)==len(set(attested.values()))==32 and len(old_history)==745 and not old_history&set(attested.values())
    assert history==old_history|set(attested.values()) and len(history)==777 and sorted(history)==inv['history_token_hashes']
    assert inv['T21_identity_attestations']==chain['T21_identity_attestations'] and inv['T21_used_spans']==chain['T21_used_spans']
    assert inv['inherited_bindings']==chain['inherited_bindings']
    for c,rows in exclusions.items():
        for r in rows:assert 0<=r['char_start']<r['char_end']<=template['sources'][c]['characters']
    cfg=read(ROOT/'review/native_value_portable_private_config_draft_v1.json');texts={};sources=template['sources']
    for c in PHYSICAL:
        path=Path(cfg['sources'][c]);spec=sources[c];assert sha(path)==spec['file_sha256']
        if c=='wiki_validation':
            with pa.memory_map(str(path),'r')as source:table=ipc.open_stream(source).read_all()
            value='\n\n'.join(table.column('text').to_pylist())
        elif c in ('utilitarianism','ethics'):
            raw=path.read_text();lo=raw.index('*** START OF');hi=raw.rindex('*** END OF');value=raw[raw.index('\n',lo)+1:hi]
        else:value=path.read_text()
        assert len(value)==spec['characters']and textsha(value)==spec['processed_text_sha256'];texts[c]=value
    texts['optimizers_new']='\n\n'.join(texts[c]for c in ('adadelta','adamax','asgd'))
    assert textsha(texts['optimizers_new'])==sources['optimizers_new']['processed_text_sha256']
    components=[];cursor=0
    for c in ('adadelta','adamax','asgd'):
        components.append({'source':c,'char_start':cursor,'char_end':cursor+len(texts[c]),'file_sha256':sources[c]['file_sha256']});cursor+=len(texts[c])+2
    assert components==sources['optimizers_new']['components']
    assert dm['sources']==inv['sources']==sources
    modeldir=Path(cfg['models']['gpt2']['path'])
    for name,h in dm['tokenizer_sha256'].items():assert sha(modeldir/name)==h
    assert dm['tokenizer_sha256']==template['tokenizer_sha256']
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True);assert tok.is_fast
    bos=tok.bos_token_id;assert type(bos)is int and 0<=bos<50257
    literals={name:{'literal':value,'ids':tok.encode(value,add_special_tokens=False)}for name,value in [('the','The'),('space',' '),('newline','\n'),('double_newline','\n\n')]};literals['start_special']={'label':'bos','id':bos}
    assert dm['literals']==literals
    identity=read(package/'identity.json');assert hashlib.sha256(json.dumps(literals,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==identity['private_literals_commitment']
    inherited=frozenset(history);selected=[];rejections={};items=[];candidate_count=0
    for c in CORPORA:
        fresh=[];rejections[c]=[]
        for w,target in enumerate((125,375,625,875)):
            seen=set()
            for j in grid(target):
                a=(len(texts[c])-8192)*j//1000
                if a in seen:continue
                seen.add(a);candidate_count+=1;buffer=texts[c][a:a+8192];e=tok(buffer,add_special_tokens=False,return_offsets_mapping=True)
                reason=None;b=None;hits=[];component=None;natural=None;BOS=None
                if len(e['input_ids'])<512:reason='fewer_than512_tokens'
                else:
                    ids=e['input_ids'][:512];pairs=e['offset_mapping'][:512]
                    assert len(ids)==512 and all(type(x)is int and 0<=x<50257 for x in ids)
                    assert all(0<=lo<=hi<=8192 for lo,hi in pairs)
                    b=a+max(hi for lo,hi in pairs);natural=th(ids);BOS=th([bos]+ids[1:])
                    hits=[r for r in exclusions[c]if overlap(a,b,r['char_start'],r['char_end'])]
                    if hits:reason='historical_span_overlap'
                    elif any(overlap(a,b,lo,hi)for lo,hi in fresh):reason='new_panel_overlap'
                    elif natural in history:reason='historical_or_new_exact_token_window'
                    elif BOS in history:reason='historical_or_new_exact_BOS_token_window'
                    elif c=='optimizers_new':
                        valid=[v for v in components if v['char_start']<=a and b<=v['char_end']]
                        if len(valid)!=1:reason='used_span_crosses_component_boundary'
                        else:component=valid[0]['source']
                if reason:
                    rejections[c].append({'window':w,'preferred_numerator':target,'numerator':j,'char_start':a,'used_char_end':b,'tested_char_end':b,'reason':reason,'overlap_origins':sorted({r['origin']for r in hits}),'BOS_duplicate_scope':('historical'if BOS in inherited else'previously_selected')if reason=='historical_or_new_exact_BOS_token_window'else None});continue
                assert not any(overlap(a,b,r['char_start'],r['char_end'])for r in exclusions[c])
                fresh.append((a,b));history.update((natural,BOS));items.append({'corpus':c,'window':w,'ids':ids})
                selected.append({'corpus':c,'window':w,'fraction_numerator':j,'fraction_denominator':1000,'preferred_numerator':target,'char_start':a,'used_char_end':b,'extraction_char_end':a+8192,'passage_sha256':textsha(buffer),'window_token_sha256':natural,'first_token_id':ids[0],'bos_window_token_sha256':BOS,'identical_start_pair':natural==BOS,'code_component':component,'exclusion_extent':'used512_token_span'})
                break
            else:raise AssertionError('Grid exhausted')
    assert selected==dm['passages']and rejections==dm['rejections']and len(selected)==16
    stored=torch.load(OUT/'tokens.pt',weights_only=True,map_location='cpu');assert len(stored)==16
    for x,y,r in zip(items,stored,selected):
        assert (x['corpus'],x['window'])==(y['corpus'],y['window'])and y['tokens'].dtype==torch.int64 and y['tokens'].shape==(512,)
        assert y['tokens'].numpy().tobytes()==np.asarray(x['ids'],dtype=np.int64).tobytes()
        n=y['tokens'].clone();n[0]=bos;assert torch.equal(n[1:],y['tokens'][1:])and th(n.numpy())==r['bos_window_token_sha256']
    starts=read(OUT/'start_identity_summary.json')
    assert starts=={'historical_first_token_membership':'unavailable_not_used','natural_BOS_pairing_preserved':True,'start_intervention':'Replace token0 only; retain physical512 and tokens1..511','identical_start_pairs':[{'corpus':r['corpus'],'window':r['window'],'identical':r['identical_start_pair']}for r in selected]}
    assert dm['calibration_reference_manifest_sha256']==template['calibration_reference_manifest_sha256']and dm['calibration_reference_files']==template['calibration_reference_files']
    for name,h in pre['output_sha256'].items():assert sha(OUT/name)==h
    for c in PHYSICAL:assert sha(Path(cfg['sources'][c]))==sources[c]['file_sha256']
    report={'schema':'T22_actual_input_independent_audit_v1','status':'passed','audited_data_manifest_sha256':sha(OUT/'data_manifest.json'),'audit_source_sha256':sha(__file__),
        'actual_preparation_sha256':{p.name:sha(p)for p in OUT.iterdir()if p.is_file()},'candidates_independently_reconstructed':candidate_count,'accepted':16,'rejected':sum(map(len,rejections.values())),'identity_starts':32,'known_history_hashes':777,'T21_identity_attestations_verified':32,'retained_full8192_calibration_buffers':4,'all_inherited_exclusion_records_preserved':True,'all_selected_rejected_records_exact':True,'no_selected_overlap_or_code_component_crossing':True,'BOS_replacement_only_and_literal_commitment_verified':True,'calibration_map_bindings_unchanged':True,'historical_first_token_membership':'unavailable_not_used','model_forwards':0,'checkpoint_weights_read':False,'GPU_used':False,'file_deletions':0,'wall_seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'model_launch_approved':False,'candidate_count_scope':'Reconstructed every grid candidate through each accepted window; count is observed and compared through full selected/rejected records, not hardcoded to20 or747. Greedy earlier accepts alter later new-panel exclusions; count alone is not a monotone cross-study freshness metric.','history_scope':'Known coordinated recorded chain, not host-wide/pretraining proof; preserved BUILD bookkeeping exception remains in pinned review.'}
    with (ROOT/'review/t22_actual_input_independent_audit.json').open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    try:main()
    except BaseException as error:
        import traceback,datetime
        failure=ROOT/'review'/('t22_actual_input_audit_failure_'+str(time.time_ns())+'.json')
        with failure.open('x') as f:json.dump({'status':'failed','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'audit_source_sha256':sha(__file__),'model_forwards':0,'GPU_used':False,'file_deletions':0},f,indent=2);f.write('\n')
        raise
