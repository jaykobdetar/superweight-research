"""Query-bias by source-access factorial under zero source values."""
import argparse,json,math,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Runner
from fresh_inputs_v2 import prepare_fresh
from gpt2_layer_writes import LayerWrites
from gpt2_regrowth_kv import GPT2KV

class QuerySource(GPT2KV):
    def __init__(self,m):
        self.qbias={};self.random={};self.original_bias={};self.diag={};self.pending={};self.snapshots={};self.save_snapshot=False
        self.query_mode='present';self.zero_values=False;self.effective_masks={};self.query_checks=0;self.context_checks=0
        self.reconstruction_inputs={};self.failure_reconstruction=None
        super().__init__(m)
        g=torch.Generator(device='cpu').manual_seed(20261024)
        for i in range(4,12):
            att=m.transformer.h[i].attn
            assert att.num_heads==12 and att.head_dim==64 and att.scale_attn_weights and not att.scale_attn_by_inverse_layer_idx and not att.reorder_and_upcast_attn
            b=att.c_attn.bias[:768].detach().clone();self.qbias[i]=b;cpu=b.cpu().reshape(12,64);self.original_bias[i]=cpu.clone()
            perm=torch.stack([torch.randperm(64,generator=g) for _ in range(12)]);sign=torch.randint(0,2,(12,64),generator=g).float()*2-1
            rand=cpu.gather(1,perm)*sign;assert torch.equal(rand.abs().sort(dim=1).values,cpu.abs().sort(dim=1).values)
            self.random[i]={'permutation':perm,'sign':sign,'bias':rand,'device_bias':rand.flatten().to(b.device)}
            self.handles.append(att.c_proj.register_forward_pre_hook(lambda mod,a,i=i:self.context(i,a[0])))

    def reset_query(self,zero=False,mode='present',mask='native',snapshot=False):
        assert mode in ('present','removed','random');super().reset(mask=mask)
        self.zero_values=zero;self.query_mode=mode;self.diag={};self.pending={};self.snapshots={};self.save_snapshot=snapshot
        self.effective_masks={};self.query_checks=self.context_checks=0
        self.reconstruction_inputs={};self.failure_reconstruction=None

    def attn(self,m,args,kwargs):
        result=super().attn(m,args,kwargs);use=kwargs if result is None else result[1]
        self.effective_masks[m.layer_idx]=use.get('attention_mask');return result

    def combined(self,i,o):
        old=o;o=o.clone() if self.zero_values or self.query_mode!='present' else o
        if self.zero_values:o[0,0,1536:]=0
        if self.query_mode!='present':
            bias=self.qbias[i] if self.query_mode=='removed' else self.random[i]['device_bias']
            o[:,1:,:768]=o[:,1:,:768]-bias
        assert torch.equal(o[:,0,:1536],old[:,0,:1536]) and torch.equal(o[:,:,768:1536],old[:,:,768:1536])
        assert torch.equal(o[:,1:,1536:],old[:,1:,1536:])
        if self.zero_values:assert torch.count_nonzero(o[0,0,1536:])==0
        if self.query_mode=='present':assert torch.equal(o[:,:,:768],old[:,:,:768])
        else:assert torch.equal(o[:,1:,:768],old[:,1:,:768]-bias)
        self.query_checks+=1;o=super().combined(i,o)
        n=o.shape[1];q,k,v=[z.reshape(1,n,12,64).transpose(1,2) for z in o.split(768,dim=-1)]
        scores=torch.matmul(q,k.transpose(-1,-2))/8
        causal=torch.ones((n,n),device=o.device,dtype=torch.bool).triu(1);scores=scores.masked_fill(causal,float('-inf'))
        mask=self.effective_masks[i]
        if mask is not None:
            scores=scores.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else scores+mask
        prob=scores.softmax(-1);expected=torch.matmul(prob,v).transpose(1,2).reshape(1,n,768)
        source=prob[:,:,:,:1]*v[:,:,:1,:]
        nonsource=torch.matmul(prob[:,:,:,1:],v[:,:,1:,:])
        if self.zero_values:assert torch.count_nonzero(source)==0
        if self.mask=='blocked':assert torch.count_nonzero(prob[:,:,1:,0])==0
        self.diag[i]={'source_mass':prob[0,:,:,0].detach().cpu().clone(),
                      'source_vector_norm':source[0].norm(dim=-1).detach().cpu(),
                      'nonsource_vector_norm':nonsource[0].norm(dim=-1).detach().cpu(),
                      'source_q':q[0,:,0].detach().cpu().clone()}
        self.pending[i]=expected
        self.reconstruction_inputs[i]=(q,k,v,mask)
        if self.save_snapshot and i in (4,11):
            assert mask is not None
            self.snapshots[i]={'q':q.detach().cpu().clone(),'k':k.detach().cpu().clone(),'v':v.detach().cpu().clone(),'mask':mask.detach().cpu().clone()}
        return o

    def context(self,i,actual):
        expected=self.pending.pop(i);err=float((actual-expected).abs().max())
        q,k,v,mask=self.reconstruction_inputs.pop(i)
        self.diag[i]['context_max_abs_error']=err;self.context_checks+=1
        if i in self.snapshots:self.snapshots[i]['actual_context']=actual.detach().cpu().clone()
        if not math.isfinite(err) or err>1e-4:
            self.failure_reconstruction={'layer':i,'q':q.detach().cpu().clone(),'k':k.detach().cpu().clone(),
                'v':v.detach().cpu().clone(),'mask':None if mask is None else mask.detach().cpu().clone(),
                'actual_context':actual.detach().cpu().clone(),'expected_context':expected.detach().cpu().clone(),
                'max_abs_error':err,'threshold':1e-4}
        assert math.isfinite(err) and err<=1e-4,(i,'SDPA diagnostic mismatch',err)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists()
    parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256']
    s=json.loads((parent/'selection.json').read_text())['selected'];assert s['layer']==2 and s['row']==447
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    previous=('gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run','gpt2_regrowth_kv_run','gpt2_value_geometry_run')
    items,dm=prepare_fresh(cfg,tok,parent,out,fractions=(.025,.225,.425,.625),additional_parents=tuple(ROOT/r for r in previous))
    conditions=[(w,w,False,'present','native') for w in 'CEB']
    conditions += [(w+'_zero_native',w,True,'present','native') for w in 'CEB']
    conditions += [(f'{w}_zero_{q}_{mask}',w,True,q,mask) for w in 'CEB' for q in ('present','removed','random') for mask in ('causal','blocked')]
    assert len(conditions)==24;dump(out/'conditions.json',conditions)
    names=('gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py','fresh_inputs_v2.py','GPT2_QUERY_SOURCE.md')
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa',
          'seed':20261021,'random_control_seed':20261024,'coordinates':[447,138],'layers':list(range(4,12)),
          'source_hashes':{n:sha(ROOT/n) for n in names},'parent_manifest_sha256':sha(parent/'manifest.json'),'conditions_sha256':sha(out/'conditions.json')}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()}
    p=LayerWrites(m,s,[447,138]);qs=QuerySource(m);run=Runner(m,p,out);ops=[];diagnostics={};snapshots={};source_states={}
    torch.save({'native_bias':qs.original_bias,'random':{l:{k:v for k,v in d.items() if k!='device_bias'} for l,d in qs.random.items()}},out/'bias_controls.pt')
    meta['bias_controls_sha256']=sha(out/'bias_controls.pt');dump(out/'manifest.json',meta)
    def ev(x,world,zero=False,mode='present',mask='native',special=None,bl=None,keep=False,snapshot=False):
        p.reset_layers(E=world!='C',blocked=[3] if world=='B' else [],special=special);qs.reset_query(zero,mode,mask,snapshot)
        z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);cap['selected']['writers']=p.vectors
        assert p.locality==30 and p.writer_zero_checks==(world=='B') and qs.query_checks==qs.context_checks==qs.projection_checks==8
        return z,ll,cap,norms
    try:
        for item in items:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'panel':'query_source','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity}
                key=f"query_source__{item['corpus']}__{item['window']}__512__{identity}";refs={};ss={}
                b,bl,bc,bn=ev(x,'C',keep=True)
                def record(trial,world,z,cap,norms):
                    full=key+'__'+trial;run.rec(full,{**base,'trial':trial},x,z,b,cap,norms);refs[trial]=z
                    diagnostics[full]=qs.diag
                    source_states[full]={'kv':qs.cache,'source_q':{l:d['source_q'] for l,d in qs.diag.items()}}
                    if qs.snapshots:snapshots[full]=qs.snapshots
                    ops.append({**base,'trial':trial,'projection_checks':qs.projection_checks,'query_checks':qs.query_checks,'context_checks':qs.context_checks,
                                'writer_locality_checks':p.locality,'writer_zero_checks':p.writer_zero_checks,'max_context_error':max(d['context_max_abs_error'] for d in qs.diag.values())})
                    if trial==world+'_zero_native':ss[world]=source_states[full]
                    elif '_zero_' in trial:
                        ref=ss[world];ch={**base,'trial':trial+'_source_identity','source_kv_exact':all(torch.equal(v,ref['kv'][l][k]) for l,d in qs.cache.items() for k,v in d.items()),
                            'source_q_exact':all(torch.equal(d['source_q'],ref['source_q'][l]) for l,d in qs.diag.items())}
                        run.checks.append(ch);assert ch['source_kv_exact'] and ch['source_q_exact']
                for trial,world,zero,q,mask in conditions:
                    if trial=='C':z,cap,norms=b,bc,bn
                    else:
                        snap=item['corpus']=='wiki_validation' and item['window']==0 and identity=='natural' and world=='C' and zero and mask!='native'
                        z,_,cap,norms=ev(x,world,zero,q,mask,bl=bl,snapshot=snap)
                    record(trial,world,z,cap,norms)
                for world in 'CEB':
                    z=refs[world+'_zero_present_causal'];target=refs[world+'_zero_native']
                    run.checks.append({**base,'trial':world+'_mask_reference','max_loss_drift':float((z['loss'][1:]-target['loss'][1:]).abs().max()),
                        'mean_loss_drift':float((z['loss'][1:].double()-target['loss'][1:].double()).mean()),'pred_agreement':float(z['pred'][1:].eq(target['pred'][1:]).double().mean())})
                if identity=='natural' and item['window']==0:
                    for trial,special in (('repeat',None),('identity','identity')):
                        z,_,cap,norms=ev(x,'C',special=special,bl=bl);record(trial,'C',z,cap,norms)
                        ch={**base,'trial':trial,**{k+'_exact':torch.equal(z[k][1:],b[k][1:]) for k in ('loss','pred','kl')}}
                        run.checks.append(ch);assert all(ch[k+'_exact'] for k in ('loss','pred','kl'))
                del bl,bc
            run.checkpoint();torch.save(diagnostics,out/'attention_diagnostics.pt');torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt');dump(out/'operation_checks.json',ops)
            print('QUERY_SOURCE_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,
            model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==len(run.raw)==776 and before==meta['model_hash_after'] and meta['parameter_versions_unchanged']
        dump(out/'manifest.json',meta);print('COMPLETE',run.count,meta['wall_seconds'],flush=True)
    except BaseException as error:
        run.checkpoint();torch.save(diagnostics,out/'attention_diagnostics.pt');torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt');dump(out/'operation_checks.json',ops)
        torch.save({'writers':p.vectors,'diag':qs.diag,'snapshots':qs.snapshots,'kv':qs.cache,
                    'reconstruction':qs.failure_reconstruction},out/'failure_state.pt')
        meta.pop('completed_utc',None);meta.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw));dump(out/'manifest.json',meta)
        raise
    finally:qs.close();p.close()

if __name__=='__main__':main()
