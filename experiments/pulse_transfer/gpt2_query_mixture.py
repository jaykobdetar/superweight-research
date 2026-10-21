"""Interventional allocation/conditional-mixture donor decomposition."""
import argparse,hashlib,itertools,json,math,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Runner
from fresh_inputs_v2 import prepare_fresh
from gpt2_layer_writes import LayerWrites
from gpt2_query_source import QuerySource

class QueryMixture(QuerySource):
    def __init__(self,m):
        self.mix_pending={};self.donor_cache={};self.mix_checks={};self.donor_pair=None;self.current_identity=False;self.failure_mixture=None
        super().__init__(m)

    def reset_mixture(self,zero=True,mode='present',mask='causal',pair=None,current=False,snapshot=False):
        self.reset_query(zero,mode,mask,snapshot);self.mix_pending={};self.donor_cache={};self.mix_checks={}
        self.donor_pair=pair;self.current_identity=current;self.failure_mixture=None
        assert pair is None or (zero and not current)

    def combined(self,i,o):
        o=super().combined(i,o)
        if self.zero_values:
            q,k,v,mask=self.reconstruction_inputs[i];n=q.shape[-2]
            logits=q@k.transpose(-1,-2)/8
            logits=logits.masked_fill(torch.ones(n,n,device=q.device,dtype=torch.bool).triu(1),float('-inf'))
            if mask is not None:logits=logits.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else logits+mask
            mass=logits.softmax(-1)[0,:,1:,1:].sum(-1)
            stable=(logits[:,:,1:,1:].softmax(-1)@v[:,:,1:,:])[0]
            self.mix_pending[i]=(mass,stable)
        return o

    def context(self,i,actual):
        raw_inputs=self.reconstruction_inputs[i]
        super().context(i,actual)
        if not self.zero_values:return None
        a,u=self.mix_pending.pop(i);n=actual.shape[1]
        c=actual.reshape(1,n,12,64)[0].transpose(0,1)[:,1:,:]
        state={'mass':a.detach().cpu().clone(),'stable_mixture':u.detach().cpu().clone(),'context':c.detach().cpu().clone()}
        self.donor_cache[i]=state
        good=bool(torch.isfinite(a).all() and (a>0).all() and torch.isfinite(u).all() and torch.isfinite(c).all())
        error=float((c/a[...,None]-u).abs().max()) if good else float('inf')
        recon=float((c-a[...,None]*u).abs().max()) if good else float('inf')
        result=actual;gain=1.;patch_error=0.;formula_error=0.
        if good and self.donor_pair is not None:
            mass_d,mix_d,same=self.donor_pair
            ag=mass_d[i]['mass'].to(a.device);d=mix_d[i];ad=d['mass'].to(a.device);cd=d['context'].to(c.device);ud=d['stable_mixture'].to(u.device)
            good=bool(torch.isfinite(ag).all() and (ag>0).all() and torch.isfinite(ad).all() and (ad>0).all()
                      and torch.isfinite(cd).all() and torch.isfinite(ud).all())
            if good:
                ratio=ag/ad;gain=float(ratio.max());mixed=cd if same else cd*ratio[...,None]
                patch_error=float((mixed-ag[...,None]*ud).abs().max());good=bool(torch.isfinite(mixed).all())
                result=actual.clone();result[:,1:,:]=mixed.transpose(0,1).reshape(1,n-1,768)
                formula_error=float((result[:,1:,:].reshape(n-1,12,64).transpose(0,1)-mixed).abs().max())
        elif good and self.current_identity:result=actual.clone()
        good=good and all(math.isfinite(v) for v in (gain,error,recon,patch_error,formula_error))
        check={'min_mass':float(a.min()),'max_mass':float(a.max()),'max_gain':gain,'conditional_error':error,'native_reconstruction_error':recon,
               'mixed_reconstruction_error':patch_error,'formula_error':formula_error,'finite':good,'source_context_exact':torch.equal(result[:,0,:],actual[:,0,:])}
        self.mix_checks[i]=check
        if i in self.snapshots:self.snapshots[i]['delivered_context']=result.detach().cpu().clone()
        if not good or max(error,recon,patch_error,formula_error)>1e-4 or not check['source_context_exact']:
            self.failure_mixture={'layer':i,'state':state,'actual_context':actual.detach().cpu().clone(),'delivered_context':result.detach().cpu().clone(),'check':check,
                                  'donor_pair':self.donor_pair,'raw_inputs':{k:None if v is None else v.detach().cpu().clone() for k,v in zip(('q','k','v','mask'),raw_inputs)}}
            raise AssertionError(('mixture diagnostic failed',i,check))
        return (result,) if self.donor_pair is not None or self.current_identity else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists()
    (out/'donors').mkdir(exist_ok=True);parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'selection.json')==old['selection_sha256'];s=json.loads((parent/'selection.json').read_text())['selected'];assert s['layer']==2 and s['row']==447
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    prior=('gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run','gpt2_regrowth_kv_run','gpt2_value_geometry_run','gpt2_query_source_run')
    items,dm=prepare_fresh(cfg,tok,parent,out,fractions=(.125,.325,.525,.725),additional_parents=tuple(ROOT/r for r in prior))
    trials=['C','P_native','P','R','P_blocked','R_blocked','P_current','R_current']+[f'{rec}_G{g}_D{d}' for rec,g,d in itertools.product('PR',repeat=3)]
    assert len(trials)==16;dump(out/'conditions.json',trials)
    files=('gpt2_query_mixture.py','gpt2_query_source.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py','fresh_inputs_v2.py','GPT2_QUERY_MIXTURE.md')
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa','seed':20261021,
          'layers':list(range(4,12)),'source_hashes':{n:sha(ROOT/n) for n in files},'parent_manifest_sha256':sha(parent/'manifest.json'),'conditions_sha256':sha(out/'conditions.json')}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()}
    p=LayerWrites(m,s,[447,138]);qm=QueryMixture(m);run=Runner(m,p,out);ops=[];snapshots={};source_states={};donor_files={};hidden_state={}
    def capture_hidden(mod,a,o):hidden_state['sha256']=hashlib.sha256(o.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    qm.handles.append(m.transformer.ln_f.register_forward_hook(capture_hidden))
    def ev(x,mode='present',zero=True,mask='causal',pair=None,current=False,special=None,bl=None,keep=False,snapshot=False):
        p.reset_layers(E=False,blocked=[],special=special);qm.reset_mixture(zero,mode,mask,pair,current,snapshot)
        z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);cap['selected']['writers']=p.vectors;z['final_hidden_sha256']=hidden_state['sha256']
        assert p.locality==30 and p.writer_zero_checks==0 and qm.query_checks==qm.context_checks==qm.projection_checks==8
        assert len(qm.mix_checks)==(8 if zero else 0)
        return z,ll,cap,norms
    def equal(base,trial,z,ref):
        ch={**base,'trial':trial,**{k+'_exact':torch.equal(z[k][1:],ref[k][1:]) for k in ('loss','pred','kl')}}
        ch['final_hidden_exact']=z['final_hidden_sha256']==ref['final_hidden_sha256']
        run.checks.append(ch);assert all(ch[k+'_exact'] for k in ('loss','pred','kl','final_hidden')),ch
    ds={}
    try:
        for item in items:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'panel':'query_mixture','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity}
                key=f"query_mixture__{item['corpus']}__{item['window']}__512__{identity}";refs={};ds={};src=None
                snap=item['corpus']=='wiki_validation' and item['window']==0 and identity=='natural'
                b,bl,bc,bn=ev(x,zero=False,mask='native',keep=True)
                def record(trial,z,cap,norms):
                    nonlocal src
                    full=key+'__'+trial;run.rec(full,{**base,'trial':trial},x,z,b,cap,norms);refs[trial]=z
                    state={'kv':qm.cache,'q':{l:d['source_q'] for l,d in qm.diag.items()}}
                    source_states[full]=state
                    if qm.zero_values:
                        if src is None:src=state
                        ch={**base,'trial':trial+'_source_identity','source_kv_exact':all(torch.equal(v,src['kv'][l][k]) for l,d in state['kv'].items() for k,v in d.items()),
                            'source_q_exact':all(torch.equal(v,src['q'][l]) for l,v in state['q'].items())}
                        run.checks.append(ch);assert ch['source_kv_exact'] and ch['source_q_exact']
                    ops.append({**base,'trial':trial,'final_hidden_sha256':z['final_hidden_sha256'],'mixture_checks':qm.mix_checks,'projection_checks':qm.projection_checks,'query_checks':qm.query_checks,'context_checks':qm.context_checks,
                        'writer_locality_checks':p.locality,'max_context_error':max(d['context_max_abs_error'] for d in qm.diag.values())})
                    if qm.snapshots:snapshots[full]=qm.snapshots
                record('C',b,bc,bn)
                z,_,cap,norms=ev(x,mask='native',bl=bl);record('P_native',z,cap,norms)
                for w,mode in (('P','present'),('R','removed')):
                    z,_,cap,norms=ev(x,mode=mode,bl=bl,snapshot=snap);record(w,z,cap,norms);ds[w]=qm.donor_cache
                donor_path=out/'donors'/(key+'.pt');torch.save(ds,donor_path);donor_files[donor_path.name]=sha(donor_path)
                for w,mode in (('P','present'),('R','removed')):
                    z,_,cap,norms=ev(x,mode=mode,mask='blocked',bl=bl);record(w+'_blocked',z,cap,norms)
                    z,_,cap,norms=ev(x,mode=mode,current=True,bl=bl);record(w+'_current',z,cap,norms);equal(base,w+'_current_closure',z,refs[w])
                for rec,g,d in itertools.product('PR',repeat=3):
                    trial=f'{rec}_G{g}_D{d}';z,_,cap,norms=ev(x,mode='present' if rec=='P' else 'removed',pair=(ds[g],ds[d],g==d),bl=bl,snapshot=snap)
                    record(trial,z,cap,norms)
                    if g==d:equal(base,trial+'_donor_closure',z,refs[d])
                    if rec=='R':equal(base,trial+'_recipient_closure',z,refs[f'P_G{g}_D{d}'])
                run.checks.append({**base,'trial':'mask_reference','max_loss_drift':float((refs['P']['loss'][1:]-refs['P_native']['loss'][1:]).abs().max()),
                    'pred_agreement':float(refs['P']['pred'][1:].eq(refs['P_native']['pred'][1:]).double().mean())})
                if identity=='natural' and item['window']==0:
                    for trial,special in (('repeat',None),('identity','identity')):
                        z,_,cap,norms=ev(x,zero=False,mask='native',special=special,bl=bl);record(trial,z,cap,norms);equal(base,trial,z,b)
                ds={};del bl,bc
            run.checkpoint();torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt');dump(out/'operation_checks.json',ops);dump(out/'donor_files.json',donor_files)
            print('QUERY_MIXTURE_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,
            model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==len(run.raw)==520 and before==meta['model_hash_after'] and meta['parameter_versions_unchanged'];dump(out/'manifest.json',meta);print('COMPLETE',run.count,meta['wall_seconds'],flush=True)
    except BaseException as error:
        run.checkpoint();torch.save(snapshots,out/'attention_snapshots.pt');torch.save(source_states,out/'source_states.pt');dump(out/'operation_checks.json',ops);dump(out/'donor_files.json',donor_files)
        torch.save({'mixture':qm.failure_mixture,'reconstruction':qm.failure_reconstruction,'current_donors':qm.donor_cache,
                    'completed_donors_for_input':ds,'current_snapshots':qm.snapshots},out/'failure_state.pt')
        meta.pop('completed_utc',None);meta.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw));dump(out/'manifest.json',meta);raise
    finally:qm.close();p.close()

if __name__=='__main__':main()
