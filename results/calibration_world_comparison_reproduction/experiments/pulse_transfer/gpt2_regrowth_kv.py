"""Source K/V transmission after conditional L3 regrowth; protocol frozen separately."""
import argparse,itertools,json,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Runner
from fresh_inputs_v2 import prepare_fresh
from gpt2_layer_writes import LayerWrites
from llama_source_kv import KV

class GPT2KV(KV):
    """GPT-2 native projection is [Q,K,V]; only source K/V slices can change."""
    def __init__(self,m,start=4):
        self.start=start;self.width=m.config.n_embd;self.patch={};self.mask='native'
        self.cache={};self.queries=None;self.boundary=None;self.errors=[]
        self.handles=[];self.masks={};self.projection_checks=0
        for i,layer in enumerate(m.transformer.h):
            if i<start:continue
            self.handles.append(layer.attn.c_attn.register_forward_hook(lambda mod,a,o,i=i:self.combined(i,o)))
            self.handles.append(layer.attn.register_forward_pre_hook(self.attn,with_kwargs=True))
        self.handles.append(m.transformer.h[start-1].register_forward_hook(self.capture_boundary))

    def reset(self,patch=None,mask='native'):
        super().reset(patch,mask);self.boundary=None;self.projection_checks=0

    def capture_boundary(self,m,a,o):self.boundary=o[0].detach().cpu().clone()

    def combined(self,i,o):
        assert o.shape[-1]==3*self.width and o.shape[0]==1
        old=o
        if any((i,k) in self.patch for k in ('k','v')):
            o=o.clone()
            for k,j in (('k',1),('v',2)):
                if (i,k) in self.patch:
                    d=self.patch[i,k].to(device=o.device,dtype=o.dtype)
                    assert d.shape==(self.width,)
                    o[0,0,j*self.width:(j+1)*self.width]=d
                    self.errors.append(float((o[0,0,j*self.width:(j+1)*self.width]-d).abs().max()))
        assert torch.equal(o[:,:,:self.width],old[:,:,:self.width])
        assert torch.equal(o[:,1:],old[:,1:])
        for k,j in (('k',1),('v',2)):
            sl=slice(j*self.width,(j+1)*self.width)
            if (i,k) not in self.patch:assert torch.equal(o[:,:,sl],old[:,:,sl])
            self.cache.setdefault(i,{})[k]=o[0,0,sl].detach().cpu().clone()
        self.projection_checks+=1
        if i==self.start:self.queries=o[:,:,:self.width].detach().cpu().clone()
        return o

def flat(d,keys='kv'):
    return {(l,k):v for l,dd in d['kv'].items() for k,v in dd.items() if k in keys}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True)
    assert not(out/'manifest.json').exists();parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'selection.json')==old['selection_sha256'];s=json.loads((parent/'selection.json').read_text())['selected']
    assert s['layer']==2 and s['row']==447
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    items,dm=prepare_fresh(cfg,tok,parent,out,fractions=(0,.05,.9,.95),additional_parents=(ROOT/'gpt2_groups_run',ROOT/'gpt2_writes_run',ROOT/'gpt2_layer_writes_run'))
    trials=['C','E','B']+[f'{r}_K{k}_V{v}' for r,k,v in itertools.product('EB',repeat=3)]
    trials += [f'{r}_{t}' for r in 'EB' for t in ('zeroV_currentK','zeroV_EK','mask_causal','mask_blocked')]
    assert len(trials)==19;dump(out/'conditions.json',trials)
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],
          'dtype':'float32','attention':'sdpa','seed':20261021,'coordinates':[447,138],'kv_layers':list(range(4,12)),
          'source_hashes':{name:sha(ROOT/name) for name in ('gpt2_regrowth_kv.py','gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py','fresh_inputs_v2.py','GPT2_REGROWTH_KV.md')},
          'parent_manifest_sha256':sha(parent/'manifest.json'),'conditions_sha256':sha(out/'conditions.json')}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before']
    versions={n:p._version for n,p in m.named_parameters()};p=LayerWrites(m,s,[447,138]);kv=GPT2KV(m);run=Runner(m,p,out)
    donors={};operations=[]
    def ev(x,world,patch=None,mask='native',special=None,bl=None,keep=False):
        p.reset_layers(E=world!='C',blocked=[3] if world=='B' else [],special=special);kv.reset(patch,mask)
        z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);cap['selected']['writers']=p.vectors
        d={'kv':{l:{k:v.clone() for k,v in dd.items()} for l,dd in kv.cache.items()},'queries':kv.queries,'boundary':kv.boundary}
        assert p.locality==30 and p.writer_zero_checks==(world=='B') and kv.projection_checks==8
        return z,ll,cap,norms,d
    def equal_check(base,trial,z,target):
        ch={**base,'trial':trial,**{k+'_exact':torch.equal(z[k][1:],target[k][1:]) for k in ('loss','pred','kl')}}
        run.checks.append(ch);assert all(ch[k+'_exact'] for k in ('loss','pred','kl')),ch
    try:
        for item in items:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'panel':'regrowth_kv','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity}
                key=f"regrowth_kv__{item['corpus']}__{item['window']}__512__{identity}"
                refs={};ds={};caps={};b,bl,bc,bn,bd=ev(x,'C',keep=True)
                def record(trial,z,cap,norms,d,world):
                    run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);refs[trial]=z
                    op={**base,'trial':trial,'writer_locality_checks':p.locality,'writer_zero_checks':p.writer_zero_checks,
                        'projection_checks':kv.projection_checks,'patches':len(kv.errors),'max_patch_error':max(kv.errors,default=0.)}
                    if world in ds:
                        op['first_queries_exact']=torch.equal(d['queries'],ds[world]['queries']);assert op['first_queries_exact']
                    operations.append(op)
                for world in ('C','E','B'):
                    if world=='C':z,cap,norms,d=b,bc,bn,bd
                    else:z,_,cap,norms,d=ev(x,world,bl=bl)
                    record(world,z,cap,norms,d,world);caps[world]=cap
                    if world in 'EB':ds[world]=d
                ch={**base,'trial':'first_boundary','later_boundary_exact':torch.equal(ds['E']['boundary'][:,1:],ds['B']['boundary'][:,1:]),
                    'later_queries_exact':torch.equal(ds['E']['queries'][:,1:],ds['B']['queries'][:,1:]),
                    'L3_attention_source_exact':torch.equal(caps['E']['selected']['writers'][3]['attention']['after'],caps['B']['selected']['writers'][3]['attention']['after'])}
                donors[key]=ds
                run.checks.append(ch);assert all(ch[k] for k in ('later_boundary_exact','later_queries_exact','L3_attention_source_exact'))
                for recipient,kd,vd in itertools.product('EB',repeat=3):
                    patch={**flat(ds[kd],'k'),**flat(ds[vd],'v')};trial=f'{recipient}_K{kd}_V{vd}'
                    z,_,cap,norms,d=ev(x,recipient,patch,bl=bl);record(trial,z,cap,norms,d,recipient)
                    if kd==vd:equal_check(base,trial+'_diagonal',z,refs[kd])
                    if recipient=='B':equal_check(base,trial+'_recipient',z,refs[f'E_K{kd}_V{vd}'])
                for world in 'EB':
                    for fixed in (False,True):
                        patch={k:torch.zeros_like(v) for k,v in flat(ds['E'],'v').items()}
                        if fixed:patch.update(flat(ds['E'],'k'))
                        trial=world+'_zeroV_'+('EK' if fixed else 'currentK')
                        z,_,cap,norms,d=ev(x,world,patch,bl=bl);record(trial,z,cap,norms,d,world)
                    for mask in ('causal','blocked'):
                        trial=world+'_mask_'+mask;z,_,cap,norms,d=ev(x,world,mask=mask,bl=bl);record(trial,z,cap,norms,d,world)
                    z=refs[world+'_mask_causal'];target=refs[world]
                    run.checks.append({**base,'trial':world+'_mask_reference','max_loss_drift':float((z['loss'][1:]-target['loss'][1:]).abs().max()),
                        'mean_loss_drift':float((z['loss'][1:].double()-target['loss'][1:].double()).mean()),'pred_agreement':float(z['pred'][1:].eq(target['pred'][1:]).double().mean())})
                equal_check(base,'blocked_recipient',refs['B_mask_blocked'],refs['E_mask_blocked'])
                if identity=='natural' and item['window']==0:
                    for trial,special in (('repeat',None),('identity','identity')):
                        z,_,cap,norms,d=ev(x,'C',special=special,bl=bl);record(trial,z,cap,norms,d,'C');equal_check(base,trial,z,b)
                del bl,bd,bc,caps,ds
            run.checkpoint();torch.save(donors,out/'donors.pt');dump(out/'operation_checks.json',operations)
            print('REGROWTH_KV_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),
            wall_seconds=time.time()-start,model_hash_before=before,model_hash_after=modelhash(m),
            parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==len(run.raw)==616 and before==meta['model_hash_after'] and meta['parameter_versions_unchanged']
        dump(out/'manifest.json',meta);print('COMPLETE',run.count,meta['wall_seconds'],flush=True)
    except BaseException as error:
        run.checkpoint();torch.save(donors,out/'donors.pt');dump(out/'operation_checks.json',operations)
        torch.save({'writers':p.vectors,'projection_cache':kv.cache,'queries':kv.queries,'boundary':kv.boundary},out/'failure_state.pt')
        meta.pop('completed_utc',None)
        meta.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),
                    failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw))
        dump(out/'manifest.json',meta)
        raise
    finally:kv.close();p.close()

if __name__=='__main__':main()
