"""Prospective headwise value magnitude/direction factorial; no weight changes."""
import argparse,json,resource,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Runner
from fresh_inputs_v2 import prepare_fresh
from gpt2_layer_writes import LayerWrites
from gpt2_regrowth_kv import GPT2KV,flat

def random_controls():
    g=torch.Generator(device='cpu').manual_seed(20261023);result={}
    for l in range(4,12):
        result[l]={'permutation':torch.stack([torch.randperm(64,generator=g) for _ in range(12)]),
                   'sign':torch.randint(0,2,(12,64),generator=g).float()*2-1}
    return result

def value_patch(ds,direction,magnitude,random=None):
    result={};checks=[]
    for l in range(4,12):
        a=ds[direction]['kv'][l]['v'].reshape(12,64);b=ds[magnitude]['kv'][l]['v'].reshape(12,64)
        an=a.norm(dim=1,keepdim=True);bn=b.norm(dim=1,keepdim=True);assert torch.all(an>0) and torch.all(bn>0)
        z=a.clone() if direction==magnitude else a*(bn/an)
        if random is not None:
            z=z.gather(1,random[l]['permutation'])*random[l]['sign']
            assert torch.equal(z.abs().sort(dim=1).values,a.abs().sort(dim=1).values)
        norm_error=float(((z.norm(dim=1,keepdim=True)-bn).abs()/bn).max());assert norm_error<2e-6
        result[l,'v']=z.flatten();checks.append({'layer':l,'direction':direction,'magnitude':magnitude,'randomized':random is not None,
            'max_relative_norm_error':norm_error,'target_head_norms':bn.flatten().tolist(),
            'actual_head_norms':z.norm(dim=1).tolist(),'cosine_to_direction':torch.nn.functional.cosine_similarity(z,a,dim=1).tolist()})
    return result,checks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());out=a.out;out.mkdir(parents=True,exist_ok=True);assert not(out/'manifest.json').exists()
    parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256']
    s=json.loads((parent/'selection.json').read_text())['selected'];assert s['layer']==2 and s['row']==447
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    items,dm=prepare_fresh(cfg,tok,parent,out,fractions=(.175,.375,.575,.775),additional_parents=tuple(ROOT/x for x in ('gpt2_groups_run','gpt2_writes_run','gpt2_layer_writes_run','gpt2_regrowth_kv_run')))
    random=random_controls();torch.save(random,out/'random_control.pt')
    trials=['C','E','B']+[f'B_D{d}_N{n}' for d in 'EB' for n in 'EB']+['E_DE_NE','E_DB_NB']+[f'{w}_zeroV' for w in 'CEB']+['B_EK_zeroV','C_identityKV','B_randomE','B_randomB']
    assert len(trials)==16;dump(out/'conditions.json',trials)
    names=('gpt2_value_geometry.py','gpt2_regrowth_kv.py','gpt2_layer_writes.py','gpt2_writes.py','llama_source_kv.py','run_transfer.py','fresh_inputs_v2.py','GPT2_VALUE_GEOMETRY.md')
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],'dtype':'float32','attention':'sdpa',
          'seed':20261021,'random_control_seed':20261023,'coordinates':[447,138],'kv_layers':list(range(4,12)),
          'source_hashes':{n:sha(ROOT/n) for n in names},'parent_manifest_sha256':sha(parent/'manifest.json'),
          'conditions_sha256':sha(out/'conditions.json'),'random_control_sha256':sha(out/'random_control.pt')}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    m=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    m.eval();m.requires_grad_(False);before=modelhash(m);assert before==old['model_hash_before'];versions={n:p._version for n,p in m.named_parameters()}
    p=LayerWrites(m,s,[447,138]);kv=GPT2KV(m);run=Runner(m,p,out);donors={};ops=[];geometry=[]
    def ev(x,world,patch=None,special=None,bl=None,keep=False):
        p.reset_layers(E=world!='C',blocked=[3] if world=='B' else [],special=special);kv.reset(patch)
        z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep);cap['selected']['writers']=p.vectors
        d={'kv':{l:{k:v.clone() for k,v in dd.items()} for l,dd in kv.cache.items()},'queries':kv.queries,'boundary':kv.boundary}
        assert p.locality==30 and p.writer_zero_checks==(world=='B') and kv.projection_checks==8
        return z,ll,cap,norms,d
    def equal(base,label,z,b):
        ch={**base,'trial':label,**{k+'_exact':torch.equal(z[k][1:],b[k][1:]) for k in ('loss','pred','kl')}}
        run.checks.append(ch);assert all(ch[k+'_exact'] for k in ('loss','pred','kl')),ch
    try:
        for item in items:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'panel':'value_geometry','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity}
                key=f"value_geometry__{item['corpus']}__{item['window']}__512__{identity}";refs={};ds={}
                b,bl,bc,bn,bd=ev(x,'C',keep=True)
                def record(trial,z,cap,norms,d,world):
                    run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);refs[trial]=z
                    op={**base,'trial':trial,'writer_locality_checks':p.locality,'writer_zero_checks':p.writer_zero_checks,
                        'projection_checks':kv.projection_checks,'patches':len(kv.errors),'max_patch_error':max(kv.errors,default=0.)}
                    if world in ds:op['first_queries_exact']=torch.equal(d['queries'],ds[world]['queries']);assert op['first_queries_exact']
                    ops.append(op)
                for world in 'CEB':
                    if world=='C':z,cap,norms,d=b,bc,bn,bd
                    else:z,_,cap,norms,d=ev(x,world,bl=bl)
                    record(world,z,cap,norms,d,world);ds[world]=d
                donors[key]=ds
                ch={**base,'trial':'first_boundary','later_boundary_exact':torch.equal(ds['E']['boundary'][:,1:],ds['B']['boundary'][:,1:]),'later_queries_exact':torch.equal(ds['E']['queries'][:,1:],ds['B']['queries'][:,1:])}
                run.checks.append(ch);assert ch['later_boundary_exact'] and ch['later_queries_exact']
                for recipient,direction,magnitude in [('B',d,n) for d in 'EB' for n in 'EB']+[('E','E','E'),('E','B','B')]:
                    vp,checks=value_patch(ds,direction,magnitude);patch={**flat(ds['E'],'k'),**vp};trial=f'{recipient}_D{direction}_N{magnitude}'
                    z,_,cap,norms,d=ev(x,recipient,patch,bl=bl);record(trial,z,cap,norms,d,recipient)
                    geometry.extend({**base,'trial':trial,**c} for c in checks)
                    if direction==magnitude=='E':equal(base,trial+'_E_restore',z,refs['E'])
                    if recipient=='E':equal(base,trial+'_recipient',z,refs[f'B_D{direction}_N{magnitude}'])
                for world in 'CEB':
                    patch={k:torch.zeros_like(v) for k,v in flat(ds[world],'v').items()};trial=world+'_zeroV'
                    z,_,cap,norms,d=ev(x,world,patch,bl=bl);record(trial,z,cap,norms,d,world)
                patch={**flat(ds['E'],'k'),**{k:torch.zeros_like(v) for k,v in flat(ds['E'],'v').items()}}
                z,_,cap,norms,d=ev(x,'B',patch,bl=bl);record('B_EK_zeroV',z,cap,norms,d,'B')
                z,_,cap,norms,d=ev(x,'C',flat(ds['C']),bl=bl);record('C_identityKV',z,cap,norms,d,'C');equal(base,'C_identityKV',z,b)
                for world in 'EB':
                    vp,checks=value_patch(ds,world,world,random);patch={**flat(ds['E'],'k'),**vp};trial='B_random'+world
                    z,_,cap,norms,d=ev(x,'B',patch,bl=bl);record(trial,z,cap,norms,d,'B');geometry.extend({**base,'trial':trial,**c} for c in checks)
                if identity=='natural' and item['window']==0:
                    for trial,special in (('repeat',None),('identity','identity')):
                        z,_,cap,norms,d=ev(x,'C',special=special,bl=bl);record(trial,z,cap,norms,d,'C');equal(base,trial,z,b)
                del bl,bd,bc,ds
            run.checkpoint();torch.save(donors,out/'donors.pt');dump(out/'operation_checks.json',ops);dump(out/'value_checks.json',geometry)
            print('VALUE_GEOMETRY_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),wall_seconds=time.time()-start,
            model_hash_before=before,model_hash_after=modelhash(m),parameter_versions_unchanged=versions=={n:p._version for n,p in m.named_parameters()},
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==len(run.raw)==520 and before==meta['model_hash_after'] and meta['parameter_versions_unchanged']
        dump(out/'manifest.json',meta);print('COMPLETE',run.count,meta['wall_seconds'],flush=True)
    except BaseException as error:
        run.checkpoint();torch.save(donors,out/'donors.pt');dump(out/'operation_checks.json',ops);dump(out/'value_checks.json',geometry)
        torch.save({'writers':p.vectors,'projection_cache':kv.cache,'queries':kv.queries,'boundary':kv.boundary},out/'failure_state.pt')
        meta.pop('completed_utc',None);meta.update(status='failed',failure_type=type(error).__name__,failure_message=str(error),failed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw));dump(out/'manifest.json',meta)
        raise
    finally:kv.close();p.close()

if __name__=='__main__':main()
