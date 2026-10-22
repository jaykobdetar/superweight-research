"""Layer-local conditional source writes; see GPT2_LAYER_WRITES.md."""
import argparse, itertools, json, resource, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from run_transfer import ROOT,sha,dump,modelhash,Probe,Runner
from fresh_inputs_v2 import prepare_fresh
from gpt2_writes import Writes

class LayerWrites(Writes):
    def __init__(self,m,s,coords):
        self.blocked=set();self.writer_zero_checks=0
        super().__init__(m,s,coords)

    def reset_layers(self,E=False,blocked=(),coords=None,special=None):
        self.reset(E=E,coords=coords,special=special)
        self.blocked=set(blocked);self.writer_zero_checks=0

    def down(self,i,m,a,o):
        o=Probe.down(self,i,m,a,o)
        if self.selected is None or i<=self.selected['layer']:return o
        before=o
        if i in self.blocked:
            o=o.clone();o[0,0,self.active_coords]=0
            assert torch.count_nonzero(o[0,0,self.active_coords])==0
            keep=torch.ones(o.shape[-1],dtype=torch.bool,device=o.device);keep[self.active_coords]=False
            assert torch.equal(o[0,0,keep],before[0,0,keep]);self.writer_zero_checks+=1
        self.local(before,o);self.save(i,'mlp',before,o);return o

def condition_map():
    conditions=[];keys={};aliases={}
    def add(label,E,blocked,random=False):
        blocked=tuple(sorted(blocked));key=(E,blocked,random)
        if key in keys:aliases[label]=keys[key]
        else:
            keys[key]=label
            conditions.append({'trial':label,'erase':E,'blocked_layers':list(blocked),'random_coordinates':random})
    for E,b3,blate in itertools.product((False,True),repeat=3):
        add(f'E{int(E)}B3{int(b3)}BL{int(blate)}',E,([3] if b3 else [])+(list(range(4,12)) if blate else []))
    for layer in range(3,12):add(f'E1_only_{layer}',True,[l for l in range(3,12) if l!=layer])
    for layer in range(3,12):add(f'E1_block_through_{layer}',True,range(3,layer+1))
    add('random_E1_none',True,(),True);add('random_E1_all',True,range(3,12),True)
    assert len(conditions)==24 and aliases['E1_only_3']=='E1B30BL1'
    assert aliases['E1_block_through_10']=='E1_only_11'
    return conditions,aliases

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();cfg=json.loads(args.config.read_text());out=args.out;out.mkdir(parents=True,exist_ok=True)
    assert not(out/'manifest.json').exists();parent=ROOT/'gpt2_run';old=json.loads((parent/'manifest.json').read_text())
    sel=json.loads((parent/'selection.json').read_text());assert sha(parent/'selection.json')==old['selection_sha256']
    group=json.loads((ROOT/'gpt2_groups_run/group.json').read_text());s=sel['selected'];coords=[s['row'],group['other_row']]
    random=[r['row'] for r in sel['controls'][:2]];assert coords==[447,138] and random==[588,104]
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261021);path=cfg['models']['gpt2']['path']
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    items,dm=prepare_fresh(cfg,tok,parent,out,fractions=(.25,.45,.65,.85),additional_parents=(ROOT/'gpt2_groups_run',ROOT/'gpt2_writes_run'))
    conditions,aliases=condition_map();dump(out/'conditions.json',{'conditions':conditions,'aliases':aliases})
    meta={'started_utc':time.strftime('%FT%TZ',time.gmtime()),'model_id':old['model_id'],'revision':old['revision'],
          'dtype':'float32','attention':'sdpa','seed':20261021,'coordinates':coords,'random_coordinates':random,
          'source_hashes':{name:sha(ROOT/name) for name in ('gpt2_layer_writes.py','gpt2_writes.py','run_transfer.py','fresh_inputs_v2.py','GPT2_LAYER_WRITES.md')},
          'parent_manifest_sha256':sha(parent/'manifest.json'),'group_sha256':sha(ROOT/'gpt2_groups_run/group.json'),
          'conditions_sha256':sha(out/'conditions.json')}
    dump(out/'manifest.json',meta);torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(path,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',local_files_only=True)
    model.eval();model.requires_grad_(False);before=modelhash(model);assert before==old['model_hash_before']
    versions={n:p._version for n,p in model.named_parameters()};p=LayerWrites(model,s,coords);run=Runner(model,p,out);operations=[]
    def ev(x,E=False,blocked=(),cc=None,special=None,bl=None,keep=False):
        p.reset_layers(E,blocked,cc,special);z,ll,cap,norms=run.ev(x,baseline=bl,keep=keep)
        cap['selected']['writers']=p.vectors;return z,ll,cap,norms
    try:
        for item in items:
            for identity in ('natural','bos'):
                x=item['tokens'].clone()
                if identity=='bos':x[0]=dm['literals']['start_special']['id']
                base={'panel':'layer_writes','corpus':item['corpus'],'window':item['window'],'length':512,'identity':identity}
                key=f"layer_writes__{item['corpus']}__{item['window']}__512__{identity}"
                b,bl,bc,bn=ev(x,keep=True);refs={}
                for c in conditions:
                    trial=c['trial']
                    if trial=='E0B30BL0':z,cap,norms=b,bc,bn
                    else:z,_,cap,norms=ev(x,c['erase'],c['blocked_layers'],random if c['random_coordinates'] else None,bl=bl)
                    run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms);refs[trial]=z
                    operations.append({**base,'trial':trial,'locality_checks':p.locality,'writer_zero_checks':p.writer_zero_checks})
                    assert p.locality==30 and p.writer_zero_checks==len(c['blocked_layers'])
                z,target=refs['E1_only_11'],refs['E1B31BL1']
                ch={**base,'trial':'last_layer_cannot_transmit','loss_exact':torch.equal(z['loss'][1:],target['loss'][1:]),'pred_exact':torch.equal(z['pred'][1:],target['pred'][1:])}
                run.checks.append(ch);assert ch['loss_exact'] and ch['pred_exact']
                if identity=='natural' and item['window']==0:
                    for trial,special in [('repeat',None),('identity','identity')]:
                        z,_,cap,norms=ev(x,special=special,bl=bl)
                        ch={**base,'trial':trial,'loss_exact':torch.equal(z['loss'],b['loss']),'pred_exact':torch.equal(z['pred'],b['pred'])}
                        run.checks.append(ch);assert ch['loss_exact'] and ch['pred_exact']
                        run.rec(key+'__'+trial,{**base,'trial':trial},x,z,b,cap,norms)
                del bl
            run.checkpoint();dump(out/'operation_checks.json',operations)
            print('LAYER_WRITES_DONE',item['corpus'],item['window'],run.count,flush=True)
        meta.update(completed_utc=time.strftime('%FT%TZ',time.gmtime()),forward_passes=run.count,records=len(run.raw),
                    wall_seconds=time.time()-start,model_hash_before=before,model_hash_after=modelhash(model),
                    parameter_versions_unchanged=versions=={n:p._version for n,p in model.named_parameters()},
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        assert run.count==len(run.raw)==776 and meta['model_hash_before']==meta['model_hash_after'] and meta['parameter_versions_unchanged']
        dump(out/'manifest.json',meta);print('COMPLETE',meta['forward_passes'],meta['wall_seconds'],flush=True)
    finally:p.close()

if __name__=='__main__':main()
