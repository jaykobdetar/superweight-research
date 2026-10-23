"""Staged, resumable content-donor experiment. Requires an explicitly granted CPU slot.

This script never downloads models or edits the shared core runner. All outputs
are beneath this directory. Run baseline first; inspect its all-example gates.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse, hashlib, json, sys, time, resource
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'downstream'))
from streamed_olmo import StreamedOlmo
from donor_algebra import replay_context, replacement_delta, match_output_norm


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); tmp.replace(path)

def verify_manifest():
    path=ROOT/'frozen_manifest.json'; manifest=json.loads(path.read_text()); sha=digest(path)
    assert sha==(ROOT/'frozen_manifest.sha256').read_text().strip()
    assert digest(BASE/'inputs/tokenizer.json')==manifest['tokenizer_sha256']
    assert digest(BASE/'downstream/streamed_olmo.py')==manifest['runner_sha256']
    protocol=json.loads((ROOT/'contrast_protocol.json').read_text())
    assert protocol['input_manifest_sha256']==sha
    return manifest,sha

def score(model, result, rec, correct_index):
    q=rec['query_pos']; label_ids=rec['label_ids']
    x=model.ln(result['states']['15']['post_layer'][q:q+1])[0]
    w=model.mmap('lm_head.weight')
    try: logits=np.asarray(w[label_ids]) @ x
    finally: w._mmap.close()
    # The common appended target is always label A, even in B recipients.
    observed=float(result['target_logit'][q]); error=abs(float(logits[0])-observed)
    assert error < 1e-3, ('manual observed-target logit mismatch',error)
    normalizer=float(result['log_normalizer'][q]); lps=logits.astype(np.float64)-normalizer
    margin=float(logits[correct_index]-logits[1-correct_index])
    return dict(label_logits=logits.tolist(),label_log_probs=lps.tolist(),
                correct_index=correct_index,correct_minus_foil=margin,correct_label_loss=float(-lps[correct_index]),
                correct_binary_probability=float(1/(1+np.exp(-np.clip(margin,-700,700)))),
                observed_target_logit_error=error, observed_target_loss=float(result['loss'][q]),
                full_vocab_argmax=int(result['argmax'][q]),seconds=float(result['seconds']),
                peak_rss_bytes=int(result['peak_rss']),prefix_hashes=result['prefix_hashes'])

def kwargs(rec,world):
    masks={'native':(),'postboundary_mask0':(0,),'postboundary_maskp':(rec['p'],),'postboundary_mask0p':(0,rec['p'])}
    assert world in masks
    return dict(mask_sources=masks[world], mask_layers=range(2,16),
                mask_query_start=rec['p']+1,capture_layers=(15,),prefix_hash_end=rec['p'])

def load_cache(path,expected_sha256):
    assert digest(path)==expected_sha256, ('Cache checksum mismatch',str(path))
    with np.load(path) as data: return {k:data[k].copy() for k in data.files}

class Run:
    def __init__(self,args):
        self.args=args;self.manifest,self.sha=verify_manifest()
        resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
        validation_path=BASE/'downstream/results/validation_fp32.json'
        validation=json.loads(validation_path.read_text())
        assert validation['complete'] and validation['all_pass'] and len(validation['rows'])==10
        assert all(row['pass'] and not row['half_round'] for row in validation['rows'])
        assert validation['code_sha256']==self.manifest['runner_sha256']
        self.baseline_worlds=('native','postboundary_mask0','postboundary_maskp','postboundary_mask0p')
        self.out=ROOT/'results'/args.run_name;self.out.mkdir(parents=True,exist_ok=True)
        self.cache=self.out/'cache';self.cache.mkdir(exist_ok=True)
        self.model=StreamedOlmo(cap_bytes=4_000_000_000)
        assert self.model.plan['revision']==self.manifest['revision']
        self.records=self.manifest['sequences']
        if args.pairs!='all':
            wanted=args.pairs.split(',');self.records=[r for r in self.records if r['pair_id'] in wanted]
            assert len(self.records)==len(wanted), 'Unknown or duplicate pair ID'
        runmeta=self.out/'run_meta.json'
        meta=dict(manifest_sha256=self.sha,protocol_sha256=digest(ROOT/'contrast_protocol.json'),
                  amendment_sha256=digest(ROOT/'preoutput_amendment_01.json'),
                  content_runner_sha256=digest(Path(__file__)),algebra_sha256=digest(ROOT/'donor_algebra.py'),
                  core_runner_sha256=self.manifest['runner_sha256'],validation_json_sha256=digest(validation_path),
                  validation_npz_sha256=digest(BASE/'downstream/results/validation_fp32.npz'),model_revision=self.manifest['revision'],
                  cap_bytes=4_000_000_000,target_peak_bytes=1_000_000_000,blas_threads=2,
                  note='Native and postboundary_mask0 worlds stay separate. All examples retained. No effect-driven selection.')
        self.identity=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        if runmeta.exists(): assert json.loads(runmeta.read_text())==meta
        else: write_json(runmeta,meta)
    def verified_result(self,path):
        data=json.loads(path.read_text())
        assert data['manifest_sha256']==self.sha and data['run_identity']==self.identity
        assert data['key']==path.stem
        return data
    def key(self,rec,label,world): return rec['pair_id']+'__'+str(label)+'__'+world
    def baseline(self,rec,label,world,kind='recipient'):
        key=self.key(rec,label,world) if kind=='recipient' else rec['pair_id']+'__donor_'+str(label)
        resultpath=self.out/(key+'.json');cachepath=self.cache/(key+'.npz')
        if resultpath.exists():
            data=self.verified_result(resultpath)
            assert cachepath.exists();return data,load_cache(cachepath,data['cache_sha256'])
        if kind=='recipient': seq=rec['recipients'][label];ci=label
        elif label=='irrelevant': seq=rec['irrelevant'];ci=0
        else:seq=rec['paraphrases'][label];ci=label
        arrays={k:[] for k in ('P','v0','vp','vr')}
        def hook(layer,context,probs,v):
            arrays['P'].append(probs.copy()); arrays['v0'].append(v[:,0].copy())
            arrays['vp'].append(v[:,rec['p']].copy());arrays['vr'].append(v[:,rec['fact_pos']].copy())
            return context
        result=self.model.forward(seq['ids'],attention_hook=hook,**kwargs(rec,world))
        data=dict(stage='baseline' if kind=='recipient' else 'donor_baseline',key=key,
                  pair_id=rec['pair_id'],recipient_label=label,world=world,manifest_sha256=self.sha,run_identity=self.identity,
                  sequence_sha256=hashlib.sha256(np.array(seq['ids'],dtype='<i8').tobytes()).hexdigest(),
                  metrics=score(self.model,result,rec,ci))
        stacked={k:np.stack(v) for k,v in arrays.items()}
        np.savez(cachepath,**stacked);data['cache_sha256']=digest(cachepath);write_json(resultpath,data)
        print(json.dumps({k:data[k] for k in ('stage','key')}|{'margin':data['metrics']['correct_minus_foil'],
                          'seconds':data['metrics']['seconds'],'peak_MiB':data['metrics']['peak_rss_bytes']/2**20}),flush=True)
        return data,stacked
    def baseline_stage(self):
        for rec in self.records:
            for world in self.baseline_worlds:
                for label in (0,1):self.baseline(rec,label,world)
        for rec in self.records:
            for label in (0,1):
                native=json.loads((self.out/(self.key(rec,label,'native')+'.json')).read_text())
                for world in self.baseline_worlds[1:]:
                    masked=self.verified_result(self.out/(self.key(rec,label,world)+'.json'))
                    assert native['metrics']['prefix_hashes']==masked['metrics']['prefix_hashes'], 'Maskworld prefix is not native.'
        self.summarize_baselines()
    def summarize_baselines(self):
        output=[]
        for rec in self.manifest['sequences']:
            for world in self.baseline_worlds:
                paths=[self.out/(self.key(rec,label,world)+'.json') for label in (0,1)]
                if not all(path.exists() for path in paths):continue
                margins=[json.loads(path.read_text())['metrics']['correct_minus_foil'] for path in paths]
                output.append(dict(pair_id=rec['pair_id'],world=world,reciprocal_margins=margins,
                                   both_above_chance=all(x>0 for x in margins),
                                   mean_margin=float(np.mean(margins))))
        usage=[]
        for rec in self.manifest['sequences']:
            for world, ablated in (('native','postboundary_maskp'),('postboundary_mask0','postboundary_mask0p')):
                paths=[self.out/(self.key(rec,label,w)+'.json') for label in (0,1) for w in (world,ablated)]
                if not all(path.exists() for path in paths):continue
                data=[self.verified_result(path)['metrics'] for path in paths]
                costs=[data[2*i+1]['correct_label_loss']-data[2*i]['correct_label_loss'] for i in (0,1)]
                usage.append(dict(pair_id=rec['pair_id'],world=world,reciprocal_p_access_cost_nats=costs,mean_cost_nats=float(np.mean(costs)),p_usage_gate_pass=bool(np.mean(costs)>0.01)))
        write_json(self.out/'baseline_gates.json',dict(manifest_sha256=self.sha,results=output,p_usage=usage,
                    warning='This is a baseline-only qualification, not permission to discard failed examples.'))
    def donors_stage(self):
        for rec in self.records:
            for label in (0,1,'irrelevant'): self.baseline(rec,label,'native',kind='donor')
    def intervene(self,rec,label,world,condition,scope):
        basepath=self.out/(self.key(rec,label,world)+'.json')
        assert basepath.exists(),'Run baseline stage and inspect baseline_gates.json first.'
        basedata=self.verified_result(basepath);base=load_cache(self.cache/(self.key(rec,label,world)+'.npz'),basedata['cache_sha256'])
        if condition in ('identity','p_from_source0'):
            donor=base;source=rec['p'];dkey='vp' if condition=='identity' else 'v0'
        else:
            donor_label=label if condition=='matched' else 1-label
            if condition in ('irrelevant','source0'):donor_label='irrelevant'
            donorpath=self.cache/(rec['pair_id']+'__donor_'+str(donor_label)+'.npz')
            assert donorpath.exists(),'Run donors stage first.'
            donordata=self.verified_result(self.out/(rec['pair_id']+'__donor_'+str(donor_label)+'.json'))
            donor=load_cache(donorpath,donordata['cache_sha256'])
            source=0 if condition=='source0' else (rec['fact_pos'] if condition in ('ordinary_normmatched','ordinary_raw') else rec['p'])
            dkey='v0' if condition=='source0' else ('vr' if condition in ('ordinary_normmatched','ordinary_raw') else 'vp')
        key=self.key(rec,label,world)+'__'+scope+'__'+condition
        path=self.out/(key+'.json')
        if path.exists():return self.verified_result(path)
        layers=set(range(2,16)) if scope=='frozen_all' else {self.args.single_layer}
        diagnostics=[]
        def hook(layer,context,probs,v):
            freeze=(scope=='frozen_all' or layer in layers)
            if scope=='single_layer_endogenous' and layer in layers:
                assert np.array_equal(probs,base['P'][layer]), 'Routing changed before the single-layer patch.'
            P=base['P'][layer] if freeze else probs
            rebuilt=replay_context(P,v) if freeze else context
            before_prefix=rebuilt[:rec['p']+1].copy()
            if layer in layers:
                delta=replacement_delta(P,v,source,donor[dkey][layer],rec['p']+1)
                diag=dict(layer=layer,source=source,
                          source_value_difference_l2=float(np.linalg.norm(donor[dkey][layer]-v[:,source])),
                          endogenous_vs_cached_P_max=float(np.max(np.abs(probs-base['P'][layer]))),
                          used_P_equals_baseline=bool(np.array_equal(P,base['P'][layer])))
                if condition=='ordinary_normmatched':
                    reference=replacement_delta(P,v,rec['p'],donor['vp'][layer],rec['p']+1)
                    project=lambda x:self.model.linear(x,f'model.layers.{layer}.self_attn.o_proj.weight')
                    delta,normdiag=match_output_norm(delta,reference,project);diag.update(normdiag)
                rebuilt=rebuilt+delta;diagnostics.append(diag)
            assert np.array_equal(rebuilt[:rec['p']+1],before_prefix)
            return rebuilt
        result=self.model.forward(rec['recipients'][label]['ids'],attention_hook=hook,**kwargs(rec,world))
        metrics=score(self.model,result,rec,label)
        assert result['prefix_hashes']==basedata['metrics']['prefix_hashes'],'Native recipient prefix was changed.'
        if condition in ('identity','source0'):
            assert abs(metrics['correct_minus_foil']-basedata['metrics']['correct_minus_foil'])<1e-4
        data=dict(stage='intervention',key=key,pair_id=rec['pair_id'],recipient_label=label,world=world,
                  condition=condition,scope=scope,patched_layers=sorted(layers),manifest_sha256=self.sha,run_identity=self.identity,
                  metrics=metrics,baseline_margin=basedata['metrics']['correct_minus_foil'],
                  margin_delta=metrics['correct_minus_foil']-basedata['metrics']['correct_minus_foil'],
                  prefix_hashes_equal=True,diagnostics=diagnostics)
        write_json(path,data)
        print(json.dumps(dict(stage='intervention',key=key,margin=metrics['correct_minus_foil'],
                              delta=data['margin_delta'],seconds=metrics['seconds'])),flush=True)
        return data
    def intervention_stage(self):
        conditions=self.args.conditions.split(',')
        allowed={'identity','matched','mismatched','irrelevant','source0','ordinary_normmatched','p_from_source0','ordinary_raw'}
        assert all(x in allowed for x in conditions)
        for rec in self.records:
            for world in ('native','postboundary_mask0'):
                for label in (0,1):
                    for condition in conditions:self.intervene(rec,label,world,condition,self.args.scope)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['baseline','donors','interventions'],required=True)
    p.add_argument('--cpu-slot-authorized',action='store_true',help='Only use after parent grants shared CPU slot.')
    p.add_argument('--run-name',default='preregistered_v1')
    p.add_argument('--pairs',default='all',help='all or comma-separated frozen pair IDs; no effect-driven selection')
    p.add_argument('--scope',choices=['frozen_all','single_layer_endogenous'],default='frozen_all')
    p.add_argument('--single-layer',type=int,choices=[8],default=8)
    p.add_argument('--conditions',default='identity,matched,mismatched,irrelevant,source0,p_from_source0,ordinary_raw,ordinary_normmatched')
    args=p.parse_args();assert args.cpu_slot_authorized,'Wait for shared CPU-slot authorization.'
    assert 2<=args.single_layer<=15
    assert args.run_name and all(c.isalnum() or c in '_-' for c in args.run_name)
    run=Run(args)
    {'baseline':run.baseline_stage,'donors':run.donors_stage,'interventions':run.intervention_stage}[args.phase]()
if __name__=='__main__':main()
