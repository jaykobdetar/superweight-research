"""Collect L0-neuron and residual-component traces for archived H14 probes.

Only two transformer layers, with exact cached-endpoint validation. The edge
hook recomputes a single attention context from the renormalized native row.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np

R=Path(__file__).resolve().parent
ROLE=Path('/workspace/shared/olmo_computational_role/downstream')
sys.path.insert(0,str(ROLE))
from streamed_olmo import StreamedOlmo


def blocked_edge(head,query=128,source=64):
    records={}
    def hook(layer,context,probs,values):
        if layer!=0:return context
        p=probs[head,query].copy();records['native_edge_mass']=float(p[source]);p[source]=0
        assert p.sum()>1e-6
        p/=p.sum(dtype=np.float32)
        out=context.copy();out[query,head*128:(head+1)*128]=p@values[head]
        records['effective_edge_mass']=float(p[source])
        records['other_query_max_abs_change']=float(np.max(abs(out[:query]-context[:query])))
        assert records['other_query_max_abs_change']==0
        return out
    hook.records=records
    return hook


def main():
    out=R/'h14_prefix';out.mkdir(exist_ok=True)
    path=Path('/workspace/shared/olmo_recovered/olmo_training_development_complete/olmo_training_origins/phase2/protocol.json')
    protocol=json.loads(path.read_text())
    gold=np.load(R.parent/'results/mlp1_replay.npz');labels=gold['labels'].tolist()
    model=StreamedOlmo(half_round=False)
    meta={'source_protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'model_code_sha256':hashlib.sha256((ROLE/'streamed_olmo.py').read_bytes()).hexdigest(),
          'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'model_revision':'d7cbab742d80589e714b1a2d7f838dcd21cbe143',
          'scope':'Prefix only through layer1; no LM head or downstream losses.'}
    traces={};checks=[]
    for panel in protocol['panels']:
        clean=np.array([209 if t in (187,535) else t for t in panel['token_ids']],dtype=np.int64)
        clean[0]=510;clean[128]=187
        native_prefix_hashes=None
        for condition,p_token,head in [('newline_U',209,None),('newline_R',187,None),
                                      ('newline_R_edge_H14',187,14),('newline_R_edge_H0',187,0)]:
            ids=clean.copy();ids[64]=p_token
            ah=None if head is None else blocked_edge(head)
            result=model.forward(ids,attention_hook=ah,capture_layers=(0,1),stop_layer=1)
            a,b=result['states']['0'],result['states']['1']
            prefix_hashes={str(i):hashlib.sha256(result['states'][str(i)]['post_layer'][:128].tobytes()).hexdigest() for i in (0,1)}
            if condition=='newline_R':native_prefix_hashes=prefix_hashes
            if head is not None:assert prefix_hashes==native_prefix_hashes
            expected=gold['normalized_pre_mlp1'][labels.index([panel['name'],condition])]
            norm_error=float(np.max(abs(b['normalized'][128]-expected)))
            assert norm_error<5e-4,(panel['name'],condition,norm_error)
            key=panel['name']+'__'+condition
            arrays={'input_ids':ids,'m0_normalized':a['normalized'][128],
                    'm0_activations':a['activations'][128],'m0_output':a['mlp_output'][128],
                    'l0_pre_mlp':a['pre_mlp'][128],'l0_post_layer':a['post_layer'][128],
                    'a1_output':b['pre_mlp'][128]-a['post_layer'][128],
                    'l1_pre_mlp':b['pre_mlp'][128],'l1_normalized':b['normalized'][128],
                    'l1_activations':b['activations'][128],'l1_output':b['mlp_output'][128],
                    'l1_post_layer':b['post_layer'][128]}
            traces.update({key+'__'+k:np.array(v,copy=True) for k,v in arrays.items()})
            checks.append(dict(panel=panel['name'],condition=condition,normalized_cache_max_error=norm_error,
                prefix_hashes=prefix_hashes,edge_records={} if ah is None else ah.records,
                seconds=result['seconds'],peak_rss=result['peak_rss']))
            print(key,'norm_error',norm_error,'seconds',round(result['seconds'],3),flush=True)
            del result,a,b
    np.savez_compressed(out/'traces.npz',**traces)
    meta.update(checks=checks,all_pass=True,max_cache_error=max(x['normalized_cache_max_error'] for x in checks),
                peak_rss=max(x['peak_rss'] for x in checks))
    (out/'receipt.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE',meta['max_cache_error'],meta['peak_rss'],flush=True)


if __name__=='__main__':main()
