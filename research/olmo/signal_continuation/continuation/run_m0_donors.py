"""Reciprocal L0 activation-donor necessity/sufficiency tests, prefix only.

Frozen top-four group and split1091/rest3; four discovery and four context
holdouts. Holdouts are controlled prefixes, not natural unmodified prompts.
Default freezes only; --execute requires the coordinator's CPU slot.
"""
from pathlib import Path
import argparse,hashlib,json,resource,sys
import numpy as np
from collect_h14_prefix import blocked_edge

R=Path(__file__).resolve().parent;O=R/'m0_donors';O.mkdir(exist_ok=True)
ROLE=Path('/workspace/shared/olmo_computational_role')
sys.path.insert(0,str(ROLE/'downstream'))
from streamed_olmo import StreamedOlmo
GROUP=(1091,3419,6535,4798)
TREATMENTS={'sham_reproject':(), 'single1091':(1091,), 'other_top3':(3419,6535,4798),
            'top4':GROUP, 'full_m0_output':None}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    z=np.load(R/'h14_prefix/traces.npz');windows=np.load(ROLE/'inputs/recovered_windows.npz')
    ids={};panels=[]
    for name in ('water_cycle','library','bread','programming'):
        for world in ('R','E'):
            cond='newline_R' if world=='R' else 'newline_R_edge_H14'
            ids[name+'__'+world]=z[name+'__'+cond+'__input_ids']
        panels.append(dict(name=name,split='discovery'))
    for key in ('wiki_validation__64082','dubois__42448','hobbes__73992','optimizers__9712'):
        x=windows[key][:129].copy();x[np.isin(x,(187,535))]=209;x[0]=510;x[64]=x[128]=187
        for world in ('R','E'):ids[key+'__'+world]=x.copy()
        panels.append(dict(name=key,split='context_holdout',source_window=key,
                           source_full_sha256=hashlib.sha256(windows[key].tobytes()).hexdigest()))
    protocol=dict(group=GROUP,treatments=TREATMENTS,panels=panels,
        checkpoint='d7cbab742d80589e714b1a2d7f838dcd21cbe143',q=128,p=64,
        selection='Top4 L0 neurons by mean absolute contribution to L1 neuron1710 gate numerator on4 discovery panels, frozen before donor tests. Split1091/rest3 is a2x2, not a value sweep.',
        interventions='At L0 MLP output q128, replace selected post-SwiGLU activations by opposite-world donor values and freshly project. Keep all other positions and activations; L1 attention remains endogenous. Full-output donor is positive control.',
        heldout='Four archived windows selected before donor effects; replace all187/535 tokens with209, then force0=510 and64=128=187. These are controlled heldout contexts, not naturally occurring text.',
        score='Prefix outcomes only: L1 primary gate/up/write, alternate-pair writes, source geometry, normalized state. No downstream loss.',
        input_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in ids.items()},
        runner_sha256=sha(__file__),trace_sha256=sha(R/'h14_prefix/traces.npz'),
        group_selection_sha256=sha(R/'h14_prefix/decomposition_checks.json'))
    path=O/'protocol.json';text=json.dumps(protocol,indent=2)
    if path.exists():assert path.read_text()==text
    else:path.write_text(text);np.savez_compressed(O/'inputs.npz',**ids)
    return ids,protocol


def execute():
    ids,protocol=freeze()
    model=StreamedOlmo(half_round=False)
    validation=json.loads((ROLE/'downstream/results/validation_fp32.json').read_text())
    assert validation['all_pass'] and validation['complete'] and len(validation['rows'])==10
    assert validation['code_sha256']==sha(ROLE/'downstream/streamed_olmo.py')
    identity=dict(model_sha256=validation['code_sha256'],protocol_sha256=sha(O/'protocol.json'))
    ip=O/'run_identity.json'
    if ip.exists():assert json.loads(ip.read_text())==identity
    else:ip.write_text(json.dumps(identity,indent=2))
    w=model.mmap('model.layers.0.mlp.down_proj.weight')
    gate=np.memmap(R.parent/'tensors/gate_proj.f32',mode='r',dtype='<f4',shape=(8192,2048))
    up=np.memmap(R.parent/'tensors/up_proj.f32',mode='r',dtype='<f4',shape=(8192,2048))
    down=np.memmap(R.parent/'tensors/down_proj.f32',mode='r',dtype='<f4',shape=(2048,8192))
    axis=np.full(2048,-1/2048,dtype=np.float64);axis[1764]+=1;axis/=np.linalg.norm(axis)
    archived=np.load(R/'h14_prefix/traces.npz')

    def save_result(name,world,treatment,result):
        state=result['states']['1'];h=state['normalized'][128]
        a=state['activations'][128];pre=state['pre_mlp'][128];post=state['post_layer'][128]
        aa=a.copy();aa[1710]=0;other=aa@down.T;col=pre+other;col=col-col.mean(dtype=np.float64)
        features={str(j):dict(gate=float(h@gate[j]),up=float(h@up[j]),activation=float(a[j]),
                             raw_write=float(a[j]*down[1764,j]),
                             axis_write=float(a[j]*(down[1764,j]-down[:,j].mean(dtype=np.float64))/np.sqrt(1-1/2048))) for j in (1710,3950,4786)}
        receipt=dict(panel=name,world=world,treatment=treatment,features=features,
            column_alignment=float(col@axis/np.linalg.norm(col)),
            normalized_norm=float(np.linalg.norm(h)),seconds=result['seconds'],peak_rss=result['peak_rss'],
            run_identity=identity)
        stem=name+'__'+world+'__'+treatment
        np.savez_compressed(O/(stem+'.npz'),l1_normalized=h,l1_pre_mlp=pre,l1_activations=a,
            m0_activations=result['states']['0']['activations'][128],
            m0_output=result['states']['0']['mlp_output'][128],l1_post_layer=post)
        receipt['result_sha256']=sha(O/(stem+'.npz'))
        (O/(stem+'.json')).write_text(json.dumps(receipt,indent=2))
        print(stem,features['1710']['raw_write'],receipt['column_alignment'],flush=True)
        return receipt

    for panel in protocol['panels']:
        name=panel['name'];donors={}
        # Reuse discovery native states; newly run both native worlds on holdouts.
        for world in ('R','E'):
            stem=name+'__'+world+'__native'
            resultpath=O/(stem+'.npz');receiptpath=O/(stem+'.json')
            if panel['split']=='discovery':
                condition='newline_R' if world=='R' else 'newline_R_edge_H14'
                donors[world]={k:archived[name+'__'+condition+'__'+k] for k in ('m0_activations','m0_output','l1_normalized','l1_activations')}
            elif resultpath.exists():
                receipt=json.loads(receiptpath.read_text());assert receipt['result_sha256']==sha(resultpath)
                assert receipt['run_identity']==identity
                zz=np.load(resultpath);donors[world]={k:zz[k] for k in ('m0_activations','m0_output','l1_normalized','l1_activations')}
            else:
                result=model.forward(ids[name+'__'+world],attention_hook=blocked_edge(14) if world=='E' else None,
                                     capture_layers=(0,1),stop_layer=1)
                save_result(name,world,'native',result)
                a,b=result['states']['0'],result['states']['1']
                donors[world]={'m0_activations':a['activations'][128].copy(),'m0_output':a['mlp_output'][128].copy(),
                               'l1_normalized':b['normalized'][128].copy(),'l1_activations':b['activations'][128].copy()}
                del result,a,b
        for world in ('R','E'):
            donor=donors['E' if world=='R' else 'R']
            for treatment,group in TREATMENTS.items():
                stem=name+'__'+world+'__'+treatment;path=O/(stem+'.npz');rp=O/(stem+'.json')
                if path.exists():
                    receipt=json.loads(rp.read_text());assert receipt['result_sha256']==sha(path) and receipt['run_identity']==identity
                    continue
                def hook(layer,normalized,activations,output):
                    if layer!=0:return output
                    y=output.copy()
                    if group is None:y[128]=donor['m0_output']
                    else:
                        a=activations[128].copy();a[list(group)]=donor['m0_activations'][list(group)]
                        y[128]=a@w.T
                    assert np.array_equal(y[:128],output[:128])
                    return y
                result=model.forward(ids[name+'__'+world],attention_hook=blocked_edge(14) if world=='E' else None,
                                     mlp_hook=hook,capture_layers=(0,1),stop_layer=1)
                if treatment=='sham_reproject':
                    err=float(np.max(abs(result['states']['1']['normalized'][128]-donors[world]['l1_normalized'])))
                    assert err<5e-4,(name,world,err)
                save_result(name,world,treatment,result);del result
    w._mmap.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.execute:
        resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000));execute()
    else:
        _,p=freeze();print('Frozen',len(p['panels']),'panels;80 intervention and8 heldout-native prefix passes.')
