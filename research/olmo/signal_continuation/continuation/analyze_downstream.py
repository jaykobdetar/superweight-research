"""Analyze frozen per-target losses and source states without new inference."""
from pathlib import Path
import csv,hashlib,json
import numpy as np

R=Path(__file__).resolve().parent;B=R/'downstream';A=R/'axis_controls';O=R/'results'
P=json.loads((B/'frozen_protocol.json').read_text())
axis=np.full(2048,-1/2048);axis[1764]+=1;axis/=np.linalg.norm(axis)
rows=[];interactions=[];checks=[];kvrows=[];attn=[]


def similarity(a,b):
    a=a.astype(np.float64).ravel();b=b.astype(np.float64).ravel()
    return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))


def load(name,treatment):
    stem=name+'__'+treatment
    folder=A if '_axis_remove_' in treatment or '_complement_remove_' in treatment else B
    p=folder/(stem+'.npz');receipt=json.loads((folder/(stem+'.json')).read_text())
    assert receipt['result_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    return np.load(p),receipt


for panel in P['panels']:
    name=panel['name'];g=panel['group_name'];p=panel['first_later187']
    native,_=load(name,'native_no_hook');loss=native['loss'];q=np.arange(len(loss))
    scopes={'after_source':q>=1,'source_query_only':q==0,
            'before_next_newline':(q>=1)&(q<p),'after_next_newline':q>p}
    treatments=[x['name'] for x in panel['treatments']]
    if panel['source_token']==187:treatments += [g+'_axis_remove_with1710',g+'_complement_remove_with1710']
    values={}
    for treatment in treatments:
        z,receipt=load(name,treatment);values[treatment]=z['loss']
        centered=z['source_post_l1'][0].astype(np.float64);centered-=centered.mean()
        alignment=float(centered@axis/np.linalg.norm(centered))
        for scope,mask in scopes.items():
            delta=z['loss'][mask]-loss[mask]
            rows.append(dict(panel=name,source_token=panel['source_token'],treatment=treatment,scope=scope,
                query_count=int(mask.sum()),mean_nll=float(z['loss'][mask].mean()),
                delta_nll=float(delta.mean()),ppl_ratio=float(np.exp(delta.mean())),
                argmax_agreement=float(np.mean(z['argmax'][mask]==native['argmax'][mask])),
                source_alignment=alignment))
            if scope=='after_source':
                for layer in range(2,16):
                    # Attention includes the final query, which has no scored
                    # next token inside this finite input. Align to NLL queries.
                    prob=z['attention_to_source_l'+str(layer)][:,:len(loss)][:,mask]
                    attn.append(dict(panel=name,treatment=treatment,layer=layer,
                                     mean_probability=float(prob.mean()),
                                     max_head_mean_probability=float(prob.mean(axis=1).max())))
        for kind in ('k','v'):
            a=z['source_l2_'+kind].reshape(16,128);b=native['source_l2_'+kind].reshape(16,128)
            for head in range(16):
                kvrows.append(dict(panel=name,treatment=treatment,projection=kind,head=head,
                    cosine_to_native=similarity(a[head],b[head]),norm=float(np.linalg.norm(a[head])),
                    native_norm=float(np.linalg.norm(b[head])),
                    difference_norm=float(np.linalg.norm(a[head]-b[head])),
                    clipped_fraction=float(np.mean(abs(a[head])==8))))
    for scope,mask in scopes.items():
        interaction=(values[g+'_drop_with1710']-values['drop1710']-
                     values[g+'_drop_only']+values['native_no_hook'])
        row=dict(panel=name,scope=scope,primary_by_group_interaction=float(interaction[mask].mean()))
        if panel['source_token']==187:
            comp=(values[g+'_drop_with1710']-values[g+'_axis_remove_with1710']-
                  values[g+'_complement_remove_with1710']+values['drop1710'])
            row['axis_by_complement_interaction']=float(comp[mask].mean())
        interactions.append(row)
    for left,right in [('native_no_hook','reconstruct_native'),('drop1710',g+'_same_world_addback')]:
        error=abs(values[left]-values[right]);assert error.max()<1e-3 and error.mean()<1e-4
        checks.append(dict(panel=name,left=left,right=right,max_absolute_nll_error=float(error.max()),
                           mean_absolute_nll_error=float(error.mean())))

for filename,data in [('downstream_summary',rows),('downstream_interactions',interactions),
                      ('downstream_numerical_controls',checks),('source_l2_kv_diagnostics',kvrows),
                      ('source_attention_diagnostics',attn)]:
    (O/(filename+'.json')).write_text(json.dumps(data,indent=2))
    columns=list(dict.fromkeys(k for x in data for k in x))
    with (O/(filename+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(data)

# Additional same-segment lesion validation against the pinned public FP16 run.
repo=Path('/workspace/shared/olmo_computational_role/repo/results/pulse_continuation/column_factorial_run')
index=json.loads((repo/'per_token_index.json').read_text());arch=np.load(repo/'per_token_numeric.npz');validation=[]
for old,new in [('clean','native_no_hook'),('delete_column','drop1710')]:
    record=next(x for x in index if x['condition']=='first_dubois_natural__'+old and x['window']==0)
    expected=arch[str(record['index'])+'_loss'][:254]
    actual,_=load('dubois__32768__natural187',new);got=actual['loss'][1:]
    err=got-expected
    validation.append(dict(condition=new,original_full_token_hash=record['token_sha256'],
        original_scored_queries='1..510',matched_queries='1..254',original_dtype='FP16',new_dtype='FP32',
        old_mean=float(expected.mean()),new_mean=float(got.mean()),mean_delta=float(err.mean()),
        max_abs=float(abs(err).max()),mean_abs=float(abs(err).mean())))
(O/'matched_archived_column_validation.json').write_text(json.dumps(validation,indent=2))
print('VALIDATED',len(checks),'numerical controls,',len(validation),'same-segment archived conditions')
for row in rows:
    if row['scope']=='after_source' and row['treatment'] not in ('reconstruct_native',row['treatment'].split('_')[0]+'_unused'):
        print(row['panel'],row['treatment'],row['mean_nll'],row['delta_nll'],row['source_alignment'])
print('INTERACTIONS',[x for x in interactions if x['scope']=='after_source'])
