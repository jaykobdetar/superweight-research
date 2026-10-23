"""Join exact-hash public input reconstructions to recorded target losses.

No fresh predictions, no inferred argmax identities. Target examples are
selected descriptively, not confirmatory evidence of a semantic mechanism.
"""
import os
if os.environ.get('OLMO_ALLOW_LOCAL_TEXT_JOINS') != '1':
    raise SystemExit('Optional local-corpus step. Read README_REPRODUCTION.txt, supply the tokenizer/corpora, then set OLMO_ALLOW_LOCAL_TEXT_JOINS=1. Outputs may contain source text; do not redistribute without checking licenses.')

from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from reconstruct_tokenizer import Tokenizer,load_window

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'analysis';M=json.loads((ROOT/'input_manifest.json').read_text());T=Tokenizer()
RUNS=('confirmation_run','column_factorial_run','later_uses_run','source_geometry_run')
records={};checks=[]
for run in RUNS:
    p=ROOT/'repo/results/pulse_continuation'/run
    ix=json.loads((p/'per_token_index.json').read_text());a=np.load(p/'per_token_numeric.npz')
    w=[json.loads(x) for x in (p/'windows.jsonl').read_text().splitlines()];wr={(x['group']+'__'+x['trial'],x['window']):x for x in w}
    records[run]={}
    for r in ix:
        key=(r['condition'],r['window']);records[run][key]={'loss':a[str(r['index'])+'_loss'].astype('float64'),'kl':a[str(r['index'])+'_kl'].astype('float64'),'row':wr[key]}

def writecsv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def classify(s):
    if s.isspace():return 'whitespace'
    x=s.strip()
    if x and all(c.isalpha() for c in x):return 'alphabetic_piece'
    if x and all(c.isnumeric() for c in x):return 'numeric_piece'
    if x and all(not c.isalnum() for c in x):return 'punctuation'
    return 'mixed_piece'

boundaries=[];examples=[];tokens=[];sourcemeta=[]
for corpus in ('wiki_validation','dubois','hobbes','optimizers'):
    for w,start in enumerate(M['confirmation'][corpus]['evaluation_starts'][:4]):
        x=load_window(corpus,start);g=f'confirm_{corpus}_natural';d=records['confirmation_run']
        b=d[(g+'__clean',w)];sha=hashlib.sha256(x.tobytes()).hexdigest();assert sha==b['row']['token_sha256']
        positions=np.arange(b['row']['score_start']+1,512);assert len(positions)==len(b['loss'])
        ps=np.flatnonzero(x==187);p=int(ps[0]) if len(ps) else None
        sm={'corpus':corpus,'window':w,'source_token_id':int(x[0]),'source_token_text':T.decode(x[:1]),'first_single_newline':p,'num_single_newlines':len(ps),'token_sha256':sha}
        sourcemeta.append(sm);checks.append({'group':g,'window':w,'token_sha256_exact':True})
        for trial in ('remove_source','mean_KV','deleted'):
            z=d[(g+'__'+trial,w)];delta=z['loss']-b['loss']
            subsets={'all':np.ones(len(delta),bool)}
            if p is not None:
                subsets.update(before_newline_broadcast=positions<p+2,after_newline_broadcast=positions>=p+2,after_newline_plus32=positions>=p+34)
            for scope,mask in subsets.items():
                if mask.any():boundaries.append({**sm,'trial':trial,'scope':scope,'targets':int(mask.sum()),'delta_nll':float(delta[mask].mean()),'ppl_ratio':float(np.exp(delta[mask].mean())),'fraction_delta_gt1':float((delta[mask]>1).mean()),'mean_kl':float(z['kl'][mask].mean())})
            if trial=='remove_source':
                # Largest effects are openly marked as selected examples.
                for rank,j in enumerate(np.argsort(delta)[::-1][:8],1):
                    pos=int(positions[j]);examples.append({**sm,'rank_within_window':rank,'target_position':pos,'query_position':pos-1,'target_id':int(x[pos]),'target':T.decode(x[pos:pos+1]),'prefix_excerpt':T.decode(x[max(0,pos-12):pos]),'native_target_nll':float(b['loss'][j]),'removed_target_nll':float(z['loss'][j]),'delta_nll':float(delta[j]),'meanKV_delta_nll':float(d[(g+'__mean_KV',w)]['loss'][j]-b['loss'][j])})
                for j,pos in enumerate(positions):
                    tid=int(x[pos]);s=T.decode([tid]);prior=np.flatnonzero(x[:pos]==tid)
                    repeat='none_in_prefix' if not len(prior) else 'recent32' if prior[-1]>=pos-32 else 'earlier_only'
                    tokens.append({'corpus':corpus,'window':w,'target_position':int(pos),'token_id':tid,'class':classify(s),'repeat':repeat,'delta_nll':float(delta[j]),'native_nll':float(b['loss'][j]),'meanKV_delta_nll':float(d[(g+'__mean_KV',w)]['loss'][j]-b['loss'][j]),'gt1':bool(delta[j]>1)})

writecsv('source0_input_identities.csv',sourcemeta);writecsv('source0_before_after_newline.csv',boundaries);writecsv('source0_affected_prediction_examples.csv',examples);writecsv('source0_target_level_analysis.csv',tokens)
strata=[]
for by in ('class','repeat'):
    for corpus in ('wiki_validation','dubois','hobbes','optimizers'):
        for value in sorted({t[by] for t in tokens if t['corpus']==corpus}):
            ts=[t for t in tokens if t['corpus']==corpus and t[by]==value]
            strata.append({'corpus':corpus,'stratum':by,'value':value,'targets':len(ts),'descriptive_pooled_delta_nll':float(np.mean([t['delta_nll'] for t in ts])),'fraction_gt1':float(np.mean([t['gt1'] for t in ts])),'native_nll':float(np.mean([t['native_nll'] for t in ts]))})
writecsv('source0_target_strata.csv',strata)
(OUT/'input_join_checks.json').write_text(json.dumps({'checks':checks,'scope':'16 exact-hash 512-token windows. Full corpus identity checked for books and code; Wiki partial-corpus reconstruction accepted only because all four exact window hashes match. Later trials retain no semantic role claims.'},indent=2)+'\n')
for sm in sourcemeta:
    sel=[r for r in boundaries if r['corpus']==sm['corpus'] and r['window']==sm['window'] and r['trial']=='remove_source']
    print(sm['corpus'],sm['window'],'first187',sm['first_single_newline'],'sourceID',sm['source_token_id'],[(r['scope'],r['targets'],round(r['delta_nll'],4)) for r in sel])
print('\nAffected Wiki predictions, 2 selected examples per window')
for r in examples:
    if r['corpus']=='wiki_validation' and r['rank_within_window']<=2:print(r['window'],r['target_position'],repr(r['prefix_excerpt']),'=>',repr(r['target']),'native',round(r['native_target_nll'],3),'removed',round(r['removed_target_nll'],3),'meanKVdelta',round(r['meanKV_delta_nll'],4))
