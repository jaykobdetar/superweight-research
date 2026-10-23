"""Build final first-four, matched-edit and 32-window identity summaries."""
import os
if os.environ.get('OLMO_ALLOW_LOCAL_TEXT_JOINS') != '1':
    raise SystemExit('Optional local-corpus step. Read README_REPRODUCTION.txt, supply the tokenizer/corpora, then set OLMO_ALLOW_LOCAL_TEXT_JOINS=1. Outputs may contain source text; do not redistribute without checking licenses.')

from pathlib import Path
import csv,hashlib,json,math
import numpy as np
from reconstruct_tokenizer import Tokenizer,load_window
R=Path(__file__).resolve().parent;A=R/'analysis';M=json.loads((R/'input_manifest.json').read_text());T=Tokenizer()
def csvout(name,rows):
    with (A/name).open('w') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
p=list(csv.DictReader((A/'per_window_native_dependence.csv').open()));summ=[]
for g in sorted({x['group'] for x in p if x['run']=='confirmation_run' and x['group'].endswith('_natural')}):
    for t in ('remove_source','mean_KV','deleted'):
        a=[x for x in p if x['run']=='confirmation_run' and x['group']==g and x['trial']==t and int(x['window'])<4]
        s={'group':g,'trial':t,'windows':len(a),'targets':sum(int(x['targets']) for x in a)}
        for k in ('delta_nll','agreement','kl','fraction_harmed_gt_1_nat','fraction_harmed_gt_0.1_nat'):s[k]=float(np.mean([float(x[k]) for x in a]))
        s['ppl_ratio']=math.exp(s['delta_nll']);summ.append(s)
csvout('confirmation_first4_summary.csv',summ)

out=[]
for run,offset,gprefix,t in [('column_factorial_run',0,'first','delete_column'),('later_uses_run',768,'later','column_source')]:
    rows=[json.loads(s) for s in (R/'repo/results/pulse_continuation'/run/'windows.jsonl').read_text().splitlines()]
    for c in M['confirmation']:
        for i,start in enumerate(M['confirmation'][c]['evaluation_starts'][:4]):
            x=load_window(c,start+offset);z=next(z for z in rows if z['group']==f'{gprefix}_{c}_natural' and z['trial']==t and z['window']==i)
            assert hashlib.sha256(x.tobytes()).hexdigest()==z['token_sha256']
            out.append({'run':run,'corpus':c,'window':i,'source_id':int(x[0]),'source':T.decode(x[:1]),'first187':int(np.flatnonzero(x==187)[0]) if (x==187).any() else None,'ppl_ratio':math.exp(z['nll']-z['baseline_nll']),'agreement':z['agreement'],'token_sha256':z['token_sha256']})
csvout('source_identity_column_failure.csv',out)

p=R/'repo/results/pulse_continuation/confirmation_run';ix=json.loads((p/'per_token_index.json').read_text());a=np.load(p/'per_token_numeric.npz');idx={(x['condition'],x['window']):x['index'] for x in ix};matched=[]
for window,q in ((0,16),(6,17)):
    for ident in ('natural','The'):
        g=f'confirm_dubois_{ident}';base=a[str(idx[(g+'__clean',window)])+'_loss'].astype(float);z=a[str(idx[(g+'__remove_source',window)])+'_loss'].astype(float);d=z-base
        matched.append({'source_identity':ident,'window':window,'q':q,'first_post_broadcast_target':q+2,'native_nll':float(base.mean()),'masked_nll':float(z.mean()),'ppl_ratio':math.exp(d.mean()),'pre_q_cost':float(d[:q].mean()),'post_q_cost':float(d[q:].mean())})
(A/'matched_first_token_edit.json').write_text(json.dumps(matched,indent=2)+'\n')
extras=[];byrow={(z['group']+'__'+z['trial'],z['window']):z for z in map(json.loads,(p/'windows.jsonl').read_text().splitlines())}
for c in ('dubois','hobbes','optimizers'):
    for w,start in enumerate(M['confirmation'][c]['evaluation_starts']):
        if w<4 or (c!='optimizers' and w==7):continue
        x=load_window(c,start);g=f'confirm_{c}_natural';sha=hashlib.sha256(x.tobytes()).hexdigest();assert sha==byrow[(g+'__clean',w)]['token_sha256']
        ps=np.flatnonzero(x==187);p0=int(ps[0]) if len(ps) else None
        base=a[str(idx[(g+'__clean',w)])+'_loss'].astype(float);d=a[str(idx[(g+'__remove_source',w)])+'_loss'].astype(float)-base
        extras.append({'corpus':c,'window':w,'source_id':int(x[0]),'first187':p0,'overall_delta_nll':float(d.mean()),'before_delta_nll':float(d[:p0].mean()) if p0 is not None and p0>0 else None,'after_delta_nll':float(d[p0:].mean()) if p0 is not None else None,'token_sha256':sha})
csvout('source0_additional_windows.csv',extras)
summary={'three_source187_cases':[z for z in out if z['source_id']==187],
         'other_book_code_count':sum(z['source_id']!=187 and z['corpus']!='wiki_validation' for z in out),
         'other_book_code_column_ratio_range':[min(z['ppl_ratio'] for z in out if z['source_id']!=187 and z['corpus']!='wiki_validation'),max(z['ppl_ratio'] for z in out if z['source_id']!=187 and z['corpus']!='wiki_validation')],
         'matched_edit':matched}
(A/'discriminating_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
