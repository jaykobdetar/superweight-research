"""NumPy-only independent-reader audit of every exported localization metric."""
import argparse,json,math,resource
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_gate_results import ROOT,read,sha,mean,close,bootstrap

CORPORA={'wiki_validation':4,'dubois':4,'hobbes':4,'optimizers':3}
IDENTITIES=('natural','bos')
FAMILIES=('remove_pc1','orthogonal_control','position')
LAYERS=tuple('L'+str(l)for l in range(4,12))+('all',)
CONDITIONS=['C','P','current','fullrank']+[f+'_'+l for f in FAMILIES for l in LAYERS]
CONTRASTS=[t for l in LAYERS for t in ('remove_pc1_'+l+'_minus_P','remove_pc1_'+l+'_minus_orthogonal_control','position_'+l+'_minus_P')]
INPUTS=[(c,w,i)for c,n in CORPORA.items()for w in range(n)for i in IDENTITIES]
def exact_keys(rows,fields,expected,label):
 keys=[tuple(r[f]for f in fields)for r in rows]
 assert len(keys)==len(set(keys)),('Duplicate',label)
 assert set(keys)==set(expected),('Incomplete or unexpected universe',label)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
 if args.output:args.output.write_text(json.dumps({'status':'checking'})+'\n')
 m=read(ROOT/'manifests/localization_numeric_export.json');p=ROOT/'results/gate_localization'
 for f,h in m['exported_files'].items():assert sha(p/f)==h,('export hash',f)
 assert read(p/'manifest.json')['status']=='completed'and read(p/'audit.json')['status']=='passed'
 rows=[json.loads(x)for x in(p/'windows.jsonl').read_text().splitlines()];idx=read(p/'per_token_index.json');rr={r['key']:r for r in rows}
 assert len(rows)==len(idx)==len(rr)==m['records']==938
 expected={(f'gate__{c}__{w}__512__{i}__{t}',)for c,w,i in INPUTS for t in CONDITIONS}
 expected|={(f'gate__{c}__0__512__natural__{t}',)for c in CORPORA for t in ('repeat','identity')}
 exact_keys(rows,('key',),expected,'records');exact_keys(idx,('key',),expected,'index')
 for r in rows:
  assert r['key']==f"gate__{r['corpus']}__{r['window']}__512__{r['identity']}__{r['trial']}"
  assert r['panel']=='gate'and r['length']==r['input_tokens']==512 and r['target_start']==2 and r['scored_tokens']==510
 assert {r['key']for r in idx}==set(rr)and sorted(r['index']for r in idx)==list(range(938))
 values={};scalars=0
 with np.load(p/'per_token_numeric.npz',allow_pickle=False)as z:
  expected=set()
  for e in idx:
   r=rr[e['key']];assert e['score_index']==r['score_index']==1 and e['token_sha256']==r['token_sha256'];v={}
   for suffix,field in(('loss','nll'),('kl','kl'),('kl_p','kl_p'),('matches_native','agreement'),('matches_p','agreement_p')):
    name=f"{e['index']}_{suffix}";expected.add(name);x=z[name];assert x.shape==(510,)and np.isfinite(x).all()
    assert x.dtype==np.bool_ if suffix.startswith('matches_')else np.issubdtype(x.dtype,np.floating)
    close(mean(x),r[field]);v[field]=x.astype(np.float64);scalars+=1
   values[e['key']]=v
  assert set(z.files)==expected
 for e in idx:
  stem=e['key'].rsplit('__',1)[0]+'__'
  assert e['baseline_key']==stem+'C'and e['p_baseline_key']==stem+'P'
  for k,f in(('baseline_key','baseline_nll'),('p_baseline_key','p_baseline_nll')):close(mean(values[e[k]]['nll']),rr[e['key']][f]);scalars+=1
 conditions=read(p/'conditions.json');assert conditions==CONDITIONS;groups=defaultdict(list)
 for r in rows:
  if r['trial']in conditions:groups[r['corpus'],r['identity'],r['trial']].append(r)
 summarychecks=0
 for filename,segments in(('summary.json',False),('segment_summary.json',True)):
  summaries=read(p/filename);assert len(summaries)==(496 if segments else 248)
  universe=[(c,i,t)for c in CORPORA for i in IDENTITIES for t in CONDITIONS]
  fields=('corpus','identity','trial')
  if segments:universe=[k+(s,)for k in universe for s in ('early_2_127','later_128_511')];fields+=('segment',)
  exact_keys(summaries,fields,universe,filename)
  for s in summaries:
   gs=groups[s['corpus'],s['identity'],s['trial']];assert len(gs)==(3 if s['corpus']=='optimizers'else 4)
   sl=slice(None)if not segments else slice(0,126)if s['segment']=='early_2_127'else slice(126,None)
   for ref,prefix in(('C',''),('P','p_')):
    delta=[];kl=[];agree=[]
    for r in gs:
     v=values[r['key']];b=values[r['key'].rsplit('__',1)[0]+'__'+ref]
     delta.append(mean(v['nll'][sl])-mean(b['nll'][sl]));kl.append(mean(v['kl'if ref=='C'else'kl_p'][sl]));agree.append(mean(v['agreement'if ref=='C'else'agreement_p'][sl]))
    assert s[prefix+'windows']==len(gs);close(s[prefix+'mean_delta_nll'],mean(delta));close(s[prefix+'ppl_ratio'],math.exp(mean(delta)));close(s[prefix+'ppl_ratio_ci95'],np.exp(bootstrap(delta)));close(s[prefix+'mean_kl'],mean(kl));close(s[prefix+'mean_agreement'],mean(agree));close(s[prefix+'per_window_delta_nll'],delta);summarychecks+=1
 contrasts=read(p/'paired_contrasts.json');paired=read(p/'paired_summary.json');assert len(contrasts)==810 and len(paired)==216;cg=defaultdict(list)
 exact_keys(contrasts,('corpus','window','identity','trial'),[(c,w,i,t)for c,w,i in INPUTS for t in CONTRASTS],'contrasts')
 exact_keys(paired,('corpus','identity','trial'),[(c,i,t)for c in CORPORA for i in IDENTITIES for t in CONTRASTS],'paired summaries')
 for c in contrasts:
  assert c['length']==512
  left,right=c['trial'].split('_minus_');right=right+'_'+left.rsplit('_',1)[1]if right=='orthogonal_control'else right
  stem=f"gate__{c['corpus']}__{c['window']}__512__{c['identity']}__";delta=mean(values[stem+left]['nll'])-mean(values[stem+right]['nll']);close(delta,c['delta_nll']);cg[c['corpus'],c['identity'],c['trial']].append(delta)
 def check_groups(summaries,g,key):
  for s in summaries:
   ds=g[s['corpus'],s['identity'],s[key]];assert s['n']==len(ds);close(s['mean_delta_nll'],mean(ds));close(s['ci95_delta_nll'],bootstrap(ds));close(s['per_window_delta_nll'],ds)
 check_groups(paired,cg,'trial')
 nonadd=read(p/'nonadditivity.json');ns=read(p/'nonadditivity_summary.json');assert len(nonadd)==90 and len(ns)==24;ng=defaultdict(list)
 exact_keys(nonadd,('corpus','window','identity','family'),[(c,w,i,f)for c,w,i in INPUTS for f in FAMILIES],'nonadditivity records')
 exact_keys(ns,('corpus','identity','family'),[(c,i,f)for c in CORPORA for i in IDENTITIES for f in FAMILIES],'nonadditivity summaries')
 for r in nonadd:
  assert r['length']==512
  stem=f"gate__{r['corpus']}__{r['window']}__512__{r['identity']}__";base=mean(values[stem+'P']['nll']);joint=mean(values[stem+r['family']+'_all']['nll'])-base;singles=[mean(values[stem+r['family']+'_L'+str(l)]['nll'])-base for l in range(4,12)]
  close(r['joint_delta_nll'],joint);close(r['singleton_delta_nll'],singles);close(r['sum_singleton_delta_nll'],sum(singles));close(r['joint_minus_sum_delta_nll'],joint-sum(singles));ng[r['corpus'],r['identity'],r['family']].append(joint-sum(singles))
 check_groups(ns,ng,'family')
 out={'status':'passed','records':938,'scalar_reductions':scalars,'dual_reference_summary_checks':summarychecks,'paired_contrasts':810,'paired_groups':216,'nonadditivity_records':90,'nonadditivity_groups':24,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':'Audits exported loss/KL/boolean agreement arrays and aggregations, not original KL-from-logits, private tokens, hook arithmetic or absent complete contexts.'}
 if args.output:args.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
