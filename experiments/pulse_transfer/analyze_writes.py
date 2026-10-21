"""CPU audit and conditional writer effects from the complete source factorial."""
import argparse,json,math,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from analyze_transfer import aggregate

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);a=ap.parse_args();p=a.run;rows=[json.loads(l) for l in (p/'windows.jsonl').read_text().splitlines()];raw=torch.load(p/'per_token.pt',weights_only=True);traces=torch.load(p/'traces.pt',weights_only=True);m=json.loads((p/'manifest.json').read_text());assert len(rows)==len(raw)==len(traces)==len({r['key'] for r in rows})==m['forward_passes']==m['records']==360;assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'];groups=defaultdict(list);byinput=defaultdict(dict);geom=[];err=0.
 for r in rows:
  key=r['key'];pref='__'.join(key.split('__')[:5]);z=raw[key];b=raw[pref+'__E0A0M0'];j=1;assert r['score_index']==z['score_index']==j and r['scored_tokens']==510;assert z['token_sha256']==b['token_sha256']==r['token_sha256'];vals={'nll':float(z['loss'][j:].double().mean()),'baseline_nll':float(b['loss'][j:].double().mean()),'agreement':float(z['pred'][j:].eq(b['pred'][j:]).double().mean()),'kl':float(z['kl'][j:].double().mean())}
  for k,v in vals.items():assert math.isfinite(v);err=max(err,abs(v-r[k]));assert abs(v-r[k])<1e-12
  groups[r['corpus'],r['identity'],r['trial']].append(r);byinput[r['corpus'],r['identity'],r['window']][r['trial']]=r
  for l,d in traces[key]['capture']['writers'].items():
   for component,vec in d.items():
    geom.append({'corpus':r['corpus'],'identity':r['identity'],'window':r['window'],'trial':r['trial'],'layer':l,'component':component,'before_coords':vec['before'][m['coordinates']].tolist(),'after_coords':vec['after'][m['coordinates']].tolist(),'before_norm':float(vec['before'].norm()),'after_norm':float(vec['after'].norm())})
 checks=json.loads((p/'checks.json').read_text());assert len(checks)==8 and all(x['loss_exact'] and x['pred_exact'] for x in checks);ops=json.loads((p/'operation_checks.json').read_text());assert len(ops)==352 and all(x['locality_checks']==30 for x in ops)
 for x in ops:assert x['zero_checks']==(10 if x['trial'] in ('E1A1M1','random_E1A1M1') else 0)
 summary=[{'corpus':c,'identity':i,'trial':t,**aggregate(rs)} for (c,i,t),rs in groups.items() if t not in ('repeat','identity')];dump(p/'summary.json',summary);dump(p/'geometry.json',geom);contrasts=[]
 pairs=[('MLP_given_erasure','E1A0M1','E1A0M0'),('attention_given_erasure','E1A1M0','E1A0M0'),('both_given_erasure','E1A1M1','E1A0M0'),('MLP_without_erasure','E0A0M1','E0A0M0'),('attention_without_erasure','E0A1M0','E0A0M0'),('both_without_erasure','E0A1M1','E0A0M0')]
 for (c,i,w),rr in byinput.items():
  for label,t,b in pairs:contrasts.append({'corpus':c,'identity':i,'window':w,'trial':label,'delta_nll':rr[t]['nll']-rr[b]['nll']})
 dump(p/'paired_contrasts.json',contrasts);cg=defaultdict(list)
 for r in contrasts:cg[r['corpus'],r['identity'],r['trial']].append(r['delta_nll'])
 cs=[]
 for (c,i,t),ds in cg.items():
  ds=np.array(ds);rng=np.random.default_rng(20261021);bs=ds[rng.integers(0,len(ds),(2000,len(ds)))].mean(1);cs.append({'corpus':c,'identity':i,'trial':t,'mean_delta_nll':float(ds.mean()),'ci95_delta_nll':np.quantile(bs,[.025,.975]).tolist(),'exp_mean_delta_nll':float(np.exp(ds.mean()))})
 dump(p/'paired_summary.json',cs);dump(p/'audit.json',{'status':'passed','records':360,'scalar_error':err,'closures':8,'operation_locality_checks':sum(x['locality_checks'] for x in ops),'raw_coordinate_zero_checks':sum(x['zero_checks'] for x in ops),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
 for r in summary:
  if r['identity']=='natural':print(r['corpus'],r['trial'],round(r['ppl_ratio'],4),round(r['mean_agreement'],4))
 print('FIRST WIKI RAW447 AFTER L3')
 for r in geom:
  if r['corpus']=='wiki_validation' and r['window']==0 and r['identity']=='natural' and r['layer']==3 and r['component']=='residual':print(r['trial'],r['after_coords'],r['after_norm'])
if __name__=='__main__':main()
