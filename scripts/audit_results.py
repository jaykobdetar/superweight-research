"""CPU-only audit of exported per-token losses/KL against window records."""
import argparse,json,math,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();rows=[]
 expected=json.loads((ROOT/'manifests/numeric_export.json').read_text());assert expected,'Empty export manifest';expected={r['run']:r['records'] for r in expected}
 actual={p.parent.name for p in (ROOT/'results/pulse_continuation').glob('*/per_token_numeric.npz')};assert actual==set(expected),'Missing or unexpected numeric batches'
 for p in sorted((ROOT/'results/pulse_continuation').glob('*/per_token_numeric.npz')):
  run=p.parent;index=json.loads((run/'per_token_index.json').read_text());window_rows=list(map(json.loads,(run/'windows.jsonl').read_text().splitlines()));records={(x['group']+'__'+x['trial'],x['window']):x for x in window_rows};assert len(records)==len(window_rows),'Duplicate window record';keys=[(e['condition'],e['window']) for e in index];assert len(keys)==len(set(keys))==expected[run.name] and set(keys)==set(records),'Index coverage mismatch';assert sorted(e['index'] for e in index)==list(range(len(index))),'Duplicate/missing numeric index';worst=0
  with np.load(p,allow_pickle=False) as arrays:
   assert set(arrays.files)=={str(e['index'])+'_'+f for e in index for f in ('loss','kl')},'Missing/unexpected NPZ array'
   for ent in index:
    r=records[ent['condition'],ent['window']];assert ent['token_sha256']==r['token_sha256']
    for f,key in [('loss','nll'),('kl','kl_clean_to_condition')]:
     x=arrays[str(ent['index'])+'_'+f];assert len(x)==r['scored_tokens'];assert np.isfinite(x).all();err=abs(float(x.astype(np.float64).mean())-r[key]);worst=max(worst,err);assert err<1e-10,(run.name,key,err)
  rows.append({'run':run.name,'records':len(index),'max_scalar_error':worst})
 out={'passed':True,'runs':rows,'records':sum(r['records'] for r in rows),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'loss and KL numerical replay; prediction IDs are intentionally excluded, so agreement is audited in original local reviewer artifacts, not reconstructed here'}
 if args.output:args.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
