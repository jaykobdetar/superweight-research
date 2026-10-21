"""CPU-only replay of loss, KL and prediction-agreement reductions; NumPy only."""
import argparse, hashlib, json, math, resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args()
    manifest=json.loads((ROOT/'manifests/transfer_numeric_export.json').read_text())
    assert manifest and len({m['run'] for m in manifest})==len(manifest)
    actual={p.parent.name for p in (ROOT/'results/pulse_transfer').glob('*/per_token_numeric.npz')}
    assert actual=={m['run'] for m in manifest}
    results=[]
    for m in manifest:
        run=ROOT/'results/pulse_transfer'/m['run'];path=run/'per_token_numeric.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==m['export_sha256']
        index=json.loads((run/'per_token_index.json').read_text())
        rows=[json.loads(l) for l in (run/'windows.jsonl').read_text().splitlines()]
        records={r['key']:r for r in rows};indexed={e['key']:e for e in index}
        assert len(records)==len(rows)==len(index)==len(indexed)==m['records']>0
        assert set(indexed)==set(records) and sorted(e['index'] for e in index)==list(range(len(index)))
        worst=0.
        with np.load(path,allow_pickle=False) as arrays:
            assert set(arrays.files)=={f"{e['index']}_{f}" for e in index for f in ('loss','kl','matches_native')}
            for e in index:
                r=records[e['key']];b=indexed[e['baseline_key']]
                assert e['token_sha256']==r['token_sha256']==b['token_sha256']
                assert e['score_index']==r['score_index']==b['score_index']
                fields={'nll':arrays[f"{e['index']}_loss"], 'kl':arrays[f"{e['index']}_kl"],
                        'agreement':arrays[f"{e['index']}_matches_native"], 'baseline_nll':arrays[f"{b['index']}_loss"]}
                for field,a in fields.items():
                    assert a.ndim==1 and len(a)==r['scored_tokens'] and np.isfinite(a).all()
                    if field=='agreement':assert a.dtype==np.bool_
                    else:assert np.issubdtype(a.dtype,np.floating)
                    error=abs(float(a.astype(np.float64).mean())-r[field]);worst=max(worst,error)
                    assert error<1e-12,(m['run'],e['key'],field,error)
        results.append({'run':m['run'],'records':len(rows),'max_scalar_error':worst})
    out={'passed':True,'runs':results,'records':sum(r['records'] for r in results),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'scope':'Reaggregate saved numeric losses/KL and equality masks. Does not reconstruct KL from full logits, reselect candidates, validate private input tensors, or prove hook locality; use full rerun and independent local review for those.'}
    if args.output:args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
