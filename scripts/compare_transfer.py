"""Compare a full private rerun's scalar record reductions with the archived run."""
import argparse,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--atol',type=float,default=0.);args=ap.parse_args()
    assert math.isfinite(args.atol) and args.atol>=0
    manifest=json.loads((ROOT/'manifests/transfer_numeric_export.json').read_text());out=[]
    expected={'gpt2_run','llama_run','gpt2_groups_run','llama_source_kv_run','gpt2_writes_run','llama_replacements_run'}
    assert len(manifest)==6 and {m['run'] for m in manifest}==expected,'Expected all six unique batches'
    for m in manifest:
        saved=ROOT/'results/pulse_transfer'/m['run'];fresh=args.work/'pulse_transfer'/m['run']
        aa=list(map(json.loads,(saved/'windows.jsonl').read_text().splitlines()))
        bb=list(map(json.loads,(fresh/'windows.jsonl').read_text().splitlines()))
        a={r['key']:r for r in aa};b={r['key']:r for r in bb}
        assert len(aa)==len(a)==len(bb)==len(b)==m['records']>0,'Duplicate/missing records'
        assert set(a)==set(b);worst=0.
        for key,x in a.items():
            y=b[key];assert x['token_sha256']==y['token_sha256'] and x['score_index']==y['score_index']
            for k in ('nll','baseline_nll','kl','agreement'):
                assert math.isfinite(y[k]);error=abs(x[k]-y[k]);worst=max(worst,error)
                assert error<=args.atol,(m['run'],key,k,error,args.atol)
        mm=json.loads((fresh/'manifest.json').read_text());archived=json.loads((saved/'manifest.json').read_text())
        assert isinstance(mm.get('completed_utc'),str) and mm['completed_utc']
        assert mm['records']==len(b) and mm['forward_passes']==archived['forward_passes']
        for field in ('model_hash_before','model_hash_after'):
            assert isinstance(mm.get(field),str) and re.fullmatch('[0-9a-f]{64}',mm[field])
            assert mm[field]==archived[field]
        assert mm['model_hash_before']==mm['model_hash_after'] and mm['parameter_versions_unchanged'] is True
        out.append({'run':m['run'],'records':len(a),'max_scalar_error':worst})
    print(json.dumps({'passed':True,'atol':args.atol,'runs':out,'scope':'Scalar comparison only; each run also executes its protocol closures.'},indent=2))

if __name__=='__main__':main()
