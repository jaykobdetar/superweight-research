"""Read-only result aggregation; every frozen example remains visible."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='preregistered_v1');args=p.parse_args()
    out=ROOT/'results'/args.run_name
    manifest=json.loads((ROOT/'frozen_manifest.json').read_text())
    meta=json.loads((out/'run_meta.json').read_text())
    identity=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    by_key={}
    for path in out.glob('*.json'):
        row=json.loads(path.read_text())
        if 'key' in row:
            assert row['run_identity']==identity and row['key']==path.stem
            assert row['manifest_sha256']==meta['manifest_sha256']
            by_key[row['key']]=row
    def key(rec,label,world):return rec['pair_id']+'__'+str(label)+'__'+world
    rows=[];pair_rows=[]
    for rec in manifest['sequences']:
        for world in manifest['worlds']:
            bases=[by_key.get(key(rec,i,world)) for i in (0,1)]
            qualified=all(b and b['metrics']['correct_minus_foil']>0 for b in bases)
            for scope in ('frozen_all','single_layer_endogenous'):
                deltas=[];transfers=[]
                for label in (0,1):
                    base=bases[label]
                    if base is None:continue
                    for condition in ('identity','matched','mismatched','irrelevant','source0','p_from_source0','ordinary_raw','ordinary_normmatched'):
                        row=by_key.get(key(rec,label,world)+'__'+scope+'__'+condition)
                        if row:
                            rows.append(dict(pair_id=rec['pair_id'],world=world,scope=scope,label=label,
                                condition=condition,baseline_qualified_pair=qualified,
                                margin=row['metrics']['correct_minus_foil'],baseline_margin=row['baseline_margin'],
                                margin_delta=row['margin_delta'],correct_label_loss=row['metrics']['correct_label_loss'],
                                prefix_hashes_equal=row['prefix_hashes_equal']))
                    matched=by_key.get(key(rec,label,world)+'__'+scope+'__matched')
                    mismatched=by_key.get(key(rec,label,world)+'__'+scope+'__mismatched')
                    if matched and mismatched:
                        m=matched['metrics']['correct_minus_foil'];mm=mismatched['metrics']['correct_minus_foil']
                        deltas.append(m-mm)
                        transfers.append(base['metrics']['correct_minus_foil']>0 and m>0 and mm<0)
                pair_rows.append(dict(pair_id=rec['pair_id'],world=world,scope=scope,
                    baseline_qualified=qualified,reciprocal_directions_completed=len(deltas),
                    matched_minus_mismatched=deltas,
                    pair_mean_matched_minus_mismatched=float(np.mean(deltas)) if len(deltas)==2 else None,
                    strong_bidirectional_sign_transfer=all(transfers) if len(transfers)==2 else None))
    aggregates=[]
    for world in manifest['worlds']:
        for scope in ('frozen_all','single_layer_endogenous'):
            for qualified_only in (False,True):
                subset=[x for x in pair_rows if x['world']==world and x['scope']==scope and x['reciprocal_directions_completed']==2
                        and (not qualified_only or x['baseline_qualified'])]
                values=[x['pair_mean_matched_minus_mismatched'] for x in subset]
                aggregates.append(dict(world=world,scope=scope,baseline_qualified_only=qualified_only,
                    completed_pair_count=len(subset),planned_pair_count=8,
                    mean_pair_donor_label_effect=float(np.mean(values)) if values else None,
                    positive_pair_count=sum(x>0 for x in values),
                    strong_bidirectional_sign_transfers=sum(x['strong_bidirectional_sign_transfer'] for x in subset)))
    report=dict(note='No pair excluded from archive. Completed subsets are not evidence about missing results. No formal population inference from eight hand-built pairs.',
                result_file_count=len(by_key),pairs=pair_rows,aggregates=aggregates)
    (out/'donor_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        with (out/'all_intervention_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(aggregates,indent=2))
if __name__=='__main__':main()
