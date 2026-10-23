"""Audit completed baseline results without running the model."""
from pathlib import Path
import argparse, hashlib, json, csv
import numpy as np
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='preregistered_v1');args=p.parse_args()
    out=ROOT/'results'/args.run_name; manifest=json.loads((ROOT/'frozen_manifest.json').read_text())
    meta=json.loads((out/'run_meta.json').read_text())
    identity=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    rows=[];effects=[];attention=[]
    for rec in manifest['sequences']:
        pair=[]
        for label in (0,1):
            data={}
            for world in ('native','postboundary_mask0','postboundary_maskp','postboundary_mask0p'):
                key=f"{rec['pair_id']}__{label}__{world}";path=out/(key+'.json')
                if not path.exists():continue
                r=json.loads(path.read_text());assert r['run_identity']==identity and r['key']==key
                cache=out/'cache'/(key+'.npz');assert sha(cache)==r['cache_sha256']
                data[world]=r['metrics'];m=r['metrics']
                rows.append(dict(pair_id=rec['pair_id'],label=label,world=world,margin=m['correct_minus_foil'],
                                 correct_loss=m['correct_label_loss'],target_logit_error=m['observed_target_logit_error'],
                                 seconds=m['seconds'],peak_rss_bytes=m['peak_rss_bytes']))
                if world=='native':
                    with np.load(cache) as z:
                        q=rec['query_pos']
                        for L in range(2,16):
                            cosine=np.sum(z['v0'][L]*z['vp'][L],axis=-1)/(np.linalg.norm(z['v0'][L],axis=-1)*np.linalg.norm(z['vp'][L],axis=-1))
                            attention.append(dict(pair_id=rec['pair_id'],label=label,layer=L,
                                final_query_mean_p_mass=float(z['P'][L,:,q,rec['p']].mean()),
                                final_query_mean_source0_mass=float(z['P'][L,:,q,0].mean()),
                                final_query_mean_fact_mass=float(z['P'][L,:,q,rec['fact_pos']].mean()),
                                source0_p_mean_V_cosine=float(cosine.mean())))
            if len(data)==4:
                assert all(m['prefix_hashes']==data['native']['prefix_hashes'] for m in data.values())
                n,m0,mp,m0p=[data[w]['correct_label_loss'] for w in ('native','postboundary_mask0','postboundary_maskp','postboundary_mask0p')]
                effect=dict(pair_id=rec['pair_id'],label=label,p_usage_native_nats=mp-n,p_usage_mask0_nats=m0p-m0,
                            source0_p_interaction_nats=m0p-m0-mp+n)
                effects.append(effect);pair.append(effect)
        if len(pair)!=2:continue
    for filename,items in [('baseline_metrics.csv',rows),('source_access_effects.csv',effects),('native_routing_diagnostics.csv',attention)]:
        if items:
            with (out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    def mean(k,subset):return float(np.mean([r[k] for r in subset])) if subset else None
    report=dict(completed_baselines=len(rows),planned_baselines=64,complete=len(rows)==64,
        forward_seconds_sum=sum(r['seconds'] for r in rows),peak_rss_bytes=max((r['peak_rss_bytes'] for r in rows),default=None),
        max_manual_target_logit_error=max((r['target_logit_error'] for r in rows),default=None),
        native_positive_margin_count=sum(r['margin']>0 for r in rows if r['world']=='native'),
        mask0_positive_margin_count=sum(r['margin']>0 for r in rows if r['world']=='postboundary_mask0'),
        mean_p_usage_native_nats=mean('p_usage_native_nats',effects),mean_p_usage_mask0_nats=mean('p_usage_mask0_nats',effects),
        mean_source0_p_interaction_nats=mean('source0_p_interaction_nats',effects),
        all_four_world_complete_group_prefix_controls_exact=True,prefix_groups_checked=len(effects),cache_checksums_verified=len(rows),
        note='Usage costs are correct-label NLL differences, not generic next-token loss. Routing diagnostics are descriptive only.')
    (out/'baseline_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
