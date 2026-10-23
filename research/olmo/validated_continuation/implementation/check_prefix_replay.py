from pathlib import Path
import json,numpy as np,resource
from streamed_olmo import StreamedOlmo
R=Path(__file__).resolve().parent;resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
a=np.load('/workspace/shared/olmo_signal_production/results/mlp1_replay.npz');model=StreamedOlmo();rows=[]
for i in range(5):
 token=int(a['labels'][i,1]);results=[model.forward([token,suffix],stop_layer=1,capture_layers=(1,)) for suffix in (15,209)]
 state=results[0]['states']['1'];ref=a['normalized_pre_mlp1'][i]
 row={'token':token,'normalized_max_abs_error':float(np.max(np.abs(state['normalized'][0]-ref))),'centered_post_layer_max_abs_error':float(np.max(np.abs((state['post_layer'][0].astype(np.float64)-state['post_layer'][0].astype(np.float64).mean())-a['clean'][i]))),'suffix_change_source0_exact':bool(np.array_equal(state['post_layer'][0],results[1]['states']['1']['post_layer'][0]))}
 row['pass']=row['normalized_max_abs_error']<1e-4 and row['centered_post_layer_max_abs_error']<.001 and row['suffix_change_source0_exact'];rows.append(row)
print(json.dumps(rows,indent=2));(R/'results/prefix_replay.json').write_text(json.dumps({'rows':rows,'all_pass':all(x['pass'] for x in rows),'peak_rss':model.peak_rss},indent=2))
