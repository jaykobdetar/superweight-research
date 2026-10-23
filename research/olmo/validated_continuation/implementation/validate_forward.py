"""Archive every validation result; do not silently loosen the frozen gate."""
from pathlib import Path
import json,time,resource,argparse,hashlib
import numpy as np
from streamed_olmo import StreamedOlmo
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=10);ap.add_argument('--half-round',action='store_true');ap.add_argument('--stop-layer',type=int);ap.add_argument('--tag',default='fp32');args=ap.parse_args()
resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
p=json.loads((R/'protocols/factorial_protocol.json').read_text());data=np.load(R/'protocols/frozen_inputs.npz');model=StreamedOlmo(half_round=args.half_round)
rows=[];out={};start=time.time()
for v in p['validation'][:args.limit]:
 ids=data[v['id']+'_ids'];mask=(0,) if v['trial']=='remove_source' else ()
 result=model.forward(ids,mask_sources=mask,stop_layer=args.stop_layer,progress=True,capture_layers=(1,) if args.stop_layer is not None else ())
 if args.stop_layer is not None:
  for k,a in result['states'].get('1',{}).items():out[v['id']+'_'+k]=a
  rows.append({'id':v['id'],'stop_layer':args.stop_layer,'seconds':result['seconds'],'peak_rss':result['peak_rss']})
 else:
  got=result['loss'][1:];expected=data[v['id']+'_archived_loss'];assert got.shape==expected.shape
  delta=got-expected;threshold=p['validation_gate']
  row={'id':v['id'],'half_round':args.half_round,'native_mean':float(got.mean()),'archived_mean':float(expected.mean()),'mean_delta':float(delta.mean()),'mae':float(np.abs(delta).mean()),'p95_abs':float(np.quantile(np.abs(delta),.95)),'max_abs':float(np.abs(delta).max()),'seconds':result['seconds'],'peak_rss':result['peak_rss']}
  row['pass']=abs(row['mean_delta'])<=threshold['mean_abs_window_NLL_difference_max'] and row['mae']<=threshold['per_target_MAE_max'] and row['p95_abs']<=threshold['per_target_p95_abs_difference_max']
  rows.append(row);out[v['id']+'_loss']=result['loss'];out[v['id']+'_argmax']=result['argmax']
 print('VALIDATION',json.dumps(rows[-1]),flush=True)
 (R/'results'/f'validation_{args.tag}.json').write_text(json.dumps({'rows':rows,'seconds':time.time()-start,'complete':len(rows)==args.limit,'all_pass':all(x.get('pass',False) for x in rows),'code_sha256':hashlib.sha256((R/'streamed_olmo.py').read_bytes()).hexdigest()},indent=2))
 np.savez_compressed(R/'results'/f'validation_{args.tag}.npz',**out)
