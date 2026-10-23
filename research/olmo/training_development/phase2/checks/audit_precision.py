"""Compare stored FP32 and FP16-approx runs, without inference or downloads."""
from pathlib import Path
import json,numpy as np,hashlib
R=Path(__file__).resolve().parents[1];P=json.loads((R/'protocol.json').read_text());out=[]
for f in sorted((R/'results').glob('*/float32.json')):
 half=f.with_name('fp16_approx.json')
 if not half.exists():continue
 a=json.loads(f.read_text());b=json.loads(half.read_text());ao={(o['panel'],o['condition']):o for o in a['observations']};bo={(o['panel'],o['condition']):o for o in b['observations']}
 row={'checkpoint':a['name'],'commit':a['revision'],'fp32_causal_prefix_scaled_error':a['causal_prefix_max_scaled_error'],'fp16_approx_causal_prefix_scaled_error':b['causal_prefix_max_scaled_error'],'peak_rss_bytes':max(a['peak_rss_bytes'],b['peak_rss_bytes'])}
 for key in ['gate','up','activation','contribution','l0_attention_q_to_0_p_q','l1_attention_q_to_0_p_q']:
  x=np.concatenate([np.asarray(ao[k][key]).ravel() for k in ao]);y=np.concatenate([np.asarray(bo[k][key]).ravel() for k in ao]);d=abs(x-y);mask=abs(x)>1
  row[key+'_max_absolute_difference']=float(d.max());row[key+'_max_scaled_difference']=float((d/np.maximum(abs(x),1)).max());row[key+'_large_count']=int(mask.sum());row[key+'_large_sign_agreement']=bool(np.all(np.sign(x[mask])==np.sign(y[mask])))
 for key in ['gate','up','activation','contribution']:
  x=np.concatenate([np.asarray(o[key]).ravel() for o in a['single_tokens'].values()]);y=np.concatenate([np.asarray(o[key]).ravel() for o in b['single_tokens'].values()]);mask=abs(x)>1
  row['single_tokens_'+key+'_max_scaled_difference']=float((abs(x-y)/np.maximum(abs(x),1)).max());row['single_tokens_'+key+'_large_sign_agreement']=bool(np.all(np.sign(x[mask])==np.sign(y[mask])))
 isolate=[]
 for p in P['panels']:
  o=ao[(p['name'],'empty_isolated')];x=np.array(o['contribution'][0]);y=np.array(o['contribution'][-1]);isolate.append(float(np.max(abs(x-y)/np.maximum(abs(x),1))))
 row['fp32_isolated_equal_token_max_scaled_error']=max(isolate)
 out.append(row)
(R/'deliverables'/'precision_audit.json').write_text(json.dumps(out,indent=2))
for o in out:print(o['checkpoint'],'scaled contribution diff',o['contribution_max_scaled_difference'],'causal',o['fp32_causal_prefix_scaled_error'],'isolated',o['fp32_isolated_equal_token_max_scaled_error'],'RSS',o['peak_rss_bytes'])
