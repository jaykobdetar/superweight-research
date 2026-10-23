from pathlib import Path
import json,sys,time,resource,hashlib,os,numpy as np
from prefix_model import Prefix,P,ROOT
name=sys.argv[1] if len(sys.argv)>1 else 'main';half='--half' in sys.argv;dest=ROOT/'results'/name;dest.mkdir(exist_ok=True);m=Prefix(name,half=half);start=time.time();observations=[]
for panel in P['panels']:
 ids=panel['token_ids'];ordinary=ids.copy();ordinary[0]=510
 clean=[209 if i in (187,535) else i for i in ids];clean[0]=510;clean[128]=187
 u=clean.copy();u[64]=209;r=clean.copy();r[64]=187;e=ids.copy();e[0]=e[128]=510
 variants=[('ordinary',ordinary,{}),('newline_U',u,{}),('newline_R',r,{}),('newline_R_edge_H14',r,{'edges':[(14,128,64)]}),('newline_R_edge_H0',r,{'edges':[(0,128,64)]}),('newline_R_edge_all',r,{'edges':[(h,128,64) for h in range(16)]}),('newline_R_mlp0_q_zero',r,{'mlp0_q_zero':True}),('newline_U_mlp0_q_zero',u,{'mlp0_q_zero':True}),('empty_ordinary',e,{}),('empty_isolated',e,{'isolated_layers':(0,1)})]
 donor=None
 for condition,toks,kwargs in variants:
  o,state=m.forward(toks,return_states=True,**kwargs);o.update(panel=panel['name'],condition=condition)
  if condition=='empty_isolated':donor=state['l1_context_q'];o['self_only_max_activation_abs_diff']=float(np.max(abs(state['activation'][0]-state['activation'][-1])))
  observations.append(o)
 for head in [12,0]:
  o=m.forward(e,patch_l1=(head,donor));o.update(panel=panel['name'],condition=f'empty_patch_H{head}');observations.append(o)
 o=m.forward(e,mlp0_q_zero=True);o.update(panel=panel['name'],condition='empty_mlp0_q_zero');observations.append(o)
 print(name,'fp16approx' if half else 'fp32',panel['name'],'U',observations[-12]['contribution'][-1][0] if False else next(v['contribution'][-1][0] for v in observations if v['panel']==panel['name'] and v['condition']=='newline_U'),'R',next(v['contribution'][-1][0] for v in observations if v['panel']==panel['name'] and v['condition']=='newline_R'),flush=True)
singles={str(i):m.forward([i]) for i in [510,15,187,535,209]}
# Numeric causal-prefix consistency, independent of attention/intervention conclusions.
full,fs=m.forward(P['panels'][0]['token_ids'],return_states=True);short,ss=m.forward(P['panels'][0]['token_ids'][:65],return_states=True)
causal_abs=float(np.max(abs(fs['activation'][:65]-ss['activation'])));causal_rel=float(np.max(abs(fs['activation'][:65]-ss['activation'])/(1+abs(ss['activation']))))
result={'name':name,'revision':m.meta['commit'],'mode':'fp16_rounding_approximation' if half else 'float32','protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),'observations':observations,'single_tokens':singles,'causal_prefix_max_abs_error':causal_abs,'causal_prefix_max_scaled_error':causal_rel,'seconds':time.time()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
file=dest/('fp16_approx.json' if half else 'float32.json');assert not file.exists();file.write_text(json.dumps(result,indent=2));print('DONE',file,'seconds',result['seconds'],'RSS',result['peak_rss_bytes'],'singleThe',singles['510']['contribution'][0][0],'singlePeriod',singles['15']['contribution'][0][0],flush=True)

# Precision sensitivity for every earlier checkpoint, before moving to another revision.
if not half and name!="main":
 import subprocess
 subprocess.run([sys.executable,__file__,name,"--half"],check=True)
