"""Streaming CPU FP64 tied-unembedding audit. Loads one tensor, never a model.
Actual runtime GPU-logit metrics remain primary; reconstruction errors are reported.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import argparse,hashlib,json,math,time,resource
from pathlib import Path
from collections import defaultdict
import torch
from safetensors import safe_open
def j(p):return json.loads(p.read_text())

def load(p):return torch.load(p,map_location='cpu',weights_only=True)

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def th(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
P=None;WEIGHT=None # Bound explicitly by CLI; scientific functions unchanged.
WEIGHT_SHA='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
FAMILIES=('remove_pc1','orthogonal_control','position');LAYERS=tuple(range(4,12));QROWS=(0,1,126,127,254,510)
def loss(z,t):return z.logsumexp(-1)-z.gather(-1,t[:,None]).flatten()
def kl(a,b):
 la=a.log_softmax(-1);lb=b.log_softmax(-1);return(la.exp()*(la-lb)).sum(-1)
def center(d):return d-d.mean(-1,keepdim=True)
def family_metrics(p,c,singles,joint,t):
 lp=loss(p,t);ls=torch.stack([loss(x,t)for x in singles]);add=p.clone()
 for z in singles:add.add_(z-p)
 lj=loss(joint,t);la=loss(add,t);ss=(ls-lp).sum(0);jj=lj-la;rr=la-lp-ss;nn=lj-lp-ss;diff=center(joint-add)
 d={'loss_p':lp,'loss_joint':lj,'loss_additive':la,'singleton_losses':ls,'sum_singleton_delta':ss,'J':jj,'R':rr,'joint_minus_sum':nn,'identity_error':(nn-jj-rr).abs(),
 'kl_joint_additive':kl(joint,add),'kl_additive_joint':kl(add,joint),'kl_p_additive':kl(p,add),'kl_native_additive':kl(c,add),
 'centered_logit_l2':diff.norm(dim=-1),'centered_logit_rms':diff.square().mean(-1).sqrt(),'centered_logit_max_abs':diff.abs().max(-1).values,
 'joint_delta_centered_l2':center(joint-p).norm(dim=-1),'additive_delta_centered_l2':center(add-p).norm(dim=-1),
 'agreement_joint_additive':joint.argmax(-1).eq(add.argmax(-1)),'agreement_p_additive':p.argmax(-1).eq(add.argmax(-1)),'agreement_native_additive':c.argmax(-1).eq(add.argmax(-1))}
 # An elementary independent target-cancellation check.
 rpartition=add.logsumexp(-1)+7*p.logsumexp(-1)-sum(z.logsumexp(-1)for z in singles)
 jpartition=joint.logsumexp(-1)-add.logsumexp(-1)-(joint-add).gather(-1,t[:,None]).flatten()
 return d,add,max(float((rpartition-rr).abs().max()),float((jpartition-jj).abs().max()))
def main():
 started=time.time();m=j(P/'manifest.json');assert m.get('status')=='completed'and j(P/'audit.json')['status']=='passed'
 assert sha(WEIGHT)==WEIGHT_SHA;torch.set_num_threads(4)
 with safe_open(str(WEIGHT),framework='pt',device='cpu')as f:
  keys=[k for k in ('wte.weight','transformer.wte.weight')if k in f.keys()];assert len(keys)==1;w32=f.get_tensor(keys[0])
 assert w32.shape==(50257,768)and w32.dtype==torch.float32
 wf={'shape':list(w32.shape),'dtype':str(w32.dtype),'sha256':th(w32)};assert wf==m['readout']['weight']and m['readout']['bias']is None and m['readout']['tied_embedding_storage']
 w=w32.double();del w32
 raw=load(P/'per_token.pt');items=load(P/'tokens.pt');tokens={(x['corpus'],x['window']):x['tokens']for x in items};bos=j(P/'data_manifest.json')['literals']['start_special']['id']
 stat={};sample={'rows':0,'max_absolute_logit_error':0.,'max_RMS_logit_error':0.,'max_relative_L2_error':0.,'max_gamma_bound_ratio':0.,'max_NLL_error':0.,'max_KL_either_direction':0.,'zero_reference_norm_rows':0,'zero_reference_nonzero_error_rows':0,'relative_L2_definition':'L2(CPU_FP64 minus saved_GPU_FP32)/L2(saved_GPU_FP32), reported only for nonzero reference norm; zero-norm rows counted separately'}
 signs=defaultdict(lambda:{'changes':0,'max_runtime_absolute_at_changed_sign':0.});partitionerr=0.;sampled_add_error=0.;allrows=0;inputcount=0
 def observe(label,a,b,where):
  assert a.shape==b.shape and torch.isfinite(a).all()and torch.isfinite(b).all()
  d=(a.double()-b.double()).abs();mx=float(d.max());entry=stat.setdefault(label,{'max_abs':0.,'sum_abs':0.,'elements':0,'max_location':None,'nonzero':0})
  if mx>entry['max_abs']:entry['max_abs']=mx;entry['max_location']=where
  entry['sum_abs']+=float(d.sum());entry['elements']+=d.numel();entry['nonzero']+=int(torch.count_nonzero(d))
 def sampled(a,b,h,where):
  # Higham-style conservative FP32 dot-product bound, frozen prospectively.
  nonlocal sample
  assert a.shape==b.shape;d=a-b.double();absolute=d.abs();u=2**-24;n=2*768+2;gamma=n*u/(1-n*u)
  bound=gamma*(h.double().abs()@w.abs().T)+n*2**-126+1e-12
  assert bool((absolute<=bound).all()),('FP32 accumulation bound',where,float((absolute-bound).max()))
  delta=absolute.max(-1).values
  k1,k2=kl(a,b.double()),kl(b.double(),a);assert bool((k1<=delta.square()/2+1e-10).all())and bool((k2<=delta.square()/2+1e-10).all())
  denom=b.double().norm(dim=-1);valid=denom>0;relative=d.norm(dim=-1)[valid]/denom[valid]
  sample['zero_reference_norm_rows']+=int((~valid).sum());sample['zero_reference_nonzero_error_rows']+=int(((~valid)&(d.norm(dim=-1)>0)).sum())
  sample['rows']+=len(a);sample['max_absolute_logit_error']=max(sample['max_absolute_logit_error'],float(delta.max()));sample['max_RMS_logit_error']=max(sample['max_RMS_logit_error'],float(d.square().mean(-1).sqrt().max()));sample['max_relative_L2_error']=max(sample['max_relative_L2_error'],float(relative.max())if relative.numel()else 0.);sample['max_gamma_bound_ratio']=max(sample['max_gamma_bound_ratio'],float((absolute/bound).max()));sample['max_KL_either_direction']=max(sample['max_KL_either_direction'],float(torch.maximum(k1,k2).max()))
  return delta
 for name in j(P/'hidden_files.json'):
  inputcount+=1;s=Path(name).stem;_,c,window,_,ident=s.split('__');window=int(window);hs=load(P/'hidden'/name);ad=load(P/'additive'/name);x=tokens[c,window].clone()
  if ident=='bos':x[0]=bos
  targets=x[1:];samplefile=P/'readout_rows'/name;gpu=load(samplefile)if samplefile.exists()else None
  if gpu is not None:
   assert gpu['query_rows'].tolist()==list(QROWS)
   for mode in tuple(hs)+('current','fullrank'):
    hh=hs['P'if mode in('current','fullrank')else mode][0,list(QROWS)];z=hh.double()@w.T;actual=gpu[mode];delta=sampled(z,actual,hh,(s,mode))
    nerr=(loss(z,gpu['targets'])-loss(actual.double(),gpu['targets'])).abs();assert bool((nerr<=2*delta+1e-10).all());sample['max_NLL_error']=max(sample['max_NLL_error'],float(nerr.max()))
   for family in FAMILIES:
    recons={mode:hs[mode][0,list(QROWS)].double()@w.T for mode in ['C','P',family+'_all']+[family+'_L'+str(l)for l in LAYERS]}
    d,add,pe=family_metrics(recons['P'],recons['C'],[recons[family+'_L'+str(l)]for l in LAYERS],recons[family+'_all'],gpu['targets']);partitionerr=max(partitionerr,pe);sampled_add_error=max(sampled_add_error,float((add-gpu[family+'_additive']).abs().max()));del recons,d,add
  for q0 in range(0,511,32):
   q1=min(q0+32,511);tt=targets[q0:q1];zz={mode:hs[mode][0,q0:q1].double()@w.T for mode in ('C','P')}
   def direct(mode,z):
    zraw=raw[s+'__'+mode];where=(s,mode,q0,q1)
    observe('model_NLL',loss(z,tt),zraw['loss'][q0:q1],where);observe('model_KL_C',kl(zz['C'],z),zraw['kl'][q0:q1],where);observe('model_KL_P',kl(zz['P'],z),zraw['kl_p'][q0:q1],where);observe('model_prediction',z.argmax(-1),zraw['pred'][q0:q1],where)
   direct('C',zz['C']);direct('P',zz['P'])
   for family in FAMILIES:
    singles=[]
    for l in LAYERS:
     mode=family+'_L'+str(l);z=hs[mode][0,q0:q1].double()@w.T;direct(mode,z);singles.append(z)
    mode=family+'_all';joint=hs[mode][0,q0:q1].double()@w.T;direct(mode,joint);d,add,pe=family_metrics(zz['P'],zz['C'],singles,joint,tt);partitionerr=max(partitionerr,pe)
    for k,v in d.items():
     saved=ad[family][k][:,q0:q1]if k=='singleton_losses'else ad[family][k][q0:q1];observe('additive_'+k,v,saved,(s,family,q0,q1))
     if k in('J','R','joint_minus_sum'):
      flip=v.sign()!=saved.sign();a=signs[k];a['changes']+=int(flip.sum())
      if bool(flip.any()):a['max_runtime_absolute_at_changed_sign']=max(a['max_runtime_absolute_at_changed_sign'],float(saved[flip].abs().max()))
    del singles,joint,d,add
   allrows+=(q1-q0)*29
  del hs,ad,gpu
  print('CPU_READOUT_INPUT_DONE',inputcount,s,'RSS_KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<3*1024**2
 for entry in stat.values():entry['mean_abs']=entry.pop('sum_abs')/entry['elements']
 predstats=stat.pop('model_prediction');stat['model_prediction_agreement']={'mismatches':predstats['nonzero'],'rows':predstats['elements'],'fraction_agreement':1-predstats['nonzero']/predstats['elements']}
 assert inputcount==32 and allrows==928*511 and sample['rows']==744
 return {'status':'passed','weight_file_sha256':WEIGHT_SHA,'loaded_tensor_key':keys[0],'loaded_weight_fingerprint':wf,'inputs':32,'hidden_trajectories':928,'query_rows_reconstructed':allrows,'sampled_actual_GPU_comparison':sample,'sampled_additive_logit_max_error':sampled_add_error,'all_query_runtime_metric_errors':stat,'per_query_sign_changes':dict(signs),'R_target_cancellation_J_partition_max_error':partitionerr,'wall_seconds':time.time()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'pretrained_model_loaded':False,'single_weight_tensor_loaded':True,'gpu_used':False,'limits':'CPU FP64 hidden-to-logit reconstruction is approximate. Direct raw-GPU bound validated only on six frozen queries of four natural inputs. Runtime saved metrics remain primary; all-query errors and any sign/argmax changes are disclosed, not silently substituted. Successful complete GPU logits/context arrays outside fixed snapshots are absent.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True)
 ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 P=args.run.resolve();WEIGHT=args.checkpoint.resolve()
 assert P.is_dir() and WEIGHT.is_file() and not args.output.exists(),'Existing raw run, checkpoint and new output path required'
 args.output.write_text(json.dumps({'status':'checking','model_loaded':False,'gpu_used':False})+'\n')
 try:
  result=main();args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 except BaseException as error:
  args.output.write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error),'model_loaded':False,'gpu_used':False},indent=2)+'\n');raise
