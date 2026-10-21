"""Independent statistics/record-guard fixture; no model or source-token load."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,copy,json,hashlib,time,resource
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import analyze_gate_wiki_calibration as a
from gate_wiki_calibration import mode_spec

def main():
 start=time.time();torch.set_num_threads(1);shards=[]
 for ci,c in enumerate(a.CORPORA):
  for i in ['natural','bos']:
   for w in range(4):
    pref=f'gate__{c}__{w}__512__{i}__';raw={};scale=w+1
    for mode in list(a.CONDITIONS)+(['repeat','identity']if i=='natural'and w==0 else []):
     delta=0
     if mode in a.CONDITIONS:
      family,center,basis,layers=mode_spec(mode)
      if family:delta=(10 if center=='M'else 2*(center=='F')+3*(basis=='F')+5*(center==basis=='F'))*scale*(2 if family=='orthogonal'else 1)*(3 if len(layers)==8 else 1)
     loss=torch.arange(511,dtype=torch.float32)/512+2+delta
     raw[pref+mode]={'loss':loss,'kl':torch.zeros(511),'kl_p':torch.zeros(511),'pred':torch.zeros(511,dtype=torch.long),'agreement_p':torch.ones(511,dtype=torch.bool),'score_index':1,'final_hidden_sha256':'1'*64,'token_sha256':'2'*64}
    shards.append({'metadata':{'corpus':c,'identity':i,'window':w},'per_token':raw,'norms':{}})
 result=a.summarize(shards);assert len(result['window_metrics'])==2328 and len(result['summary'])==576 and len(result['contrasts'])==912
 specs={x['name']:x for x in result['contrasts']if x['family']=='remove'and x['layer_set']=='L10'and x['view']=='full'and x['identity']=='natural'and x['corpus']==a.CORPORA[0]}
 expected={'center_at_O':2,'center_at_F':7,'direction_at_O':3,'direction_at_F':8,'interaction':5,'M8_minus_OO':10,'M8_minus_FF':0}
 bootstrap=np.random.default_rng(20261021).integers(0,4,(2000,4));contrast_cases=0
 for row in result['contrasts']:
  if row['family']not in ['remove','orthogonal']:continue
  unit=expected[row['name']]*(2 if row['family']=='orthogonal'else 1)*(3 if row['layer_set']=='all'else 1);v=np.arange(1,5)*unit
  assert row['delta_nll']['per_window']==v.tolist() and row['delta_nll']['mean']==v.mean();assert row['delta_nll']['ci95']==np.quantile(v[bootstrap].mean(1),[.025,.975]).tolist();contrast_cases+=1
 prefix='gate__wiki_validation__0__512__natural__';rejected=[]
 for case in ['unknown_corpus','unknown_identity','unknown_window','prefix','missing_mode','extra_mode','loss_shape','loss_dtype','pred_dtype','pred_range','agreement_dtype','nonfinite','input_token_hash','current_mean_preserving_permutation','current_hidden','repeat_loss','identity_loss','C_KL','P_KL','agreement','missing_input','duplicate_input']:
  ss=list(shards);ss[0]=copy.deepcopy(shards[0]);z=ss[0]['per_token'];p=z[prefix+'P'];cur=z[prefix+'current']
  if case=='unknown_corpus':ss[0]['metadata']['corpus']='unregistered'
  elif case=='unknown_identity':ss[0]['metadata']['identity']='other'
  elif case=='unknown_window':ss[0]['metadata']['window']=4
  elif case=='prefix':z['incorrect__C']=z.pop(prefix+'C')
  elif case=='missing_mode':del z[prefix+'remove_OF_all']
  elif case=='extra_mode':z[prefix+'unregistered']=copy.deepcopy(p)
  elif case=='loss_shape':z[prefix+'remove_OF_all']['loss']=torch.ones(510)
  elif case=='loss_dtype':z[prefix+'remove_OF_all']['loss']=z[prefix+'remove_OF_all']['loss'].double()
  elif case=='pred_dtype':z[prefix+'remove_OF_all']['pred']=torch.zeros(511,dtype=torch.int32)
  elif case=='pred_range':z[prefix+'remove_OF_all']['pred'][1]=50257
  elif case=='agreement_dtype':z[prefix+'remove_OF_all']['agreement_p']=torch.ones(511)
  elif case=='nonfinite':z[prefix+'remove_OF_all']['loss'][1]=float('nan')
  elif case=='input_token_hash':z[prefix+'remove_OF_all']['token_sha256']='3'*64
  elif case=='current_mean_preserving_permutation':cur['loss'][1:127]=cur['loss'][1:127].flip(0)
  elif case=='current_hidden':cur['final_hidden_sha256']='3'*64
  elif case=='repeat_loss':z[prefix+'repeat']['loss'][0]+=1
  elif case=='identity_loss':z[prefix+'identity']['loss'][510]+=1
  elif case=='C_KL':z[prefix+'C']['kl'][1]=.1
  elif case=='P_KL':p['kl_p'][1]=.1
  elif case=='agreement':z[prefix+'remove_OF_all']['agreement_p'][1]=False
  elif case=='missing_input':ss.pop()
  else:ss[-1]=ss[0]
  try:a.summarize(ss)
  except (AssertionError,KeyError):rejected.append(case)
  else:raise AssertionError(('invalid fixture accepted',case))
 out={'status':'passed','source_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'window_rows':2328,'summary_rows':576,'contrast_rows':912,'independently_checked_primary_and_M8_contrasts':contrast_cases,'common_bootstrap_indices_recomputed':True,'negative_cases_rejected':rejected,'scope':'Statistics and record guards only; operation/snapshot/suffix/provenance checks remain separately required. Empty norm fixture is deliberate, not a full production-shard validation.','model_forwards':0,'gpu_used':False,'wall_seconds':time.time()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (ROOT/'review/gate_wiki_statistics_independent.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
