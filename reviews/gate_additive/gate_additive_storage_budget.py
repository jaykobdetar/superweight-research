"""Prospective serialization sizing only; no model, tokenizer or corpus execution."""
import json,resource,tempfile
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
N,V,D,Q,H=32,50257,768,511,12
with tempfile.TemporaryDirectory(prefix='additive-storage-unit-')as td:
 p=Path(td)
 def size(name,value):
  f=p/(name+'.pt');torch.save(value,f);return f.stat().st_size
 unit={}
 unit['main_hidden']=size('main_hidden',{'hidden':torch.zeros(1,512,D)})
 unit['main_gate_state']=size('main_gate_state',{k:torch.zeros(H,Q)for k in('b','mass','target_b','target_mass')})
 unit['suffix_gate_state']=size('suffix_gate_state',{k:torch.zeros(H,127)for k in('b','mass','target_b','target_mass')})
 unit['raw_logit_rows']=size('raw_logit_rows',{'queries':torch.tensor([0,1,31,63,126,127,254,510]),'logits':torch.zeros(8,V)})
 unit['additive_logit_rows']=size('additive_logit_rows',{'queries':torch.tensor([0,1,31,63,126,127,254,510]),'logits':torch.zeros(8,V,dtype=torch.float64)})
 # Exact primary record shapes, with extra per-row FP64 CE independently recomputed.
 unit['main_metrics']=size('main_metrics',{'loss':torch.zeros(Q),'kl':torch.zeros(Q),'kl_p':torch.zeros(Q),
  'pred':torch.zeros(Q,dtype=torch.int64),'agreement_p':torch.zeros(Q,dtype=torch.bool),'loss64':torch.zeros(Q,dtype=torch.float64)})
 # Existing readout helper has18 scalar FP64 vectors,8 singleton loss vectors,and3 bool vectors.
 unit['additive_metrics']=size('additive_metrics',{'float_vectors':torch.zeros(26,Q,dtype=torch.float64),'bool_vectors':torch.zeros(3,Q,dtype=torch.bool)})
 unit['p_source_states']=size('p_source_states',{l:{k:torch.zeros(1,H,1,64)for k in('q','k','v')}for l in range(4,12)})
# Conservative allowance uses actual prior snapshot size per state, rounded up to12MiB.
counts={'main_hidden':32*29,'main_gate_state':32*56,'suffix_gate_state':32*56,
 'raw_logit_rows':4*31,'additive_logit_rows':4*3,'main_metrics':1000,'additive_metrics':32*3,'p_source_states':32}
components={k:{'unit_bytes':unit[k],'count':n,'bytes':unit[k]*n}for k,n in counts.items()}
allowances={'four_attention_states':4*12*1024**2,'metadata_and_scalar_json':48*1024**2,
 'copied_calibration_inputs_code':8*1024**2,'bounded_failure_capture':64*1024**2}
total=sum(v['bytes']for v in components.values())+sum(allowances.values())
prior=root/'gpt2_gate_localization_run';prior_sizes={}
for name in ('gates','suffix','prefix','traces.pt','attention_snapshots.pt','local_norms','source_states.pt'):
 f=prior/name;prior_sizes[name]=sum(q.stat().st_size for q in f.rglob('*')if q.is_file())if f.is_dir()else f.stat().st_size
m=json.loads((prior/'manifest.json').read_text())
result={'status':'prospective_unit_serialization_measured','no_model_or_tokenizer':True,'gpu_used':False,
 'components':components,'allowances_bytes':allowances,'projected_bytes_including_failure_allowance':total,
 'cap_bytes':2*1024**3,'cap_margin_bytes':2*1024**3-total,'reserve_bytes':2*1024**3,
 'prior_measured_artifacts':prior_sizes,'prior_peak_process_rss_kib':m['peak_process_rss_kib'],
 'prior_peak_gpu_allocated_bytes':m['peak_gpu_allocated_bytes'],
 'prospective_readout_cpu_payload_bytes':{'fp64_accumulator':Q*V*8,'P_logits_fp32':Q*V*4,'C_logits_fp32':Q*V*4,
 'current_logits_fp32':Q*V*4,'chunk32_float64_matrix':32*V*8},
 'limitations':'Unit tensor serialization plus explicit allowances, not a measurement of the future complete writer. Final writer must measure exact full synthetic layout before production. No statement that future process RSS is measured.',
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
assert result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
