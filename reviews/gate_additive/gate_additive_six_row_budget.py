"""Storage-only prospective six-row revision; no inference or tokenizer work.
Re-serialize only the four sampled readout shards, first reproducing the exact
original eight-row sizes from the full actual-writer fixture.
"""
import json,resource,sys,tempfile,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));torch.set_num_threads(1)
from gpt2_gate_localization import CONDITIONS,FAMILIES,mode_spec
from run_transfer import sha
OLD=(0,1,31,63,126,127,254,510);NEW=(0,1,126,127,254,510)
full=json.loads((ROOT/'review/gate_additive_full_layout.json').read_text())
V=50257;rowsizes={};started=time.time()
with tempfile.TemporaryDirectory(prefix='gate-additive-six-row-sizing-')as td:
 p=Path(td)
 for corpus in('wiki_validation','mill','wollstonecraft','optimizers_new'):
  filename=f'gate__{corpus}__0__512__natural.pt';sizes={}
  for name,rows in(('original_eight',OLD),('proposed_six',NEW)):
   data={'query_rows':torch.tensor(rows),'targets':torch.zeros(len(rows),dtype=torch.long)}
   for mode in CONDITIONS:
    data[mode]=torch.zeros(len(rows),V)
    family,affected=mode_spec(mode)
    if family and len(affected)==8:data[family+'_additive']=torch.zeros(len(rows),V,dtype=torch.float64)
   f=p/filename;torch.save(data,f);sizes[name]=f.stat().st_size
   if name=='original_eight':assert sizes[name]==full['file_sizes']['readout_rows/'+filename],(filename,sizes[name],full['file_sizes']['readout_rows/'+filename])
   f.unlink();del data
  sizes['saved_bytes']=sizes['original_eight']-sizes['proposed_six'];rowsizes[corpus]=sizes
saved=sum(s['saved_bytes']for s in rowsizes.values());normal=full['total_serialized_bytes']-saved
cap=2*1024**3;failure=64*1024**2;float_json=16*1024**2;analysis=32*1024**2
checkpoints=json.loads((ROOT/'review/gate_additive_full_layout_increments.json').read_text())
updated=[]
for r in checkpoints:
 reduction=rowsizes[r['corpus']]['saved_bytes']if r['window']==0 and r['identity']=='natural'else 0
 updated.append({**r,'six_row_increment_bytes':r['increment_bytes']-reduction})
result={'status':'passed_prospective_storage_revision','actual_model_forwards':0,'gpu_used':False,
 'original_rows':list(OLD),'proposed_rows':list(NEW),'four_original_row_files_exact_sizes_reproduced':True,
 'files':rowsizes,'saved_bytes':saved,'original_full_writer_bytes':full['total_serialized_bytes'],
 'projected_six_row_full_writer_bytes':normal,'cap_bytes':cap,'failure_reserve_bytes':failure,
 'additional_float_json_growth_allowance_bytes':float_json,'additional_analysis_output_allowance_bytes':analysis,
 'all_in_projection_bytes':normal+failure+float_json+analysis,
 'remaining_margin_bytes':cap-(normal+failure+float_json+analysis),
 'proposed_next_input_bound_bytes':{'snapshot_and_rows':146000000,'rows_only':104000000,'ordinary':58000000},
 'measured_class_maximum_bytes':{'snapshot_and_rows':max(r['six_row_increment_bytes']for r in updated if r['corpus']=='wiki_validation'and r['window']==0 and r['identity']=='natural'),
 'rows_only':max(r['six_row_increment_bytes']for r in updated if r['corpus']!='wiki_validation'and r['window']==0 and r['identity']=='natural'),
 'ordinary':max(r['six_row_increment_bytes']for r in updated if not(r['window']==0 and r['identity']=='natural'))},
 'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'wall_seconds':time.time()-started,
 'dependencies':{'full_layout_json_sha256':sha(ROOT/'review/gate_additive_full_layout.json'),
 'runner_measured_sha256':full['source_hashes']['gpt2_gate_additive.py'],
 'failure_measured_sha256':full['source_hashes']['gate_additive_failure.py']},
 'limits':'Exact serialized readout replacement, projected onto the measured full writer. Extra float-JSON and analysis reserves are explicit prospective allowances, not measurements of future results. No scientific inputs/arms/hidden coverage changed. No full writer rerun.'}
assert result['remaining_margin_bytes']>0 and result['peak_rss_kib']<3*1024**2
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
