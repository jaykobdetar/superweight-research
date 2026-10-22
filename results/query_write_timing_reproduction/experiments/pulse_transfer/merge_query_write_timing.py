"""Immutable T21 planned-segment merger; no tokenizer/model/GPU operations.

Both324-return segments must complete all final guards before any composite
is created. All32 shards are verified and copied byte-for-byte, never linked.
"""
import argparse,copy,json,os,resource,time
from pathlib import Path
from query_write_timing_storage import EvidenceWriter,NORMAL_LIMIT,FAILURE_LIMIT,DISK_RESERVE,METADATA_ALLOWANCE,sha,json_write,read_npz

ROOT=Path(__file__).resolve().parent
CORPORA={'A':('wiki_validation','utilitarianism'),'B':('ethics','optimizers_new')}
EQUAL_FIELDS=('source_hashes','input_dependencies','metadata_dependencies','freeze_digest','model_id','revision','dtype','attention','runtime_versions','precision_settings_before','arithmetic_runtime','calibration_map_digest','calibration_fit','physical_execution_lengths','boundaries','arms','score_views','normal_limit_bytes','failure_limit_bytes','disk_reserve_bytes')
def read(path):return json.loads(Path(path).read_text())
def copy_exact(source,dest):
 source,dest=Path(source),Path(dest);assert source.is_file() and not source.is_symlink()
 assert all(not p.is_symlink() for p in source.absolute().parents),'Symlink source ancestor'
 dest.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as a,dest.open('xb') as b:
  for block in iter(lambda:a.read(4*1024**2),b''):b.write(block)
  b.flush();os.fsync(b.fileno())
 assert sha(source)==sha(dest)
def validate(segment,label):
 assert segment.is_dir() and not segment.is_symlink();m=read(segment/'manifest.json');inventory=read(segment/'numeric_inventory.json')
 assert not any((segment/name).exists() for name in ('failure_state.npz','checkpoint_failure.json','merge_failure.json')),'Segment has preserved failure marker'
 assert not list((segment/'inputs').glob('*__partial.npz')),'Segment has preserved partial boundary'
 for checkpoint in (segment/'checkpoints').glob('*/manifest.json'):assert read(checkpoint)['status']!='failed','Segment has failed versioned checkpoint'
 assert m['status']=='completed' and m['segment']==label and m['execution_type']=='planned_segment' and m['all_final_guards_passed'] is True
 assert m['forward_attempts']==m['forward_passes']==m['fully_processed_forwards']==324
 assert m['evaluation_forwards']==m['suffix_forwards']==160 and m['anchor_forwards']==4 and m['calibration_forwards']==0
 assert m['model_load_attempted'] is True and m['model_object_returned'] is True and m['model_loaded'] is True
 assert m['model_hash_before']==m['model_hash_after'] and m['parameter_versions_unchanged'] is True
 assert m['precision_settings_before']==m['precision_settings_after']
 assert m['normal_limit_bytes']==NORMAL_LIMIT and m['failure_limit_bytes']==FAILURE_LIMIT and m['disk_reserve_bytes']==DISK_RESERVE
 expected={f'inputs/{c}__{w}__{i}__{b}.npz' for c in CORPORA[label] for w in range(4) for i in ('natural','bos') for b in ('000',)}
 assert set(inventory)==expected and {p.relative_to(segment).as_posix() for p in (segment/'inputs').iterdir()}==expected
 for name,entry in inventory.items():
  p=segment/name;assert entry=={'sha256':sha(p),'bytes':p.stat().st_size}
 for name,h in m['source_hashes'].items():assert sha(segment/'frozen_sources'/name)==h
 for name,h in m['input_dependencies'].items():assert sha(segment/name)==h
 dm=read(segment/'data_manifest.json');reference=segment/'calibration_reference'
 assert {p.name for p in reference.iterdir()}=={'manifest.json','calibration_maps.npz','calibration_directions.npz'}
 assert all(p.is_file() and not p.is_symlink() for p in reference.iterdir())
 assert sha(reference/'manifest.json')==dm['calibration_reference_manifest_sha256']
 for name in ('calibration_maps.npz','calibration_directions.npz'):assert sha(reference/name)==dm['calibration_reference_files'][name]
 for name,h in m['metadata_dependencies'].items():assert sha(segment/'metadata_dependencies'/name.removeprefix('../').replace('/','__'))==h
 return m,inventory

def merge(a):
 started=time.time();writer=None
 for source in (a.segment_a,a.segment_b):assert not a.out.resolve().is_relative_to(source.resolve()),'Output inside immutable segment'
 am,ai=validate(a.segment_a,'A');bm,bi=validate(a.segment_b,'B')
 assert bm['previous_segment_A']=={'manifest_sha256':sha(a.segment_a/'manifest.json'),'numeric_inventory_sha256':sha(a.segment_a/'numeric_inventory.json')},'B predecessor binding differs from supplied A'
 for key in EQUAL_FIELDS:assert am[key]==bm[key],('Segment provenance mismatch',key)
 assert am['model_hash_before']==bm['model_hash_before'],'Segment parameter identity differs'
 for name,h in am['source_hashes'].items():assert sha(ROOT/name)==h,('Frozen local source changed',name)
 # Every non-input dependency is exactly shared, including all three frozen F
 # references and the parameter/selection/checkpoint metadata used by anchors.
 reference_files={p.relative_to(a.segment_a/'calibration_reference').as_posix():sha(p) for p in (a.segment_a/'calibration_reference').rglob('*') if p.is_file()}
 assert set(reference_files)=={'manifest.json','calibration_maps.npz','calibration_directions.npz'}
 for name,h in reference_files.items():assert sha(a.segment_b/'calibration_reference'/name)==h
 try:
  writer=EvidenceWriter(a.out);writer.guard('complete_composite_before_copy',sum(e['bytes'] for e in ai.values())+sum(e['bytes'] for e in bi.values())+METADATA_ALLOWANCE)
  for name,h in am['source_hashes'].items():copy_exact(a.segment_a/'frozen_sources'/name,a.out/'frozen_sources'/name)
  executed_merger=sha(__file__);copy_exact(__file__,a.out/'execution_provenance'/'merge_query_write_timing.py')
  for name,h in am['input_dependencies'].items():copy_exact(a.segment_a/name,a.out/name)
  for name,h in am['metadata_dependencies'].items():
   relative=name.removeprefix('../').replace('/','__');copy_exact(a.segment_a/'metadata_dependencies'/relative,a.out/'metadata_dependencies'/relative)
  for name,h in reference_files.items():copy_exact(a.segment_a/'calibration_reference'/name,a.out/'calibration_reference'/name)
  provenance={}
  for label,segment,manifest,inventory in (('A',a.segment_a,am,ai),('B',a.segment_b,bm,bi)):
   journal=(segment/'journal.jsonl').read_bytes().splitlines();chain='0'*64
   import hashlib
   for line in journal:
    event=json.loads(line);assert event['previous_sha256']==chain;chain=hashlib.sha256(line).hexdigest()
   for name in ('manifest.json','numeric_inventory.json','journal.jsonl','disk_checks.json'):copy_exact(segment/name,a.out/'execution_provenance'/label/name)
   provenance[label]={'manifest_sha256':sha(segment/'manifest.json'),'numeric_inventory_sha256':sha(segment/'numeric_inventory.json'),'journal_sha256':sha(segment/'journal.jsonl'),'journal_chain_sha256':chain,'forward_attempts':manifest['forward_attempts'],'forward_passes':manifest['forward_passes'],'fully_processed_forwards':manifest['fully_processed_forwards'],'all_final_guards_passed':True,'resources':{k:manifest[k] for k in ('wall_seconds','peak_process_rss_kib','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')},'model_hash_before':manifest['model_hash_before'],'model_hash_after':manifest['model_hash_after'],'parameter_versions_unchanged':manifest['parameter_versions_unchanged'],'precision_settings_before':manifest['precision_settings_before'],'precision_settings_after':manifest['precision_settings_after']}
   for name,entry in inventory.items():
    copy_exact(segment/name,a.out/name);assert entry=={'sha256':sha(a.out/name),'bytes':(a.out/name).stat().st_size};writer.inventory[name]=entry;writer.journal({'event':'copied_verified_boundary_shard','segment':label,'path':name,**entry})
   writer.checkpoint({'phase':'copied_segment','segment':label,'copied_shards':len(writer.inventory)})
  assert set(writer.inventory)==set(ai)|set(bi) and len(writer.inventory)==32
  assert all(sha(a.out/'frozen_sources'/name)==h for name,h in am['source_hashes'].items()) and sha(__file__)==executed_merger
  assert validate(a.segment_a,'A')==(am,ai) and validate(a.segment_b,'B')==(bm,bi),'Segment changed during composite assembly'
  manifest={k:copy.deepcopy(am[k]) for k in EQUAL_FIELDS}
  manifest.update(study='query_write_timing',status='completed',execution_type='planned_sequential_segments',forward_attempts=648,forward_passes=648,fully_processed_forwards=648,evaluation_forwards=320,suffix_forwards=320,anchor_forwards=8,calibration_forwards=0,original_segments=provenance,all_final_guards_passed=True,model_hash_before=am['model_hash_before'],model_hash_after=bm['model_hash_after'],parameter_versions_unchanged=True,precision_settings_after=bm['precision_settings_after'],merger_sha256=executed_merger,merger_model_forwards=0,resource_scope='Individual complete segments; no inferred simultaneous global peak.',merge_wall_seconds=time.time()-started,merge_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,normal_limit_bytes=NORMAL_LIMIT)
  writer.checkpoint({'phase':'completed','copied_shards':32},manifest=manifest,final=True);return manifest
 except BaseException as error:
  if writer is not None and writer.owns():json_write(a.out/'merge_failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'copied_shards':len(writer.inventory),'model_forwards':0,'merger_sha256':sha(__file__)})
  raise

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('segment-a','segment-b','out'):ap.add_argument('--'+name,type=Path,required=True)
 merge(ap.parse_args())
if __name__=='__main__':main()
