"""Separately reviewed T19 bounded recovery adapter; original sources untouched.

Default prepares new staged sources only. --execute is an explicit later launch.
All scientific helpers and the complete inner condition/control loop retain their
original AST. Only roster, counters, commit gate and provenance are instrumented.
"""
import argparse,ast,collections,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ORIGINAL_RUNNER_SHA='7ec659a8ddd786e77b1c2f2f19fe0b86178af2824d20915eac5fe784cbaf384f'
FORENSIC_SHA='e48c04df3786518377918c50b7ab98a3497ca3cd5fe73e9e1da62bc6587339b3'
PREFIX_SHA='39283ce12a732069a35e470f4f521ee643117521cb4a09dbbe489e1c8040b541'
PROPOSAL_SHA='57486de4c0eaa076b6d5c097ffdbbf28ba7fa0b276856c204d28467b9ce2105c'
CHECKPOINT='checkpoints/000018'
CONTINUITY='wiki_validation__0__natural'
RESERVE=2*1024**3
COUNTS={'A':{'total':616,'main':306,'suffix':306,'anchors':4,'new_scientific':546,'continuity':70},
        'B':{'total':546,'main':272,'suffix':272,'anchors':2,'new_scientific':546,'continuity':0}}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def copy_exact(p,q,h=None):
 p,q=Path(p),Path(q);assert p.is_file() and not p.is_symlink();h=h or sha(p);assert sha(p)==h
 q.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as src,q.open('xb') as dst:shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
 assert sha(q)==h

def journal(path):
 counts=collections.Counter();chain='0'*64;events=[]
 for line in Path(path).read_bytes().splitlines():
  d=json.loads(line);assert d['previous_sha256']==chain
  assert line==json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  chain=hashlib.sha256(line).hexdigest();counts[d['event']]+=1;events.append(d)
 return dict(counts),chain,events

def verify_original(original,forensic,prefix):
 original=Path(original);assert sha(forensic)==FORENSIC_SHA and sha(prefix)==PREFIX_SHA
 f=read(forensic);p=read(prefix);assert f['status']=='interrupted_no_final_manifest' and p['status']=='passed'
 assert not (original/'manifest.json').exists(),'Never reinterpret an existing final outcome'
 for name,entry in f['files'].items():
  path=original/name;assert path.is_file() and path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],('Original evidence changed',name)
 cp=read(original/CHECKPOINT/'manifest.json');inventory=read(original/CHECKPOINT/'numeric_inventory.json')
 assert sha(original/CHECKPOINT/'numeric_inventory.json')==p['committed_inventory_sha256']
 assert len(inventory)==16 and {Path(n).stem.rsplit('__',2)[0] for n in inventory}=={'wiki_validation','utilitarianism'}
 for n,v in inventory.items():assert (original/n).stat().st_size==v['bytes'] and sha(original/n)==v['sha256']
 assert sum(v['bytes'] for v in inventory.values())==p['committed_bytes']==233391179
 for n,h in cp['source_hashes'].items():assert sha(original/'frozen_sources'/n)==h
 assert cp['source_hashes']['gpt2_gate_native_value.py']==ORIGINAL_RUNNER_SHA
 counts,chain,events=journal(original/'journal.jsonl')
 assert counts==f['journal_counts'] and chain==f['journal_final_sha256']
 assert counts['model_returned']==1110 and counts['forward_completed']==1109 and counts['committed']==16
 assert cp['forward_passes']==cp['forward_attempts']==1092
 return cp,inventory,f

def context(root):
 c=read(Path(root)/'recovery_context.json');assert c['schema']=='native_value_bounded_recovery_v1'
 assert c['adapter_sha256']==sha(Path(root)/'gpt2_gate_native_value_recovery.py')
 return c

def verify_segment(path,segment):
 path=Path(path);m=read(path/'manifest.json');c=COUNTS[segment]
 assert m['status']==m['phase']=='completed' and m['execution_type']=='bounded_recovery_segment' and m['recovery_segment']==segment
 assert m['forward_attempts']==m['forward_passes']==c['total']
 assert (m['evaluation_forwards'],m['suffix_forwards'],m['anchor_forwards'])==(c['main'],c['suffix'],c['anchors'])
 assert m['parameter_versions_unchanged'] is True and m['model_hash_before']==m['model_hash_after']
 assert m['precision_settings_before']==m['precision_settings_after']
 for n,h in m['source_hashes'].items():assert sha(path/'frozen_sources'/n)==h
 inv=read(path/'numeric_inventory.json')
 for n,v in inv.items():assert (path/n).stat().st_size==v['bytes'] and sha(path/n)==v['sha256']
 expected={'ethics'} if segment=='A' else {'optimizers_new'}
 scientific={n for n in inv if Path(n).stem.rsplit('__',2)[0] in expected}
 assert len(scientific)==8 and len(inv)==(9 if segment=='A' else 8)
 counts,chain,events=journal(path/'journal.jsonl')
 assert counts['forward_attempt']==counts['model_returned']==counts['forward_completed']==c['total']
 if segment=='A':
  gate=read(path/'continuity_gate.json');assert gate['passed'] is True and gate['forward_passes_at_gate']==70
  assert gate['group']==CONTINUITY
  completed=[e for e in events if e['event']=='forward_completed']
  assert all(e['identity']['corpus']=='wiki_validation' for e in completed[:70])
  assert all(e['identity']['corpus']=='ethics' for e in completed[70:])
 return m,inv,{'counts':counts,'final_sha256':chain}

def recovery_manifest(root):
 c=context(root);cp,inventory,f=verify_original(c['original_run'],c['forensic'],c['prefix'])
 if c['segment']=='B':
  assert sha(Path(c['segment_a'])/'manifest.json')==c['segment_a_manifest_sha256']
  verify_segment(c['segment_a'],'A')
 return {'execution_type':'bounded_recovery_segment','recovery_segment':c['segment'],
  'recovery_expected_counts':COUNTS[c['segment']],'original_source_hashes':cp['source_hashes'],
  'recovery_adapter_sha256':c['adapter_sha256'],'recovery_proposal_sha256':PROPOSAL_SHA,
  'original_interrupted_run_sha256':FORENSIC_SHA,'original_after_guards_available':False,
  'original_missing_guards':f['missing_guards'],'retained_original_forwards':1092,
  'discarded_original_uncommitted_returns':18,'original_fully_processed_forwards':1109,
  'adapted_runner_ast_proof_sha256':sha(Path(root)/'recovery_ast_proof.json')}

def selected_group(root,corpus,window,identity):
 c=context(root)
 if c['segment']=='A':return corpus=='ethics' or (corpus,window,identity)==('wiki_validation',0,'natural')
 return corpus=='optimizers_new'

def recursive_mismatches(a,b,path=(),limit=16):
 import torch
 if isinstance(a,torch.Tensor):
  return [] if isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b) else [{'path':list(path),'kind':'tensor_type_shape_dtype_or_bytes'}]
 if isinstance(a,dict):
  if not isinstance(b,dict) or list(a)!=list(b):return [{'path':list(path),'kind':'dictionary_members_or_order'}]
  out=[]
  for k in a:
   out+=recursive_mismatches(a[k],b[k],path+(str(k),),limit)
   if len(out)>=limit:return out[:limit]
  return out
 if isinstance(a,(tuple,list)):
  if type(a)!=type(b) or len(a)!=len(b):return [{'path':list(path),'kind':'sequence_type_or_length'}]
  out=[]
  for i,(x,y) in enumerate(zip(a,b)):
   out+=recursive_mismatches(x,y,path+(str(i),),limit)
   if len(out)>=limit:return out[:limit]
  return out
 return [] if type(a)==type(b) and a==b else [{'path':list(path),'kind':'scalar_type_or_value'}]

def compare_npz_payload(reference,current,chunk_bytes=1024**2):
    """Compare every canonical uncompressedNPZ byte, not just fingerprints.

    .npy headers encode tensor dtype/shape; metadata.npy holds the complete
    ordered recursive scalar/container/type tree. ZIP compression headers and
    timestamps are the only excluded bytes. Streaming bounds extra RAM.
    """
    compared=0
    with zipfile.ZipFile(reference) as left,zipfile.ZipFile(current) as right:
        ln,rn=left.namelist(),right.namelist()
        if ln!=rn:return {'passed':False,'reason':'member_names_or_order','left_members':ln,'right_members':rn,'compared_uncompressed_bytes':0}
        assert len(ln)==len(set(ln)) and all(n.endswith('.npy') and '/' not in n for n in ln) and 'metadata.npy' in ln
        for name in ln:
            if left.getinfo(name).file_size!=right.getinfo(name).file_size:
                return {'passed':False,'reason':'member_uncompressed_size','member':name,'compared_uncompressed_bytes':compared}
            offset=0
            with left.open(name) as a,right.open(name) as b:
                while True:
                    aa,bb=a.read(chunk_bytes),b.read(chunk_bytes)
                    if aa!=bb:
                        pos=next((i for i,(x,y) in enumerate(zip(aa,bb)) if x!=y),min(len(aa),len(bb)))
                        return {'passed':False,'reason':'uncompressed_payload_byte','member':name,'member_offset':offset+pos,'compared_uncompressed_bytes':compared}
                    compared+=len(aa);offset+=len(aa)
                    if not aa:break
    return {'passed':True,'members':ln,'compared_uncompressed_bytes':compared,
            'comparison':'All ordered members including .npy dtype/shape headers, complete recursive metadata bytes, and all actual tensor bytes. Only ZIP container compression headers/timestamps excluded.'}

def finalize_group(root,writer,group,payload):
    c=context(root)
    if c['segment']=='A' and group!=CONTINUITY:
        gate=read(writer.out/'continuity_gate.json');assert gate['passed'] is True,'Continuity gate required before new scientific group'
    writer.save('inputs/'+group+'.npz',payload)
    if c['segment']=='A' and group==CONTINUITY:
        reference=Path(c['original_run'])/'inputs'/(CONTINUITY+'.npz')
        current=writer.out/'inputs'/(CONTINUITY+'.npz');comparison=compare_npz_payload(reference,current)
        _,_,events=journal(writer.out/'journal.jsonl');returns=sum(e['event']=='model_returned' for e in events)
        result={'group':group,**comparison,'reference_sha256':sha(reference),'replay_sha256':sha(current),
                'reference_actual':str(reference),'replay_actual':str(current),'forward_passes_at_gate':returns}
        assert returns==70;writer.guard('continuity_verdict',1024**2);put(writer.out/'continuity_gate.json',result)
        assert comparison['passed'],('Exact continuity mismatch; both actual full artifacts preserved',result)


def replace_once(text,old,new):
 assert text.count(old)==1,('Expected one exact adaptation point',old)
 return text.replace(old,new,1)

def instrument(source,segment):
 assert hashlib.sha256(source.encode()).hexdigest()==ORIGINAL_RUNNER_SHA
 changes=[]
 def edit(old,new,label):
  nonlocal source
  source=replace_once(source,old,new);changes.append(label)
 edit('ROOT=Path(__file__).resolve().parent\n','ROOT=Path(__file__).resolve().parent\nfrom gpt2_gate_native_value_recovery import recovery_manifest, selected_group, finalize_group\n','Import recovery orchestration')
 edit('    def progress():', '    meta.update(recovery_manifest(ROOT))\n    def progress():','Verify original provenance and label bounded segment before weights')
 edit("            for identity in ('natural','bos'):\n                x=item['tokens'].clone()", "            for identity in ('natural','bos'):\n                if not selected_group(ROOT,item['corpus'],item['window'],identity):continue\n                x=item['tokens'].clone()",'Filter only physical input/start execution roster after full input validation')
 edit("writer.save('inputs/'+group_name+'.npz',payload);group_committed=True", "finalize_group(ROOT,writer,group_name,payload);group_committed=True",'Persist exact continuity gate at committed group boundary')
 c=COUNTS[segment]
 edit('assert attempts==completed==2184 and main_count==suffix_count==1088 and anchor_count==8',
      f"assert attempts==completed=={c['total']} and main_count==suffix_count=={c['main']} and anchor_count=={c['anchors']}",'Segment final counters only')
 edit("'GPT2_NATIVE_VALUE_IMPLEMENTATION_ADDENDUM.md','gpt2_layer_writes.py'", "'GPT2_NATIVE_VALUE_IMPLEMENTATION_ADDENDUM.md','gpt2_layer_writes.py','gpt2_gate_native_value_recovery.py','GPT2_NATIVE_VALUE_RECOVERY_PROPOSAL.md','recovery_context.json','recovery_ast_proof.json'",'Include actual adapter/context/protocol/proof in executed closure')
 return source,changes

def ast_proof(original,adapted,segment,changes):
 a,b=ast.parse(original),ast.parse(adapted)
 functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 aa,bb=functions(a),functions(b);names=sorted(set(aa)-{'main'})
 assert set(aa)==set(bb) and all(aa[n]==bb[n] for n in names)
 main_a=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 main_b=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 # Scientific per-condition body from `for boundary` through all suffix/control
 # calls is identical, as are nested ev/anchors and every guard expression.
 def nested(tree,name):return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
 for name in ('ev','anchors','assert_check','progress','resources','checkpoint'):assert nested(main_a,name)==nested(main_b,name)
 boundary=lambda tree:[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='boundary']
 assert boundary(main_a)==boundary(main_b)
 return {'status':'passed','segment':segment,'original_runner_sha256':ORIGINAL_RUNNER_SHA,
  'adapted_runner_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'allowed_changes':changes,
  'top_level_scientific_helpers_ast_exact':names,'nested_forward_anchor_control_helpers_ast_exact':['ev','anchors','assert_check','progress','resources','checkpoint'],
  'complete_boundary_condition_and_suffix_loop_ast_exact':True,'thresholds_calibration_scoring_controls_unchanged':True}

def prepare(a):
 assert not a.work.exists(),'Fresh recovery work directory; no overwrites'
 cp,inv,f=verify_original(a.original,a.forensic,a.prefix)
 assert sha(a.proposal)==PROPOSAL_SHA
 if a.segment=='B':assert a.segment_a is not None;verify_segment(a.segment_a,'A')
 assert shutil.disk_usage(a.work.parent).free>=RESERVE+1024**3
 a.work.mkdir(parents=True);a._created_work=True;dest=a.work/'pulse_transfer';dest.mkdir()
 for name,h in cp['source_hashes'].items():
  copy_exact(a.original/'frozen_sources'/name,a.work/'original_frozen_sources'/name,h)
  if name!='gpt2_gate_native_value.py':copy_exact(a.original/'frozen_sources'/name,dest/name,h)
 copy_exact(__file__,dest/'gpt2_gate_native_value_recovery.py')
 copy_exact(a.proposal,dest/'GPT2_NATIVE_VALUE_RECOVERY_PROPOSAL.md',PROPOSAL_SHA)
 c={'schema':'native_value_bounded_recovery_v1','segment':a.segment,'original_run':str(a.original.resolve()),
  'forensic':str(a.forensic.resolve()),'prefix':str(a.prefix.resolve()),'adapter_sha256':sha(__file__),
  'segment_a':None if a.segment_a is None else str(a.segment_a.resolve()),
  'segment_a_manifest_sha256':None if a.segment_a is None else sha(a.segment_a/'manifest.json')}
 put(dest/'recovery_context.json',c)
 original=(a.original/'frozen_sources/gpt2_gate_native_value.py').read_text();adapted,changes=instrument(original,a.segment)
 proof=ast_proof(original,adapted,a.segment,changes);put(dest/'recovery_ast_proof.json',proof)
 with (dest/'gpt2_gate_native_value.py').open('x') as out:out.write(adapted)
 # Original relative metadata reads require these exact sibling locations.
 for name,h in cp['metadata_dependencies'].items():copy_exact(ROOT/name,dest/name,h)
 command=[sys.executable,str(dest/'gpt2_gate_native_value.py'),'--config',str(a.config.resolve()),'--inputs',str(a.inputs.resolve()),
          '--calibration-reference',str(a.reference.resolve()),'--out',str(a.work/'run')]
 put(a.work/'preparation.json',{'status':'prepared','segment':a.segment,'model_forwards':0,'pretrained_model_loaded':False,
  'expected_counts':COUNTS[a.segment],'command':command,'original_source_hashes':cp['source_hashes'],'proof':proof,
  'adapter_sha256':sha(__file__),'original_final_guards_missing':f['missing_guards']})
 print('RECOVERY_PREPARED',a.segment,a.work,flush=True)
 if a.execute:subprocess.run(command,check=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--segment',choices=['A','B'],required=True)
 for n in ('original','forensic','prefix','proposal','work','config','inputs','reference'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--segment-a',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 a._created_work=False
 try:prepare(a)
 except BaseException as e:
  if a._created_work and a.work.is_dir() and not (a.work/'preparation_failure.json').exists():put(a.work/'preparation_failure.json',{'status':'failed','error_type':type(e).__name__,'error':str(e),'execute_requested':a.execute})
  raise
if __name__=='__main__':main()
