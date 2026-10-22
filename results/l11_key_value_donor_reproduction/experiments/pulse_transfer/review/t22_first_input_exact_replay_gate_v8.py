"""Prespecified T22 first-input exact replay gate; stdlib streaming, no models.

Run ONLY after complete A_r3, before B or efficacy interpretation. All numeric
array occurrences and26 actual complete hidden/readout fingerprints are exact.
Only manifest-bound freeze_digest scalar differences are permitted. No tolerance,
old-shard reuse, averaging, scientific data export or source-artifact mutation.
"""
import argparse,ast,hashlib,json,math,resource,struct,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD_MANIFEST_SHA='52b418d965d2a98ce514f6bedf423e97fca433e971dfc386649784913bbd09c9'
OLD_SHARD_SHA='6b97a1fbb0dd7d31a1029be911e79b133fc9d3cccda00fdf26d8a0c91b8f7358'
OLD_SHARD_BYTES=228062734
SHARD='inputs/wiki_validation__0__natural__000.npz'
PARSER_ANCESTRY_SHA='8515ef18460cb1eb7fcb24fd22e0b0d0dd1e7e6dbccb183f73cf14abf58505a9'
DT={'float32':('<f4',4),'float64':('<f8',8),'int64':('<i8',8),'bool':('|b1',1),'uint8':('|u1',1)}
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched_early','remove_l11_kv','orth_write_matched_early','orth_l11_kv','remove_l11_k','remove_l11_v','orth_l11_k','orth_l11_v')
BLOCK=4*1024**2


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
 return h.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def need(value, message):
    if not value:
        raise AssertionError(message)

def npy_header(stream):
    need(stream.read(6) == b'\x93NUMPY', 'NPY magic')
    version = tuple(stream.read(2))
    need(version in ((1, 0), (2, 0), (3, 0)), 'NPY version')
    count = 2 if version == (1, 0) else 4
    size = struct.unpack('<H' if count == 2 else '<I', stream.read(count))[0]
    need(size + 8 + count <= 256, 'declared bounded NPY header')
    result = ast.literal_eval(stream.read(size).decode('utf8' if version == (3, 0) else 'latin1'))
    need(set(result) == {'descr', 'fortran_order', 'shape'}, 'NPY header fields')
    need(result['fortran_order'] is False, 'C-order NPY')
    return result

def metadata(archive):
    with archive.open('metadata.npy') as stream:
        header = npy_header(stream)
        need(header['descr'] == '|u1', 'metadata uint8')
        data = stream.read()
        need(header['shape'] == (len(data),), 'metadata length')
    value = json.loads(data)
    need(set(value) == {'schema', 'object'} and value['schema'] == 'wiki_gate_lossless_npz_v1', 'codec schema')
    return value['object'], len(data)

def journal(path):
    previous = '0' * 64
    result = []
    for raw in Path(path).read_bytes().splitlines(keepends=True):
        need(raw.endswith(b'\n'), 'complete journal line')
        event = json.loads(raw)
        body = raw[:-1]
        need(body == json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), 'canonical journal')
        need(event['previous_sha256'] == previous, 'journal hash chain')
        previous = hashlib.sha256(body).hexdigest()
        result.append(event)
    return result


def scalar_equal(a,b):
 return type(a) is type(b) and (struct.pack('>d',a)==struct.pack('>d',b) if isinstance(a,float) else a==b)


def compare_archives(old,new,old_freeze,new_freeze,counts):
 counts.update(arrays=0,raw_bytes=0,max_array_bytes=0,scalars=0,permitted_freeze_scalar_differences=0)
 paths=hashlib.sha256();exclusions=[];members=[]
 with zipfile.ZipFile(old) as za,zipfile.ZipFile(new) as zb:
  for z in (za,zb):
   need(len(z.namelist())==len(set(z.namelist()))==1367,'exact1366array+metadata unique ZIP members')
   need(z.getinfo('metadata.npy').file_size<=8*1024**2+256,'bounded normal metadata')
  a,_=metadata(za);b,_=metadata(zb)
  def walk(a,b,path=()):
   need(a['type']==b['type'],'node kind at '+str(path));kind=a['type']
   if kind=='tensor':
    need(set(a)==set(b)=={'type','array','dtype','shape','sha256'},'tensor metadata fields')
    need(a==b,'array semantic descriptor/internal SHA at '+str(path))
    name='a'+str(counts['arrays'])+'.npy';need(a['array']+'.npy'==name,'ordered array identity')
    dtype=a['dtype'].removeprefix('torch.');need(dtype in DT,'numeric dtype allowlist')
    shape=tuple(a['shape']);need(all(type(n) is int and n>=0 for n in shape),'shape dimensions')
    nbytes=math.prod(shape)*DT[dtype][1];need(nbytes<=102926336,'known largest array')
    need(type(a['sha256']) is str and len(a['sha256'])==64 and all(c in '0123456789abcdef' for c in a['sha256']),'internal SHA format')
    h1=hashlib.sha256();h2=hashlib.sha256();length=0
    with za.open(name) as fa,zb.open(name) as fb:
     for f in (fa,fb):
      header=npy_header(f);need(header['shape']==shape and header['descr']==DT[dtype][0],'actual NPY dtype/shape')
     while True:
      x=fa.read(BLOCK);y=fb.read(BLOCK)
      need(x==y,'uncompressed C bytes at '+str(path)+' offset '+str(length))
      if not x:break
      h1.update(x);h2.update(y);length+=len(x)
    # Reading both streams to EOF independently verifies both ZIP CRCs.
    need(length==nbytes and h1.hexdigest()==h2.hexdigest()==a['sha256'],'complete C-byte SHA/length')
    members.append(name);counts['arrays']+=1;counts['raw_bytes']+=length;counts['max_array_bytes']=max(counts['max_array_bytes'],length)
    paths.update(json.dumps([path,a['dtype'],shape,a['sha256']],separators=(',',':')).encode()+b'\n');return
   if kind=='scalar':
    need(set(a)==set(b)=={'type','value'},'scalar node fields');x=a['value'];y=b['value'];counts['scalars']+=1
    need(x is None or type(x) in (bool,int,float,str),'scalar type')
    if not scalar_equal(x,y):
     need(path and path[-1]==('key','freeze_digest') and x==old_freeze and y==new_freeze,'exact scientific scalar at '+str(path))
     counts['permitted_freeze_scalar_differences']+=1;exclusions.append(path)
    return
   need(kind in ('dict','list','tuple') and set(a)==set(b)=={'type','items'},'container fields')
   need(len(a['items'])==len(b['items']),'container length at '+str(path))
   if kind=='dict':
    seen=set()
    for (ka,va),(kb,vb) in zip(a['items'],b['items']):
     need(ka==kb and ka.get('type')=='scalar','ordered exact scalar dictionary keys')
     k=ka['value'];need(type(k) in (str,int,bool) and k not in seen,'unique scalar key');seen.add(k)
     walk(va,vb,path+(('key',k),))
   else:
    for i,(va,vb) in enumerate(zip(a['items'],b['items'])):walk(va,vb,path+(('index',i),))
  walk(a,b)
  need(za.namelist()==zb.namelist()==members+['metadata.npy'],'complete ordered member universe')
 need(counts['arrays']==1366 and counts['raw_bytes']==253652704,'selected fixed full array universe')
 counts['semantic_array_commitment_sha256']=paths.hexdigest();counts['excluded_scalar_paths']=exclusions
 return counts


def expected_roles():
 roles=[]
 for arm in ARMS:
  roles.append((arm,'main'))
  if arm=='baseline':roles.extend((name,'main') for name in ('native_repeat','native_identity'))
  roles.append((arm,'suffix'))
 return roles


def completion_prefix(run,total):
 events=journal(run/'journal.jsonl');done=[e for e in events if e['event']=='forward_completed']
 need(len(done)==total,'all journal completion count');roles=expected_roles();result=[]
 for i,(e,(arm,phase)) in enumerate(zip(done[:26],roles),1):
  ident=e['identity'];need(set(ident)=={'corpus','window','identity','length','token_sha256','boundary','arm','phase'},'complete identity keys')
  need(tuple(ident[k] for k in ('corpus','window','identity','length','boundary','arm','phase'))==('wiki_validation',0,'natural',512,'000',arm,phase),'preselected ordered first26 roles')
  need(e['attempt']==e['completed']==e['processed']==i,'actual first26 completion counters')
  for key,shape in (('hidden_fingerprint',[1,512,768]),('logits_fingerprint',[512,50257])):
   fp=e[key];need(set(fp)=={'sha256','shape','dtype'} and fp['shape']==shape and fp['dtype']=='torch.float32','full-output fingerprint extent')
   need(type(fp['sha256']) is str and len(fp['sha256'])==64 and all(c in '0123456789abcdef' for c in fp['sha256']),'full-output SHA')
  result.append({k:e[k] for k in ('attempt','completed','processed','identity','hidden_fingerprint','logits_fingerprint')})
 need(len(result)==26,'all26 actual full-output records')
 return result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',type=Path,default=ROOT/'gpt2_l11_key_value_donor_run_A_r2');p.add_argument('--new-run',type=Path,required=True);p.add_argument('--new-manifest-sha256',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 old=args.old_run.resolve();new=args.new_run.resolve();out=args.out.resolve()
 need(not out.exists() and out.parent.is_dir(),'fresh receipt in existing parent')
 need(all(out!=x and x not in out.parents for x in (old,new)),'receipt outside immutable runs')
 started=time.monotonic();receipt={'status':'failed','gate_source_sha256':sha(__file__),'old_manifest_sha256':OLD_MANIFEST_SHA,'old_shard_sha256':OLD_SHARD_SHA,'new_manifest_sha256':args.new_manifest_sha256,'replay_gate_passed':False}
 try:
  need(old!=new,'distinct immutable runs');need(sha(old/'manifest.json')==OLD_MANIFEST_SHA,'pinned failed A_r2manifest')
  need(sha(new/'manifest.json')==args.new_manifest_sha256,'caller-bound completed A_r3manifest')
  om=load(old/'manifest.json');nm=load(new/'manifest.json');oi=load(old/'numeric_inventory.json');ni=load(new/'numeric_inventory.json')
  need(om['status']=='failed' and nm['status']=='completed' and nm['segment']=='A' and nm['all_final_guards_passed'] is True,'failed old / complete guarded new A')
  need(all(om[k]==26 and nm[k]==388 for k in ('forward_attempts','model_outputs_observed','forward_passes','fully_processed_forwards')),'complete old26/new388 accounting')
  need([om[k] for k in ('evaluation_forwards','suffix_forwards','anchor_forwards')]==[12,12,2] and [nm[k] for k in ('evaluation_forwards','suffix_forwards','anchor_forwards')]==[192,192,4],'category accounting')
  for k in ('input_dependencies','metadata_dependencies','model_id','revision','dtype','attention','runtime_versions','precision_settings_before','model_hash_before','calibration_map_digest'):
   need(om[k]==nm[k],'same scientific dependency '+k)
  need(nm['model_hash_before']==nm['model_hash_after'] and nm['parameter_versions_unchanged'] is True and nm['precision_settings_before']==nm['precision_settings_after'],'new final parameter/precision guards')
  need(len(ni)==16 and set(oi)=={SHARD,'failure_state.npz'},'actual old/new inventory membership')
  for run in (old,new):
   need(not any(q.is_symlink() for q in (run,run/'manifest.json',run/'numeric_inventory.json',run/'journal.jsonl',run/SHARD)),'no symlink evidence paths')
  for run,entry in ((old,oi[SHARD]),(new,ni[SHARD])):
   f=run/SHARD;need(f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],'inventory-bound complete shard')
  need(oi[SHARD]=={'bytes':OLD_SHARD_BYTES,'sha256':OLD_SHARD_SHA},'literal old shard pin')
  need(not any((new/name).exists() for name in ('failure_state.npz','checkpoint_failure.json','merge_failure.json')) and not list((new/'inputs').glob('*__partial.npz')),'no new failure artifacts')
  need(not any(load(q)['status']=='failed' for q in (new/'checkpoints').glob('*/manifest.json')),'no failed versioned checkpoint')
  old_done=completion_prefix(old,26);new_done=completion_prefix(new,388)
  need(old_done==new_done,'ordered26 actual identity/full-hidden/full-logit fingerprints')
  receipt['full_output_comparisons']={'ordered_forwards':26,'actual_hidden_fingerprints':26,'actual_full_logits_fingerprints':26,'canonical_records_sha256':hashlib.sha256(json.dumps(old_done,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
  for digest in (om['freeze_digest'],nm['freeze_digest']):need(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest),'manifest freeze SHA')
  receipt['array_comparisons']={}
  compare_archives(old/SHARD,new/SHARD,om['freeze_digest'],nm['freeze_digest'],receipt['array_comparisons'])
  need(sha(old/'manifest.json')==OLD_MANIFEST_SHA and sha(new/'manifest.json')==args.new_manifest_sha256,'manifest end recheck')
  need(sha(old/SHARD)==OLD_SHARD_SHA and sha(new/SHARD)==ni[SHARD]['sha256'],'shard end recheck')
  receipt.update(status='passed_exact_first_input_replay_gate',replay_gate_passed=True,B_and_interpretation_hard_blocked=False,new_shard_sha256=ni[SHARD]['sha256'],new_inventory_sha256=sha(new/'numeric_inventory.json'),old_journal_sha256=sha(old/'journal.jsonl'),new_journal_sha256=sha(new/'journal.jsonl'),scope='Exact first-input technical invariance only; prior26 are not an added sample or retained reused shard. Full-panel audits and separate B launch approval remain required.')
 except BaseException as error:
  receipt.update(error_type=type(error).__name__,error=str(error),B_and_interpretation_hard_blocked=True)
 receipt['wall_seconds']=time.monotonic()-started;receipt['peak_RSS_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 if receipt['peak_RSS_kib']*1024>3*1024**3:receipt.update(status='failed',replay_gate_passed=False,B_and_interpretation_hard_blocked=True,error='Independent replay-gate RSS exceeded3GiB')
 with out.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':receipt['status'],'receipt_sha256':sha(out)}))
 return 0 if receipt['status']=='passed_exact_first_input_replay_gate' else 1


if __name__=='__main__':sys.exit(main())
