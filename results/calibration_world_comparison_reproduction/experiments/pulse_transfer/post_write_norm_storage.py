"""T20 immutable lossless boundary shards and bounded failure evidence.

No model loading. The audited T18 codec is reused without changing old sources.
All caps remain subject to a complete new high-entropy layout measurement.
"""
import ast,hashlib,json,os,shutil
from pathlib import Path
import torch
import numpy as np
from gate_wiki_storage import dumps_npz,read_npz
from gate_native_value import fingerprint

NORMAL_LIMIT=512*1024**2
FAILURE_LIMIT=384*1024**2
DISK_RESERVE=2*1024**3
METADATA_ALLOWANCE=32*1024**2
READOUT_ROWS=(1,126,127,510)
FAILURE_ROWS=(0,1,31,63,126,127,254,510)
SCHEMA='post_write_norm_explicit_v1'
ARMS=('baseline','current','remove_full','orth_full','remove_write_matched','orth_write_matched')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def json_write(path,value):
 path=Path(path);data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def codec_tree(value):
 """Lossless ndarray representation for the unchanged Torch-only NPZ codec.

 Runtime matching caches stay NumPy. Serialized arrays become CPU tensors with
 the same shape, dtype and bytes; no values are summarized, clipped or omitted.
 """
 if isinstance(value,np.ndarray):
  assert not value.dtype.hasobject
  array=np.array(value,copy=True,order='C',subok=False);tensor=torch.from_numpy(array)
  assert tensor.numpy().dtype==value.dtype and tensor.numpy().shape==value.shape and tensor.numpy().tobytes()==value.tobytes(order='C')
  return tensor
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,dict):return {codec_tree(k):codec_tree(v) for k,v in value.items()}
 if isinstance(value,list):return [codec_tree(v) for v in value]
 if isinstance(value,tuple):return tuple(codec_tree(v) for v in value)
 return value
def byte_exact(left,right):
 """Preserve signed zero and every storage bit; numeric equality is insufficient."""
 if isinstance(left,torch.Tensor):
  return isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),right.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
 if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and left.dtype==right.dtype and left.shape==right.shape and left.tobytes(order='C')==right.tobytes(order='C')
 if isinstance(left,dict):return isinstance(right,dict) and set(left)==set(right) and all(byte_exact(v,right[k]) for k,v in left.items())
 if isinstance(left,(tuple,list)):return type(left)is type(right) and len(left)==len(right) and all(byte_exact(a,b) for a,b in zip(left,right))
 return type(left)is type(right) and left==right
def nested_fp(value):
 if isinstance(value,torch.Tensor):return fingerprint(value)
 if isinstance(value,dict):return {k:nested_fp(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [nested_fp(v) for v in value]
 return value
def summarize_diag(diag):
 return {layer:{key:({'fingerprint':fingerprint(v),'min':float(v.double().min()),'max':float(v.double().max()),'mean':float(v.double().mean()),'rms':float(v.double().square().mean().sqrt())} if isinstance(v,torch.Tensor) else v) for key,v in values.items()} for layer,values in diag.items()}
def empty_payload(metadata):
 return {'schema':SCHEMA,'metadata':metadata,**{k:{} for k in ('metrics','suffix','gates','suffix_gates','gate_norms','suffix_gate_norms','write_norms','suffix_write_norms','operations','suffix_operations','source_reference','snapshots','readouts')},'checks':[]}
def trim_gate(gates):return {layer:{key:v[:127].clone() if v.ndim==1 else v[:,:127].clone() for key,v in vals.items()} for layer,vals in gates.items()}
def trim_gate_norm(norms):return {layer:{key:v.clone() if key=='source_v_norm' else v[:127].clone() for key,v in vals.items()} for layer,vals in norms.items()}
def failure_bundle(bundle):
 if bundle is None:return None
 return {key:bundle[key] for key in ('metrics','hidden','source','gates','gate_norms','write_norms','snapshots','write_snapshot','operations','prefix_contexts','prefix_writes','returned_write') if key in bundle}
def logit_rows(left,right=None):
 if left is None:return None
 assert left.ndim==2 and (right is None or right.shape==left.shape)
 maxima=[];bad=[]
 for start in range(0,len(left),32):
  a=left[start:start+32].detach().cpu().double();b=None if right is None else right[start:start+32].detach().cpu().double()
  finite=torch.isfinite(a).all(-1) if b is None else torch.isfinite(a).all(-1)&torch.isfinite(b).all(-1)
  bad.extend((~finite).nonzero().flatten().add(start).tolist());delta=a.abs() if b is None else (a-b).abs()
  maxima.append(torch.nan_to_num(delta,nan=float('inf'),posinf=float('inf'),neginf=float('inf')).amax(-1))
 maxima=torch.cat(maxima);rows=sorted({q for q in FAILURE_ROWS if q<len(left)}|{int(maxima.argmax())}|set(bad[:1]))
 return {'query_rows':torch.tensor(rows),'left':left[rows].detach().cpu().clone(),'right':None if right is None else right[rows].detach().cpu().clone(),'left_full_fingerprint':fingerprint(left),'right_full_fingerprint':None if right is None else fingerprint(right),'per_query_max_abs':maxima,'nonfinite_query_indices':bad,'scope':'Fixed rows, maximum discrepancy row, first nonfinite row and all per-query maxima.'}
def source_closure(root,seeds):
 root=Path(root);pending=list(seeds);found={}
 while pending:
  name=pending.pop()
  if name in found:continue
  path=root/name;assert path.is_file() and not path.is_symlink(),name;found[name]=sha(path)
  if path.suffix!='.py':continue
  for node in ast.walk(ast.parse(path.read_text())):
   modules=[node.module] if isinstance(node,ast.ImportFrom) and node.level==0 else [v.name for v in node.names] if isinstance(node,ast.Import) else []
   for module in modules:
    candidate=(module or '').split('.')[0]+'.py'
    if (root/candidate).is_file():pending.append(candidate)
 return dict(sorted(found.items()))

class EvidenceWriter:
 def __init__(self,out,*,normal_limit=NORMAL_LIMIT,failure_limit=FAILURE_LIMIT,reserve=DISK_RESERVE,outstanding_other_caps=0):
  self.out=Path(out);self.owned=False
  assert not self.out.exists() and not self.out.is_symlink(),'Fresh immutable output directory required'
  assert all(not p.is_symlink() for p in self.out.absolute().parents),'Symlink output ancestor'
  self.out.mkdir(parents=True);self.owned=True;st=self.out.stat();self.identity=(st.st_dev,st.st_ino)
  (self.out/'inputs').mkdir();(self.out/'frozen_sources').mkdir()
  self.normal_limit=normal_limit;self.failure_limit=failure_limit;self.reserve=reserve;self.outstanding_other_caps=outstanding_other_caps
  self.inventory={};self.disk_checks=[];self.chain='0'*64;self.checkpoint_index=0
 def owns(self):return self.owned and self.out.is_dir() and not self.out.is_symlink() and (self.out.stat().st_dev,self.out.stat().st_ino)==self.identity
 def used(self):return sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
 def guard(self,stage,projected=0,*,failure=False):
  assert self.owns();free=shutil.disk_usage(self.out).free;used=self.used()
  remaining=max(0,self.normal_limit-used)
  required=self.reserve+self.outstanding_other_caps+(0 if failure else self.failure_limit+remaining)
  if failure:required+=projected
  record={'stage':stage,'free_bytes':free,'used_bytes':used,'projected_bytes':projected,'reserve_bytes':self.reserve,'outstanding_other_caps':self.outstanding_other_caps,'remaining_normal_cap':remaining,'required_free_bytes':required,'failure_write':failure}
  self.disk_checks.append(record)
  assert failure or used+projected<=self.normal_limit,('Normal cap',record)
  assert free>=required,('Complete remaining-cap reserve',record)
  return record
 def journal(self,event):
  assert self.owns();data=json.dumps({'previous_sha256':self.chain,**event},sort_keys=True,separators=(',',':'),allow_nan=False).encode();next_chain=hashlib.sha256(data).hexdigest()
  with (self.out/'journal.jsonl').open('ab') as f:f.write(data+b'\n');f.flush();os.fsync(f.fileno())
  self.chain=next_chain
 def save(self,name,value,*,failure=False):
  assert self.owns() and not Path(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/'))
  data=dumps_npz(codec_tree(value));limit=self.failure_limit if failure else self.normal_limit-self.used();assert len(data)<=limit,('Serialized cap',name,len(data),limit)
  self.guard('write:'+name,len(data),failure=failure);path=self.out/name;assert not path.exists() and not path.is_symlink();path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  entry={'bytes':len(data),'sha256':sha(path)};self.inventory[name]=entry;self.journal({'event':'committed','path':name,**entry});return entry
 def checkpoint(self,progress,*,manifest=None,failure=False,final=False):
  assert self.owns();self.checkpoint_index+=1;stem=self.out/'checkpoints'/f'{self.checkpoint_index:06d}'
  json_write(stem/'progress.json',progress);json_write(stem/'numeric_inventory.json',self.inventory);json_write(stem/'disk_checks.json',self.disk_checks)
  if manifest is not None:json_write(stem/'manifest.json',manifest)
  if final:
   assert manifest is not None
   json_write(self.out/'numeric_inventory.json',self.inventory);json_write(self.out/'disk_checks.json',self.disk_checks)
   final_bytes=len((json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode())
   guard=self.guard('final_manifest_last',final_bytes+4096,failure=failure)
   # Guard metadata is bounded here. The canonical manifest is the final
   # fallible operation: rejection or an earlier I/O failure cannot leave a
   # completed manifest that the merger would accept.
   assert len((json.dumps(guard,indent=2,allow_nan=False)+'\n').encode())<=4096
   json_write(stem/'finalization_guard.json',guard)
   json_write(self.out/'manifest.json',manifest)
  else:self.guard('checkpoint',failure=failure)
