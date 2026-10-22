"""T19 immutable lossless evidence writer and bounded failure selection.

Uses the frozen audited T18 NPZ codec unchanged. No inferred aliases or tensor
truncation: every successful saved array is explicit; scalar summaries are named.
"""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import torch
from gate_wiki_storage import dumps_npz, read_npz, write_npz
from gate_native_value import fingerprint

NORMAL_LIMIT = 512*1024**2
FAILURE_LIMIT = 256*1024**2
DISK_RESERVE = 2*1024**3
METADATA_ALLOWANCE = 32*1024**2
READOUT_ROWS = (1,126,127,510)
FAILURE_ROWS = (0,1,31,63,126,127,254,510)
SCHEMA = 'native_value_explicit_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()

def json_write(path, value):
    """Exclusive immutable JSON evidence; version checkpoints instead of replace."""
    path=Path(path); data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())


def nested_fp(value):
    if isinstance(value,torch.Tensor): return fingerprint(value)
    if isinstance(value,dict): return {k:nested_fp(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [nested_fp(v) for v in value]
    return value

def exact(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(exact(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)): return isinstance(b,type(a)) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def summarize_diag(diag):
    """Save all scalar checks and tensor fingerprints; full L10 vector norms separately."""
    return {l:{k:({'fingerprint':fingerprint(v),'min':float(v.double().min()),'max':float(v.double().max()),'mean':float(v.double().mean()),'rms':float(v.double().square().mean().sqrt())} if isinstance(v,torch.Tensor) else v) for k,v in d.items()} for l,d in diag.items()}

def empty_payload(metadata):
    return {'schema':SCHEMA,'metadata':metadata,**{k:{} for k in ('metrics','suffix','gates','norms','suffix_gates','suffix_norms','operations','suffix_operations','source_reference','snapshots','readouts')},'checks':[]}

def trim_gate(gates):
    return {l:{k:v[:127].clone() if v.ndim==1 else v[:,:127].clone() for k,v in d.items()} for l,d in gates.items()}

def trim_norm(norms):
    return {l:{k:v[:127].clone() for k,v in d.items()} for l,d in norms.items()}

def logit_rows(left,right=None):
    if left is None: return None
    assert left.ndim==2 and (right is None or right.shape==left.shape)
    maxima=[]; bad=[]
    for start in range(0,len(left),32):
        a=left[start:start+32].detach().cpu().double(); b=None if right is None else right[start:start+32].detach().cpu().double()
        finite=torch.isfinite(a).all(-1) if b is None else torch.isfinite(a).all(-1)&torch.isfinite(b).all(-1)
        bad.extend((~finite).nonzero().flatten().add(start).tolist())
        d=a.abs() if b is None else (a-b).abs()
        maxima.append(torch.nan_to_num(d,nan=float('inf'),posinf=float('inf'),neginf=float('inf')).amax(-1))
    maxima=torch.cat(maxima); indices=sorted({q for q in FAILURE_ROWS if q<len(left)}|{int(maxima.argmax())}|set(bad[:1]))
    return {'query_rows':torch.tensor(indices),'left':left[indices].detach().cpu().clone(),
            'right':None if right is None else right[indices].detach().cpu().clone(),
            'left_full_fingerprint':fingerprint(left),'right_full_fingerprint':None if right is None else fingerprint(right),
            'per_query_max_abs':maxima,'nonfinite_query_indices':bad,
            'scope':'Fixed rows, maximum discrepancy row, first nonfinite row; all per-query maxima.'}

def failure_bundle(bundle):
    if bundle is None:return None
    return {k:bundle[k] for k in ('metrics','hidden','source','gates','norms','snapshots','operations','prefix_contexts','direct_state') if k in bundle}

def source_closure(root, names):
    root=Path(root); pending=list(names); found={}
    while pending:
        name=pending.pop()
        if name in found:continue
        p=root/name; assert p.is_file(),name; found[name]=sha(p)
        if p.suffix!='.py':continue
        for node in ast.walk(ast.parse(p.read_text())):
            modules=[node.module] if isinstance(node,ast.ImportFrom) and node.level==0 else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                candidate=(module or '').split('.')[0]+'.py'
                if (root/candidate).is_file():pending.append(candidate)
    return dict(sorted(found.items()))

class EvidenceWriter:
    def __init__(self,out,*,normal_limit=NORMAL_LIMIT,failure_limit=FAILURE_LIMIT,reserve=DISK_RESERVE):
        self.out=Path(out); assert not self.out.exists(),'Fresh immutable output directory required'
        self.out.mkdir(parents=True); (self.out/'inputs').mkdir(); (self.out/'frozen_sources').mkdir()
        self.normal_limit=normal_limit;self.failure_limit=failure_limit;self.reserve=reserve
        self.inventory={};self.disk_checks=[];self.chain='0'*64;self.checkpoint_index=0
    def used(self): return sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
    def guard(self,stage,projected=0,failure=False):
        free=shutil.disk_usage(self.out).free; used=self.used()
        record={'stage':stage,'free_bytes':free,'normal_bytes':used,'projected_bytes':projected,'reserve_bytes':self.reserve,'failure_limit_bytes':self.failure_limit}
        self.disk_checks.append(record)
        assert failure or used+projected<=self.normal_limit,('Normal cap',record)
        assert free-projected>=self.reserve+(0 if failure else self.failure_limit),('Disk reserve',record)
        return record
    def journal(self,event):
        data=json.dumps({'previous_sha256':self.chain,**event},sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.chain=hashlib.sha256(data).hexdigest()
        with (self.out/'journal.jsonl').open('ab') as f:f.write(data+b'\n');f.flush();os.fsync(f.fileno())
    def save(self,name,value,*,failure=False):
        data=dumps_npz(value);limit=self.failure_limit if failure else self.normal_limit-self.used()
        assert len(data)<=limit,('Serialized cap',name,len(data),limit)
        self.guard('write:'+name,len(data),failure=failure)
        p=self.out/name;assert not p.exists();p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        self.inventory[name]={'bytes':len(data),'sha256':sha(p)}
        self.journal({'event':'committed','path':name,**self.inventory[name]});return self.inventory[name]
    def checkpoint(self,progress,*,manifest=None,failure=False,final=False):
        self.checkpoint_index+=1
        stem=self.out/'checkpoints'/f'{self.checkpoint_index:06d}'
        json_write(stem/'progress.json',progress)
        json_write(stem/'numeric_inventory.json',self.inventory)
        json_write(stem/'disk_checks.json',self.disk_checks)
        if manifest is not None:json_write(stem/'manifest.json',manifest)
        if final:
            assert manifest is not None
            json_write(self.out/'manifest.json',manifest)
            json_write(self.out/'numeric_inventory.json',self.inventory)
            json_write(self.out/'disk_checks.json',self.disk_checks)
        self.guard('checkpoint',failure=failure)
