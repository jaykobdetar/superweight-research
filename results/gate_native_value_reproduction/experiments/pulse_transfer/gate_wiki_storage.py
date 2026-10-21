"""Lossless tensor/scalar NPZ codec and audited singleton incoming-state dedup.

No pickle. Aliases are verified by content, dtype and shape on both write/read.
All distinct delivered contexts, final hidden states and logits remain explicit.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch

SCHEMA='wiki_gate_lossless_npz_v1'
SHARED_PATHS=(('q',),('k',),('v',),('mask',),('actual_context',),('stable_mixture',),('gate_state','b'),('gate_state','mass'))

def tensor_identity(x):
    assert isinstance(x,torch.Tensor)
    y=x.detach().cpu().contiguous()
    return {'dtype':str(y.dtype),'shape':list(y.shape),'sha256':hashlib.sha256(y.numpy().tobytes()).hexdigest()}

def _encode(obj,arrays):
    if isinstance(obj,torch.Tensor):
        x=obj.detach().cpu().contiguous().numpy();assert x.dtype.kind in 'biufc' and not x.dtype.hasobject
        name='a'+str(len(arrays));arrays[name]=x
        return {'type':'tensor','array':name,**tensor_identity(obj)}
    if isinstance(obj,dict):return {'type':'dict','items':[[_encode(k,arrays),_encode(v,arrays)]for k,v in obj.items()]}
    if isinstance(obj,(list,tuple)):return {'type':type(obj).__name__,'items':[_encode(v,arrays)for v in obj]}
    assert obj is None or type(obj) in (bool,int,float,str),type(obj)
    return {'type':'scalar','value':obj}

def _decode(obj,arrays,seen):
    kind=obj['type']
    if kind=='tensor':
        name=obj['array'];assert name not in seen;seen.add(name)
        x=torch.from_numpy(arrays[name].copy());assert tensor_identity(x)=={k:obj[k]for k in ('dtype','shape','sha256')}
        return x
    if kind=='dict':
        items=[(_decode(k,arrays,seen),_decode(v,arrays,seen))for k,v in obj['items']]
        out=dict(items);assert len(out)==len(items),'duplicate dictionary keys';return out
    if kind in ('list','tuple'):
        values=[_decode(v,arrays,seen)for v in obj['items']];return tuple(values)if kind=='tuple'else values
    assert kind=='scalar';return obj['value']

def dumps_npz(obj):
    arrays={};header={'schema':SCHEMA,'object':_encode(obj,arrays)}
    arrays['metadata']=np.frombuffer(json.dumps(header,separators=(',',':')).encode(),dtype=np.uint8)
    buf=io.BytesIO();np.savez_compressed(buf,**arrays);return buf.getvalue()

def loads_npz(data):
    with np.load(io.BytesIO(data),allow_pickle=False)as arrays:
        header=json.loads(arrays['metadata'].tobytes());assert header['schema']==SCHEMA
        seen=set();result=_decode(header['object'],arrays,seen)
        assert set(arrays.files)==seen|{'metadata'}
    return result

def write_npz(path,obj,limit_bytes=None):
    """Serialize once in RAM and reject before disk write if over declared cap."""
    data=dumps_npz(obj)
    if limit_bytes is not None:assert len(data)<=limit_bytes,('Serialized payload exceeds declared cap',len(data),limit_bytes)
    p=Path(path);assert not p.exists(),'Evidence files are immutable; choose a new checkpoint path'
    with p.open('xb')as f:f.write(data)
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def read_npz(path):return loads_npz(Path(path).read_bytes())

def getpath(obj,path):
    for key in path:obj=obj[key]
    return obj

def flatten(obj,prefix=()):
    out={}
    for k,v in obj.items():
        if isinstance(v,dict):out.update(flatten(v,prefix+(k,)))
        else:out[prefix+(k,)]=v
    return out

def unflatten(flat):
    result={}
    for path,value in flat.items():
        branch=result
        for key in path[:-1]:branch=branch.setdefault(key,{})
        assert path[-1]not in branch;branch[path[-1]]=value
    return result

class SnapshotMismatch(AssertionError):
    def __init__(self,payload):
        self.payload=payload;super().__init__('Singleton snapshot input differs from P; actual sides retained')

class SnapshotStore:
    def __init__(self):self.payload={'schema':'wiki_gate_singleton_dedup_v1','base':None,'states':{}}
    def add(self,mode,snapshot):
        assert mode not in self.payload['states']
        actual=flatten(snapshot)
        assert all(path in actual for path in SHARED_PATHS)
        if mode=='P':
            assert self.payload['base']is None
            self.payload['base']=snapshot
            self.payload['states'][mode]={'base':True,'identities':{path:tensor_identity(v)for path,v in actual.items()}}
        else:
            assert self.payload['base']is not None
            base=flatten(self.payload['base']);assert set(actual)==set(base)
            mismatch=[]
            for path in SHARED_PATHS:
                a,b=actual[path],base[path]
                if a.shape!=b.shape or a.dtype!=b.dtype or not torch.equal(a,b)or tensor_identity(a)!=tensor_identity(b):mismatch.append(path)
            if mismatch:raise SnapshotMismatch({'mode':mode,'mismatching_paths':mismatch,'reference_actual':self.payload['base'],'trial_actual':snapshot})
            self.payload['states'][mode]={'base':False,'aliases':{path:{'reference_mode':'P','reference_path':path,**tensor_identity(actual[path])}for path in SHARED_PATHS},
                'unique':{path:v for path,v in actual.items()if path not in SHARED_PATHS},
                'identities':{path:tensor_identity(v)for path,v in actual.items()}}
        # No equality-by-assumption: independently reconstruct and rehash now.
        restored=restore_snapshot(self.payload,mode)
        assert all(torch.equal(v,flatten(restored)[p])for p,v in actual.items())

def restore_snapshot(payload,mode):
    assert payload['schema']=='wiki_gate_singleton_dedup_v1'
    base=flatten(payload['base']);state=payload['states'][mode]
    if state['base']:assert mode=='P';flat=base
    else:
        assert mode!='P' and set(state['aliases'])==set(SHARED_PATHS)
        flat=dict(state['unique']);assert not(set(flat)&set(SHARED_PATHS))
        for path,alias in state['aliases'].items():
            assert alias['reference_mode']=='P'and alias['reference_path']==path
            x=base[path];assert tensor_identity(x)=={k:alias[k]for k in ('dtype','shape','sha256')};flat[path]=x
        assert set(flat)==set(base)
    assert set(flat)==set(state['identities'])
    for path,x in flat.items():assert tensor_identity(x)==state['identities'][path]
    return unflatten(flat)
