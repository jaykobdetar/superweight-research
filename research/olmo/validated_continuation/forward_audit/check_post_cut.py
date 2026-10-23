"""Actual 16-block control flow on cheap deterministic synthetic projections, not pretrained OLMo."""
from pathlib import Path
import numpy as np,json,importlib.util,hashlib
R=Path(__file__).resolve().parent;P=Path('/workspace/shared/olmo_computational_role/downstream/streamed_olmo.py');sp=importlib.util.spec_from_file_location('stream',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);s=m.StreamedOlmo.__new__(m.StreamedOlmo);s.half=False;s.check=lambda:0;s.peak_rss=0;rng=np.random.default_rng(64128);L,D=9,2048;p=4;ids=[2,7,8,3,2,11,12,13,4];E=rng.normal(size=(71,D)).astype('f4');s.embed=lambda ids:E[np.asarray(ids)].copy()
def linear(x,key):
 if key.endswith('q_proj.weight'):return np.roll(x,3,axis=-1)*np.float32(.2)
 if key.endswith('k_proj.weight'):return np.roll(x,7,axis=-1)*np.float32(.2)
 if key.endswith('v_proj.weight'):return x*np.float32(.4)
 if key.endswith('o_proj.weight'):return x*np.float32(.5)
 if key.endswith('gate_proj.weight'):return np.tile(x,(1,4))*np.float32(.3)
 if key.endswith('up_proj.weight'):return np.tile(x,(1,4))*np.float32(.2)
 if key.endswith('down_proj.weight'):return x.reshape(len(x),4,D).sum(1)*np.float32(.4)
 raise ValueError(key)
s.linear=linear;wp=R/'synthetic_head.f32';s.mmap=lambda key:np.memmap(wp,mode='r',dtype='<f4',shape=(71,D))
native=s.forward(ids,prefix_hash_end=p);post=s.forward(ids,mask_sources=(0,p),mask_query_start=p+1,prefix_hash_end=p);globalcut=s.forward(ids,mask_sources=(0,p),prefix_hash_end=p);maskp=s.forward(ids,mask_sources=(p,),prefix_hash_end=p)
out={'scope':__doc__,'implementation_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'post_prefix_hashes_exact':post['prefix_hashes']==native['prefix_hashes'],'post_precut_nll_exact':bool(np.array_equal(post['loss'][:p+1],native['loss'][:p+1])),'post_after_cut_nll_changed':bool(np.any(post['loss'][p+1:]!=native['loss'][p+1:])),'global_mask_changes_later_prefix':globalcut['prefix_hashes']!=native['prefix_hashes'],'maskp_prefix_hashes_exact':maskp['prefix_hashes']==native['prefix_hashes'],'maskp_precut_nll_exact':bool(np.array_equal(maskp['loss'][:p+1],native['loss'][:p+1])),'prefix_hash_count':len(native['prefix_hashes'])};assert all(v for k,v in out.items() if isinstance(v,bool)) and out['prefix_hash_count']==16;(R/'post_cut_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
