import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,resource
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from gate_wiki_storage import read_npz
from gate_wiki_calibration import fit_geometry
from fit_gate_calibration import fit_maps_n
p=R/'gpt2_gate_wiki_calibration_run';torch.set_num_threads(4)
m=read_npz(p/'calibration_maps.npz');s=read_npz(p/'calibration_spectra.npz');saved=read_npz(p/'calibration_directions.npz');gs=read_npz(p/'calibration_geometry.npz')
cal={f.stem:read_npz(f)['raw'] for f in sorted((p/'calibration').glob('*.npz'))};fresh,sp=fit_maps_n(cal,4)
def diffs(a,b,path=''):
 out=[]
 if isinstance(a,torch.Tensor):
  if not torch.equal(a,b):out.append(dict(path=path,max_error=float((a-b).abs().max()),fp32_exact=torch.equal(a.float(),b.float()),shape=list(a.shape)))
 elif isinstance(a,dict):
  for k in a:out+=diffs(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,(list,tuple)):
  for i,(x,y)in enumerate(zip(a,b)):out+=diffs(x,y,path+'/'+str(i))
 elif a!=b:out.append(dict(path=path,a=a,b=b))
 return out
out={'fresh_maps_value_differences':diffs(fresh,m['F']),'fresh_report_differences':diffs(sp,s['F']),'fresh_strides':{str(l):{k:[list(v.stride()),list(m['F'][l][k].stride())]for k,v in d.items()}for l,d in fresh.items()}}
ref=R/'gpt2_gate_subspace_run'
raw=torch.load(ref/'calibration_raw.pt',weights_only=True,map_location='cpu');old,_=fit_maps_n({f'wiki_test__{i}':raw[f'wiki_test__{i}']for i in range(4)},4)
mixture=torch.load(ref/'calibration_maps.pt',weights_only=True,map_location='cpu')
out['all_original_strides']={label:{str(l):{k:list(v.stride())for k,v in d.items()}for l,d in mm.items()}for label,mm in {'O':old,'F':fresh,'M':mixture}.items()}
restored={label:{l:{k:(v.T.contiguous().T if k=='basis_qr'else v)for k,v in d.items()}for l,d in mm.items()}for label,mm in m.items()}
out['restored_maps_value_differences']=diffs(restored,m)
out['old_maps_value_differences']=diffs(old,m['O']);out['mixture_maps_value_differences']=diffs(mixture,m['M'])
for name,mm in [('decoded',m),('refitted_fresh',{**m,'F':fresh}),('original_layout',{'O':old,'F':fresh,'M':mixture}),('restore_qr_stride',restored)]:
 d,g=fit_geometry(mm,s);out[name]={'direction_differences':diffs(d,saved),'geometry_differences':diffs(g,gs)}
out['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(R/'review/gate_wiki_direction_replay_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
