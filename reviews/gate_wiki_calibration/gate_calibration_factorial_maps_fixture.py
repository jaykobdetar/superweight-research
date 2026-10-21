"""Synthetic algebra fixtures for prospective T18; no pretrained model/inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gate_calibration_factorial_maps import factorial_map,geometry,common_state_center_check,CONDITIONS
from gpt2_gate_localization import localization_map
from gate_localization_inputs import directions

def main():
    started=time.time();torch.set_num_threads(1);g=torch.Generator().manual_seed(20261026)
    original={};fresh={};spec={}
    for l in range(4,12):
        q=torch.linalg.qr(torch.randn(12,12,generator=g,dtype=torch.float64))[0]
        o=torch.linalg.qr(torch.randn(12,12,generator=g,dtype=torch.float64))[0]
        f=torch.linalg.qr(torch.randn(12,12,generator=g,dtype=torch.float64))[0]
        original[l]={'alpha':torch.randn(12,generator=g,dtype=torch.float64),'basis_r':o,'basis_qr':q}
        fresh[l]={'alpha':torch.randn(12,generator=g,dtype=torch.float64),'basis_r':f,'basis_qr':q.clone()}
        spec[l]={'r':{'eigenvalues':list(range(12,0,-1))}}
    ds,report=geometry(original,fresh,spec);errors=[];compared=0;unchanged=0
    assert len(CONDITIONS)==len(set(CONDITIONS))==20
    for dtype in (torch.float32,torch.float64):
        for l in range(4,12):
            mm={k:{n:v.to(dtype)for n,v in src[l].items()}for k,src in(('O',original),('F',fresh))}
            dd={k:{n:ds[k][l][n].to(dtype)for n in('u','v')}for k in('O','F')}
            b=torch.randn(12,511,generator=g,dtype=dtype)*3;mass=b.sigmoid()
            for family,old in(('remove','remove_pc1'),('orthogonal','orthogonal_control')):
                a=factorial_map(family+'_OO_all',l,b,mass,mm,dd)
                z=localization_map(old+'_all',l,b,mass,mm['O'],dd['O'])
                assert torch.equal(a[0],z[0])and torch.equal(a[1],z[1])and torch.equal(a[4]['s'],z[4]['s'])
                compared+=1
                for basis in('O','F'):
                    err=common_state_center_check(family,basis,b,mass,mm,dd);assert err<=1e-4;errors.append(err)
            for mode in CONDITIONS:
                a=factorial_map(mode,l,b,mass,mm,dd)
                assert torch.isfinite(a[0]).all()and torch.isfinite(a[1]).all()and a[5]<=1e-6
                if l!=10 and mode.endswith('_L10'):
                    assert a[0] is b and a[1] is mass;unchanged+=1
    failures=[]
    for name,mutate in(('zero_eigenvalue',lambda d:d[4]['r'].update(eigenvalues=[0.]*12)),
                       ('tied_leading_eigenvalues',lambda d:d[4]['r'].update(eigenvalues=[12.,12.]+list(range(10,0,-1)))),
                       ('nonfinite_eigenvalue',lambda d:d[4]['r'].update(eigenvalues=[float('nan')]+list(range(11,0,-1))))):
        bad=copy.deepcopy(spec);mutate(bad)
        try:geometry(original,fresh,bad)
        except AssertionError:failures.append(name)
        else:raise AssertionError(('Invalid geometry accepted',name))
    out={'status':'passed','old_formula_exact_cases':compared,'unaffected_identity_cases':unchanged,
        'common_state_center_checks':len(errors),'max_center_formula_error':max(errors),
        'rejected_geometry_cases':failures,'source_sha256':hashlib.sha256((ROOT/'gate_calibration_factorial_maps.py').read_bytes()).hexdigest(),
        'model_forwards':0,'model_loaded':False,'gpu_used':False,'wall_seconds':time.time()-started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
