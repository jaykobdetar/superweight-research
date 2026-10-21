"""Runtime-safe exact directions algebra copied from reviewed input helper. No input selection."""
import torch
from run_transfer import th

def directions(maps):
    result={}
    for l in range(4,12):
        u0=maps[l]['basis_r'][:,0].double();norm=float(u0.norm())
        assert torch.isfinite(u0).all()and abs(norm-1)<=1e-10
        u=u0/u0.norm();chosen=None
        for j in range(12):
            q=maps[l]['basis_qr'][:,j].double();assert torch.isfinite(q).all()
            w=q-u*torch.dot(u,q);wn=float(w.norm())
            assert torch.isfinite(w).all()
            if wn>=1e-6:chosen=(j,w,wn);break
        assert chosen is not None
        j,w,wn=chosen;v=w/w.norm();pivot=int(v.abs().argmax());sign=-1 if float(v[pivot])<0 else 1;v=v*sign
        assert abs(float(torch.dot(u,v)))<=1e-12 and abs(float(v.norm())-1)<=1e-12
        uf,vf=u.float(),v.float()
        assert abs(float(uf.norm()-vf.norm()))<=1e-6 and abs(float(torch.dot(uf,vf)))<=1e-6
        result[l]={'u':u,'v':v,'qr_column':j,'residual_norm':wn,'sign_pivot':pivot,'sign_multiplier':sign,
            'u_shape':list(u.shape),'v_shape':list(v.shape),'dtype':str(u.dtype),'u_sha256':th(u),'v_sha256':th(v),
            'u_norm_error':abs(float(u.norm())-1),'v_norm_error':abs(float(v.norm())-1),
            'orthogonality_error':abs(float(torch.dot(u,v))),
            'fp32_norm_difference':abs(float(uf.norm()-vf.norm())),
            'fp32_orthogonality_error':abs(float(torch.dot(uf,vf)))}
    return result
