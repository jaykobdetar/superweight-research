"""Prospective T18 algebra/geometry only; no input selection or model execution."""
import torch
from gpt2_gate_subspace import LAYERS, mapped_gate
from gate_direction_geometry import directions

FAMILIES=('remove','orthogonal')
CELLS=('OO','OF','FO','FF') # center first, direction second
SETS=('L10','all')
CONDITIONS=('C','P','current','fullrank')+tuple(f+'_'+c+'_'+s for f in FAMILIES for c in CELLS for s in SETS)
GAP_FLOOR=1e-12
NORM_TOLERANCE=1e-6

def mode_spec(mode):
    if mode in ('C','P','current','fullrank','calibration'):return None,None,None,()
    family,cell,layer_set=mode.split('_')
    assert family in FAMILIES and cell in CELLS and layer_set in SETS
    return family,cell[0],cell[1],LAYERS if layer_set=='all' else (10,)

def geometry(original,fresh,fresh_spectra):
    """Frozen fit diagnostics; no outcome-based direction/rank choice."""
    assert set(original)==set(fresh)==set(fresh_spectra)==set(LAYERS)
    ds={'O':directions(original),'F':directions(fresh)};report={}
    for l in LAYERS:
        o,f=original[l],fresh[l];u,v=ds['O'][l]['u'],ds['F'][l]['u']
        eig=torch.tensor(fresh_spectra[l]['r']['eigenvalues'],dtype=torch.float64)
        assert eig.shape==(12,) and torch.isfinite(eig).all() and eig[0]>0
        gap=float((eig[0]-eig[1])/eig[0]);assert gap>GAP_FLOOR,('Numerically unresolved fresh direction',l,gap)
        assert torch.equal(o['basis_qr'],f['basis_qr']),'Original/fresh QR matrices must be exactly equal'
        delta=f['alpha']-o['alpha'];assert delta.shape==(12,)and delta.dtype==torch.float64 and torch.isfinite(delta).all()
        report[l]={'leading_eigenvalue':float(eig[0]),'relative_leading_gap':gap,
            'gap_floor':GAP_FLOOR,'signed_pc1_cosine':float(u@v),'absolute_pc1_cosine':abs(float(u@v)),
            'mean_displacement_l2':float(delta.norm()),'mean_displacement_projection_original':float(delta@u),
            'mean_displacement_projection_fresh':float(delta@v),
            'original_qr_column':ds['O'][l]['qr_column'],'fresh_qr_column':ds['F'][l]['qr_column'],
            'scope':'One fresh calibration; signs canonicalized by inherited fit, no outcome-based selection.'}
    return ds,report

def factorial_map(mode,layer,b,mass,maps,ds):
    """Return proposed b/mass and per-query diagnostics in the current state.

    maps/dirs are already cast to the attention device/dtype by the caller.
    Arrays b/mass have head x query shape; q0 is handled by the context hook.
    The inherited fullrank copy uses O solely as an identity diagnostic.
    """
    family,center,basis,affected=mode_spec(mode)
    assert b.shape==mass.shape and b.ndim==2 and b.shape[0]==12
    if mode=='fullrank':
        bb,aa,le,me=mapped_gate(mode,b,mass,maps['O'])
        return bb,aa,le,me,{},0.
    if layer not in affected:return b,mass,0.,0.,{},0.
    logq=torch.arange(1,b.shape[1]+1,device=b.device,dtype=b.dtype).log()[:,None]
    alpha=maps[center]['alpha'];u=ds[basis]['u'];v=ds[basis]['v']
    assert alpha.shape==u.shape==v.shape==(12,)
    s=(b.T-logq-alpha)@u
    du=s[:,None]*u;dv=s[:,None]*v
    nu,nv=du.double().norm(dim=-1),dv.double().norm(dim=-1)
    error=float(((nu-nv).abs()/torch.maximum(torch.ones_like(nu),torch.maximum(nu,nv))).max())
    bb=(b.T-(du if family=='remove' else dv)).T
    return bb,bb.sigmoid(),0.,0.,{'s':s,'hypothetical_pc1_norm':nu,'hypothetical_orthogonal_norm':nv},error

def common_state_center_check(family,basis,b,mass,maps,ds):
    """Algebraic check at one shared input, never compare downstream trajectories."""
    left=factorial_map(family+'_O'+basis+'_L10',10,b,mass,maps,ds)[0]
    right=factorial_map(family+'_F'+basis+'_L10',10,b,mass,maps,ds)[0]
    u=ds[basis]['u'];v=u if family=='remove'else ds[basis]['v']
    expected=((maps['F']['alpha']-maps['O']['alpha'])@u)*v
    return float(((right-left).T-expected).abs().max())
