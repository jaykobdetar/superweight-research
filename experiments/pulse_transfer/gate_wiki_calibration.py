"""T18 matched Wiki4 center/direction factorial with explicit unchanged M8 anchors."""
import math
import torch
from gpt2_gate_subspace import LAYERS, ERROR_TOLERANCE, mapped_gate
from gpt2_gate_subspace_v2 import GateV2
from gpt2_query_source import QuerySource
from gpt2_gate_localization import GateLocalization, NORM_TOLERANCE
from gate_factorial_algebra import (FAMILIES, CELLS, SETS, GAP_FLOOR,
    factorial_map as primary_map, geometry, common_state_center_check)
from gate_direction_geometry import directions

CONDITIONS=('C','P','current','fullrank')+tuple(f+'_'+c+'_'+s for f in FAMILIES for c in CELLS for s in SETS)+tuple(f+'_M8_'+s for f in FAMILIES for s in SETS)
SNAPSHOT_MODES=('P',)+tuple(m for m in CONDITIONS if m.endswith('_L10'))
READOUT_ROWS=(1,126,127,510)

def mode_spec(mode):
    if mode in ('C','P','current','fullrank','calibration'):return None,None,None,()
    family,cell,which=mode.split('_')
    assert family in FAMILIES and cell in (*CELLS,'M8') and which in SETS,mode
    return family,('M' if cell=='M8' else cell[0]),('M' if cell=='M8' else cell[1]),LAYERS if which=='all' else (10,)

def wiki_map(mode,layer,b,mass,maps,ds):
    family,center,basis,affected=mode_spec(mode)
    if mode=='fullrank':
        bb,aa,le,me=mapped_gate(mode,b,mass,maps['M']);return bb,aa,le,me,{},0.
    if center=='M':
        return primary_map(family+'_OO_'+('all' if len(affected)==8 else 'L10'),layer,b,mass,{'O':maps['M']},{'O':ds['M']})
    return primary_map(mode,layer,b,mass,maps,ds)

def fit_geometry(maps,spectra):
    assert set(maps)==set(spectra)=={'O','F','M'}
    ds,report=geometry(maps['O'],maps['F'],spectra['F']);ds['M']=directions(maps['M'])
    for layer in LAYERS:
        for label in ('O','F'):
            eig=torch.tensor(spectra[label][layer]['r']['eigenvalues'],dtype=torch.float64)
            gap=float((eig[0]-eig[1])/eig[0])
            assert torch.isfinite(eig).all() and eig[0]>0 and gap>GAP_FLOOR,(label,layer,gap)
            report[layer][label+'_leading_eigenvalue']=float(eig[0]);report[layer][label+'_relative_leading_gap']=gap
        assert torch.equal(maps['O'][layer]['basis_qr'],maps['M'][layer]['basis_qr'])
    return ds,report

class GateWiki(GateV2):
    def install_sets(self,maps,ds,digest):
        assert set(maps)==set(ds)=={'O','F','M'}
        self.map_sets=maps;self.direction_sets=ds;self.set_digest=digest
        self.map_device={};self.direction_device={}
        self.install_maps(maps['M'],digest)
        # Validate each direction with the inherited FP64/FP32 checks.
        for label in ('O','F','M'):GateLocalization.install_directions(self,ds[label],digest)
    def reset_gate(self,mode='P',zero=True,mask='causal',snapshot=False):
        assert mode in CONDITIONS or mode=='calibration'
        # QuerySource's historical snapshots use L4/L11; disable those explicitly.
        super().reset_gate('C' if mode=='C' else 'P',zero,mask,False)
        self.mode=mode;self.capture_l10=snapshot;self.current_contexts={};self.localization_norms={}
    def combined(self,layer,output):
        result=super().combined(layer,output)
        if self.capture_l10 and layer==10:
            q,k,v,mask=self.reconstruction_inputs[layer]
            assert mask is not None
            self.snapshots[layer]={k:v.detach().cpu().clone() for k,v in zip(('q','k','v','mask'),(q,k,v,mask))}
        return result
    def context(self, layer, actual):
        raw_inputs = self.reconstruction_inputs[layer]
        QuerySource.context(self, layer, actual)
        self.current_contexts[layer]=actual.detach().cpu().clone()
        self.prefix_contexts[layer]={"current":actual[:,:128].detach().cpu().clone(),"delivered":actual[:,:128].detach().cpu().clone()}
        if not self.zero_values:
            return None
        b, a, u = self.gate_pending.pop(layer)
        n = actual.shape[1]
        c = actual.reshape(1,n,12,64)[0].transpose(0,1)[:,1:]
        mode=self.mode;family,center,basis,affected=mode_spec(mode);active=layer in affected
        assert self.maps is not None and self.calibration_sha256 and self.direction_sha256
        if layer not in self.map_device:
            self.map_device[layer]={name:{k:v.to(device=b.device,dtype=b.dtype) for k,v in mapping[layer].items()} for name,mapping in self.map_sets.items()}
            self.direction_device[layer]={name:{k:d[layer][k].to(device=b.device,dtype=b.dtype) for k in ('u','v')} for name,d in self.direction_sets.items()}
        proposed_b,target,full_error,full_mass_error,nd,norm_error=wiki_map(
            mode,layer,b,a,self.map_device[layer],self.direction_device[layer])
        center_errors={}
        if layer==10 and family is not None and center!='M':
            center_errors={f+'_'+d:common_state_center_check(f,d,b,a,self.map_device[layer],self.direction_device[layer]) for f in FAMILIES for d in ('O','F')}
        good = bool(torch.isfinite(b).all() and torch.isfinite(a).all() and (a>0).all()
                    and torch.isfinite(c).all() and torch.isfinite(u).all() and torch.isfinite(target).all())
        if proposed_b is not None: good = good and bool(torch.isfinite(proposed_b).all())
        if active: good = good and bool((target>=0).all() and (target<=1).all())
        errors = {'conditional_error': float('inf'), 'native_reconstruction_error': float('inf'),
                  'mixed_reconstruction_error': float('inf'), 'sigmoid_context_error': float('inf'),
                  'fullrank_logit_error': full_error, 'fullrank_mass_error': full_mass_error,
                  'fullrank_context_error': 0., 'formula_error': 0.}
        result = actual; gain = float('inf')
        if good:
            ratio = target/a
            gain = float(ratio.max())
            mixed = c*ratio[...,None] if active else c
            errors.update(conditional_error=float((c/a[...,None]-u).abs().max()),
                          native_reconstruction_error=float((c-a[...,None]*u).abs().max()),
                          mixed_reconstruction_error=float((mixed-target[...,None]*u).abs().max()),
                          sigmoid_context_error=float((c-b.sigmoid()[...,None]*u).abs().max()))
            if mode == 'fullrank':
                alternate = c*(proposed_b.sigmoid()/a)[...,None]
                errors['fullrank_context_error'] = float((alternate-c).abs().max())
            if active or mode in ('current','fullrank'):
                result = actual.clone()
                result[:,1:] = mixed.transpose(0,1).reshape(1,n-1,768)
                errors['formula_error'] = float((result[:,1:].reshape(n-1,12,64).transpose(0,1)-mixed).abs().max())
            good = bool(torch.isfinite(mixed).all() and torch.isfinite(ratio).all())
        good = good and math.isfinite(norm_error) and math.isfinite(gain) and all(math.isfinite(v) for v in errors.values())
        check = {'min_mass':float(a.min()), 'max_mass':float(a.max()),
                 'min_b':float(b.min()), 'max_b':float(b.max()),
                 'min_target_mass':float(target.min()), 'max_target_mass':float(target.max()),
                 'min_target_b':None if proposed_b is None else float(proposed_b.min()),
                 'max_target_b':None if proposed_b is None else float(proposed_b.max()),
                 'max_gain':gain, 'target_zeros':int((target==0).sum()),
                 'target_ones':int((target==1).sum()), 'finite':good,
                 'source_context_exact':torch.equal(result[:,0],actual[:,0]),
                 'unaffected_context_exact':active or torch.equal(result,actual),
                 'norm_match_error':norm_error,'active':active, 'center_shift_errors':center_errors, **errors}
        delivered=result.reshape(1,n,12,64)[0].transpose(0,1)[:,1:]
        nd.update(logit_delta_norm=(proposed_b-b).T.double().norm(dim=-1),
                  mass_delta_norm=(target-a).T.double().norm(dim=-1),
                  context_delta_norm=(delivered-c).transpose(0,1).reshape(n-1,768).double().norm(dim=-1))
        self.localization_norms[layer]={k:v.detach().cpu().clone() for k,v in nd.items()}
        self.prefix_contexts[layer]['delivered']=result[:,:128].detach().cpu().clone()
        state = {'b':b.detach().cpu().clone(), 'mass':a.detach().cpu().clone(),
                 'target_b':None if proposed_b is None else proposed_b.detach().cpu().clone(),
                 'target_mass':target.detach().cpu().clone()}
        self.gate_cache[layer] = state; self.gate_checks[layer] = check
        if layer in self.snapshots:
            self.snapshots[layer].update(gate_state=state, stable_mixture=u.detach().cpu().clone(),
                                         delivered_context=result.detach().cpu().clone())
        if not good or max(center_errors.values(),default=0.)>ERROR_TOLERANCE or max(errors.values())>ERROR_TOLERANCE or norm_error>NORM_TOLERANCE or not check['source_context_exact'] or not check['unaffected_context_exact']:
            self.failure_gate = {'layer':layer,'mode':self.mode,'calibration_sha256':self.calibration_sha256,
                'state':state,'check':check,'norms':self.localization_norms[layer],'direction_sha256':self.direction_sha256,'raw_inputs':{k:None if v is None else v.detach().cpu().clone()
                for k,v in zip(('q','k','v','mask'),raw_inputs)},'stable_mixture':u.detach().cpu().clone(),
                'actual_context':actual.detach().cpu().clone(),'delivered_context':result.detach().cpu().clone()}
            raise AssertionError(('gate diagnostic failed',layer,self.mode,check))
        return (result,) if active or mode in ('current','fullrank') else None

