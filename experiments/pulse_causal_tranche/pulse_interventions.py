"""Position-selective ablation and direct additive pulses; model weights stay fixed."""
from __future__ import annotations
import json
import torch
import torch.nn.functional as F

class PulseIntervention:
    """Remove selected scalar(s), then add one fixed pulse per active layer.

    Selected coordinates must share one output row per layer. A temporary copy
    of each projection with selected weights zeroed computes the removed-row
    result. The original module parameters and all other output rows are kept.
    Pulses add to that row after ablation; they never read evaluation inputs.
    """
    def __init__(self, modules, coordinates):
        self.modules = modules
        self.coordinates = [tuple(c) for c in coordinates]
        self.selected = self.coordinates.copy()
        self.mode = 'baseline'
        self.ablation_positions = 'all'
        self.pulse_position = 0
        self.pulse_values = {}
        self.pulse_rows = {}
        self.ablation_method = 'zero_projection'
        self.handles = []
        self._cache = {}
        self.reset_diagnostics()

    def configure(self, condition):
        self.selected = [tuple(c) for c in condition.get('selected',self.coordinates)]
        self.mode = condition.get('mode','baseline')
        self.ablation_positions = condition.get('ablation_positions','all')
        self.pulse_position = int(condition.get('pulse_position',0))
        self.pulse_values = {int(l):float(v) for l,v in condition.get('pulse_values',{}).items()}
        self.pulse_rows = {int(l):int(v) for l,v in condition.get('pulse_rows',{}).items()}
        self.reset_diagnostics()

    def reset_diagnostics(self):
        if self.mode not in ['baseline','ablate']:raise ValueError('Invalid intervention mode')
        if self.ablation_positions not in ['all','first','later','none']:raise ValueError('Invalid position mask')
        if self.pulse_position < 0:raise ValueError('pulse_position must be nonnegative')
        self.stats = {l:{'abs_sum':0.,'count':0,'first_abs_sum':0.,'first_count':0,'later_abs_sum':0.,'later_count':0,'nonfinite':False} for l in self.modules}
        for layer in self.modules:
            selected=[c for c in self.selected if c[0]==layer]
            if len(set(c[1] for c in selected))>1:raise ValueError('One selected output row per layer is required')
            if len(set(selected))!=len(selected):raise ValueError('Duplicate coordinates')
            module=self.modules[layer]
            for l,r,c in selected:
                if r<0 or r>=module.out_features or c<0 or c>=module.in_features:raise ValueError('Coordinate out of bounds')
            pulse_row=self.pulse_rows.get(layer,selected[0][1] if selected else 0)
            if pulse_row<0 or pulse_row>=module.out_features:raise ValueError('Pulse row out of bounds')
        if any(c[0] not in self.modules for c in self.selected):raise ValueError('Unknown target layer')

    def __enter__(self):
        if self.handles:raise RuntimeError('Intervention is already installed')
        for layer,module in self.modules.items():
            self.handles.append(module.register_forward_hook(lambda mod,inputs,output,l=layer:self._hook(l,mod,inputs,output)))
        return self

    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[]

    def _hook(self,layer,module,inputs,output):
        selected=[c for c in self.selected if c[0]==layer]
        row=selected[0][1] if selected else next((r for l,r,c in self.coordinates if l==layer),0)
        result=output
        if self.mode=='ablate' and selected:
            result=output.clone()
            if self.ablation_positions!='none':
                columns=[c[2] for c in selected]
                if self.ablation_method=='zero_projection':
                    key=(layer,row,tuple(columns))
                    if key not in self._cache:
                        weights=module.weight.detach().clone();weights[row,columns]=0
                        self._cache[key]=weights
                    removed=F.linear(inputs[0],self._cache[key],module.bias)[...,row]
                elif self.ablation_method=='contribution':
                    old=(inputs[0][...,columns].float()*module.weight[row,columns].float()).sum(-1)
                    removed=(output[...,row].float()-old).to(output.dtype)
                else:raise ValueError('Unknown ablation method')
                if self.ablation_positions=='all':result[...,row]=removed
                elif self.ablation_positions=='first':result[...,0,row]=removed[...,0]
                elif self.ablation_positions=='later':result[...,1:,row]=removed[...,1:]
            if layer in self.pulse_values and self.pulse_position<output.shape[-2]:
                target_row=self.pulse_rows.get(layer,row)
                result[...,self.pulse_position,target_row]=(result[...,self.pulse_position,target_row].float()+self.pulse_values[layer]).to(output.dtype)
        values=result[...,row].detach().float().abs();s=self.stats[layer]
        s['abs_sum']+=float(values.sum());s['count']+=values.numel()
        s['first_abs_sum']+=float(values[...,0].sum());s['first_count']+=values[...,0].numel()
        s['later_abs_sum']+=float(values[...,1:].sum());s['later_count']+=values[...,1:].numel()
        s['nonfinite']|=not bool(torch.isfinite(values).all())
        return result

    def diagnostics(self):
        diagnostics={str(l):dict(down_proj_abs_mean=s['abs_sum']/max(1,s['count']),down_proj_first_abs_mean=s['first_abs_sum']/max(1,s['first_count']),down_proj_later_abs_mean=s['later_abs_sum']/max(1,s['later_count']),projection_nonfinite=s['nonfinite']) for l,s in self.stats.items()}
        return {**diagnostics[str(min(self.modules))],'layer_diagnostics':json.dumps(diagnostics)}
