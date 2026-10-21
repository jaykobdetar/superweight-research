"""FP64 additive-logit diagnostic; no model forward or fitted parameters.

All model logits arrive as FP32. Add singleton deltas in fixed L4..L11 order,
using CPU FP64 in query chunks. J and R are descriptive exact loss terms, not
mediation or feedback estimates. Keep tokenwise quantities before aggregation.
"""
import torch

LAYERS=tuple(range(4,12))
CHUNK=32
IDENTITY_TOLERANCE=1e-12

def cpu_logits(z):
    assert z.ndim==2 and z.dtype==torch.float32 and torch.isfinite(z).all()
    return z.detach().cpu().contiguous()

def loss64(z,targets):
    assert z.ndim==2 and targets.shape==(len(z),) and targets.dtype==torch.long
    assert torch.isfinite(z).all()
    targets=targets.cpu();out=[]
    for start in range(0,len(z),CHUNK):
        a=z[start:start+CHUNK].detach().cpu().double()
        out.append(a.logsumexp(-1)-a.gather(-1,targets[start:start+CHUNK,None]).squeeze(-1))
    result=torch.cat(out);assert torch.isfinite(result).all();return result

class AdditiveReadout:
    def __init__(self,baseline,targets):
        self.base=cpu_logits(baseline);self.targets=targets.cpu().clone()
        self.additive=self.base.double();self.loss_p=loss64(self.base,self.targets)
        self.single_losses={};self.layers=[]

    def add_singleton(self,layer,logits):
        assert len(self.layers)<8 and layer==LAYERS[len(self.layers)],'Wrong singleton order or duplicate'
        x=cpu_logits(logits);assert x.shape==self.base.shape
        self.single_losses[layer]=loss64(x,self.targets)
        for start in range(0,len(x),CHUNK):
            self.additive[start:start+CHUNK]+=x[start:start+CHUNK].double()-self.base[start:start+CHUNK].double()
        assert torch.isfinite(self.additive).all()
        self.layers.append(layer)

    def finish(self,joint,native):
        assert self.layers==list(LAYERS),'All eight singletons required before the joint readout'
        joint=cpu_logits(joint);native=cpu_logits(native)
        assert joint.shape==native.shape==self.base.shape
        lj=loss64(joint,self.targets);la=loss64(self.additive,self.targets)
        single=torch.stack([self.single_losses[l] for l in LAYERS])
        summed=(single-self.loss_p).sum(0)
        j=lj-la;r=la-self.loss_p-summed;nonadd=lj-self.loss_p-summed
        error=(nonadd-j-r).abs();assert float(error.max())<=IDENTITY_TOLERANCE
        data={'loss_p':self.loss_p,'loss_joint':lj,'loss_additive':la,'singleton_losses':single,
              'sum_singleton_delta':summed,'J':j,'R':r,'joint_minus_sum':nonadd,'identity_error':error}
        parts={k:[] for k in ('kl_joint_additive','kl_additive_joint','kl_p_additive','kl_native_additive',
            'centered_logit_l2','centered_logit_rms','centered_logit_max_abs',
            'joint_delta_centered_l2','additive_delta_centered_l2',
            'agreement_joint_additive','agreement_p_additive','agreement_native_additive')}
        for start in range(0,len(joint),CHUNK):
            a=self.additive[start:start+CHUNK];z=joint[start:start+CHUNK].double()
            p=self.base[start:start+CHUNK].double();c=native[start:start+CHUNK].double()
            laa=a.log_softmax(-1);lz=z.log_softmax(-1);lp=p.log_softmax(-1);lc=c.log_softmax(-1)
            for name,left,right in (('kl_joint_additive',lz,laa),('kl_additive_joint',laa,lz),
                                   ('kl_p_additive',lp,laa),('kl_native_additive',lc,laa)):
                parts[name].append((left.exp()*(left-right)).sum(-1))
            delta=z-a;delta-=delta.mean(-1,keepdim=True)
            parts['centered_logit_l2'].append(delta.norm(dim=-1))
            parts['centered_logit_rms'].append(delta.square().mean(-1).sqrt())
            parts['centered_logit_max_abs'].append(delta.abs().max(-1).values)
            for name,x in (('joint_delta_centered_l2',z-p),('additive_delta_centered_l2',a-p)):
                x-=x.mean(-1,keepdim=True);parts[name].append(x.norm(dim=-1))
            for name,x in (('agreement_joint_additive',z),('agreement_p_additive',p),('agreement_native_additive',c)):
                parts[name].append(x.argmax(-1).eq(a.argmax(-1)))
        data.update({k:torch.cat(v)for k,v in parts.items()})
        assert all(torch.isfinite(v).all()for v in data.values())
        # Floating-point KL roundoff is reported, never clipped.
        assert min(float(data[k].min())for k in data if k.startswith('kl_'))>=-1e-10
        return data
