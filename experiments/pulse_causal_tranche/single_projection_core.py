#!/usr/bin/env python3
"""Inference-only super-weight interventions, reconstructed from the supplied README.

No model parameters are changed. Scalar operations replace selected contributions
in a Linear forward hook. `shuffle` is a causal one-position lag: the first
position remains unchanged, and every later position receives its predecessor.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
import torch.nn.functional as F

DEFAULT_MODEL = 'allenai/OLMo-1B-0724-hf'
DEFAULT_WEIGHTS = '1:1764:1710,1:1764:8041'
DEFAULT_SCALES = '-1,-0.5,0,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.75,0.9,1,1.1,1.25,1.5,2,3'
OPERATIONS = ['constant_mean','centered','shuffle','sign_only','clip','tanh','signed_square','cube','sqrt','addition']


def transform_input(x, operation, mean, r):
    """FP32 feature transforms. mean and r broadcast over the feature axis."""
    x = x.float()
    mean = mean.float()
    r = r.float().clamp_min(1e-6)
    if operation == 'baseline': return x
    if operation == 'constant_mean': return mean.expand_as(x)
    if operation == 'centered': return x - mean
    if operation == 'shuffle':
        if x.ndim < 2: raise ValueError('shuffle requires a sequence dimension')
        return torch.cat((x[..., :1, :], x[..., :-1, :]), dim=-2)
    if operation == 'sign_only': return r * x.sign()
    if operation == 'clip': return x.clamp(-r, r)
    if operation == 'tanh': return r * torch.tanh(x / r) / math.tanh(1)
    if operation == 'signed_square': return x * x.abs() / r
    if operation == 'cube': return x.pow(3) / r.square()
    if operation == 'sqrt': return x.sign() * torch.sqrt(r * x.abs())
    raise ValueError(f'Unsupported transform: {operation}')


class ProjectionProbe:
    def __init__(self, module, row, columns, mean, r):
        self.module, self.row, self.columns = module, int(row), list(columns)
        self.mean, self.r = mean, r
        self.operation, self.scale, self.scope = 'baseline', 1.0, 'scalar'
        self.handle = None
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.abs_sum = self.count = self.first_abs_sum = self.first_count = 0
        self.later_abs_sum = self.later_count = 0
        self.nonfinite = False

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        self.handle = None

    def _hook(self, module, inputs, output):
        x = inputs[0]
        operation = self.operation
        if operation == 'baseline' or (operation == 'scale' and self.scale == 1):
            result = output
        elif operation == 'direct_zero':
            # Independent positive control: a temporary projection copy, never model weights.
            key = tuple(self.columns)
            if getattr(self, '_ablated_columns', None) != key:
                self._ablated_weight = module.weight.detach().clone()
                self._ablated_weight[self.row, self.columns] = 0
                self._ablated_columns = key
            result = F.linear(x, self._ablated_weight, module.bias)
        elif self.scope == 'layer':
            if operation == 'addition':
                z = module.weight.float().sum(-1) + x.float().sum(-1, keepdim=True)
                if module.bias is not None: z = z + module.bias.float()
            else:
                z = x.float() * self.scale if operation == 'scale' else transform_input(x, operation, self.mean, self.r)
                z = F.linear(z, module.weight.float(), None if module.bias is None else module.bias.float())
            result = z.to(output.dtype)
        else:
            cols = self.columns
            values = x[..., cols].float()
            weights = module.weight[self.row, cols].float()
            original = values * weights
            if operation == 'scale': replacement = original * self.scale
            elif operation == 'addition': replacement = values + weights
            else:
                replacement = weights * transform_input(values, operation, self.mean[cols], self.r[cols])
            result = output.clone()
            result[..., self.row] = (output[..., self.row].float() + (replacement-original).sum(-1)).to(output.dtype)
        channel = result[..., self.row].detach().float().abs()
        self.nonfinite |= not bool(torch.isfinite(channel).all())
        self.abs_sum += float(channel.sum())
        self.count += channel.numel()
        first = channel[..., 0]
        self.first_abs_sum += float(first.sum())
        self.first_count += first.numel()
        later = channel[..., 1:]
        self.later_abs_sum += float(later.sum())
        self.later_count += later.numel()
        return result

    def diagnostics(self):
        return dict(down_proj_abs_mean=self.abs_sum/max(1,self.count),
                    down_proj_first_abs_mean=self.first_abs_sum/max(1,self.first_count),
                    down_proj_later_abs_mean=self.later_abs_sum/max(1,self.later_count),
                    projection_nonfinite=self.nonfinite)


def shifted_nll(logits, ids):
    """One negative log likelihood per predictable next token; omit final logits."""
    return F.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
                           ids[..., 1:].reshape(-1), reduction='none')


def json_write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def hash_model(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def score(model, windows, device, probe, baseline_predictions=None):
    probe.reset_diagnostics()
    chunks, predictions = [], []
    nll_sum, count, agreement, correct = 0., 0, 0, 0
    start = time.perf_counter()
    with torch.inference_mode():
        for index, window in enumerate(windows):
            ids = window.unsqueeze(0).to(device)
            logits = model(input_ids=ids, use_cache=False).logits
            loss = shifted_nll(logits, ids)
            if not bool(torch.isfinite(loss).all()):
                return dict(status='nonfinite',nll=None,perplexity=None,agreement=None,
                            token_accuracy=None,scored_tokens=count,seconds=time.perf_counter()-start,
                            **probe.diagnostics()), chunks, predictions
            prediction = logits[:, :-1].argmax(-1).cpu().flatten()
            predictions.append(prediction)
            n = loss.numel()
            block_sum = float(loss.sum())
            nll_sum += block_sum
            count += n
            block_correct = int((prediction == window[1:]).sum())
            correct += block_correct
            block_agreement = n if baseline_predictions is None else int((prediction == baseline_predictions[index]).sum())
            agreement += block_agreement
            chunks.append(dict(window=index,scored_tokens=n,nll=block_sum/n,
                               agreement=block_agreement/n,token_accuracy=block_correct/n))
            del logits, loss
    nll = nll_sum/count
    return dict(status='ok',nll=nll,perplexity=math.exp(nll) if nll<700 else None,
                agreement=agreement/count,token_accuracy=correct/count,scored_tokens=count,
                seconds=time.perf_counter()-start,**probe.diagnostics()), chunks, predictions


def load_text(args):
    if args.text_file:
        text = Path(args.text_file).read_text()
        source = dict(kind='text_file',path=str(Path(args.text_file).resolve()))
    else:
        from datasets import load_dataset
        dataset = load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='validation',cache_dir=args.cache_dir)
        text = '\n\n'.join(dataset['text'])
        source = dict(kind='huggingface_dataset',name='Salesforce/wikitext',config='wikitext-2-raw-v1',split='validation',fingerprint=dataset._fingerprint)
    source['text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    return text, source


def make_windows(ids, calibration_tokens, eval_tokens, context, offset):
    if offset < 0 or eval_tokens <= 0: raise ValueError('offset must be nonnegative and eval-tokens positive')
    if context < 2: raise ValueError('context must be >=2')
    if calibration_tokens < context: raise ValueError('calibration-tokens must be >= context')
    calibration_end = offset + calibration_tokens
    evaluation_start = calibration_end + context  # unused guard segment
    calibration = [ids[i:min(i+context,calibration_end)] for i in range(offset,calibration_end,context)]
    calibration = [w for w in calibration if len(w)>=2]
    blocks = max(1, eval_tokens // context)
    available = (len(ids)-evaluation_start)//context
    if blocks>available: raise ValueError(f'Need {blocks} held-out windows but corpus has {available}')
    # Sample disjoint windows evenly across the held-out corpus, not one passage.
    block_indices = torch.linspace(0,available-1,steps=blocks).round().long().tolist()
    starts = [evaluation_start + b*context for b in block_indices]
    windows = [ids[i:i+context] for i in starts]
    return calibration, windows, dict(calibration_start=offset,calibration_end=calibration_end,
                                     guard_tokens=context,evaluation_window_starts=starts,context=context,
                                     evaluation_input_tokens=sum(map(len,windows)),scored_tokens=sum(len(w)-1 for w in windows))


def calibrate(model, module, windows, device):
    activations = []
    def capture(_, inputs): activations.append(inputs[0].detach().float().cpu().reshape(-1,inputs[0].shape[-1]))
    hook = module.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            for w in windows: model(w.unsqueeze(0).to(device),use_cache=False)
    finally: hook.remove()
    values = torch.cat(activations)
    mean = values.mean(0)
    r = values.abs().median(0).values.clamp_min(1e-6)
    return mean.to(device), r.to(device), values


def bootstrap_ratio(chunks, baseline_chunks, seed):
    import numpy as np
    delta = np.array([a['nll']-b['nll'] for a,b in zip(chunks,baseline_chunks)])
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0,len(delta),size=(2000,len(delta)))].mean(-1)
    low, high = np.quantile(means,[.025,.975])
    return math.exp(float(low)), math.exp(float(high))


def plot_results(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':140,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors = {'both':'#3564bc','first':'#df7541','second':'#359875'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for target in ['both','first','second']:
        values=sorted([r for r in rows if r['target']==target and r['operation']=='scale' and r['scope']=='scalar' and r['status']=='ok'],key=lambda r:r['scale'])
        if not values: continue
        axes[0].plot([r['scale'] for r in values],[r['ppl_ratio'] for r in values],'.-',label=target,color=colors[target])
        axes[1].plot([r['scale'] for r in values],[r['agreement']*100 for r in values],'.-',label=target,color=colors[target])
    for ax in axes:
        ax.set_xlabel('Contribution multiplier α'); ax.axvline(1,color='gray',lw=.7,ls='--');ax.grid(alpha=.15);ax.legend()
    axes[0].set_yscale('log');axes[0].set_ylabel('Perplexity / baseline (log scale)')
    axes[1].set_ylabel('Top-1 agreement with baseline (%)')
    fig.suptitle('OLMo-1B: sensitivity of the two published super weights')
    fig.tight_layout();fig.savefig(out/'sensitivity.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for offset,target in enumerate(['both','first','second']):
        values={r['operation']:r for r in rows if r['scope']=='scalar' and r['target']==target and r['operation'] in OPERATIONS}
        if not values: continue
        x=list(range(len(OPERATIONS))); y=[values.get(op,{}).get('ppl_ratio',float('nan')) or float('nan') for op in OPERATIONS]
        axes[0].bar([v+(offset-1)*.24 for v in x],y,width=.24,label=target,color=colors[target])
        axes[1].plot(x,[(values.get(op,{}).get('agreement') or 0)*100 for op in OPERATIONS],'.-',label=target,color=colors[target])
    for ax in axes:
        ax.set_xticks(range(len(OPERATIONS)),[s.replace('_',' ') for s in OPERATIONS],rotation=48,ha='right');ax.grid(axis='y',alpha=.15);ax.legend()
    axes[0].set_yscale('log');axes[0].set_ylabel('Perplexity / baseline (log scale)');axes[0].axhline(1,color='gray',lw=.7)
    axes[1].set_ylabel('Top-1 agreement with baseline (%)')
    fig.suptitle('Scalar operator substitutions (calibration statistics held fixed)')
    fig.tight_layout();fig.savefig(out/'operations.png');plt.close(fig)

