"""CPU-only synthetic checks; AST extraction avoids runner/model imports."""
import ast
import hashlib
import json
import os
import resource
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

torch.set_num_threads(1)
torch.manual_seed(91)
root = Path(__file__).resolve().parents[1]
source = (root / 'run_tranche.py').read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Hooks')
namespace = {'torch': torch, 'LAYERS': [2, 12, 15]}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<Hooks AST>', 'exec'), namespace)
Hooks = namespace['Hooks']
h = Hooks.__new__(Hooks)
h.capture = False
h.cache = {}
results = {}

recipient = torch.randn(1, 12, 2048, dtype=torch.float16)
donor = torch.randn_like(recipient)
h.patch = {'type': 'kv', 'heads': [(2, 15)], 'kinds': 'k', 'positions': 'zero', 'donor': {(2, 'k'): donor}}
expected = recipient.clone()
expected[:, :1, 1920:] = donor[:, :1, 1920:]
assert torch.equal(h.proj(2, 'k', recipient), expected)
assert torch.equal(h.proj(2, 'v', recipient), recipient)
assert torch.equal(h.proj(12, 'k', recipient), recipient)
results['kv_zero_localization'] = True
h.patch['positions'] = 'all'
expected[:, :, 1920:] = donor[:, :, 1920:]
assert torch.equal(h.proj(2, 'k', recipient), expected)
h.patch['donor'][(2, 'k')] = recipient
assert torch.equal(h.proj(2, 'k', recipient), recipient)
results['kv_all_and_identity'] = True
h.capture = True
h.patch = {}
output = h.proj(2, 'k', donor.clone())
saved = h.cache[(2, 'k')].clone()
output.clamp_(-0.1, 0.1)
assert torch.equal(h.cache[(2, 'k')], saved)
results['preclip_capture_clone'] = True
h.capture = False
h.patch = {'type': 'context', 'heads': [(2, 15)], 'donor': {(2, 'context'): donor}}
expected = recipient.clone()
expected[:, :, 1920:] = donor[:, :, 1920:]
assert torch.equal(h.context(2, recipient), expected)
assert torch.equal(h.context(12, recipient), recipient)
results['context_localization'] = True

delta = donor.float() - recipient.float()
q, _ = torch.linalg.qr(torch.randn(2048, 32))
random, _ = torch.linalg.qr(torch.randn(2048, 32))
base = {'type': 'direction', 'layer': 2, 'delta': delta, 'mean': delta[0, 9:].mean(0),
        'vectors': q, 'reference': q[:, :8], 'mode': 'full', 'sign': 1}
h.patch = base
actual = h.residual(2, recipient)
assert torch.equal(actual[:, :9], recipient[:, :9])
assert torch.equal(actual[:, 9:], donor[:, 9:])
h.patch = base | {'sign': -1}
actual = h.residual(2, donor)
assert torch.equal(actual[:, :9], donor[:, :9])
assert torch.equal(actual[:, 9:], recipient[:, 9:])
results['full_patch_reconstruction_both_directions'] = True

# Deliberately disparate magnitude finite fp16 values challenge cancellation.
clean = torch.tensor([1., 2048., 65504., 1e-5], dtype=torch.float16)
deleted = torch.tensor([.0003, -.5, .0003, 65504.], dtype=torch.float16)
diff = deleted.float() - clean.float()
reconstructed = (clean.float() + diff).half()
results['float32_additive_extreme_reconstruction'] = {
    'clean': clean.tolist(), 'donor': deleted.tolist(), 'reconstructed': reconstructed.tolist(),
    'exact': torch.equal(reconstructed, deleted),
}
# Per-token norm matching is measured in float32 before output cast.
raw = delta[0, 9:]
a = (raw @ random[:, :8]) @ random[:, :8].T
reference = (raw @ q[:, :8]) @ q[:, :8].T
a *= reference.norm(dim=-1, keepdim=True) / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
assert torch.allclose(a.norm(dim=-1), reference.norm(dim=-1), rtol=1e-6)
results['random8_per_token_norm_match'] = True

results['runner_sha256'] = hashlib.sha256(source.encode()).hexdigest()
results['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
results['device'] = str(recipient.device)
destination = Path(__file__).with_name('cpu_hook_checks.json')
destination.write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
