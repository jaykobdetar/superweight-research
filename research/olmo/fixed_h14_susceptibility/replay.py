"""Optional offline GPU replay of one 71-token L1/E1 window, compared with exported data."""
import argparse
import hashlib
import json
import math
import os
import struct
import zipfile
from pathlib import Path

from audit import ROOT, DOSES, RANDOM, check, group, load_vectors


def source_window(archive, tokenizer_json, book, anchor):
    from tokenizers import Tokenizer
    meta = json.loads((ROOT / "data/literary_manifest.json").read_text())
    row, = [r for r in meta["rows"] if r["module"] == book and r["anchor"] == anchor]
    for file, expected in ((archive, meta["archive"]), (tokenizer_json, meta["tokenizer_sha256"])):
        with file.open("rb") as f:
            check(hashlib.file_digest(f, "sha256").hexdigest() == expected, "input digest mismatch")
    with zipfile.ZipFile(archive) as z:
        raw = z.read(row["member"])
    check(hashlib.sha256(raw).hexdigest() == row["member_sha256"], "book digest")
    text = raw.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    ids = Tokenizer.from_file(str(tokenizer_json)).encode(text, add_special_tokens=False).ids
    k = row["token_offset"]
    window = ids[k - 32:k + 39]
    check(hashlib.sha256(struct.pack("<71i", *window)).hexdigest() == row["ids_sha256"], "window hash")
    check(window[32] == 187 and window[32-row["prior_distance"]] == 187, "newline position")
    return window, row["near"]


def model_digest(model):
    # Stream blocks to host; the digest uses unique registered names, shape, dtype and little-endian bytes.
    h = hashlib.sha256()
    for name, p in model.named_parameters(remove_duplicate=True):
        for v in (name, str(tuple(p.shape)), str(p.dtype)):
            h.update(v.encode() + b"\0")
        for block in p.detach().reshape(-1).split(2**20):
            h.update(memoryview(block.cpu().contiguous().numpy()).cast("B"))
    return h.hexdigest()


def replay(args):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ids, offset = source_window(args.corpus, args.model / "tokenizer.json", args.book, args.anchor)
    if not args.run:
        return {"verified_window": [args.book, args.anchor], "model_forwards": 0}
    import torch
    from transformers import AutoModelForCausalLM
    check(torch.cuda.is_available() and torch.cuda.mem_get_info()[0] >= 6 * 2**30, "requires 6 GiB free VRAM")
    torch.cuda.set_per_process_memory_fraction(.55)
    torch.set_num_threads(1)
    torch.manual_seed(1117)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        attn_implementation="eager", dtype=torch.float16, low_cpu_mem_usage=True, device_map={"":"cuda:0"}
    ).eval()
    check(model_digest(model) == "5ad3e5d6f3dd2e671b23d2d6b530e30aab7b22438be14fd7c2bf71f18527dda5",
          "not the recorded FP16 parameter identity")
    l0, l1 = model.model.layers[:2]
    oproj = l0.self_attn.o_proj
    w = float(l1.mlp.down_proj.weight[1764,1710].detach().item())
    vectors = {k:torch.tensor(v["values"], dtype=torch.float64, device="cuda")
               for k,v in load_vectors(ROOT / "data/vectors.json").items()}
    query_rows = group(ROOT / "data/literary_queries.csv", ["book","anchor_percent","query"], "intervention")
    dose_rows = group(ROOT / "data/literary_doses.csv", ["book","anchor_pct"], "arm")
    key = (args.book, str(args.anchor))
    forwards, comparisons, worst = 0, 0, 0.

    def match(x, saved):
        nonlocal comparisons, worst
        e = abs(x - float(saved))
        worst = max(worst, e)
        check(math.isfinite(e) and e <= args.atol, f"absolute error {e:g} exceeds {args.atol:g}")
        comparisons += 1

    def world(tokens, pos, delta=None, lesion=None):
        nonlocal forwards
        cache = {}
        def head(_, inp):
            if delta is not None:
                ctx = inp[0].clone()
                h = ctx[0,pos,14*128:15*128]
                h.copy_((h.double()+delta).to(h.dtype))
                return (ctx, *inp[1:])
        def read_z(_, inp):
            cache["z"] = inp[0][0,pos,1710].item()
        def lesion_hook(_, _inp, out):
            if lesion is not None:
                y = out.clone()
                y[0,pos,1764] = (out[0,pos,1764].double()-lesion*w).to(out.dtype)
                return y
        hooks = [oproj.register_forward_pre_hook(head), l1.mlp.down_proj.register_forward_pre_hook(read_z),
                 l1.register_forward_hook(lesion_hook)]
        try:
            logits = model(tokens,use_cache=False).logits[0].detach().cpu()
        finally:
            for hook in hooks:
                hook.remove()
        forwards += 1
        return logits, cache["z"]

    def score(a, b, pos, tokens):
        p = a[pos+1:pos+33].double().log_softmax(-1)
        q = b[pos+1:pos+33].double().log_softmax(-1)
        target = tokens[0,pos+2:pos+34].cpu().reshape(-1,1)
        return ((p.exp()*(p-q)).sum(-1).mean().item(),
                (p.gather(-1,target)-q.gather(-1,target)).mean().item(),
                (p.argmax(-1)!=q.argmax(-1)).double().mean().item())

    with torch.inference_mode():
        changed = ids.copy()
        changed[32] = ids[32+offset]
        for kind, seq, pos in (("newline", ids, 32), ("neighbor", ids, 32+offset), ("substitute", changed, 32)):
            tokens = torch.tensor([seq],device="cuda")
            native, z0 = world(tokens,pos)
            arms = {k:float(k)*vectors["template"] for k in (DOSES if kind=="newline" else ["0","1"])}
            for name,delta in {**arms, **{k:vectors[k] for k in RANDOM}}.items():
                intact, z = world(tokens,pos,delta)
                lesioned, z1 = world(tokens,pos,delta,z)
                check(z == z1 and torch.equal(native[:pos],intact[:pos]) and
                      torch.equal(native[:pos],lesioned[:pos]), "causal-prefix or observer mismatch")
                if name == "0":
                    check(torch.equal(native,intact) and z == z0, "zero-edit identity")
                V = score(intact,lesioned,pos,tokens)
                if name not in ("0.25","0.50","0.75"):
                    r = query_rows[(*key,kind)][{"0":"native","1":"template"}.get(name,name)]
                    for v,col in zip((z,*V),("z","same_world_scalar_lesion_KL","delta_NLL","argmax_change")):
                        match(v,r[col])
                if kind == "newline":
                    r = dose_rows[key][name]
                    for scores,cols in (
                        (score(native,intact,pos,tokens), ("intact_C_KL_native_to_steered","C_delta_NLL","C_argmax_change")),
                        (V, ("vulnerability_V_KL_steered_to_steered_lesion","V_delta_NLL","V_argmax_change")),
                        (score(native,lesioned,pos,tokens), ("joint_J_KL_native_to_steered_lesion","J_delta_NLL","J_argmax_change")),
                    ):
                        for v,col in zip(scores,cols):
                            match(v,r[col])
    return {"window":key,"forwards":forwards,"comparisons":comparisons,"maximum_absolute_error":worst}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="local, compatible safetensors checkpoint directory")
    p.add_argument("--corpus", type=Path, required=True, help="local, checksum-pinned NLTK gutenberg.zip")
    p.add_argument("--book", required=True, help="manifest book ID, e.g. austen-emma")
    p.add_argument("--anchor", required=True, type=int, choices=(20,50,80))
    p.add_argument("--run", action="store_true", help="opt in to GPU model loading and forward comparison")
    p.add_argument("--atol", type=float, default=1e-8, help="absolute per-scalar tolerance, not an accuracy metric")
    print(json.dumps(replay(p.parse_args()),indent=2))
