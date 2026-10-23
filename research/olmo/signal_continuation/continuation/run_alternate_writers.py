"""Frozen, small downstream primary-by-alternate factorial.

Default only freezes a protocol. Full forward requires explicit --execute and
the coordinator's validated-forward/CPU-slot approval; no downloads performed.
"""
from pathlib import Path
from dataclasses import asdict
import argparse, hashlib, json, sys, time
import numpy as np
from alternate_writer_hooks import (make_hook, group_factorial,
    SOURCE_NEGATIVE_GROUP, NEWLINE_POSITIVE_GROUP)

R = Path(__file__).resolve().parent
ROLE = Path("/workspace/shared/olmo_computational_role")
OUT = R / "downstream"


def digest_bytes(b):
    return hashlib.sha256(b).hexdigest()


def frozen_protocol():
    original = np.load(ROLE / "inputs/recovered_windows.npz")
    source_manifest = json.loads((ROLE / "inputs/recovered_window_manifest.json").read_text())
    hashes = {f"{x['corpus']}__{x['start']}": x["token_sha256"] for x in source_manifest}
    seqs, panels = {}, []
    for key in ("dubois__32768", "optimizers__12000"):
        source = original[key].astype(np.int64)
        assert digest_bytes(source.tobytes()) == hashes[key]
        assert source[0] == 187 and len(source) == 512
        for token, group_name, group in ((187, "newline_positive", NEWLINE_POSITIVE_GROUP),
                                        (510, "source_negative", SOURCE_NEGATIVE_GROUP)):
            ids = source[:256].copy()
            ids[0] = token
            name = key + ("__natural187" if token == 187 else "__controlled510")
            seqs[name] = ids
            later = np.flatnonzero(ids[1:] == 187) + 1
            specs = group_factorial(group, prefix=group_name)
            # First-stage 2x2, numerical reconstruction and add-back controls.
            chosen = specs[:5]
            panels.append(dict(name=name, source_window=key,
                source_window_sha256=hashes[key], input_sha256=digest_bytes(ids.tobytes()),
                source_token=token, first_later187=int(later[0]) if len(later) else None,
                input_kind="archived natural prefix" if token == 187 else "controlled source-token replacement; suffix fixed",
                group_name=group_name, group=group,
                treatments=[dict(name="native_no_hook", hook=None)] +
                           [dict(name=x.name, hook=asdict(x)) for x in chosen]))
    return seqs, dict(model="allenai/OLMo-1B-0724-hf",
        revision="d7cbab742d80589e714b1a2d7f838dcd21cbe143",
        dtype="FP32 NumPy; not bit-equivalent to original FP16 experiments",
        selection="Two previously known natural newline-column failures plus matched controlled The-source counterparts. Exploratory mechanism test, not new population estimate.",
        source_position=0, sequence_length=256,
        conditions="No attention masks; source-local L1 output changes only.",
        score="Query indices1..254 primary; query0 and query relative to first later187 separately. NLL[t] predicts ids[t+1].",
        stopping="No interpretation if numerical reconstruction/add-back max NLL error exceeds1e-3 or mean absolute exceeds1e-4; pause for diagnosis.",
        panels=panels,
        hook_sha256=digest_bytes((R/"alternate_writer_hooks.py").read_bytes()),
        runner_sha256=digest_bytes(Path(__file__).read_bytes()))


def freeze():
    OUT.mkdir(exist_ok=True)
    seqs, protocol = frozen_protocol()
    text = json.dumps(protocol, indent=2)
    target = OUT / "frozen_protocol.json"
    if target.exists():
        assert target.read_text() == text, "Frozen protocol changed; use a distinct output directory."
    else:
        target.write_text(text)
        np.savez_compressed(OUT / "frozen_inputs.npz", **seqs)
    print("Frozen", len(protocol["panels"]), "panels and",
          sum(len(x["treatments"]) for x in protocol["panels"]), "forward conditions.")
    return seqs, protocol


def execute():
    from alternate_writer_hooks import Intervention
    sys.path.insert(0, str(ROLE / "downstream"))
    from streamed_olmo import StreamedOlmo
    seqs, protocol = freeze()
    model = StreamedOlmo(half_round=False)
    for job in model.plan["jobs"]:
        assert Path(job["file"]).stat().st_size == job["size"]
    model_code = digest_bytes((ROLE/"downstream/streamed_olmo.py").read_bytes())
    run_identity = dict(protocol_sha256=digest_bytes((OUT/"frozen_protocol.json").read_bytes()),
                        model_code_sha256=model_code)
    identity_path = OUT / "run_identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == run_identity
    else:
        identity_path.write_text(json.dumps(run_identity, indent=2))
    down = np.memmap(R.parent/"tensors/down_proj.f32", mode="r", dtype="<f4", shape=(2048,8192))
    for panel in protocol["panels"]:
        ids = seqs[panel["name"]]
        for treatment in panel["treatments"]:
            stem = panel["name"] + "__" + treatment["name"]
            path = OUT / (stem+".npz")
            receipt_path = OUT / (stem+".json")
            if path.exists() or receipt_path.exists():
                assert path.exists() and receipt_path.exists(), "Partial result requires diagnosis."
                receipt = json.loads(receipt_path.read_text())
                assert receipt["result_sha256"] == digest_bytes(path.read_bytes())
                assert receipt["run_identity"] == run_identity
                continue
            spec = treatment["hook"]
            hook = None if spec is None else make_hook(down, Intervention(**spec))
            result = model.forward(ids, mlp_hook=hook, capture_layers=(1,), diagnostic_sources=(0,))
            # Layer2 source0 K/V are computed separately from the captured state.
            # This is diagnostic source-local replay, not an additional mediator intervention.
            source = result["states"]["1"]["post_layer"][:1]
            norm = model.ln(source)
            kv = {k: np.clip(model.linear(norm, "model.layers.2.self_attn."+k+"_proj.weight"), -8, 8)
                  for k in ("k", "v")}
            data = {k:result[k] for k in ("loss", "argmax", "target_logit", "log_normalizer")}
            data.update(source_post_l1=source, source_l2_normalized=norm,
                        source_l2_k=kv["k"], source_l2_v=kv["v"])
            for layer, diag in result["attention"].items():
                data["attention_to_source_l"+layer] = diag["0"]
            np.savez_compressed(path, **data)
            receipt = dict(panel=panel["name"], treatment=treatment["name"],
                input_sha256=panel["input_sha256"], seconds=result["seconds"],
                peak_rss=result["peak_rss"], run_identity=run_identity,
                result_sha256=digest_bytes(path.read_bytes()))
            receipt_path.write_text(json.dumps(receipt, indent=2))
            print(stem, "seconds", round(result["seconds"],3),
                  "mean_NLL_excluding_q0", float(result["loss"][1:].mean()), flush=True)
            del result, data
        # Check each panel immediately rather than completing an invalid batch.
        load = lambda name: np.load(OUT/(panel["name"]+"__"+name+".npz"))["loss"]
        checks=[]
        for a,b in (("native_no_hook","reconstruct_native"),
                    ("drop1710",panel["group_name"]+"_same_world_addback")):
            delta=np.abs(load(a)-load(b))
            checks.append(dict(left=a,right=b,max_abs=float(delta.max()),mean_abs=float(delta.mean())))
            assert delta.max()<1e-3 and delta.mean()<1e-4, (panel["name"],a,b,checks[-1])
        (OUT/(panel["name"]+"__numerical_controls.json")).write_text(json.dumps(checks,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args=ap.parse_args()
    if args.execute:
        execute()
    else:
        freeze()
