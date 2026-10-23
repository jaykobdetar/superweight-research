"""Local structural controls and factor diagnostics; never downstream inference."""
from pathlib import Path
from dataclasses import asdict
import csv, hashlib, json, resource
import numpy as np
from alternate_writer_hooks import (D, N, PRIMARY, COORDINATE, make_hook,
    group_factorial, SOURCE_NEGATIVE_GROUP, NEWLINE_POSITIVE_GROUP, H14_QUERY_GROUP)

R = Path(__file__).resolve().parent
O = R / "results"
O.mkdir(exist_ok=True)
z = np.load(R.parent / "results/mlp1_replay.npz")
labels = z["labels"].tolist()
down = np.memmap(R.parent / "tensors/down_proj.f32", mode="r", dtype="<f4", shape=(D, N))
axis = np.full(D, -1 / D)
axis[COORDINATE] += 1
axis /= np.linalg.norm(axis)


def centered(v):
    return v - v.mean(axis=-1, keepdims=True)


def geometry(v):
    v = centered(v)
    a = float(v @ axis)
    n = float(np.linalg.norm(v))
    return dict(axis=a, norm=n, alignment=a / n,
                orthogonal_norm=float(np.linalg.norm(v - a * axis)),
                ln_coordinate=float(v[COORDINATE] / np.sqrt(np.mean(v * v) + 1e-5)))


def write_rows(name, rows):
    (O / (name + ".json")).write_text(json.dumps(rows, indent=2))
    with (O / (name + ".csv")).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


groups = dict(source_negative=SOURCE_NEGATIVE_GROUP,
              newline_positive=NEWLINE_POSITIVE_GROUP, h14_query=H14_QUERY_GROUP)
geometry_rows = []
for i, (panel, condition) in enumerate(labels):
    carry, base = z["carry"][i], z["column"][i]
    for name, group in groups.items():
        v = centered(np.asarray(down[:, group], dtype=np.float64) @
                     z["activation"][i, list(group)].astype(np.float64))
        a = float(v @ axis) * axis
        orth = v - a
        for treatment, vector in dict(
                column=base, column_group_removed=base-v,
                column_group_axis_removed=base-a,
                column_group_complement_removed=base-orth,
                group_only=carry+v, carry_only=carry).items():
            geometry_rows.append(dict(panel=panel, condition=condition, group=name,
                                      treatment=treatment, **geometry(vector)))
write_rows("axis_complement_controls", geometry_rows)

# Gate/up factor swaps are node-level arithmetic interventions, not necessarily
# realizable by natural normalized inputs. Evaluate fixed groups, not value sweeps.
factor_rows = []
pairs = [("source_The_vs_newline", 0, 2)]
for panel in ("water_cycle", "library", "bread", "programming"):
    pairs.append((panel, labels.index([panel, "newline_R"]),
                  labels.index([panel, "newline_R_edge_H14"])))
for panel, r, e in pairs:
    for group_name, group in groups.items():
        for j in group:
            w = float((down[COORDINATE, j] - down[:, j].mean(dtype=np.float64)) /
                      np.sqrt(1 - 1 / D))
            for gd in ("R", "E"):
                for ud in ("R", "E"):
                    g = float(z["gate"][r if gd == "R" else e, j])
                    u = float(z["up"][r if ud == "R" else e, j])
                    act = g * np.exp(-np.logaddexp(0, -g)) * u
                    factor_rows.append(dict(panel=panel, group=group_name, neuron=j,
                                            gate_donor=gd, up_donor=ud, gate=g,
                                            up=u, axis_write=act*w))
write_rows("alternate_gate_up_factorials", factor_rows)

# Real-tensor callback checks. Every treatment uses a fresh projection, and an
# independent FP64 masked projection is the numerical oracle.
checks = []
for i, group_name in ((0, "source_negative"), (2, "newline_positive"),
                      (7, "h14_query")):
    group = groups[group_name]
    a = np.stack([z["activation"][i], np.zeros(N, dtype=np.float32)])
    h = np.stack([z["normalized_pre_mlp1"][i], np.zeros(D, dtype=np.float32)])
    native = a @ down.T
    # Distinct untreated output sentinel checks exact target-row locality.
    native[1] = np.arange(D, dtype=np.float32) / 1024
    original_a, original_h, original_y = a.copy(), h.copy(), native.copy()
    for spec in group_factorial(group, prefix=group_name):
        hook = make_hook(down, spec)
        actual = hook(1, h, a, native)
        assert hook(0, h, a, native) is native
        assert np.array_equal(actual[1], native[1])
        assert np.array_equal(a, original_a) and np.array_equal(h, original_h)
        assert np.array_equal(native, original_y)
        oa = a[0].astype(np.float64).copy()
        if spec.keep is not None:
            oa.fill(0)
            oa[list(spec.keep)] = a[0, list(spec.keep)]
        else:
            oa[list(spec.drop)] = 0
        # Row streaming avoids creating a 128 MiB FP64 copy of the matrix.
        expected = np.empty(D)
        for lo in range(0, D, 64):
            expected[lo:lo+64] = down[lo:lo+64].astype(np.float64) @ oa
        if spec.partial_group:
            gw = down[:, spec.partial_group].astype(np.float64) @ a[0, list(spec.partial_group)]
            ax = float(gw @ axis) * axis
            expected -= ax if spec.remove_part == "axis" else gw-ax
        if spec.addback:
            expected += down[:, spec.addback].astype(np.float64) @ a[0, list(spec.addback)]
        err = float(np.max(np.abs(actual[0] - expected)))
        alignment_error = abs(geometry(z["carry"][i] + actual[0])["alignment"] -
                              geometry(z["carry"][i] + expected)["alignment"])
        assert err < 0.001 and alignment_error < 0.00001
        checks.append(dict(panel=labels[i], intervention=spec.name,
                           max_output_abs_error=err, alignment_error=alignment_error,
                           input_unchanged=True, non_target_exact=True,
                           other_layers_identity=True))

protocol = dict(
    scope="Source-local L1-output treatments. Not parameter ablations or new training.",
    callback="mlp_hook(layer_idx, normalized_input, activations, native_output)",
    array_convention="batch1 [sequence,width]; activation is post-SwiGLU",
    groups=groups,
    selection="Exploratory groups frozen from the preceding 21-state analysis, not held-out discoveries.",
    primary_stage="2x2 primary1710 presence x selected-group presence; same-world add-back on one representative each.",
    conditional_stage="Only after effect: axis/complement decomposition, group-only versus carry-only sufficiency.",
    falsifier="Local alignment improvement without loss/KV rescue does not establish a behavioral mechanism.",
    output_indexing="NLL index t uses query t to predict target t+1. Local edit p may affect t>=p; report t>p separately.",
    specifications={name: [asdict(s) for s in group_factorial(group, prefix=name)]
                    for name, group in groups.items()},
    code_sha256=hashlib.sha256((R/"alternate_writer_hooks.py").read_bytes()).hexdigest())
(R / "INTERVENTION_MANIFEST.json").write_text(json.dumps(protocol, indent=2))
result = dict(all_pass=True, tests=checks,
              maximum_output_abs_error=max(x["max_output_abs_error"] for x in checks),
              maximum_alignment_error=max(x["alignment_error"] for x in checks),
              peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              source_state_count=len(labels),
              downstream_forward_executed=False)
(O / "hook_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != "tests"}, indent=2))
