"""Audit recorded metrics. Standard library only; does not run inference or write files."""
import csv
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
RANDOM = [f"r{i}" for i in range(8)]
DOSES = ["0", "0.25", "0.50", "0.75", "1"]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def group(path, keys, factor):
    out = defaultdict(dict)
    with path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            key = tuple(r[k] for k in keys)
            label = r[factor]
            check(label not in out[key], f"duplicate: {key}, {label}")
            out[key][label] = r
    return out


def load_vectors(path):
    vectors = json.loads(path.read_text())
    check(set(vectors) == set(["template", *RANDOM]), "unexpected vector labels")
    for name, entry in vectors.items():
        data = struct.pack("<128d", *entry["values"])
        check(hashlib.sha256(data).hexdigest() == entry["sha256"], f"vector hash: {name}")
        check(abs(entry["postnorm"] / vectors["template"]["postnorm"] - 1) < 1e-10,
              f"write length: {name}")
    return vectors


def audit(data=ROOT / "data"):
    for line in (data / "SHA256SUMS").read_text().splitlines():
        sha, filename = line.split()
        check(hashlib.sha256((data / filename).read_bytes()).hexdigest() == sha, filename)
    load_vectors(data / "vectors.json")
    meta = json.loads((data / "literary_manifest.json").read_text())
    books = set(meta["members"])
    expected = {(b, str(a)) for b in books for a in (20, 50, 80)}
    check(len(books) == 8 and not meta["ineligible"], "wrong book panel")
    check({(r["module"], str(r["anchor"])) for r in meta["rows"]} == expected, "manifest")
    q = group(data / "literary_queries.csv", ["book", "anchor_percent", "query"], "intervention")
    d = group(data / "literary_doses.csv", ["book", "anchor_pct"], "arm")
    kinds = ["newline", "neighbor", "substitute"]
    check(set(d) == expected and set(q) == {(*k, y) for k in expected for y in kinds}, "panel")
    kg = "same_world_scalar_lesion_KL"
    hits = Counter()
    natural, low, monotonic, onsets = {}, {}, [], []
    for (b, a, y), modes in q.items():
        check(set(modes) == set(["native", "template", *RANDOM]), "missing query arm")
        nt, t = modes["native"], modes["template"]
        z = float(nt["z"]) - float(t["z"])
        rz, rk, wn = [], [], float(t["write_norm"])
        for label, r in modes.items():
            check(int(r["n_positions"]) == 32 and math.isfinite(float(r[kg])), "query metrics")
            if label != "native":
                check(float(r["change_toward_negative_z"]) == float(nt["z"]) - float(r["z"]), "z")
            if label in RANDOM:
                rz.append(float(r["change_toward_negative_z"]))
                rk.append(float(r[kg]))
                check(abs(float(r["write_norm"]) / wn - 1) <= .02, "write dose mismatch")
        zg = z >= max(100, median(rz) + 100)
        fg = float(t[kg]) >= max(.25, float(nt[kg]) + .10, median(rk) + .10)
        joint = zg and fg
        for r in modes.values():
            check([r[k] for k in ("z_gate_for_template", "KL_gate_for_template", "joint_gate_for_template")]
                  == list(map(str, [zg, fg, joint])), f"query gate: {b} {a} {y}")
        hits[y, "joint"] += joint
        hits[y, "functional"] += fg
        if y == "newline":
            natural[b, a] = joint

    c, v, j = ("intact_C_KL_native_to_steered", "vulnerability_V_KL_steered_to_steered_lesion",
               "joint_J_KL_native_to_steered_lesion")
    for (b, a), modes in d.items():
        check(set(modes) == set(DOSES + RANDOM), "dose panel")
        check(float(modes["0"][c]) == 0 and modes["0"][v] == modes["0"][j], "identity")
        query = q[b, a, "newline"]
        for lab, r in modes.items():
            check(int(r["scored_positions"]) == 32 and all(math.isfinite(float(r[key])) for key in (c,v,j)), "dose metrics")
            if lab not in ("0.25", "0.50", "0.75"):
                qr = query[{"0": "native", "1": "template"}.get(lab, lab)]
                for dx, qx in ((v,kg), ("z","z"), ("V_delta_NLL","delta_NLL"), ("V_argmax_change","argmax_change")):
                    check(float(r[dx]) == float(qr[qx]), "shared endpoint mismatch")
        vals = [float(modes[k][v]) for k in DOSES]
        full = modes["1"]
        ok = natural[b, a] and float(full[c]) <= min(.10, .10 * float(full[v]))
        check(all(r["full_amplitude_low_cost_gate"] == str(ok) for r in modes.values()), "cost gate")
        low[b, a] = ok
        monotonic.append(all(x <= z + .01 for x, z in zip(vals, vals[1:])))
        onsets.append(next((k for k, val in zip(DOSES, vals) if val >= .25), "none"))
    by_book = {b: sum(low[b, str(a)] for a in (20,50,80)) for b in sorted(books)}
    return {
        "query_joint_hits": {k:hits[k,"joint"] for k in kinds},
        "query_functional_hits": {k:hits[k,"functional"] for k in kinds},
        "full_dose_low_cost_hits_by_book": by_book,
        "per_dose_median_C_V_J": {k:[median(float(m[k][s]) for m in d.values()) for s in (c,v,j)] for k in DOSES},
        "first_tested_vulnerability_over_0.25": dict(Counter(onsets)),
        "monotone_within_0.01": sum(monotonic),
        "total_windows": len(expected),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
