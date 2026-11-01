#!/usr/bin/env python3
"""Check this supplement's saved scalar reductions; no model is loaded."""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "optimizer_control_8041"
PRIMARY = HERE.parent / "optimizer_axes"


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_manifest(directory):
    manifest = load(directory / "SOURCE-MANIFEST.json")
    for name, record in manifest["files"].items():
        path = directory / name
        data = path.read_bytes()
        require(len(data) == record["bytes"], f"size: {path}")
        require(hashlib.sha256(data).hexdigest() == record["sha256"], f"hash: {path}")
    if "pdf_sha256" in manifest:
        data = (directory / "adaptive-8041-summary.pdf").read_bytes()
        require(hashlib.sha256(data).hexdigest() == manifest["pdf_sha256"], "PDF hash")


def main():
    verify_manifest(HERE)
    verify_manifest(CONTROL)
    causal = load(HERE / "h10-causal-audit.json")
    require(causal["execution_count"] == 16 and causal["execution_all_complete"] is True, "causal count")
    require(causal["controls_all_pass"] is True and causal["predictions_all_pass"] is True, "causal flags")
    require(len(causal["results"]) == 2, "causal backgrounds")
    reductions = []
    for row in causal["results"]:
        pulse = row["pulse"]
        base = abs(pulse["Rtop4"])
        require(base > 0, "zero pulse denominator")
        removed = 1 - abs(pulse["top4_H10_clamped"]) / base
        only_error = abs(pulse["top4_nonH10_clamped"] - pulse["Rtop4"]) / base
        require(math.isclose(1 - removed, row["H10_clamp_magnitude_ratio"], rel_tol=1e-12), "clamp reduction")
        require(math.isclose(only_error, row["H10_only_pulse_relative_error"], rel_tol=1e-12), "H10-only reduction")
        require(pulse["Rtop4"] == pulse["sameworld_H10_sham"], "same-world sham")
        reductions.append({"context": row["context"], "removed_fraction": removed, "H10_only_error": only_error})

    scope = load(HERE / "scope-outcome-audit.json")
    require(scope["execution_count"] == 24 and scope["all_complete"] is True, "scope execution count")
    require(scope["controls_all_pass"] is True, "scope controls")
    counts = {name: {"pass": 0, "total": 0} for name in ("lambda1", "gate_minus2", "gate_plus2", "selfonly")}
    for row in scope["results"]:
        name = row["condition"]
        require(name in counts, "unknown scope condition")
        flags = row["passes"]
        require(flags and all(type(v) is bool for v in flags.values()), "invalid result flags")
        require(row["all_pass"] == all(flags.values()), "scope joint decision")
        if name in ("gate_minus2", "gate_plus2"):
            require(flags["gate_sign"] is True and flags["up"] is False, "near-threshold outcome")
        counts[name]["total"] += 1
        counts[name]["pass"] += int(row["all_pass"])
    require(counts == scope["criteria_counts"], "scope recorded counts")
    require(counts == {"lambda1": {"pass": 4, "total": 4}, "gate_minus2": {"pass": 0, "total": 4},
                       "gate_plus2": {"pass": 0, "total": 4}, "selfonly": {"pass": 2, "total": 2}}, "scope frozen counts")

    control = load(CONTROL / "results.json")
    primary = load(PRIMARY / "results.json")
    comparison = load(CONTROL / "comparison.json")
    protocol = load(CONTROL / "protocol.json")
    require((primary["column_neuron"], control["column_neuron"]) == (1710, 8041), "neuron identity")
    require(primary["focal_residual_coordinate"] == control["focal_residual_coordinate"] == 1764, "focal axis")
    require(primary["sampled_other_coordinates"] == control["sampled_other_coordinates"] == 128, "sample count")
    require(protocol["column_neuron"] == 8041 and protocol["focal_row"] == 1764, "control protocol")
    for obj, key in ((primary, "primary"), (control, "control")):
        v, gain = obj["metrics"]["v"], obj["metrics"]["gain"]
        require(v["sample_less"] == comparison[f"{key}_sampled_axes_with_lower_v"], "v count")
        require(math.isclose(v["focal_over_sample_median"], comparison[f"{key}_focal_v_over_sample_median"], rel_tol=1e-15), "v ratio")
        require(math.isclose(gain["focal_over_sample_median"], comparison[f"{key}_focal_gain_over_sample_median"], rel_tol=1e-15), "gain ratio")
        require(obj["epsilon_dominant_sample_count"] == comparison[f"{key}_epsilon_dominant_sample_count"], "epsilon count")
    print(json.dumps({"status": "PASS_SAVED_SUMMARY_ONLY", "causal_reductions": reductions,
                      "scope_counts": counts, "optimizer_axes": 128,
                      "limitations": "No model, original tensors, hook validation or input reconstruction."},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
