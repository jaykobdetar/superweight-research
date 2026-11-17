"""Discover and run published, read-only CPU audits. Does not run model inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Audit:
    key: str
    scope: str
    script: str
    report_flag: str | None = None  # None: stdout is the full JSON report.


# Keep a positive list: many other scripts acquire data, regenerate evidence in
# place, or run model forwards. Do not dispatch by recursively discovering files.
AUDITS = (
    Audit("olmo-core", "OLMo historical loss/KL", "scripts/audit_results.py", "--output"),
    Audit("transfer", "GPT-2/Llama transfer and follow-ups", "scripts/audit_transfer.py", "--output"),
    Audit("gate", "GPT-2 gate subspaces", "scripts/audit_gate_results.py", "--output"),
    Audit("localization", "GPT-2 gate localization", "scripts/audit_localization_results.py", "--output"),
    Audit("additive", "GPT-2 additive output/loss", "scripts/audit_additive_results.py", "--output"),
    Audit("wiki", "GPT-2 Wiki calibration", "scripts/audit_wiki_calibration_results.py", "--output"),
    Audit("native-value", "GPT-2 native source values", "scripts/audit_native_value_results.py", "--out"),
    Audit("write-norm", "GPT-2 actual-write norms (16)", "scripts/audit_post_write_norm_results.py", "--report"),
    Audit("write-timing", "GPT-2 write timing (17)", "scripts/audit_query_write_timing_results.py", "--report"),
    Audit("kv-donor", "GPT-2 L11 K/V donors (18)", "scripts/audit_l11_key_value_donor_results.py", "--report"),
    Audit("calibration-worlds", "GPT-2 calibration worlds (19)",
          "scripts/audit_calibration_world_comparison_results.py", "--report"),
    Audit("olmo-files", "OLMo package manifest byte identities", "scripts/verify_olmo_bundles.py"),
    Audit("olmo-summary", "OLMo conditional algorithm/optimizer control",
          "research/olmo/algorithm_update/check_summary.py"),
    Audit("olmo-h14", "OLMo fixed-H14 holdout",
          "research/olmo/fixed_h14_susceptibility/audit.py"),
)


def new_output_dir(path: Path) -> Path:
    """Require a fresh destination outside the checkout, including through symlinks."""
    path = path.resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Pick a new output directory outside the repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def run(check: Audit, dest: Path) -> None:
    report = dest / f"{check.key}.json"
    command = [sys.executable, "-B", str(ROOT / check.script)]
    if check.report_flag:
        command.extend([check.report_flag, str(report)])
    # Disable inherited optimization: historical auditors use Python assert.
    env = dict(os.environ, PYTHONOPTIMIZE="0", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (dest / f"{check.key}.log").open("x") as log:
        if check.report_flag:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=log, check=True)
        else:
            with report.open("x") as output:
                subprocess.run(command, env=env, cwd=ROOT, stdout=output, stderr=log, check=True)
    # Reject a silent exit or broken/missing report. The original auditor's
    # exit status, not a guessed common JSON schema, determines numerical success.
    if not isinstance(json.loads(report.read_text()), dict):
        raise ValueError(f"{check.key}: expected a JSON object")


def main() -> int:
    by_key = {check.key: check for check in AUDITS}
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="show CPU-only checks; no NumPy needed")
    execute = commands.add_parser("run", help="run selected checks, stopping at the first failure")
    execute.add_argument("checks", nargs="+", choices=("all", *by_key))
    execute.add_argument("--out-dir", required=True, type=Path, help="new directory outside the checkout")
    args = parser.parse_args()
    if args.command == "list":
        for check in AUDITS:
            print(f"{check.key:18} {check.scope}")
        return 0
    if "all" in args.checks and len(args.checks) != 1:
        parser.error("use 'all' alone")
    if len(set(args.checks)) != len(args.checks):
        parser.error("specify each check once")
    selected = AUDITS if args.checks == ["all"] else [by_key[key] for key in args.checks]
    try:
        dest = new_output_dir(args.out_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for check in selected:
        print(f"Running {check.key}...", flush=True)
        try:
            run(check, dest)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"FAILED {check.key}: {error}. Inspect {dest / (check.key + '.log')}", file=sys.stderr)
            return 1
        print(f"PASS {check.key}: {dest / (check.key + '.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
