"""Launch a frozen numerical auditor without changing its code or the parent process."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


def run_pinned(
    *, root: Path, study: str, auditor: str, sha256: str,
    description: str, default_report: Path | None = None,
) -> int:
    """Preserve the historical --report CLI; never run unchecked bytes or with -O."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--report", type=Path, default=default_report,
                        required=default_report is None)
    args = parser.parse_args()
    code = root / "results" / f"{study}_reproduction" / "experiments" / "pulse_transfer"
    source = code / auditor
    if hashlib.sha256(source.read_bytes()).hexdigest() != sha256:
        raise RuntimeError(f"Pinned public auditor source differs: {source.name}")

    # Historical auditors rely on assertions. A clean interpreter also isolates
    # their process-global sys.path, BLAS settings, and imported analysis modules.
    env = dict(os.environ, PYTHONOPTIMIZE="0")
    command = [sys.executable, "-B", str(source), "--code", str(code),
               "--package", str(root / "results" / study), "--report", str(args.report)]
    return subprocess.run(command, env=env).returncode
