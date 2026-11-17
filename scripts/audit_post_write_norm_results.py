"""Run the Report 16 safe-array audit. No model, private inputs or network required."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/post_write_norm_matching"
REPRODUCTION = ROOT / "results/post_write_norm_reproduction"
CODE = REPRODUCTION / "experiments/pulse_transfer"


def validate_destination(report: Path) -> None:
    if report.exists():
        raise FileExistsError(f"Preserve existing report: {report}")
    if any(report.resolve().is_relative_to(root) for root in (PACKAGE, REPRODUCTION)):
        raise ValueError("Report outside immutable package required")


def audit() -> dict:
    # Configure NumPy's runtime before the backend imports it. The frozen code
    # manages its sibling imports; load the auditor itself by absolute filename.
    if not __debug__:
        raise RuntimeError("Run without Python optimization; the archived auditor uses assertions")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[name] = "1"
    spec = importlib.util.spec_from_file_location("audit_post_write_norm_public",
                                                 CODE / "audit_post_write_norm_public.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the published Report 16 auditor")
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    start = time.monotonic()
    result = backend.audit(PACKAGE, CODE)
    result.update(wall_seconds=time.monotonic() - start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    report = parser.parse_args().report
    if report is not None:
        validate_destination(report)
    payload = json.dumps(audit(), indent=2, allow_nan=False)
    if report is not None:
        with report.open("x") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
