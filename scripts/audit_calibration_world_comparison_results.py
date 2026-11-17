"""Reanalyze Report 19 restricted observations with the pinned NumPy-only auditor."""

from pathlib import Path

from _pinned_audit import run_pinned

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return run_pinned(
        root=ROOT,
        study="calibration_world_comparison",
        auditor="audit_calibration_world_public.py",
        sha256="8693b778cd8409a14eb894afdac667f1ea90e1fd021a523756b958a87f876eee",
        description=__doc__,
    )


if __name__ == "__main__":
    raise SystemExit(main())
