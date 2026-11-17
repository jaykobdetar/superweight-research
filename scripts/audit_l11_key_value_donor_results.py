"""Audit Report 18's safe numerical results with the pinned NumPy-only auditor."""

from pathlib import Path

from _pinned_audit import run_pinned

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    return run_pinned(
        root=ROOT,
        study="l11_key_value_donor",
        auditor="audit_l11_key_value_donor_public.py",
        sha256="019a1935bd1b95e3a19f965de9d4db878159e23164d53d8b7e63ff8e889118f3",
        description=__doc__,
        default_report=ROOT / "l11_key_value_donor_public_audit.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
