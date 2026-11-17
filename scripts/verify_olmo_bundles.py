"""Verify size and SHA-256 of the local files listed in ten OLMo package manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research/olmo"
# These are versioned publication packages, not a discovery glob: an absent
# directory or manifest must fail, not disappear from a successful report.
MANIFESTS = (
    "training_development/BUNDLE_MANIFEST.json",
    "native_provenance_historical/MANIFEST.json",
    "signal_production/BUNDLE_MANIFEST.json",
    "signal_continuation/MANIFEST.json",
    "native_function/EXPORT_MANIFEST.json",
    "validated_continuation/EXPORT_MANIFEST.json",
    "optimizer_state/BUNDLE_MANIFEST.json",
    "optimizer_axes/BUNDLE_MANIFEST.json",
    "algorithm_update/SOURCE-MANIFEST.json",
    "optimizer_control_8041/SOURCE-MANIFEST.json",
)


def unique_object(pairs: list[tuple]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate manifest key: {key}")
        obj[key] = value
    return obj


def members(manifest: Path):
    """Normalize the three published layouts without rewriting their manifests."""
    payload = json.loads(manifest.read_text(), object_pairs_hook=unique_object)
    if manifest.parent.name == "native_provenance_historical":
        entries = payload
    else:
        entries = payload["files"]
    if not entries:
        raise ValueError(f"Empty inventory: {manifest}")
    if isinstance(entries, dict):
        yield from entries.items()
    elif isinstance(entries, list):
        for entry in entries:
            yield entry["path"], entry
    else:
        raise ValueError(f"Unexpected inventory layout: {manifest}")


def verify(manifest: Path) -> int:
    root = manifest.parent.resolve()
    seen = set()
    for name, entry in members(manifest):
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name:
            raise ValueError(f"Invalid relative member in {manifest}: {name}")
        path = root / rel
        if path.resolve() != path or rel in seen or not path.is_file():
            raise ValueError(f"Missing, duplicate or linked member: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Byte count differs: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"SHA-256 differs: {path}")
        seen.add(rel)
    return len(seen)


def main() -> None:
    packages = {name: verify(ROOT / name) for name in MANIFESTS}
    print(json.dumps({
        "verified_files": sum(packages.values()), "manifests": packages,
        "scope": "Byte identity against listed local members only; not model replay, "
                 "authentication of manifests, or a check for unlisted files.",
    }, indent=2))


if __name__ == "__main__":
    main()
