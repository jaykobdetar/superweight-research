"""Offline contract tests for the maintained tooling, without NumPy or a checkpoint."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import _pinned_audit
import audit_post_write_norm_results as write_norm
import audits
import verify_olmo_bundles as bundles


class Workspace(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.root = self.work / "repo"
        self.root.mkdir()

    def main(self, *args):
        with patch.object(sys, "argv", ["audits", *args]), patch.object(audits, "ROOT", self.root):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return audits.main()


class AuditRunnerTests(Workspace):
    def test_all_registry_paths_exist_and_names_are_unique(self):
        self.assertEqual(len(audits.AUDITS), len({c.key for c in audits.AUDITS}))
        for check in audits.AUDITS:
            self.assertTrue((audits.ROOT / check.script).is_file(), check.key)

    def test_rejects_existing_and_symlinked_in_tree_destination(self):
        with patch.object(audits, "ROOT", self.root):
            with self.assertRaises(FileExistsError):
                audits.new_output_dir(self.work)
            alias = self.work / "alias"
            alias.symlink_to(self.root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                audits.new_output_dir(alias / "reports")
        self.assertFalse((self.root / "reports").exists())

    def test_selection_errors_create_no_reports(self):
        dest = self.work / "receipts"
        for choices in (("all", "gate"), ("gate", "gate"), ("typo",)):
            with self.subTest(choices=choices), self.assertRaises(SystemExit):
                self.main("run", *choices, "--out-dir", str(dest))
        self.assertFalse(dest.exists())

    def test_real_subprocess_output_modes_and_optimization(self):
        (self.root / "stdout.py").write_text("assert __debug__\nprint('{\u0022stdout\u0022: true}')\n")
        (self.root / "flag.py").write_text(
            "import json, sys\nfrom pathlib import Path\nassert sys.argv[1] == '--out'\n"
            "Path(sys.argv[2]).write_text(json.dumps({'flag': True}))\nprint('log message')\n"
        )
        entries = (audits.Audit("s", "", "stdout.py"), audits.Audit("f", "", "flag.py", "--out"))
        with patch.object(audits, "AUDITS", entries), patch.dict(os.environ, PYTHONOPTIMIZE="2"):
            self.assertEqual(self.main("run", "all", "--out-dir", str(self.work / "out")), 0)
        self.assertEqual(json.loads((self.work / "out/s.json").read_text()), {"stdout": True})
        self.assertEqual(json.loads((self.work / "out/f.json").read_text()), {"flag": True})
        self.assertIn("log message", (self.work / "out/f.log").read_text())

    def test_failure_stops_following_audits_and_keeps_diagnostics(self):
        (self.root / "bad.py").write_text("raise RuntimeError('fixture failure')\n")
        entries = (audits.Audit("bad", "", "bad.py"), audits.Audit("next", "", "absent.py"))
        dest = self.work / "out"
        with patch.object(audits, "AUDITS", entries):
            self.assertEqual(self.main("run", "all", "--out-dir", str(dest)), 1)
        self.assertIn("fixture failure", (dest / "bad.log").read_text())
        self.assertFalse((dest / "next.log").exists())

    def test_silent_success_is_not_a_report(self):
        (self.root / "empty.py").write_text("pass\n")
        with patch.object(audits, "AUDITS", (audits.Audit("empty", "", "empty.py"),)):
            self.assertEqual(self.main("run", "empty", "--out-dir", str(self.work / "out")), 1)


class PinnedLauncherTests(Workspace):
    def setUp(self):
        super().setUp()
        code = self.root / "results/demo_reproduction/experiments/pulse_transfer"
        code.mkdir(parents=True)
        self.source = code / "audit.py"
        self.source.write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "Path(sys.argv[-1]).write_text(json.dumps([__debug__, *sys.argv[1:-1]]))\n"
            "raise SystemExit(7)\n"
        )
        self.options = dict(root=self.root, study="demo", auditor="audit.py",
                            sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                            description="test")

    def test_passes_report_arguments_unoptimized_and_propagates_exit(self):
        report = self.work / "report"
        with patch.object(sys, "argv", ["wrapper", "--report", str(report)]):
            with patch.dict(os.environ, PYTHONOPTIMIZE="2"):
                self.assertEqual(_pinned_audit.run_pinned(**self.options), 7)
        self.assertEqual(json.loads(report.read_text()),
                         [True, "--code", str(self.source.parent), "--package", str(self.root / "results/demo"),
                          "--report"])

    def test_corrupt_source_is_rejected_before_execution(self):
        self.source.write_text("raise RuntimeError('backend executed')")
        with patch.object(sys, "argv", ["wrapper"]):
            with self.assertRaisesRegex(RuntimeError, "Pinned public auditor source differs"):
                _pinned_audit.run_pinned(**self.options, default_report=self.work / "out")
        self.assertFalse((self.work / "out").exists())

    def test_write_norm_protects_existing_report_and_bundles(self):
        with self.assertRaises(FileExistsError):
            write_norm.validate_destination(self.source)
        for folder in (write_norm.PACKAGE, write_norm.REPRODUCTION):
            with self.assertRaises(ValueError):
                write_norm.validate_destination(folder / "not-a-report.json")


class BundleTests(Workspace):
    def setUp(self):
        super().setUp()
        self.member = self.root / "member.txt"
        self.member.write_bytes(b"data")
        self.entry = {"bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()}
        self.manifest = self.root / "inventory.json"

    def save(self, value):
        self.manifest.write_text(json.dumps(value))

    def test_mapping_list_and_historical_layouts(self):
        for files in ({"member.txt": self.entry}, [{"path": "member.txt", **self.entry}]):
            self.save({"files": files})
            self.assertEqual(bundles.verify(self.manifest), 1)
        hist = self.work / "native_provenance_historical"
        self.root.rename(hist)
        manifest = hist / self.manifest.name
        manifest.write_text(json.dumps({"member.txt": self.entry}))
        self.assertEqual(bundles.verify(manifest), 1)

    def test_missing_empty_and_duplicate_members_fail(self):
        with self.assertRaises(FileNotFoundError):
            bundles.verify(self.manifest)
        for files in ([], [{"path": "member.txt", **self.entry}] * 2, {"gone.txt": self.entry}):
            self.save({"files": files})
            with self.assertRaises(ValueError):
                bundles.verify(self.manifest)
        self.manifest.write_text('{"files":{},"files":{}}')
        with self.assertRaisesRegex(ValueError, "Duplicate manifest key"):
            bundles.verify(self.manifest)

    def test_same_length_tampering_and_truncation_fail(self):
        self.save({"files": {"member.txt": self.entry}})
        for contents, message in ((b"date", "SHA-256"), (b"dat", "Byte count")):
            self.member.write_bytes(contents)
            with self.assertRaisesRegex(ValueError, message):
                bundles.verify(self.manifest)

    def test_links_and_paths_outside_the_bundle_are_rejected(self):
        (self.root / "link").symlink_to(self.member)
        for name in ("link", "../sibling", str(self.member.resolve()), "./member.txt"):
            self.save({"files": {name: self.entry}})
            with self.subTest(name=name), self.assertRaises(ValueError):
                bundles.verify(self.manifest)


if __name__ == "__main__":
    unittest.main()
