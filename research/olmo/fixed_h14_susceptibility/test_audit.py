import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from audit import ROOT, audit


class EvidenceTests(unittest.TestCase):
    def test_recorded_screens(self):
        r = audit()
        self.assertEqual(r["query_joint_hits"], {"newline":24,"neighbor":0,"substitute":0})
        self.assertEqual(r["query_functional_hits"]["neighbor"], 1)
        self.assertEqual(sum(r["full_dose_low_cost_hits_by_book"].values()),23)
        self.assertEqual(r["monotone_within_0.01"],13)

    def test_manifest_and_numerical_crosscheck_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory) / "data"
            shutil.copytree(ROOT / "data", d)
            p = d / "literary_doses.csv"
            old = p.read_bytes()
            p.write_bytes(old.replace(b"6.0468995242130915e-06",b"1.0468995242130915e-06",2))
            with self.assertRaises(ValueError):
                audit(d)
            f = d / "SHA256SUMS"
            f.write_text(f.read_text().replace(hashlib.sha256(old).hexdigest(), hashlib.sha256(p.read_bytes()).hexdigest()))
            with self.assertRaisesRegex(ValueError,"shared endpoint mismatch"):
                audit(d)


if __name__ == "__main__":
    unittest.main()
