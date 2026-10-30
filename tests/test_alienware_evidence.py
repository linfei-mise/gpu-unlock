# SPDX-License-Identifier: GPL-2.0-only
"""Validate public evidence integrity and catch accidental private/binary files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/hardware/alienware-m18-r2/evidence"

class EvidenceTests(unittest.TestCase):
    def test_matched_summary_reproduces_from_saved_runs(self):
        spec=importlib.util.spec_from_file_location("summary", ROOT/"diagnostics/summarize_alienware.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        matched=EVIDENCE/"matched-v2"
        summary=module.summarize(matched)
        self.assertEqual(summary,json.loads((matched/"comparison.json").read_text()))
        self.assertTrue(summary["all_compared_outputs_identical"])
        status=json.loads((matched/"unlocked/status.json").read_text())
        for name,digest in status["source_sha256"].items():
            path=ROOT/"diagnostics/alienware_m18_r2"/name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)

    def test_manifest_covers_exactly_the_public_files(self):
        manifest=json.loads((EVIDENCE/"SHA256SUMS.json").read_text())
        actual={str(p.relative_to(EVIDENCE)) for p in EVIDENCE.rglob("*") if p.is_file() and p.name!="SHA256SUMS.json"}
        self.assertEqual(set(manifest),actual)
        for name,digest in manifest.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest(),digest)

    def test_no_private_identifiers_or_binary_payloads(self):
        sensitive=re.compile(r"/home/[^/\s]+|GPU-[0-9a-fA-F-]{16,}|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b|\b[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{30,}")
        for path in EVIDENCE.rglob("*"):
            if not path.is_file():continue
            with self.subTest(file=str(path.relative_to(EVIDENCE))):
                self.assertIn(path.suffix,(".json",".jsonl",".txt",".csv",".log"))
                self.assertLess(path.stat().st_size,1024*1024)
                self.assertIsNone(sensitive.search(path.read_text()))

    def test_completed_statuses_and_finite_results(self):
        def reject_constant(value):raise ValueError("Non-finite JSON value: "+value)
        for path in EVIDENCE.rglob("*.json"):
            with self.subTest(file=str(path.relative_to(EVIDENCE))):
                obj=json.loads(path.read_text(),parse_constant=reject_constant)
                if isinstance(obj,dict) and "complete" in obj:self.assertTrue(obj["complete"])

if __name__=="__main__":unittest.main()
