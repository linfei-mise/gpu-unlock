# SPDX-License-Identifier: GPL-2.0-only
"""CPU-only acceptance guards: reject mislabeled boots and altered probes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

PROFILE=Path(__file__).resolve().parents[1]/"diagnostics/alienware_m18_r2"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

collector=load("collector",PROFILE/"collect.py")
runner=load("tensor_runner",PROFILE/"vendor/tensor_modes_driver_runner.py")

class ReachedCUDA(Exception):pass

class ProfileTests(unittest.TestCase):
    def test_boot_must_match_label(self):
        collector.check_boot("quiet gpu_unlock.targets=GPU", "DynamicPowerManagement: 0", "unlocked")
        collector.check_boot("quiet", "DynamicPowerManagement: 0", "stock-rtd3off")
        for command,state in (("quiet","unlocked"),("gpu_unlock.targets=GPU","stock-rtd3off")):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                collector.check_boot(command,"DynamicPowerManagement: 0",state)

    def test_runtime_power_policy_must_match(self):
        for params in ("", "DynamicPowerManagement: 3", "DynamicPowerManagement: 00"):
            with self.subTest(params=params),self.assertRaises(RuntimeError):
                collector.check_boot("quiet",params,"stock-rtd3off")

    def test_getter_must_confirm_real_state(self):
        for state,expected in (("unlocked",0),("stock-rtd3off",1)):
            collector.check_rates({"rates":{"fmla32":expected}},state)
            for wrong in (1-expected,2,255):
                with self.subTest(state=state,wrong=wrong),self.assertRaises(RuntimeError):
                    collector.check_rates({"rates":{"fmla32":wrong}},state)

    def test_vendor_files_are_unmodified(self):
        manifest=json.loads((PROFILE/"vendor/PROVENANCE.json").read_text())
        for name,record in manifest["files"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((PROFILE/"vendor"/name).read_bytes()).hexdigest(),record["sha256"])

    def run_runner(self,actual_uuid="GPU-test",power="[N/A]",sha=None):
        with tempfile.NamedTemporaryFile() as cubin:
            argv=["runner",cubin.name,"--sha256",sha or hashlib.sha256().hexdigest(),
                  "--expect-uuid","GPU-test","--expect-power-limit","N/A"]
            smi=subprocess.CompletedProcess([],0,f"{actual_uuid}, NVIDIA GeForce, PCI-REDACTED, 580.178.04, {power}\n")
            with mock.patch.object(sys,"argv",argv),mock.patch.object(runner.subprocess,"run",return_value=smi),mock.patch.object(runner.C,"CDLL",side_effect=ReachedCUDA):
                runner.main()

    def test_identity_cap_and_hash_guards(self):
        with self.assertRaises(ReachedCUDA):self.run_runner()
        for kwargs in ({"actual_uuid":"GPU-other"},{"power":"125.0"},{"sha":"0"*64}):
            with self.subTest(kwargs=kwargs),self.assertRaises(AssertionError):self.run_runner(**kwargs)

if __name__=="__main__":unittest.main()
