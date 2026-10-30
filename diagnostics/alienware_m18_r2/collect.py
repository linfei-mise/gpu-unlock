#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Collect a matched run, refusing incorrect boot/rate/power/binary states."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent

def capture(cmd,env=None,timeout=120):
    result=subprocess.run(list(map(str,cmd)),text=True,capture_output=True,env=env,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr or result.stdout or "Command failed")
    return result.stdout.strip()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def check_boot(cmdline,params,state):
    if ("gpu_unlock.targets=" in cmdline)!=(state=="unlocked"):
        raise RuntimeError("Wrong boot state")
    if not re.search(r"^DynamicPowerManagement: 0$",params,re.M):
        raise RuntimeError("RTD3 must be disabled in BOTH states")

def check_rates(value,state):
    expected=0 if state=="unlocked" else 1
    if value["rates"]["fmla32"]!=expected:
        raise RuntimeError("Rate selector does not match the requested state")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--state",choices=("unlocked","stock-rtd3off"),required=True)
    parser.add_argument("--bench-dir",type=Path,required=True)
    parser.add_argument("--torch-python",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("Refusing to overwrite existing evidence")
    cmdline=Path("/proc/cmdline").read_text()
    params=Path("/proc/driver/nvidia/params").read_text()
    check_boot(cmdline,params,args.state)
    if os.uname().release!="6.8.0-147-generic":raise RuntimeError("Kernel differs from this profile")
    info=capture(["nvidia-smi","--query-gpu=uuid,name,pci.device_id,driver_version,power.limit","--format=csv,noheader,nounits"])
    fields=[value.strip() for value in info.split(",")]
    if len(fields)!=5 or fields[3]!="580.178.04" or "4090 Laptop" not in fields[1]:
        raise RuntimeError("Unexpected GPU or driver")
    if fields[4].strip("[]").upper()!="N/A":raise RuntimeError("Power-limit reporting changed")
    uuid=fields[0]
    if capture(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"]):
        raise RuntimeError("Other CUDA workloads are active; stop this collection")
    manifest=json.loads((args.bench_dir/"manifest.json").read_text())
    for name,value in manifest["binaries"].items():
        if sha(args.bench_dir/name)!=value:raise RuntimeError("Binary changed: "+name)
    spec=importlib.util.spec_from_file_location("sm_rates",HERE/"vendor/read_sm_rates.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    def rates():
        value=module.read_rates(uuid)
        value["gpu"]=fields[1]
        check_rates(value,args.state)
        return value
    initial=rates()
    env={**os.environ,"CUDA_VISIBLE_DEVICES":uuid,"CUDA_DEVICE_ORDER":"PCI_BUS_ID"}
    args.output.mkdir(parents=True)
    report={"complete":False,"state":args.state,"kernel":os.uname().release,
            "driver":fields[3],"gpu":fields[1],"power_limit":fields[4],"rtd3":0,
            "clock_policy":"unchanged dynamic clocks; desktop active", "manifest":manifest,
            "source_sha256":{p.name:sha(p) for p in HERE.glob("*.py")},"rates_before":initial}
    sampler=None
    try:
        with (args.output/"telemetry.csv").open("w") as stream:
            sampler=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,power.draw,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu","--format=csv","--loop-ms=100"],stdout=stream,stderr=subprocess.DEVNULL)
            for name,options,filename in (("validate_gemm",[],"numerical.txt"),("mma_bench",["20000","5"],"mma.txt"),("gemm_bench",[],"gemm.jsonl")):
                result=capture([args.bench_dir/name,*options],env=env)
                (args.output/filename).write_text(result+"\n")
                if name=="validate_gemm" and result.count("failures=0")!=6:raise RuntimeError("Seeded check failed")
                if name=="gemm_bench":
                    rows=[json.loads(line) for line in result.splitlines()]
                    if len(rows)!=6 or any(r["failures"] for r in rows):raise RuntimeError("GEMM check failed")
            cubin=args.bench_dir/"tensor_modes_issue_sm89.cubin"
            result=json.loads(capture([sys.executable,HERE/"vendor/tensor_modes_driver_runner.py",cubin,"--sha256",manifest["binaries"][cubin.name],"--expect-uuid",uuid,"--expect-power-limit","N/A"],env=env))
            result["gpu"]=fields[1]
            (args.output/"tensor-modes.json").write_text(json.dumps(result,indent=2)+"\n")
            capture([args.torch_python,HERE/"validate_torch.py","--state",args.state,"--output",args.output/"torch-checks.json"],env=env,timeout=180)
            report["rates_after"]=rates()
            errors=subprocess.run(["journalctl","-k","-b","--no-pager","-g","NVRM: Xid|API mismatch"],text=True,capture_output=True,env={**os.environ,"LANG":"C"})
            if errors.returncode not in (0,1) or "-- No entries --" not in errors.stdout:
                raise RuntimeError("GPU kernel error found; inspect journal locally")
            report["kernel_gpu_errors"]=0
            report["complete"]=True
    except BaseException as exc:
        # Keep a concise error without publishing the GPU UUID or private paths.
        message=str(exc).replace(uuid,"GPU-REDACTED").replace(str(Path.home()),"HOME")
        report["error"]=message
        raise RuntimeError(message) from None
    finally:
        if sampler is not None:
            sampler.terminate();sampler.wait(timeout=10)
        (args.output/"status.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Matched collection passed:",args.state)

if __name__=="__main__":main()
