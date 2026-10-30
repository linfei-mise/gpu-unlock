#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Build persistent matched probes without installing anything."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cuda-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cuda=args.cuda_root.resolve(); source=HERE/"vendor"
    provenance=json.loads((source/"PROVENANCE.json").read_text())
    for name,record in provenance["files"].items():
        if sha(source/name)!=record["sha256"]:raise RuntimeError("Vendor source hash mismatch: "+name)
    compiler=cuda/"bin/nvcc"
    version=subprocess.check_output([str(compiler),"--version"],text=True).strip()
    for name in ("mma_bench","validate_gemm","gemm_bench"):
        cmd=[str(compiler),"-O3","-arch=sm_89","-std=c++17","-o",str(args.output/name),str(source/(name+".cu"))]
        if name!="mma_bench":cmd += ["-lcublas","-Xlinker","-rpath","-Xlinker",str(cuda/"lib")]
        subprocess.run(cmd,check=True)
    ptx=args.output/"tensor_modes_issue_sm89.ptx"; cubin=args.output/"tensor_modes_issue_sm89.cubin"
    subprocess.run([str(compiler),"-O3","-arch=sm_89","-std=c++17","-ptx","-o",str(ptx),str(source/"tensor_modes_issue.cu")],check=True)
    subprocess.run([str(cuda/"bin/ptxas"),"-O3","-arch=sm_89","-o",str(cubin),str(ptx)],check=True)
    manifest={"nvcc":version,"vendor":provenance,
              "binaries":{name:sha(args.output/name) for name in ("mma_bench","validate_gemm","gemm_bench",cubin.name)}}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Built four hash-pinned probes. Reuse this exact directory across boots.")

if __name__=="__main__":main()
