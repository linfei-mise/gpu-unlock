#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Bounded, synthetic GPU checks. Never changes clocks, power caps or drivers."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def errors(actual, reference, bound):
    actual = actual.detach().double().cpu()
    reference = reference.detach().double().cpu()
    diff = (actual - reference).abs()
    bad = (~torch.isfinite(actual)) | (~torch.isfinite(reference)) | (diff > bound)
    return {"failures": int(bad.sum()), "cells": actual.numel(),
            "max_abs_error": float(diff.max()),
            "relative_l2_error": float(torch.linalg.vector_norm(actual-reference) /
                                       torch.linalg.vector_norm(reference).clamp_min(1e-30))}


def guard():
    ac = Path("/sys/class/power_supply/AC/online")
    if ac.exists() and ac.read_text().strip() != "1":
        raise RuntimeError("AC power disconnected")
    values = subprocess.check_output([
        "nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.sm",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(", ")
    sample = dict(zip(("temperature_c", "power_w", "utilization_pct", "sm_mhz"), map(float, values)))
    if sample["temperature_c"] >= 83:
        raise RuntimeError("Conservative test stop at 83 C")
    return sample


def numerical(emit):
    # Negative controls: NaN, infinity and a corrupted value must be rejected.
    for bad in (float("nan"), float("inf"), 2.0):
        if errors(torch.tensor([bad]), torch.ones(1), 1e-6)["failures"] != 1:
            raise RuntimeError("Numerical checker failed its negative control")
    emit({"kind": "checker_negative_controls", "passed": 3})
    modes = [("fp16_fp32", torch.float16), ("bf16_fp32", torch.bfloat16),
             ("fp32_ieee", torch.float32), ("tf32_allowed", torch.float32)]
    for mode, dtype in modes:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if mode == "tf32_allowed" else "ieee"
        for m, k, n in ((32, 64, 48), (127, 95, 65), (256, 256, 256), (257, 513, 129)):
            for seed in (7, 2026):
                for pattern in ("dyadic", "scaled_normal"):
                    for layout in ("contiguous", "transposed"):
                        g = torch.Generator().manual_seed(seed)
                        def make(rows, cols):
                            shape = (cols, rows) if layout == "transposed" else (rows, cols)
                            x = (torch.randint(-8, 9, shape, generator=g).float() / 8 if pattern == "dyadic"
                                 else torch.randn(shape, generator=g) * 0.25)
                            return (x.T if layout == "transposed" else x).to(dtype)
                        a, b = make(m, k), make(k, n)
                        if pattern == "scaled_normal":
                            scale = torch.pow(2.0, torch.arange(k) % 9 - 4).to(dtype)
                            a = (a * scale).to(dtype)
                            b = (b / scale[:, None]).to(dtype)
                        ref = a.double() @ b.double()
                        magnitude = a.double().abs() @ b.double().abs()
                        gamma = k * 2**-24 / (1 - k * 2**-24)
                        bound = (4 * gamma + (2**-9 if mode == "tf32_allowed" else 0)) * magnitude + 2e-6
                        ga, gb = a.cuda(), b.cuda()
                        result = (torch.mm(ga, gb, out_dtype=torch.float32) if dtype != torch.float32
                                  else torch.mm(ga, gb))
                        record = {"kind": "gemm_numerical", "mode": mode, "shape_mkn": [m,k,n],
                                  "seed": seed, "pattern": pattern, "layout": layout,
                                  "input_strides": [list(ga.stride()), list(gb.stride())],
                                  "sha256": digest(result), **errors(result, ref, bound)}
                        if pattern == "dyadic" and record["max_abs_error"] != 0:
                            raise RuntimeError("Exactly representable dyadic GEMM differed")
                        emit(record)
                        if record["failures"]:
                            raise RuntimeError("GEMM numerical mismatch")
            guard()


def workloads(emit):
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    for dtype in (torch.float16, torch.bfloat16):
        for length in (127, 256):
            g = torch.Generator().manual_seed(41 + length)
            inputs = [(torch.randn((2,4,length,64), generator=g) * 0.5).to(dtype) for _ in range(3)]
            gpu = [x.cuda().requires_grad_() for x in inputs]
            cpu = [x.double().requires_grad_() for x in inputs]
            dq = torch.randn((2,4,length,64), generator=g).to(dtype)
            logits = cpu[0] @ cpu[1].transpose(-2,-1) / 8
            mask = torch.ones((length,length),dtype=torch.bool).tril()
            ref = logits.masked_fill(~mask,float("-inf")).softmax(-1) @ cpu[2]
            (ref * dq.double()).sum().backward()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                result = F.scaled_dot_product_attention(*gpu, dropout_p=0.0, is_causal=True)
                (result * dq.cuda()).sum().backward()
                timings = []
                with torch.no_grad():
                    for _ in range(5):
                        F.scaled_dot_product_attention(*gpu, dropout_p=0.0, is_causal=True)
                    for _ in range(5):
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):
                            F.scaled_dot_product_attention(*gpu, dropout_p=0.0, is_causal=True)
                        stop.record(); stop.synchronize()
                        timings.append(start.elapsed_time(stop) / 20)
            atol, rtol = ((5e-4, 5e-3) if dtype == torch.float16 else (4e-3, 3e-2))
            checks = [errors(result, ref, atol + rtol*ref.detach().abs())]
            gatol, grtol = ((2e-3,2e-2) if dtype == torch.float16 else (1e-2,8e-2))
            checks += [errors(a.grad, b.grad, gatol + grtol*b.grad.abs()) for a,b in zip(gpu,cpu)]
            emit({"kind":"flash_attention", "dtype":str(dtype), "shape":[2,4,length,64],
                  "backend":"FLASH_ATTENTION forced", "causal":True, "checks":checks,
                  "forward_atol":atol,"forward_rtol":rtol,"gradient_atol":gatol,"gradient_rtol":grtol,
                  "forward_times_ms":timings,"median_forward_ms":statistics.median(timings),
                  "output_sha256":digest(result)})
            if any(row["failures"] for row in checks):
                raise RuntimeError("Flash attention output/gradient mismatch")
        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(512,1024), torch.nn.GELU(), torch.nn.Linear(1024,128)).to(dtype)
        reference = copy.deepcopy(model).double()
        model = model.cuda()
        x = torch.randn(64,512).to(dtype)
        target = torch.zeros(64,128,dtype=torch.float64)
        prediction = reference(x.double())
        (prediction-target).square().mean().backward()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses, times = [], []
        gradient_checks = []
        for step in range(50):
            optimizer.zero_grad(set_to_none=True)
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = model(x.cuda())
            loss = prediction.float().square().mean()
            loss.backward()
            if step == 0:
                for actual, ref in zip(model.parameters(),reference.parameters()):
                    atol, rtol = ((2e-5,0.03) if dtype == torch.float16 else (2e-4,0.1))
                    gradient_checks.append(errors(actual.grad, ref.grad, atol+rtol*ref.grad.abs()))
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise RuntimeError("Non-finite training gradient")
            optimizer.step()
            stop.record();stop.synchronize()
            times.append(start.elapsed_time(stop));losses.append(loss.item())
        emit({"kind":"synthetic_mlp_training", "dtype":str(dtype), "steps":50,
              "losses":losses,"step_times_ms":times,"gradient_checks":gradient_checks,
              "final_weights_sha256":[digest(p) for p in model.parameters()],
              "timing_scope":"includes validation synchronization; not application throughput"})
        if not all(math.isfinite(x) for x in losses) or losses[-1] >= losses[0]:
            raise RuntimeError("Training loss did not decrease")
        if any(row["failures"] for row in gradient_checks):
            raise RuntimeError("Initial MLP gradient differed from CPU FP64 reference")
        guard()


def soak(seconds, emit):
    n = 4096
    r = (torch.arange(n,device="cuda") % 13 - 6).float()/8
    c = (torch.arange(n,device="cuda") % 11 - 5).float()/8
    a, b = r[:,None].expand(n,n).contiguous(), c[None,:].expand(n,n).contiguous()
    pairs = [(a.to(t),b.to(t)) for t in (torch.float16,torch.bfloat16)]
    expected = n*r[:,None]*c[None,:]
    output = torch.empty_like(expected)
    start = time.monotonic(); batch = 0; checked = 0
    while time.monotonic()-start < seconds:
        left,right = pairs[batch % 2]
        first,last = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record()
        for _ in range(512):
            torch.mm(left,right,out_dtype=torch.float32,out=output)
        last.record();last.synchronize()
        bad = int((output != expected).sum().item())
        checked += output.numel()
        emit({"kind":"continuous_gemm_batch", "batch":batch,"dtype":str(left.dtype),
              "elapsed_s":time.monotonic()-start,"iterations":512,"gpu_ms":first.elapsed_time(last),
              "cells_checked":output.numel(),"failures":bad,**guard()})
        if bad:raise RuntimeError("Sustained GEMM mismatch")
        batch += 1
    emit({"kind":"continuous_gemm_summary","elapsed_s":time.monotonic()-start,
          "batches":batch,"cells_checked":checked})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--state",choices=("unlocked","stock-rtd3off"),required=True)
    parser.add_argument("--suite",choices=("checks","soak"),default="checks")
    parser.add_argument("--soak-seconds",type=int,default=600)
    args=parser.parse_args()
    if args.output.exists():parser.error("Refusing to overwrite results")
    if not 1 <= args.soak_seconds <= 1800:parser.error("Soak must be bounded to 1..1800 seconds")
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability()!=(8,9):
        raise RuntimeError("Expected one visible SM89 GPU")
    cmdline=Path("/proc/cmdline").read_text()
    if ("gpu_unlock.targets=" in cmdline)!=(args.state=="unlocked"):
        raise RuntimeError("Boot state does not match the requested label")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.fp32_precision="ieee"
    report={"complete":False,"state":args.state,"suite":args.suite,"torch":torch.__version__,
            "cuda":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(),"seed_policy":"fixed CPU generators",
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"records":[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def emit(record):
        report["records"].append(record)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if record["kind"]!="gemm_numerical":print(record["kind"],len(report["records"]),flush=True)
    try:
        guard()
        if args.suite=="checks":numerical(emit);workloads(emit)
        else:soak(args.soak_seconds,emit)
        report["complete"]=True
    except BaseException as exc:
        report["error"]=str(exc)
        raise
    finally:args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("PASS",flush=True)


if __name__=="__main__":main()
