#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
"""Summarize the three saved runs using only the Python standard library."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

RUNS = ("unlocked", "stock-rtd3off", "unlocked-repeat")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(path.read_text(), parse_constant=reject)


def check_failures(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("failures", "mismatches"):
                require(child == 0, "A saved numerical check failed")
            check_failures(child)
    elif isinstance(value, list):
        for child in value:
            check_failures(child)


def comparison(values):
    stock = values[1]
    require(stock > 0, "Invalid stock performance")
    gains = [(values[i] / stock - 1) * 100 for i in (0, 2)]
    return {"by_run": dict(zip(RUNS, values)),
            "unlocked_change_pct_range": [min(gains), max(gains)]}


def summarize(root):
    statuses = [read_json(root / run / "status.json") for run in RUNS]
    for run, status in zip(RUNS, statuses):
        state = "stock-rtd3off" if run == "stock-rtd3off" else "unlocked"
        require(status["complete"] and status["state"] == state, "Incomplete or mislabeled run")
        require(status["rtd3"] == 0 and status["kernel_gpu_errors"] == 0, "Policy or kernel error")
        expected = int(state == "stock-rtd3off")
        for phase in ("rates_before", "rates_after"):
            for selector in ("fmla32", "imla2", "imla4"):
                require(status[phase]["rates"][selector] == expected, "Wrong RM rate selector")
    matched = ("manifest", "source_sha256", "kernel", "driver", "gpu", "power_limit", "rtd3", "clock_policy")
    for key in matched:
        require(all(s[key] == statuses[0][key] for s in statuses), "Run mismatch: " + key)

    torch_runs = [read_json(root / run / "torch-checks.json") for run in RUNS]
    expected_counts = {"checker_negative_controls": 1, "gemm_numerical": 128,
                       "flash_attention": 4, "synthetic_mlp_training": 2}
    for status, report in zip(statuses, torch_runs):
        require(report["complete"] and report["state"] == status["state"], "Incomplete PyTorch run")
        require(report["script_sha256"] == status["source_sha256"]["validate_torch.py"], "PyTorch source mismatch")
        require(Counter(r["kind"] for r in report["records"]) == expected_counts, "Missing PyTorch cases")
        check_failures(report)
    for key in ("torch", "cuda", "gpu_name", "seed_policy", "script_sha256", "suite"):
        require(all(t[key] == torch_runs[0][key] for t in torch_runs), "PyTorch environment mismatch")
    identity = []
    for records in zip(*(t["records"] for t in torch_runs)):
        kind = records[0]["kind"]
        require(all(r["kind"] == kind for r in records), "Record order changed")
        if kind == "gemm_numerical":
            case_keys, outputs = ("mode", "shape_mkn", "seed", "pattern", "layout", "input_strides"), ("sha256",)
        elif kind == "flash_attention":
            case_keys, outputs = ("dtype", "shape", "backend", "causal"), ("output_sha256",)
        elif kind == "synthetic_mlp_training":
            case_keys, outputs = ("dtype", "steps"), ("final_weights_sha256", "losses")
        else:
            require(all(r["passed"] == 3 for r in records), "Negative controls failed")
            continue
        require(all(r[k] == records[0][k] for r in records for k in case_keys), "Different numerical cases")
        identity.append({"kind": kind, **{k: records[0][k] for k in case_keys},
                         "identical_fields": {k: all(r[k] == records[0][k] for r in records) for k in outputs}})

    mma_runs, gemm_runs, cubin_runs, telemetry = [], [], [], {}
    for run in RUNS:
        path = root / run
        mma = {}
        for line in (path / "mma.txt").read_text().splitlines()[2:]:
            mode, best, trials = line.split()
            require(len(trials.split("/")) == 5, "Missing MMA repetitions")
            mma[mode] = float(best)
        require(len(mma) == 7, "Missing MMA modes")
        mma_runs.append(mma)
        gemms = [json.loads(line) for line in (path / "gemm.jsonl").read_text().splitlines()]
        require(len(gemms) == 6, "Missing GEMMs")
        for row in gemms:
            require(row["failures"] == 0 and row["cells_checked"] == row["n"] ** 2, "GEMM failure")
        gemm_runs.append({(row["mode"], row["n"]): row["median_TF_s"] for row in gemms})
        cubin = read_json(path / "tensor-modes.json")
        check_failures(cubin)
        require(set(cubin["modes"]) == {"f16_f32", "tf32", "u4_s32"}, "Missing cubin modes")
        require(cubin["cubin_sha256"] == statuses[0]["manifest"]["binaries"]["tensor_modes_issue_sm89.cubin"], "Cubin mismatch")
        cubin_runs.append(cubin)
        require((path / "numerical.txt").read_text().count("failures=0") == 6, "Seeded GEMM failure")
        with (path / "telemetry.csv").open() as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        telemetry[run] = {"samples": len(rows)}
        for field in rows[0]:
            if field == "timestamp":
                continue
            values = [float(row[field].split()[0]) for row in rows]
            telemetry[run][field] = {"min": min(values), "mean": statistics.mean(values), "max": max(values)}
    for key in ("iters", "reps", "blocks", "threads", "warps", "chains"):
        require(all(c[key] == cubin_runs[0][key] for c in cubin_runs), "Cubin launch mismatch")
    require(all(set(m) == set(mma_runs[0]) for m in mma_runs), "MMA mode mismatch")
    require(all(set(g) == set(gemm_runs[0]) for g in gemm_runs), "GEMM shape mismatch")
    return {"complete": True, "run_order": list(RUNS), "matching_fields": list(matched),
            "rate_selectors_fmla32_imla2_imla4": [0, 1, 0],
            "mma_best_Top_s": {mode: comparison([m[mode] for m in mma_runs]) for mode in mma_runs[0]},
            "gemm_median_TF_s": [{"mode": mode, "n": n, **comparison([g[(mode, n)] for g in gemm_runs])}
                                 for mode, n in gemm_runs[0]],
            "cubin_best_Top_s": {mode: comparison([c["modes"][mode]["best_Top_per_s"] for c in cubin_runs])
                                 for mode in cubin_runs[0]["modes"]},
            "numerical_identity": identity, "all_compared_outputs_identical":
                all(all(row["identical_fields"].values()) for row in identity),
            "telemetry": telemetry,
            "limits": ["One stock boot, two unlocked boots; five within-run timing repetitions are not five independent boots.",
                       "Dynamic clocks and active desktop; initial temperatures differ; no clock-normalized performance or energy-efficiency claim.",
                       "Hash identity applies only to recorded outputs/weights; gradients passed tolerance checks, with no gradient hashes recorded.",
                       "Synthetic arithmetic and operator tests do not establish real-model speedups."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path, help="Directory containing the three matched-v2 runs")
    args = parser.parse_args()
    print(json.dumps(summarize(args.evidence), indent=2, allow_nan=False))
