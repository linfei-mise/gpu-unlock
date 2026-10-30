# Alienware m18 R2 validation tools

These are read-only hardware validation programs for the documented RTX 4090
Laptop profile. They do not install, unload or reset drivers, write device
registers, alter clocks or power limits, flash firmware, suspend, or reboot.

The profile and evidence are in [the hardware report](../../docs/hardware/alienware-m18-r2/README.md).
The installed unlock implementation is **unmodified upstream 2.0.0**.

## Build once, collect across boots

Use a local CUDA 13.0 toolkit with SM89 support and an existing PyTorch CUDA
environment. The validated environment used PyTorch 2.14.1+cu130. Installation
of a different Python environment is not part of these tools.

```sh
python3 diagnostics/alienware_m18_r2/build.py \
  --cuda-root /path/to/cuda-toolkit \
  --output build/alienware-m18-r2
python3 -m unittest discover -s tests -p test_alienware_profile.py -v
```

Reuse the **same build directory** and Python environment in both boot states.
Run with AC power connected, no other CUDA jobs, and comparable desktop activity.
Do not change clock/power policy between runs. These tools intentionally require
kernel 6.8.0-147-generic, driver 580.178.04, and RTD3 disabled in both states.

```sh
python3 diagnostics/alienware_m18_r2/collect.py \
  --state unlocked --bench-dir build/alienware-m18-r2 \
  --torch-python /path/to/pytorch-environment/bin/python \
  --output build/validation/unlocked

# After a separately arranged stock boot with RTD3 still disabled:
python3 diagnostics/alienware_m18_r2/collect.py \
  --state stock-rtd3off --bench-dir build/alienware-m18-r2 \
  --torch-python /path/to/pytorch-environment/bin/python \
  --output build/validation/stock-rtd3off

# After returning to the default unlocked entry:
python3 diagnostics/alienware_m18_r2/collect.py \
  --state unlocked --bench-dir build/alienware-m18-r2 \
  --torch-python /path/to/pytorch-environment/bin/python \
  --output build/validation/unlocked-repeat
```

The collector checks the boot label, actual RM rate selector, GPU identity,
driver, kernel, RTD3 policy and binary hashes. It refuses to overwrite evidence.
A stock boot reporting full rate is **not** a valid stock control: stop and
inspect whether a complete power-off is needed. Do not run the firmware loader
manually against an NVIDIA-owned GPU.

Compare the saved A/B/A runs without GPU access:

```sh
python3 diagnostics/summarize_alienware.py build/validation
# Or reproduce the summary from the included sanitized evidence:
python3 diagnostics/summarize_alienware.py docs/hardware/alienware-m18-r2/evidence/matched-v2
python3 -m unittest discover -s tests -p 'test_alienware_*.py' -v
```

The summary checks matching environments, manifests, rate selectors and
numerical results before computing ratios. It reports the two unlocked
measurements separately and preserves slower/unchanged instruction modes.
The recorded collection sources were frozen across all three runs; the
CPU-only summarizer was added afterward, outside the collector directory.

`validate_torch.py` tests 128 GEMMs against CPU FP64 references, causal
FlashAttention outputs and input gradients, and synthetic MLP forward/backward
training. TF32 is an allowed precision policy; cuBLAS may choose a different
kernel for small shapes. The separate cubin probe explicitly executes TF32 MMA.
Tolerances are in the source and fixed before comparison. Exact dyadic inputs
must match exactly; general floating-point calculations use error bounds.
Three negative controls verify rejection of NaN, infinity and corrupted data.

The FlashAttention timing is a small operator measurement. The synthetic MLP
timing includes checks and synchronization and is **not** an end-to-end training
throughput benchmark. No downloaded model, private data or saved user workload
is used.

## Bounded continuous load

```sh
/path/to/pytorch-environment/bin/python diagnostics/alienware_m18_r2/validate_torch.py \
  --state unlocked --suite soak --soak-seconds 600 \
  --output build/validation/continuous-gemm.json
```

This alternates FP16/BF16 4096-square GEMMs, 512 operations per batch, and checks
every output after each batch against an exact separable reference. Temperature,
power and utilization are sampled between batches. It stops on a mismatch,
AC removal, or a conservative sampled 83 C limit. This is a bounded validation
run, not a guarantee of electrical safety or long-term stability.

## Source provenance

`vendor/PROVENANCE.json` records the original repository, exact commit, paths
and SHA-256 for each unchanged imported file from
[JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada/tree/d28c485418c3d9d034cd4bac19200bc33dd3453b).
The imported sources and these additions are GPL-2.0-only; the parent
[license text](../../licenses/GPL-2.0.txt) is retained. No NVIDIA firmware,
compiled CUDA binaries, installers or DKMS signing material is included in this
profile. Firmware provenance remains governed by the upstream project's notices.

The collector avoids recording UUIDs, PCI locations and home directories in its
normal outputs. Review all locally generated failure logs before publishing.

References:
- [Ada testing guide](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/docs/testing.md)
- [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)
