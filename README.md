# gpu-unlock: RTX 4090 Laptop Validation

This repository is a validation fork of [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock).
It documents one **Alienware m18 R2 with an NVIDIA GeForce RTX 4090 Laptop GPU**,
including controlled boot-state comparisons, numerical checks, bounded load
testing, and reproducible evidence.

The tested unlock implementation is upstream **2.0.0**, commit
[`7eb51bccae0a0b7e599e084d971170e9a1e82d0c`](https://github.com/Zagitta/gpu-unlock/commit/7eb51bccae0a0b7e599e084d971170e9a1e82d0c).
This fork adds diagnostics and documentation. The upstream kernel, firmware
sources, and packaging implementation are unchanged. No machine-specific
NVIDIA driver patch or VBIOS flash was required.

[Validation report](docs/hardware/alienware-m18-r2/README.md) ·
[Evidence index](docs/hardware/alienware-m18-r2/EVIDENCE.md) ·
[Reproduction guide](diagnostics/alienware_m18_r2/README.md) ·
[Upstream documentation](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/README.md)

## Measured results

Testing on **2026-10-30** used the sequence **unlocked A1 → stock B → unlocked A2**.
All three runs used the same binaries, collection scripts, kernel, driver,
PyTorch environment, and disabled runtime D3 power management. Each boot state
was checked against the driver's rate selectors before and after testing.

| Instruction throughput, TFLOPS | Stock B | Unlocked A1 | Unlocked A2 |
|---|---:|---:|---:|
| FP16 with FP32 accumulation | 85.1 | 170.0 | 170.6 |
| BF16 with FP32 accumulation | 84.3 | 169.6 | 170.1 |
| FP8 E4M3 | 166.9 | 336.9 | 337.2 |
| Explicit TF32 instruction probe | 42.13 | 83.61 | 82.52 |

The first three rows report the best of five repetitions; TF32 reports the
best of three using a separate cubin probe. The [full report](docs/hardware/alienware-m18-r2/README.md)
retains every tested mode, including INT4, INT8, and FP16/FP16, which did not
show a corresponding doubling. Matrix-multiplication gains varied by shape,
and the 4096-square BF16 result varied substantially across unlocked runs.

All three runs passed the recorded numerical checks. The hashes of 128 GEMM
outputs, four FlashAttention outputs, and both final MLP weight-hash lists
matched across states; both training loss sequences also matched. Gradients
were checked against CPU references with explicit tolerances.

A separate **600.24-second** continuous-load test completed **928 batches**
without an observed output mismatch. Mean sampled GPU utilization was 98.01%,
and peak sampled temperature was 66 °C. These results describe this system
and these workloads; they do not establish production-model speedups,
universal compatibility, or long-term stability.

## Tested configuration and scope

- Alienware m18 R2; BIOS 1.11.0; RTX 4090 Laptop / AD103M, 16 GB, 76 SMs.
- Ubuntu 24.04.3; Linux 6.8.0-147-generic; NVIDIA proprietary driver 580.178.04.
- CUDA toolkit 13.0.88; PyTorch 2.14.1+cu130; AC power; active desktop and external display.
- Secure Boot disabled; runtime D3 disabled; automatic unlock reapplication on resume disabled.
- Default-entry warm boot, arithmetic checks, synthetic training, CUDA sanitizer checks,
  and bounded sustained load validated within the [documented test scope](docs/hardware/alienware-m18-r2/README.md).

Sleep recovery, full power-off/cold boot, actual fallback-entry boot,
representative production workloads, battery operation, longer runs, and
future driver or kernel versions remain unvalidated. Clock and power policies
were unchanged, but dynamic clocks and initial temperatures were not fixed.

## Inspect and reproduce the evidence

Clone the repository to obtain the diagnostics and evidence. Upstream archive
export rules are retained for package-source distribution; archive downloads
may omit the validation directories.

```sh
git clone https://github.com/linfei-mise/gpu-unlock.git
cd gpu-unlock
python3 -m unittest discover -s tests -p 'test_alienware_*.py' -v
python3 diagnostics/summarize_alienware.py docs/hardware/alienware-m18-r2/evidence/matched-v2
```

These commands require only Python's standard library and do not access a GPU.
The tests verify evidence hashes, recorded source provenance, state guards,
and reproducibility of the comparison. The [reproduction guide](diagnostics/alienware_m18_r2/README.md)
provides the separate CUDA build and hardware-collection commands.

The [evidence index](docs/hardware/alienware-m18-r2/EVIDENCE.md) maps each claim
to its supporting files and explains the scope of the checks. SHA-256 manifests
provide repository-integrity checks, not independent hardware attestation.

## How to install

Use the [installation instructions for the tested upstream revision](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/README.md#how-to-install)
and review the [tested configuration and recovery limitations](docs/hardware/alienware-m18-r2/README.md).
Identify the target GPU locally and preserve a usable recovery path before
changing boot configuration. The validation tools do not install the unlock,
change system configuration, or supply a machine-specific root installer.

## Attribution and licensing

The unlock implementation is maintained by [Zagitta and upstream contributors](https://github.com/Zagitta/gpu-unlock).
The unchanged diagnostic sources imported from
[JustAResearcher/gpu-unlock-ada](https://github.com/JustAResearcher/gpu-unlock-ada/tree/d28c485418c3d9d034cd4bac19200bc33dd3453b)
are identified by exact commit, path, and SHA-256 in
[PROVENANCE.json](diagnostics/alienware_m18_r2/vendor/PROVENANCE.json).

The kernel module, packaging, and added diagnostic code are GPL-2.0-only;
see [the license text](licenses/GPL-2.0.txt) and [upstream firmware notices](firmware/NOTICE.md).
NVIDIA firmware remains subject to its own terms. The validation contribution
contains no generated firmware, compiled probes, driver installers, private
signing material, GPU UUIDs, or machine-specific boot identifiers.
