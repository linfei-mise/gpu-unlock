# Alienware m18 R2 / RTX 4090 Laptop validation

This fork adds a reproducible hardware profile and evidence for **unmodified
Zagitta/gpu-unlock 2.0.0**, base commit
`7eb51bccae0a0b7e599e084d971170e9a1e82d0c`. No machine-specific changes to
upstream kernel/firmware sources or the NVIDIA driver were needed; VBIOS was
not flashed. This report is one tested configuration,
not a general compatibility promise for every m18 R2, driver or BIOS revision.

The [evidence index](EVIDENCE.md) maps the reported findings to recorded files,
source provenance, and interpretation limits.

Status on **2026-10-30**: the unlocked → stock → unlocked comparison passed
with RTD3 disabled throughout, as did ordinary default-entry startup,
numerical checks and ten minutes of continuous load. Sleep recovery, cold
power cycles and production workloads remain unvalidated.

## Tested configuration

| Item | Value |
|---|---|
| System | Alienware m18 R2 laptop |
| BIOS | 1.11.0 |
| GPU | RTX 4090 Laptop / AD103M, 16 GB, 76 SMs |
| Device / subsystem IDs | 10de:2757 / 1028:0c9d |
| VBIOS | 95.03.3B.00.2B |
| OS / kernel | Ubuntu 24.04.3 / 6.8.0-147-generic |
| Driver | NVIDIA 580.178.04, proprietary kernel module |
| Secure Boot | Disabled |
| CUDA / PyTorch | CUDA toolkit 13.0.88 / PyTorch 2.14.1+cu130 |
| Power | AC; unchanged default policy, 125 W default / 175 W reported maximum; power.limit N/A |
| Display | Desktop active; NVIDIA also drives an external display |
| Unlock boot parameters | Exact GPU target; gpu_unlock.resume=0; nvidia.NVreg_DynamicPowerManagement=0x00 |
| Firmware input version | 610.43.03, used only to extract pinned firmware; no 610 driver installation |

The earlier loaded-driver/NVML version mismatch was fixed by reboot before
any baseline was taken. Loaded and installed NVIDIA modules both report
580.178.04. No clocks, power cap or BIOS settings were changed for these runs.

## Observed performance and correctness

### Matched RTD3-off comparison (primary evidence)

The chronological order was **A1 unlocked → B stock → A2 unlocked**, with two
normal reboots. All three runs used identical four-binary manifests, collector
source hashes, GPU, kernel, driver, PyTorch environment and RTD3=0. The stock
entry omitted the unlock target; A2 used the ordinary default Ubuntu entry.
RM fmla32/imla2/imla4 were **0 → 1 → 0**, unchanged before/after each run.
The driver loaded from disk matched NVML and the in-memory module at 580.178.04.

| Instruction probe, best of 5 | A1 unlocked | B stock | A2 unlocked | Unlocked change vs B |
|---|---:|---:|---:|---:|
| FP16 / FP32 accumulation, TFLOPS | 170.0 | 85.1 | 170.6 | +99.8% to +100.5% |
| BF16 / FP32 accumulation, TFLOPS | 169.6 | 84.3 | 170.1 | +101.2% to +101.8% |
| FP8 E4M3, TFLOPS | 336.9 | 166.9 | 337.2 | +101.9% to +102.0% |
| FP16 / FP16 accumulation, TFLOPS | 166.7 | 169.5 | 170.3 | -1.7% to +0.5% |
| INT8 / INT32, TOPS | 337.4 | 339.3 | 337.4 | -0.6% in both |
| DP4A, TOPS | 84.4 | 83.9 | 83.0 | -1.1% to +0.6% |
| FP32 FFMA, TFLOPS | 40.4 | 40.2 | 42.0 | +0.5% to +4.5% |

| Explicit cubin, best of 3 | A1 unlocked | B stock | A2 unlocked | Unlocked change vs B |
|---|---:|---:|---:|---:|
| FP16 / FP32, TFLOPS | 167.39 | 84.55 | 168.01 | +98.0% to +98.7% |
| TF32, TFLOPS | 83.61 | 42.13 | 82.52 | +95.9% to +98.5% |
| Unsigned INT4 / INT32, TOPS | 667.19 | 677.85 | 651.98 | -3.8% to -1.6% |

The two instruction tools use different launch configurations; compare each
row across states, not absolute throughput between tools. INT4, INT8 and
FP16/FP16 did not show the doubling seen in FP16/FP32, BF16/FP32, FP8 and TF32.

| cuBLAS GEMM, 5-round median TFLOPS | Size | A1 unlocked | B stock | A2 unlocked | Change vs B |
|---|---:|---:|---:|---:|---:|
| FP16 / FP32 | 1024 | 102.81 | 63.74 | 102.89 | +61.3% to +61.4% |
| BF16 / FP32 | 1024 | 102.80 | 63.84 | 102.60 | +60.7% to +61.0% |
| FP16 / FP32 | 4096 | 122.70 | 78.17 | 113.63 | +45.4% to +57.0% |
| BF16 / FP32 | 4096 | 155.28 | 76.44 | 116.23 | +52.1% to +103.1% |
| FP16 / FP32 | 8192 | 107.12 | 74.92 | 110.91 | +43.0% to +48.1% |
| BF16 / FP32 | 8192 | 107.81 | 72.58 | 108.83 | +48.5% to +49.9% |

Clocks remained dynamic and the desktop/external display remained active.
Starting sampled temperatures were 54/49/47 C for A1/B/A2, with sampled peaks
64/59/57 C. The 4096 BF16 result varies substantially between the two unlocked
runs; its best gain is not a stable performance promise. These are two
unlocked boots and one stock boot, not independent multi-boot statistics or
confidence intervals. Telemetry covers different-duration test phases; its
means do not establish energy efficiency.

All three runs passed six native seeded GEMMs, six whole-output large GEMMs,
three explicit instruction modes, 128 PyTorch GEMMs, four FlashAttention
forward/backward cases and two 50-step MLP training cases. All 128 recorded
GEMM output hashes, four attention output hashes, both final model weight-hash
lists and both 50-step loss sequences were **identical across A1/B/A2**.
Gradients passed their CPU-reference tolerance checks; gradient hashes were
not recorded. No Xid/API-mismatch entries were found during collection.

Raw repetitions and telemetry are retained in [matched-v2](evidence/matched-v2/).
[comparison.json](evidence/matched-v2/comparison.json) is reproducible with
the [CPU-only summarizer](../../../diagnostics/summarize_alienware.py). These synthetic results establish no
whole-model, game, rendering or production training speedup.

### Historical comparison (different RTD3 policies)

The **historical** before/after pair used identical binary hashes, GPU, driver
and kernel. However stock used RTD3=3 (GPU remained active), while unlocked
used RTD3=0. Therefore these numbers are observations, not a completely
matched single-variable experiment. Dynamic clocks and desktop activity are
additional sources of timing variation. The matched-v2 collection keeps RTD3
disabled in both states and refuses mislabeled RM rate selectors.

| Instruction probe, best of 5 | Stock | Unlocked | Change |
|---|---:|---:|---:|
| FP16 / FP32 accumulation, TFLOPS | 84.7 | 170.0 | +100.7% |
| BF16 / FP32 accumulation, TFLOPS | 84.7 | 169.6 | +100.2% |
| FP8 E4M3, TFLOPS | 168.1 | 345.5 | +105.5% |
| FP16 / FP16 accumulation, TFLOPS | 169.3 | 162.8 | -3.8% |
| INT8 / INT32, TOPS | 325.7 | 346.7 | +6.4% |
| DP4A, TOPS | 84.0 | 86.4 | +2.9% |
| FP32 FFMA, TFLOPS | 42.2 | 42.2 | 0.0% |

| cuBLAS GEMM, 5-round median TFLOPS | Size | Stock | Unlocked |
|---|---:|---:|---:|
| FP16 / FP32 | 1024 | 63.74 | 102.55 |
| BF16 / FP32 | 1024 | 63.74 | 102.37 |
| FP16 / FP32 | 4096 | 78.64 | 118.92 |
| BF16 / FP32 | 4096 | 76.82 | 116.37 |
| FP16 / FP32 | 8192 | 74.08 | 107.67 |
| BF16 / FP32 | 8192 | 72.74 | 108.25 |

All six seeded numerical cases and all output elements of the six large GEMMs
passed in both states. RM fmla32/imla2/imla4 changed from 1 to 0. The 600.93 s
historical loop test passed 117 cycles (702 seeded checks and 702 large GEMM
checks), sampled peak 62 C. That loop included CPU validation and idle gaps.
After it, rate selectors remained full and FP16/BF16 issue rates were 168.1
TFLOPS. GPU kernel logs had no Xid/API-mismatch entries.

## Additional validation

| Check | Result | Scope |
|---|---|---|
| Compile with KCFLAGS=-Werror | Pass on installed 6.8.0-142 and 6.8.0-147 headers | Builds only; 142 was not boot-tested |
| Pinned NVIDIA input SHA-256 | Pass | No firmware redistribution |
| CUDA memcheck + leak check | Pass, 0 errors, 0 leaked bytes | Six seeded GEMMs |
| CUDA initcheck | Pass, 0 errors | Six seeded GEMMs |
| CUDA synccheck | Pass, 0 errors | Six seeded GEMMs |
| Explicit FP16, TF32, unsigned INT4 cubin | 270,336 outputs per mode, 0 mismatches | Issue-rate arithmetic; all-one operands |
| Random/edge GEMMs | 128 cases pass CPU FP64 reference checks | FP16→FP32, BF16→FP32, IEEE FP32, TF32 allowed; four shapes, two seeds, two patterns, two layouts |
| Numerical checker negative controls | 3 pass | NaN, infinity and corrupt finite result rejected |
| FlashAttention | 4 cases pass output and dQ/dK/dV checks | Forced Flash backend, FP16/BF16, causal sequence lengths 127/256 |
| Synthetic MLP training | FP16 and BF16, 50 steps each; finite gradients and reduced loss | Initial gradients checked against CPU FP64; not a real model benchmark |
| Continuous GEMM load | Pass, 600.24 s / 928 batches, 0 mismatches | Alternating FP16/BF16; whole output checked each batch |
| Profile guard unit tests | 5 pass locally | Wrong boot, RTD3, rate, UUID, power reporting and cubin hash rejected; vendor hashes checked |
| Evidence checks | 4 pass locally | Public-file hashes/privacy, complete finite JSON, reproducible matched comparison |
| Ordinary default-entry startup | Pass after stock comparison | Four successful unlock passes, clean handoff, RM rates 0 before/after A2 |
| GitHub-hosted CI | See the repository's Hardware profile guards workflow | CPU-only checks; a workflow result is not a hardware test |

The FP16 MLP loss decreased from 0.04016 to 0.02359; BF16 from 0.04017 to
0.03286. These are correctness/smoke-test results. They do not establish a
speedup for a production training or inference workload. Random floating-point
results use explicitly recorded tolerances; exactly representable dyadic
GEMMs must match exactly. The current small-shape FlashAttention timings do
not represent whole-model token throughput.

The additional continuous test checked 15,569,256,448 output cells across 928
batches (repeated workloads, not independent random samples). CUDA event time
inside GEMM batches totaled 580.32 s of 600.24 s wall time, with approximately
98.0% mean sampled utilization, 66 C peak sampled temperature and 128.49 W peak
sampled power. There were no numerical failures. This provides stronger
sustained-load evidence than the earlier loop with CPU-side gaps; it is still
only a bounded ten-minute run.

Each batch executed 512 GEMMs and checked the final output matrix. The
intermediate outputs of the other operations were not checked separately.

## Remaining test matrix

1. **Cold boot and recovery:** the preserved original-initramfs fallback is
   hash-checked but has not been boot-tested. Full power-off/cold startup and
   repeated cold boots remain untested; A/B/A used normal warm reboots.
2. **Sleep and idle recovery:** sleep type is s2idle. Automatic unlock reapply
   remains disabled. Check rate selectors, numerical tests and kernel logs
   after resume before claiming sleep compatibility. Do not enable reapply
   on this older driver merely because another system supports it.
3. **Representative user workload:** a fixed real model with seeded inputs,
   repeated output/quality checks, prefill/decode or training-step metrics,
   same precision, memory policy and binaries in both boot states.
4. **Longer operation and updates:** cold boot repetitions, longer workload
   runs, battery use and future driver/kernel upgrades remain unvalidated.

Further boot and sleep tests close applications or interrupt the desktop.
They must be arranged when work is saved. Do not
unload the active NVIDIA driver or issue a live firmware transaction to avoid
those restarts.

## Reproduce, evidence and recovery

See [the tools and commands](../../../diagnostics/alienware_m18_r2/README.md).
Selected numerical results and telemetry are under [evidence](evidence/).
`evidence/SHA256SUMS.json` hashes the sanitized public files. Redaction removes
GPU UUIDs, PCI locations and home paths while preserving numeric data. Hashes
support integrity checks of the published files, not independent attestation
of the machine or measurements. Logs
from unrelated user applications, private installers and signing material are
excluded. Legacy evidence is deliberately labeled separately from matched-v2.

The local deployment retains a separate **ORIGINAL pre-unlock fallback** menu
entry and a hash-verified pre-install initramfs. This is a local rollback
arrangement, not a menu entry automatically installed by upstream. Ordinary
Ubuntu startup with the tested arguments passed in A2; the sanitized
[boot receipt](evidence/matched-v2/default-boot.json) records the checks.
New kernels are built through DKMS but
must be checked before a compatibility claim is extended.

For installation use the [upstream instructions](../../../README.md#how-to-install)
and a GPU-specific target address identified locally. Do not copy a different
machine's PCI address or boot/root UUID. No machine-specific root installer
from the local experiment is included in this fork.

Sources: [upstream success-report requirements](https://github.com/Zagitta/gpu-unlock/discussions/238),
[Ada validation guidance](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/docs/testing.md),
[NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html),
[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
