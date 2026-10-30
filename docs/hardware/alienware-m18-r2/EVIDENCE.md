# Evidence Index and Interpretation

This index accompanies the [hardware validation report](README.md).
Measurements were collected locally on one Alienware m18 R2 on 2026-10-30.
The published files are sanitized observations and derived summaries, not an
independent certification of the hardware or the unlock implementation.

## Primary experiment

The primary comparison is an A/B/A sequence: `unlocked` (A1),
`stock-rtd3off` (B), and `unlocked-repeat` (A2). Runtime D3 was disabled
throughout. Both transitions used normal reboots. The stock run was accepted
only when the driver's rate selector reported the stock state.

| Claim or check | Supporting evidence | Interpretation |
|---|---|---|
| Matching test programs and environment | Each run's `status.json` in [matched-v2](evidence/matched-v2/) | Contains four binary hashes, collection-source hashes, kernel, driver, GPU model, power policy, and completion status. |
| Rate selectors change with boot state | `rates_before` and `rates_after` in each `status.json` | fmla32/imla2/imla4 are 0/1/0 across A1/B/A2 and remain unchanged within each run. |
| Instruction throughput | Each run's `mma.txt` and `tensor-modes.json` | Includes every repetition. Floating-point rows use TFLOPS; integer rows use TOPS. |
| Matrix-multiplication performance | Each run's `gemm.jsonl` | Six shape/precision combinations, five repetitions each; tables use the within-run median. |
| Native arithmetic correctness | Each run's `numerical.txt`, `gemm.jsonl`, and `tensor-modes.json` | Seeded reference checks, whole-output large GEMM checks, and explicit instruction probes. |
| PyTorch arithmetic and synthetic workloads | Each run's `torch-checks.json` | 128 GEMM cases, four FlashAttention forward/backward cases, two 50-step MLP runs, and three checker negative controls. |
| Equality of recorded outputs across states | [comparison.json](evidence/matched-v2/comparison.json) and the three `torch-checks.json` files | Compares GEMM and attention output hashes, final weight-hash lists, and training loss sequences. Gradient hashes were not collected. |
| Ordinary default-entry startup | [default-boot.json](evidence/matched-v2/default-boot.json) | Sanitized local verification record of boot arguments, four successful passes, and successful handoff; raw boot identifiers are omitted. |
| Temperature, clocks, power, utilization | Each run's `telemetry.csv` | Approximately 100 ms sampling during the full collector; samples are not aligned to individual kernels. |
| No observed Xid/API mismatch during collection | `kernel_gpu_errors: 0` in each `status.json` and the collector source | Records the result of a local kernel-journal query, not a public copy of the full journal. |

Recompute the comparison and verify the evidence using the commands in the
[reproduction guide](../../../diagnostics/alienware_m18_r2/README.md).

## Additional validation

| Validation | Evidence | Scope |
|---|---|---|
| Continuous GEMM load | [torch-soak.json](evidence/extended-unlocked/torch-soak.json), [continuous-summary.json](evidence/extended-unlocked/continuous-summary.json) | 600.24 s, 928 batches; each batch performs 512 GEMMs and checks the final output matrix. Intermediate outputs are not checked separately. |
| CUDA Compute Sanitizer | [sanitizer-summary.json](evidence/extended-unlocked/sanitizer-summary.json) and `cuda-*.log` in [extended-unlocked](evidence/extended-unlocked/) | memcheck with leak checking, initcheck, and synccheck over six native seeded GEMMs; not the entire PyTorch suite. |
| Kernel-module compilation | [build-142.log](evidence/extended-unlocked/build-142.log), [build-147.log](evidence/extended-unlocked/build-147.log) | Isolated builds against installed headers; only kernel 6.8.0-147 was boot-tested. Logs retain compiler-name and unavailable-vmlinux/BTF messages. |
| NVIDIA input provenance | [firmware-inputs-verified.json](evidence/extended-unlocked/firmware-inputs-verified.json) | SHA-256 records for the two pinned inputs; the inputs and generated firmware are not redistributed here. |
| Toolchain versions | [tool-versions.json](evidence/extended-unlocked/tool-versions.json) and run manifests | Compiler and sanitizer versions; PyTorch/CUDA versions also appear in numerical reports. |

The earlier [legacy-stock](evidence/legacy-stock/) and
[legacy-unlocked](evidence/legacy-unlocked/) runs had different RTD3 policies.
They remain available for traceability but are not the primary matched
comparison. The [legacy loop test](evidence/legacy-loop-soak/status.json)
included CPU-validation gaps and is distinct from the later continuous-load test.

## Integrity, provenance, and limitations

[SHA256SUMS.json](evidence/SHA256SUMS.json) covers the public evidence files,
excluding the manifest itself. The integrity tests check the exact file set
and hashes. Collection-source hashes are checked against the included scripts;
vendored source hashes are checked against
[PROVENANCE.json](../../../diagnostics/alienware_m18_r2/vendor/PROVENANCE.json).
The summarizer was added after collection and does not alter recorded data.

Sanitization replaces GPU UUIDs, PCI locations, home/workspace paths, and boot
UUIDs with descriptive placeholders. Numeric results are retained. Raw local
boot logs, unrelated application logs, installers, and signing material are
excluded. The publication documents use English; measured values and original
English tool output are preserved.

The study has one stock boot and two unlocked boots. Repetitions within a
boot are not independent boot-level samples. Dynamic clocks, desktop activity,
and unequal initial temperatures limit timing comparisons. Hash equality is
evidence for the recorded outputs only. The sampled thermal measurements and
ten-minute run do not establish long-term reliability or energy efficiency.
Untested configurations and recovery scenarios are listed in the
[remaining test matrix](README.md#remaining-test-matrix).
