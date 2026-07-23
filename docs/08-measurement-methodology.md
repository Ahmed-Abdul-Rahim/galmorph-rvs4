# Measurement methodology

## How per-layer counts are taken

Instruction counts come from the RISC-V **`instret`** CSR — a hardware counter of
retired instructions — read with `rdinstret` (see `profile.h`). `model_forward` reads
it before and after each layer; the delta is that layer's dynamic instruction count.

This is:
- **exact** (a hardware counter, not an estimate),
- **immune to inlining** (it counts what actually retired between two reads),
- **fast** (no per-instruction plugin/callback).

On a non-RISC-V host, `profile.h` falls back to the Linux `perf` interface
(`PERF_COUNT_HW_INSTRUCTIONS`); those numbers are real but not comparable to RISC-V
(different ISA/codegen). For the study, the target numbers were taken under
`qemu-riscv32` and cross-checked against VeeR-iSS.

## Run configuration

Everything is measured at **`VLEN = 256`**:
```
qemu-riscv32 -cpu rv32,v=true,vlen=256,elen=32  ./galaxy_app <image.bin>
```
This matches the VeeR config (`bytes_per_vec = 32`). Running at QEMU's 128-bit default
would silently process half of each 32-lane `m4` group and give wrong results.

The reported figure per optimization is the **least of `-O0`, `-O2`, `-O3`** builds.

## The surprising finding: cross-lane vector ops are expensive in QEMU

A diagnostic that bracketed a single channel's 4096-step scan showed **~3100 retired
instructions per timestep**, even though the compiled inner loop is only ~16
instructions. Isolating pieces revealed that QEMU models **cross-lane** vector
instructions — the ordered reduction `vfredosum.vs` and the permute `vslidedown` — as
long *sequential* helper chains (~3100 and ~900 retired instructions respectively for a
32-element `m4` group), while per-lane arithmetic is cheap.

Consequences, all confirmed by experiment (see [`07-negative-results.md`](07-negative-results.md)):
- A `vslidedown` tree reduction (opt #16) is **worse** than `vfredosum` (opt #14).
- Reducing through memory + a scalar sum (opt #17/#18) does **not** beat `vfredosum`.
- So the ordered `vfredosum.vs` is the **cheapest reduction available**, and opt #14
  keeps it.

This is a property of the *simulator's* cost model, not of real silicon — on hardware a
`vfredosum` is a handful of cycles. It is documented here so the numbers are
interpreted correctly.
